# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference import envs
from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import VllmSchedulerOutput

    from tpu_inference.runner.input_batch import CachedRequestState

_SAMPLING_EPS = 1e-5

# `_apply_sampling_transforms` writes this in place of every logit that top-k
# or top-p removed, so "the sampler could have drawn this token" is exactly
# "its processed logit is > _MASKED_LOGIT". Real logits never come near it.
_MASKED_LOGIT = -1e12

# Written into the slots of a `[num_reqs, width]` sampling-mask row that the
# request did not fill. Must match `UNSET_SAMPLING_MASK_ID` on the trainer side
# (tunix `rl/common.py`), which reads these rows back to replay the sampler's
# support set.
SAMPLING_MASK_UNSET_ID = -1


@dataclass
class PromptLogprobsReqSnap:
    """Per-request state snapshotted at step N for use in get_output()."""
    req_id: str
    req_state: "CachedRequestState"  # Stable request state reference; CPU buffer is pre-allocated.
    req_offset: int  # Absolute row index into the full-batch logprobs tensor.
    start_idx: int  # Number of computed tokens.
    num_logits: int  # Number of rows to copy from the TPU tensor to the CPU accumulator.
    is_last_chunk: bool  # True if this is the final chunk of the prompt logprobs.
    num_k: int  # Number of top logprobs to retain for this request.


@dataclass
class PromptLogprobsAsyncData:
    """Holds async-copied prompt logprob tensors + per-request snapshots for get_output()."""
    tensors: LogprobsTensors  # Result of _jax_logprobs_copy_to_host_async (pending transfer).
    req_snaps: List[PromptLogprobsReqSnap]


def _jax_logprobs_copy_to_host_async(
        logprobs_tensors: LogprobsTensors) -> LogprobsTensors:
    """Initiate non-blocking TPU-to-host copies for all logprobs arrays."""
    return LogprobsTensors(
        logprob_token_ids=jax.copy_to_host_async(
            logprobs_tensors.logprob_token_ids),
        logprobs=jax.copy_to_host_async(logprobs_tensors.logprobs),
        selected_token_ranks=jax.copy_to_host_async(
            logprobs_tensors.selected_token_ranks),
    )


def _transform_rows(
    logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
    top_p: jax.Array,
) -> jax.Array:
    """Temperature, top-k and top-p over whatever rows it is handed.

    Both masks are computed unconditionally and then selected with `where`, so
    a request with `top_k == 0` or `top_p == 1.0` costs exactly as much as one
    that filters -- the branches are per-row data, not per-row work.
    """
    # Temperature scaling
    temperatures = temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits = logits / temperatures

    # Only apply top-k masking if k > 0 for each token
    should_apply_topk = jnp.expand_dims(top_k > 0, axis=-1)
    topk_masked = topk_mask(logits, top_k, replace_val=_MASKED_LOGIT)
    logits = jnp.where(should_apply_topk, topk_masked, logits)

    # Only apply top-p masking if p < 1.0 for each token
    should_apply_topp = jnp.expand_dims(top_p < 1.0, axis=-1)
    topp_masked = topp_mask(logits, top_p, replace_val=_MASKED_LOGIT)
    logits = jnp.where(should_apply_topp, topp_masked, logits)

    return logits


# Chunking stops paying at large batches: at num_reqs=128 the unchunked
# reduction already amortizes well enough that the `lax.map` is a net loss
# (10.06 -> 11.50 ms with a chunk of 16, v6e / vocab 262144), while at 32 and
# 64 it is worth 2.5x and 1.4x. Since that ceiling is a property of the batch
# and not of the operator's chunk size, it lives here rather than in the env
# var -- leaving `SAMPLING_MICROBATCH_SIZE` at its default is meant to be safe
# at every batch, not just the ones it was tuned for.
_SAMPLING_MICROBATCH_MAX_NUM_REQS = 128


def _sampling_microbatch_size(num_reqs: int) -> Optional[int]:
    """Rows per chunk for the transforms, or None to run the batch in one go.

    Declines in four cases: the knob is off; the batch already fits in one
    chunk, so a `lax.map` would only add a loop; the chunk does not divide the
    batch, which would need a second padded program for the remainder; or the
    batch is large enough that chunking is a measured loss. `num_reqs` is the
    padded bucket, which is a power of two, so a power-of-two chunk divides
    every bucket above it.
    """
    microbatch_size = envs.SAMPLING_MICROBATCH_SIZE
    if microbatch_size <= 0 or num_reqs <= microbatch_size:
        return None
    if num_reqs >= _SAMPLING_MICROBATCH_MAX_NUM_REQS:
        return None
    if num_reqs % microbatch_size:
        return None
    return microbatch_size


def _apply_sampling_transforms(
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    """Apply temperature scaling, top-k, and top-p filtering to logits.

    This extracts the common logit processing logic used by both the sampling
    path and the processed-logprobs path so that the transformations are
    applied identically.

    With `SAMPLING_MICROBATCH_SIZE` set, the batch is fed through the
    transforms a fixed number of rows at a time instead of all at once. top-k
    and top-p are each a 31-iteration binary search that reduces over the whole
    `[num_reqs, vocab]` array on every iteration, and past a certain batch the
    working set stops fitting: on v6e at a 262k vocab, `sample()` costs 1.37 ms
    at num_reqs=16 and 8.24 ms at num_reqs=32 -- 6x the time for 2x the work.
    Chunking holds the searches in the efficient regime (measured 41-44% ->
    64% HBM utilization on the two reduction fusions) and makes the cost linear
    in num_reqs again: with a chunk of 16, 2.5x at num_reqs=32 and 1.4x at 64
    over the whole sample-plus-mask path. Outside that band it is a loss, so
    `_sampling_microbatch_size` declines rather than chunking; see
    `SAMPLING_MICROBATCH_SIZE` in `envs.py` for the measured table.

    This is a pure throughput change. Chunking stops at the transforms;
    `jax.random.categorical` in `sample()` still draws once over the full batch
    from the unsplit key, so the RNG stream and every sampled token are
    bit-identical to the unchunked path (verified at num_reqs 16/32/64/128).
    Chunking the draw as well was measured at parity -- it buys nothing and
    would change every token.

    Args:
        logits: (B, vocab_size) raw logits in float32.
        tpu_sampling_metadata: Sampling parameters (temperature, top_k, top_p).

    Returns:
        Processed logits with temperature, top-k, and top-p applied.
    """
    temperature = tpu_sampling_metadata.temperature
    top_k = tpu_sampling_metadata.top_k
    top_p = tpu_sampling_metadata.top_p

    num_reqs, vocab_size = logits.shape
    microbatch_size = _sampling_microbatch_size(num_reqs)
    if microbatch_size is None:
        return _transform_rows(logits, temperature, top_k, top_p)

    num_chunks = num_reqs // microbatch_size
    chunked = jax.lax.map(
        lambda xs: _transform_rows(*xs),
        (logits.reshape(num_chunks, microbatch_size, vocab_size),
         temperature.reshape(num_chunks, microbatch_size),
         top_k.reshape(num_chunks, microbatch_size),
         top_p.reshape(num_chunks, microbatch_size)))
    return chunked.reshape(num_reqs, vocab_size)


@jax.jit(static_argnames=["mesh"])
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata._cache_collision_dummy is not None:
        # Force a dependency on the dummy tensor's shape to ensure unique HLO.
        logits = logits + 0 * jnp.sum(
            tpu_sampling_metadata._cache_collision_dummy)

    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        # TODO(gxd3): revisit if the 2nd dimension of the logits can be sharded
        # instead of being replicated.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))

    greedy_tokens = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)
    if not tpu_sampling_metadata.do_sampling:
        ret_tokens = greedy_tokens
        ret_logits = logits
    else:
        processed_logits = _apply_sampling_transforms(logits,
                                                      tpu_sampling_metadata)
        # (batch_size,)
        next_tokens = jax.random.categorical(rng, processed_logits)
        # Note: avoid using the sample result when temperature < _SAMPLING_EPS
        # If temperature < 0, logits /= temperatures will flip the result, causing error.
        is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS
        ret_tokens = jnp.where(is_greedy, greedy_tokens, next_tokens)
        ret_logits = jnp.where(jnp.expand_dims(is_greedy, axis=-1), logits,
                               processed_logits)
    # Replicate the result so that in multi-controller jax setup
    # (i.e. Ray based multi-host setup), we won't hit error like
    # RuntimeError: Fetching value for `jax.Array` that spans non-addressable
    # (non process local) devices is not possible.
    next_tokens = jax.lax.with_sharding_constraint(ret_tokens,
                                                   NamedSharding(mesh, P()))
    return next_tokens, ret_logits


@jax.jit(static_argnames=["width"])
def compute_sampling_mask(
    processed_logits: jax.Array,
    width: int,
) -> tuple[jax.Array, jax.Array]:
    """Extract the token support set that `sample()` actually drew from.

    `sample()` returns the post-transform logits, in which top-k and top-p have
    replaced every rejected token with `_MASKED_LOGIT`. The surviving ids are
    the whole of the sampler's action space: `jax.random.categorical`
    normalises over them and nothing else. Handing that id list to a trainer
    lets it re-normalise over the identical subspace, which is what makes the
    importance ratio exact rather than a ratio between two different
    normalisers.

    Args:
        processed_logits: (num_reqs, vocab_size) the second return value of
            `sample()`. Rows that were sampled greedily hold raw logits and so
            keep the entire vocabulary; they always report overflow.
        width: static bound on the kept-set size. vLLM already refuses
            `return_sampling_mask` unless every request sets `top_k > 0`, so a
            finite bound exists -- but it is not `top_k` itself. `topk_mask`
            thresholds (`x >= the k-th largest value`) rather than selecting
            exactly k, so ties at the cutoff keep MORE than k; see
            `SAMPLING_MASK_WIDTH` in `envs.py` for the measured overshoot and
            the "size it above the largest top_k" rule. Nothing here reads
            `top_k`: the kept set is whatever survived the `_MASKED_LOGIT`
            fill, so per-request `top_k`, top-p narrowing and tie inflation all
            come out right on their own.

    Returns:
        (mask_ids, overflow).
        `mask_ids` is (num_reqs, width) int32 holding the kept ids, padded with
        `SAMPLING_MASK_UNSET_ID`. `overflow` is (num_reqs,) bool, True where the
        row kept strictly more than `width` ids -- i.e. where `mask_ids` is a
        *subset* of the real support and replaying it would silently normalise
        over the wrong space. Callers must raise on it, not truncate.
    """
    block = _sampling_mask_block_size(processed_logits.shape[-1], width)
    if block is None:
        # Vocabulary too small (or awkwardly shaped) for the blocked form to
        # have enough groups. A direct top_k over a few thousand entries is
        # cheap anyway; the blocked path exists for real 128k-262k vocabs.
        values, ids = _support_by_full_topk(processed_logits, width)
    else:
        values, ids = _support_by_blocks(processed_logits, width, block)
    kept = values > _MASKED_LOGIT
    mask_ids = jnp.where(kept[:, :width], ids[:, :width].astype(jnp.int32),
                         SAMPLING_MASK_UNSET_ID)
    return mask_ids, kept[:, width]


def _sampling_mask_block_size(vocab_size: int, width: int) -> Optional[int]:
    """Largest block that tiles `vocab_size` and leaves > `width` groups.

    The blocked search needs at least `width + 1` groups (see
    `_support_by_blocks`) and a block that divides the vocabulary exactly, so
    that `reshape` stays a view instead of forcing a pad over the largest array
    in the sampler. Real vocabularies are padded to a multiple of 128, so the
    first candidate almost always wins.
    """
    for block in (128, 64, 32, 16, 8):
        if vocab_size % block == 0 and vocab_size // block >= width + 1:
            return block
    return None


def _support_by_full_topk(
    processed_logits: jax.Array,
    width: int,
) -> tuple[jax.Array, jax.Array]:
    # width + 1 so the caller can tell "kept exactly width" from "kept more
    # than width"; the extra column is the only way to see the overflow.
    return jax.lax.top_k(processed_logits, width + 1)


def _support_by_blocks(
    processed_logits: jax.Array,
    width: int,
    block: int,
) -> tuple[jax.Array, jax.Array]:
    """The same `(values, ids)` a full-vocab top_k gives, ~11x cheaper.

    `lax.top_k(processed_logits, width + 1)` over a 262k vocabulary costs 15.6
    ms at num_reqs=64 on v6e -- roughly three times the entire sampler it hangs
    off. It is answering a much harder question than we asked: we do not want
    the vocabulary ranked, we want the <= width ids that survived top-k/top-p's
    `_MASKED_LOGIT` fill. That is a stream compaction of a very sparse boolean,
    and it can be done in one pass plus two tiny top_ks.

    Cut the vocabulary into `groups` blocks of `block`. A survivor's block has
    `max > _MASKED_LOGIT`, so only such blocks can contribute; take the
    `width + 1` largest block maxima (a top_k over `groups`, 2048 rather than
    262144), gather just those blocks, and rank inside the resulting
    `[num_reqs, (width + 1) * block]` candidate set.

    Overflow survives the shortcut. Let B be the number of blocks whose max is
    above the fill:
      B <= width + 1  every survivor-bearing block is selected, so the
                      candidate set holds the support exactly.
      B >  width + 1  each of the width + 1 selected blocks holds at least one
                      survivor, so the candidate set holds at least width + 1
                      of them and the (width + 1)-th value is above the fill --
                      overflow fires, which is the right answer precisely when
                      more than `width` ids survived.
    Verified against the full top_k on v6e at num_reqs 8/16/32/64: identical id
    sets and identical overflow, including the greedy full-vocabulary rows.
    """
    num_reqs, vocab_size = processed_logits.shape
    blocks = processed_logits.reshape(num_reqs, vocab_size // block, block)

    # One pass over the big array; everything after this is tiny.
    block_max = jnp.max(blocks, axis=-1)
    _, block_ids = jax.lax.top_k(block_max, width + 1)

    candidates = jnp.take_along_axis(blocks, block_ids[:, :, None], axis=1)
    candidate_ids = (block_ids[:, :, None] * block +
                     jnp.arange(block, dtype=jnp.int32))

    values, positions = jax.lax.top_k(candidates.reshape(num_reqs, -1),
                                      width + 1)
    ids = jnp.take_along_axis(candidate_ids.reshape(num_reqs, -1), positions,
                              axis=1)
    return values, ids


def compute_logprobs(logits: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)


@jax.jit(static_argnames=("max_logprobs", ))
def compute_and_gather_logprobs(
    logits: jax.Array,
    next_tokens: jax.Array,
    max_logprobs: int,
) -> LogprobsTensors:
    """Compute logprobs from logits and gather the requested top-k."""
    logprobs = compute_logprobs(logits)
    return gather_logprobs(logprobs, next_tokens, max_logprobs)


@jax.jit(static_argnames=("max_logprobs", ))
def compute_and_gather_prompt_logprobs(
    logits: jax.Array,
    input_ids: jax.Array,
    max_logprobs: int,
) -> LogprobsTensors:
    """Compute logprobs from full logits and gather the requested top-k for prompt tokens."""
    prompt_target_ids = jnp.roll(input_ids, -1, axis=0)
    return compute_and_gather_logprobs(logits, prompt_target_ids, max_logprobs)


def compute_prompt_logprobs(
    full_logits: Optional[jax.Array],
    input_ids: Optional[jax.Array],
    num_prompt_logprobs: Dict[str, int],
    requests: Dict[str, "CachedRequestState"],
    scheduler_output: "VllmSchedulerOutput",
    req_ids_dp: Optional[Dict[int, List[str]]],
    dp_size: int,
    max_logprobs: int,
) -> Optional[PromptLogprobsAsyncData]:
    """Dispatches prompt logprob computation on TPU and snapshots per-request state.
    Returns PromptLogprobsAsyncData containing the async-copied tensors and
    the snapshotted state needed to safely slice them in get_output().
    """
    if (not num_prompt_logprobs or full_logits is None or input_ids is None):
        return None

    # Gather compact [total_padded_tokens, max_logprobs+1] tensors on TPU and
    # start async transfer to host (overlaps with next step's execute_model).
    # We use the statically precompiled max_logprobs instead of the dynamic user max_k
    # to avoid triggering JAX recompilation. The correct num_k is preserved in req_snaps.
    prompt_lp_tensors = compute_and_gather_prompt_logprobs(
        full_logits, input_ids, max_logprobs)
    prompt_lp_tensors = _jax_logprobs_copy_to_host_async(prompt_lp_tensors)

    # Snapshot all mutable per-request state before update_states(N+1) runs.
    padded_tokens_per_dp = full_logits.shape[0] // dp_size
    req_snaps: List[PromptLogprobsReqSnap] = []
    if req_ids_dp:
        for dp_rank, req_id_list in req_ids_dp.items():
            dp_token_offset = dp_rank * padded_tokens_per_dp
            local_token_offset = 0
            for req_id in req_id_list:
                num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
                if req_id in num_prompt_logprobs:
                    num_k = num_prompt_logprobs[req_id]
                    req_state = requests[req_id]
                    start_idx = req_state.num_computed_tokens
                    num_remaining = req_state.num_prompt_tokens - (start_idx +
                                                                   1)
                    if num_scheduled <= num_remaining:
                        num_logits = num_scheduled
                        is_last_chunk = False
                    else:
                        num_logits = num_remaining
                        is_last_chunk = True
                    req_snaps.append(
                        PromptLogprobsReqSnap(
                            req_id=req_id,
                            req_state=req_state,
                            req_offset=dp_token_offset + local_token_offset,
                            start_idx=start_idx,
                            num_logits=num_logits,
                            is_last_chunk=is_last_chunk,
                            num_k=num_k,
                        ))
                local_token_offset += num_scheduled

    return PromptLogprobsAsyncData(tensors=prompt_lp_tensors,
                                   req_snaps=req_snaps)


def gather_logprobs(
    logprobs: jax.Array,
    token_ids: jax.Array,
    num_logprobs: int,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
        num_logprobs: minimum number of logprobs to
                    retain per token


    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    # Find the topK values.
    topk_logprobs, topk_indices = jax.lax.top_k(logprobs, k=num_logprobs)

    # Get with the logprob of the prompt or sampled token.
    token_ids = jnp.expand_dims(token_ids, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, token_ids, axis=-1)

    # Compute the ranks of the actual token.
    token_ranks = jnp.sum(logprobs >= token_logprobs, axis=-1)

    # Concatenate together with the topk.
    indices = jnp.concatenate((token_ids, topk_indices), axis=1)
    logprobs = jnp.concatenate((token_logprobs, topk_logprobs), axis=1)

    # Use int32 to reduce the tensor size.
    indices = jnp.int32(indices)

    return LogprobsTensors(indices, logprobs, token_ranks)
