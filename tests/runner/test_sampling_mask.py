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
"""Tests for the keep-sampling mask producer (vLLM `--return-sampling-mask`).

The mask exists so a trainer can re-normalise over the exact token subspace
`jax.random.categorical` drew from. Three properties make that work, and each
is pinned below:

  EXACT      -- the ids are read off the post-top-k/top-p logits, so they are
                the sampler's support set, not a reconstruction of it.
  BOUNDED    -- a row that kept more ids than the static width reports
                overflow, because a truncated mask is a *different*
                distribution and is worse than no mask at all.
  ALIGNED    -- the CSR triple indexes the way vLLM's scheduler slices it,
                including requests that generated nothing this step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vllm.v1.outputs import SamplingMaskLists

from tpu_inference import envs
from tpu_inference.layers.jax.sample.sampling import (
    SAMPLING_MASK_UNSET_ID, _apply_sampling_transforms, _sampling_mask_block_size,
    _sampling_microbatch_size, _support_by_blocks, _support_by_full_topk,
    compute_sampling_mask)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.runner.tpu_runner import _sampling_masks_materialize


def _kept(row):
    row = np.asarray(row)
    return set(row[row != SAMPLING_MASK_UNSET_ID].tolist())


class TestComputeSamplingMask:

    def test_ids_are_the_topk_support(self):
        logits = jnp.array([[0.0, 5.0, 1.0, 4.0, 2.0, 3.0]],
                           dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.array([1.0], dtype=jnp.float32),
            top_k=jnp.array([3], dtype=jnp.int32),
            top_p=jnp.array([1.0], dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)

        mask_ids, overflow = compute_sampling_mask(processed, width=4)

        # ids 1, 3, 5 hold the three largest logits.
        assert _kept(mask_ids[0]) == {1, 3, 5}
        assert not bool(overflow[0])
        # The unused column is filled, not left as a stale id.
        assert int(np.sum(np.asarray(mask_ids[0]) == SAMPLING_MASK_UNSET_ID)) == 1

    def test_topp_narrows_the_topk_support(self):
        # Softmax over [4, 3, 2, 1, 0] puts ~0.64 on id 0 and ~0.24 on id 1,
        # so top_p=0.8 keeps only those two even though top_k allows four.
        logits = jnp.array([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.array([1.0], dtype=jnp.float32),
            top_k=jnp.array([4], dtype=jnp.int32),
            top_p=jnp.array([0.8], dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)

        mask_ids, overflow = compute_sampling_mask(processed, width=4)

        assert _kept(mask_ids[0]) == {0, 1}
        assert not bool(overflow[0])

    def test_mask_matches_the_sampled_distribution(self):
        # The point of the mask: renormalising over the kept ids reproduces
        # the sampler's own probabilities, and the rejected ids are exactly
        # the ones that carry zero probability.
        rng = np.random.default_rng(0)
        logits = jnp.asarray(rng.normal(size=(4, 128)), dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.full((4, ), 0.9, dtype=jnp.float32),
            top_k=jnp.full((4, ), 8, dtype=jnp.int32),
            top_p=jnp.full((4, ), 1.0, dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)
        probs = np.asarray(jax.nn.softmax(processed, axis=-1))

        mask_ids, _ = compute_sampling_mask(processed, width=8)

        for row in range(4):
            kept = _kept(mask_ids[row])
            assert kept == set(np.flatnonzero(probs[row] > 0.0).tolist())
            np.testing.assert_allclose(probs[row, sorted(kept)].sum(),
                                       1.0,
                                       rtol=1e-5)

    def test_ties_keep_more_than_k(self):
        # `topk_mask` thresholds at the k-th largest VALUE (`x >= cutoff`), it
        # does not select exactly k entries. Four ids share the 2nd-largest
        # logit, so top_k=2 keeps five, not two. The mask must carry all five:
        # `categorical` really can draw any of them, so dropping one would
        # normalise over a smaller space than the sampler used.
        logits = jnp.array([[9.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0,
                             0.0]],
                           dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.array([1.0], dtype=jnp.float32),
            top_k=jnp.array([2], dtype=jnp.int32),
            top_p=jnp.array([1.0], dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)

        mask_ids, overflow = compute_sampling_mask(processed, width=8)

        assert _kept(mask_ids[0]) == {0, 1, 2, 3, 4}
        assert not bool(overflow[0])
        # And a width sized to `top_k` itself would have lost two of them.
        _, tight_overflow = compute_sampling_mask(processed, width=2)
        assert bool(tight_overflow[0])

    def test_per_request_top_k_may_differ_within_a_batch(self):
        # Nothing in the producer reads `top_k`; it reads the -1e12 fill. So
        # rows with different `top_k` in one batch each get their own width,
        # padded out to the static one.
        logits = jnp.tile(
            jnp.asarray(np.arange(16, dtype=np.float32))[None, :], (3, 1))
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.ones((3, ), dtype=jnp.float32),
            top_k=jnp.array([1, 4, 7], dtype=jnp.int32),
            top_p=jnp.ones((3, ), dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)

        mask_ids, overflow = compute_sampling_mask(processed, width=8)

        assert [len(_kept(mask_ids[r])) for r in range(3)] == [1, 4, 7]
        assert not bool(np.any(np.asarray(overflow)))

    def test_overflow_when_support_exceeds_width(self):
        logits = jnp.asarray(np.arange(32, dtype=np.float32)[None, :])
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.array([1.0], dtype=jnp.float32),
            top_k=jnp.array([10], dtype=jnp.int32),
            top_p=jnp.array([1.0], dtype=jnp.float32),
            do_sampling=True,
        )
        processed = _apply_sampling_transforms(logits, metadata)

        _, overflow = compute_sampling_mask(processed, width=4)
        assert bool(overflow[0])

        _, overflow = compute_sampling_mask(processed, width=10)
        assert not bool(overflow[0])

    def test_greedy_row_reports_overflow(self):
        # `sample()` hands back raw logits for greedy rows, so the whole
        # vocabulary is "kept". vLLM refuses temperature <= 0 alongside
        # return_sampling_mask; this makes the leak loud if it ever arrives.
        logits = jnp.asarray(np.arange(32, dtype=np.float32)[None, :])
        _, overflow = compute_sampling_mask(logits, width=8)
        assert bool(overflow[0])


class TestBlockedSupportSearch:
    """The blocked search must be indistinguishable from the full top_k.

    `compute_sampling_mask` does not rank the vocabulary; it compacts the <=
    width ids that survived the `_MASKED_LOGIT` fill, by taking the `width + 1`
    highest-max blocks and ranking only inside those. That is ~11x cheaper at a
    262k vocabulary but it is only worth having if it is exactly equivalent,
    overflow included -- a mask that quietly drops one id makes the trainer
    normalise over the wrong support, which is the one failure this whole
    feature exists to prevent.

    The tests above all run tiny vocabularies, which take the fallback path, so
    without this class the blocked search is never executed.
    """

    # 2048 % 128 == 0 and 2048 // 128 = 16 > width + 1, so this really does
    # take the blocked path rather than falling back.
    VOCAB = 2048
    WIDTH = 8

    def _processed(self, seed, num_reqs, top_k, top_p=1.0):
        rng = np.random.default_rng(seed)
        logits = jnp.asarray(
            rng.normal(size=(num_reqs, self.VOCAB)).astype(np.float32))
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.ones((num_reqs, ), dtype=jnp.float32),
            top_k=jnp.full((num_reqs, ), top_k, dtype=jnp.int32),
            top_p=jnp.full((num_reqs, ), top_p, dtype=jnp.float32),
            do_sampling=True,
        )
        return _apply_sampling_transforms(logits, metadata)

    def test_this_shape_actually_takes_the_blocked_path(self):
        assert _sampling_mask_block_size(self.VOCAB, self.WIDTH) == 128
        # And the real one: Gemma4's 262144 over the default width of 128.
        assert _sampling_mask_block_size(262144, 128) == 128

    def test_falls_back_when_there_are_too_few_groups(self):
        # 6 ids cannot be cut into 5 blocks of any supported size.
        assert _sampling_mask_block_size(6, 4) is None
        # Divisible by 128, but 1024 // 128 = 8 groups cannot hold width + 1.
        assert _sampling_mask_block_size(1024, 128) is None

    @pytest.mark.parametrize("top_k", [1, 4, 8, 9, 64])
    def test_matches_the_full_topk_ids_and_overflow(self, top_k):
        processed = self._processed(seed=top_k, num_reqs=4, top_k=top_k)

        ref_values, ref_ids = _support_by_full_topk(processed, self.WIDTH)
        got_values, got_ids = _support_by_blocks(processed, self.WIDTH, 128)

        ref_kept = np.asarray(ref_values) > -1e12
        got_kept = np.asarray(got_values) > -1e12
        # Overflow is the last column, and it is the reason for width + 1.
        np.testing.assert_array_equal(ref_kept[:, self.WIDTH],
                                      got_kept[:, self.WIDTH])
        for row in range(processed.shape[0]):
            ref_set = set(np.asarray(ref_ids)[row][ref_kept[row]].tolist())
            got_set = set(np.asarray(got_ids)[row][got_kept[row]].tolist())
            assert ref_set == got_set

    def test_survivors_concentrated_in_one_block(self):
        # The correctness argument turns on "at most width + 1 blocks can hold
        # a survivor". Force the opposite extreme: every survivor inside a
        # single block, so one selected block carries the whole support and the
        # other width blocks are pure fill.
        logits = np.full((1, self.VOCAB), -30.0, dtype=np.float32)
        logits[0, 300:305] = np.arange(5, dtype=np.float32)
        processed = _apply_sampling_transforms(
            jnp.asarray(logits),
            TPUSupportedSamplingMetadata(
                temperature=jnp.ones((1, ), dtype=jnp.float32),
                top_k=jnp.array([5], dtype=jnp.int32),
                top_p=jnp.ones((1, ), dtype=jnp.float32),
                do_sampling=True,
            ))

        mask_ids, overflow = compute_sampling_mask(processed, width=self.WIDTH)

        assert _kept(mask_ids[0]) == {300, 301, 302, 303, 304}
        assert not bool(overflow[0])

    def test_overflow_still_fires_when_support_exceeds_width(self):
        # More survivors than `width`, spread wide enough that more than
        # width + 1 blocks contain one. The blocked search can only see
        # width + 1 of those blocks, and must still report overflow rather than
        # hand back the subset it can see.
        logits = np.full((1, self.VOCAB), -30.0, dtype=np.float32)
        logits[0, ::64] = 1.0  # 32 survivors, one per block, 32 blocks
        processed = _apply_sampling_transforms(
            jnp.asarray(logits),
            TPUSupportedSamplingMetadata(
                temperature=jnp.ones((1, ), dtype=jnp.float32),
                top_k=jnp.array([32], dtype=jnp.int32),
                top_p=jnp.ones((1, ), dtype=jnp.float32),
                do_sampling=True,
            ))

        _, overflow = compute_sampling_mask(processed, width=self.WIDTH)
        assert bool(overflow[0])

        # Widened past the support, the same input must come back clean and
        # complete -- 2048 // 128 = 16 groups is still > 32 + 1, so this one
        # takes the fallback and the two paths meet on the same answer.
        mask_ids, overflow = compute_sampling_mask(processed, width=40)
        assert not bool(overflow[0])
        assert _kept(mask_ids[0]) == set(range(0, self.VOCAB, 64))

    def test_greedy_row_reports_overflow_on_the_blocked_path(self):
        # Raw logits, nothing filled, so every block is live and B is the whole
        # group count -- the B > width + 1 branch of the argument.
        logits = jnp.asarray(
            np.random.default_rng(0).normal(
                size=(2, self.VOCAB)).astype(np.float32))
        _, overflow = compute_sampling_mask(logits, width=self.WIDTH)
        assert bool(np.all(np.asarray(overflow)))


class TestSamplingMicrobatch:
    """`SAMPLING_MICROBATCH_SIZE` must be a pure throughput knob.

    Chunking exists because top-k and top-p each reduce over the whole
    `[num_reqs, vocab]` array 31 times and fall off a throughput cliff once the
    batch stops fitting. That is worth having only if it cannot be observed in
    the output: the chunked transforms have to produce the same processed
    logits, hence the same support set, hence the same mask, for every setting
    that is legal -- and quietly do nothing for every setting that is not.
    """

    VOCAB = 2048
    WIDTH = 8

    def _inputs(self, num_reqs, seed=0):
        rng = np.random.default_rng(seed)
        logits = jnp.asarray(
            rng.normal(size=(num_reqs, self.VOCAB)).astype(np.float32))
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.asarray(rng.uniform(0.5, 1.5, num_reqs),
                                    dtype=jnp.float32),
            # Per-request k and p, so a chunk cannot accidentally be handed
            # uniform parameters and pass for the wrong reason.
            top_k=jnp.asarray(rng.integers(1, 16, num_reqs), dtype=jnp.int32),
            top_p=jnp.asarray(rng.uniform(0.7, 1.0, num_reqs),
                              dtype=jnp.float32),
            do_sampling=True,
        )
        return logits, metadata

    @pytest.mark.parametrize("num_reqs,microbatch_size", [(16, 8), (16, 4),
                                                          (32, 8), (12, 4),
                                                          (16, 1)])
    def test_chunked_transforms_are_bit_identical(self, monkeypatch, num_reqs,
                                                  microbatch_size):
        logits, metadata = self._inputs(num_reqs)
        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", 0)
        reference = _apply_sampling_transforms(logits, metadata)

        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", microbatch_size)
        chunked = _apply_sampling_transforms(logits, metadata)

        # Bit-identical, not merely close: the mask is read off a `>` against
        # `_MASKED_LOGIT`, so a one-ulp drift on a boundary logit changes the
        # support set the trainer replays.
        assert np.array_equal(np.asarray(chunked), np.asarray(reference))

    def test_mask_is_unchanged_by_chunking(self, monkeypatch):
        logits, metadata = self._inputs(16, seed=7)
        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", 0)
        ref_ids, ref_overflow = compute_sampling_mask(
            _apply_sampling_transforms(logits, metadata), width=self.WIDTH)

        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", 4)
        ids, overflow = compute_sampling_mask(
            _apply_sampling_transforms(logits, metadata), width=self.WIDTH)

        assert [_kept(r) for r in ids] == [_kept(r) for r in ref_ids]
        assert np.array_equal(np.asarray(overflow), np.asarray(ref_overflow))

    @pytest.mark.parametrize("num_reqs,microbatch_size", [
        (16, 0),  # off
        (16, -1),  # nonsense
        (16, 16),  # one chunk is just the full batch with a loop around it
        (16, 32),  # larger than the batch
        (12, 8),  # not a divisor: the remainder would need its own program
        (128, 16),  # at the ceiling, where chunking is a measured loss
        (256, 16),  # and above it
    ])
    def test_settings_that_should_not_chunk(self, monkeypatch, num_reqs,
                                            microbatch_size):
        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", microbatch_size)
        assert _sampling_microbatch_size(num_reqs) is None

    def test_the_default_chunks_exactly_the_band_it_was_tuned_for(
            self, monkeypatch):
        # Resolve the declared default rather than reading the attribute, so a
        # developer with SAMPLING_MICROBATCH_SIZE exported does not see this
        # fail. The default being safe at *every* batch is the reason it is on
        # rather than opt-in, so it is worth pinning.
        monkeypatch.delenv("SAMPLING_MICROBATCH_SIZE", raising=False)
        default = envs.environment_variables["SAMPLING_MICROBATCH_SIZE"]()
        assert default == 16

        monkeypatch.setattr(envs, "SAMPLING_MICROBATCH_SIZE", default)
        assert _sampling_microbatch_size(32) == 16
        assert _sampling_microbatch_size(64) == 16
        # Below the chunk there is nothing to chunk; at and above 128 the
        # unchunked reduction already amortizes and `lax.map` is a net loss.
        assert _sampling_microbatch_size(16) is None
        assert _sampling_microbatch_size(8) is None
        assert _sampling_microbatch_size(128) is None


class TestSamplingMasksMaterialize:

    def _mask_ids(self, rows, width):
        out = np.full((len(rows), width), SAMPLING_MASK_UNSET_ID,
                      dtype=np.int32)
        for i, row in enumerate(rows):
            out[i, :len(row)] = row
        return jnp.asarray(out)

    def test_csr_round_trips_to_the_input_rows(self):
        rows = [[7, 9], [1, 2, 3], [4]]
        masks = _sampling_masks_materialize(
            self._mask_ids(rows, width=4),
            jnp.zeros(3, dtype=bool),
            valid_sampled_token_ids=[[7], [2], [4]],
            logits_indices_selector=None,
            num_reqs=3,
        )
        assert isinstance(masks, SamplingMaskLists)
        assert masks.to_nested_list() == rows
        assert masks.cu_num_generated_tokens == [0, 1, 2, 3]
        # Each request slices back to its own row.
        for req_idx, row in enumerate(rows):
            assert masks.slice_request(req_idx, 1).to_nested_list() == [row]

    def test_requests_that_generated_nothing_are_skipped(self):
        # A partial-prefill row has its sampled token cleared. It must not
        # consume an `offsets` entry, but it still has to advance
        # `cu_num_generated_tokens` by zero so later requests slice correctly.
        rows = [[7, 9], [1, 2, 3], [4]]
        masks = _sampling_masks_materialize(
            self._mask_ids(rows, width=4),
            jnp.zeros(3, dtype=bool),
            valid_sampled_token_ids=[[7], [], [4]],
            logits_indices_selector=None,
            num_reqs=3,
        )
        assert masks.to_nested_list() == [[7, 9], [4]]
        assert masks.cu_num_generated_tokens == [0, 1, 1, 2]
        assert masks.slice_request(0, 1).to_nested_list() == [[7, 9]]
        assert masks.slice_request(2, 1).to_nested_list() == [[4]]

    def test_selector_maps_rows_back_to_request_order(self):
        # Under DP the sampled rows come back shuffled; the mask has to be
        # reordered by the same selector as the tokens, or every request gets
        # someone else's support set.
        rows = [[10], [20], [30]]
        masks = _sampling_masks_materialize(
            self._mask_ids(rows, width=2),
            jnp.zeros(3, dtype=bool),
            valid_sampled_token_ids=[[0], [0], [0]],
            logits_indices_selector=[2, 0, 1],
            num_reqs=3,
        )
        assert masks.to_nested_list() == [[30], [10], [20]]

    def test_padding_rows_beyond_num_reqs_are_dropped(self):
        rows = [[10], [20], [99], [99]]
        masks = _sampling_masks_materialize(
            self._mask_ids(rows, width=2),
            jnp.zeros(4, dtype=bool),
            valid_sampled_token_ids=[[0], [0]],
            logits_indices_selector=None,
            num_reqs=2,
        )
        assert masks.to_nested_list() == [[10], [20]]

    def test_overflow_raises(self):
        with pytest.raises(RuntimeError, match="SAMPLING_MASK_WIDTH"):
            _sampling_masks_materialize(
                self._mask_ids([[1, 2]], width=2),
                jnp.asarray([True]),
                valid_sampled_token_ids=[[1]],
                logits_indices_selector=None,
                num_reqs=1,
            )

    def test_overflow_on_a_row_that_generated_nothing_is_ignored(self):
        masks = _sampling_masks_materialize(
            self._mask_ids([[1, 2], [3]], width=2),
            jnp.asarray([False, True]),
            valid_sampled_token_ids=[[1], []],
            logits_indices_selector=None,
            num_reqs=2,
        )
        assert masks.to_nested_list() == [[1, 2]]

    def test_empty_support_raises(self):
        with pytest.raises(RuntimeError, match="Empty sampling mask"):
            _sampling_masks_materialize(
                self._mask_ids([[]], width=2),
                jnp.zeros(1, dtype=bool),
                valid_sampled_token_ids=[[1]],
                logits_indices_selector=None,
                num_reqs=1,
            )

    def test_multiple_tokens_per_request_raises(self):
        with pytest.raises(RuntimeError, match="one sampled token"):
            _sampling_masks_materialize(
                self._mask_ids([[1, 2]], width=2),
                jnp.zeros(1, dtype=bool),
                valid_sampled_token_ids=[[1, 2]],
                logits_indices_selector=None,
                num_reqs=1,
            )

    def test_all_rows_discarded(self):
        masks = _sampling_masks_materialize(
            self._mask_ids([[1], [2]], width=2),
            jnp.zeros(2, dtype=bool),
            valid_sampled_token_ids=[[], []],
            logits_indices_selector=None,
            num_reqs=2,
        )
        assert masks.to_nested_list() == []
        assert masks.cu_num_generated_tokens == [0, 0, 0]
