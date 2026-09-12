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

from tpu_inference.layers.jax.sample.sampling import (SAMPLING_MASK_UNSET_ID,
                                                      _apply_sampling_transforms,
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
