"""CPU contracts for reference-image block selection, without loading NPU plugins."""

import importlib.util
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[2] / "mindiesd/layers/flash_attn/ref_sparse_mask.py"
SPEC = importlib.util.spec_from_file_location("ref_sparse_mask_under_test", SOURCE)
ref = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ref)


def test_noncontiguous_ownership_and_protected_rows():
    # Original positions 0,2,4,6 form one image; the permutation makes two pure blocks.
    plan = ref.build_ref_plan(
        [0, 2, 4, 6, 1, 3, 5, 7, 8],
        (),
        ((1, 2), (3, 4), (5, 6), (7, 9)),
        2,
        {2},
        ((0, 2, 4, 6),),
    )
    assert plan == (((0, 1),), (3,))  # Partial final block and protected row excluded.


def test_mixed_image_blocks_are_not_candidates():
    assert ref.build_ref_plan(list(range(9)), ((1, 5),), ((6, 9),), 2, set()) == (
        ((1,),),
        (3,),
    )


@pytest.mark.parametrize("positions", [((0, 1), (1, 2)), ((-1, 0),), ((0, 9),), ((),)])
def test_invalid_image_ownership(positions):
    with pytest.raises(ValueError):
        ref.build_ref_plan(list(range(8)), (), ((6, 8),), 2, set(), positions)


def test_image_target_overlap():
    with pytest.raises(ValueError):
        ref.build_ref_plan(list(range(8)), ((0, 4),), ((2, 8),), 2, set())


def test_per_head_per_row_topk_preserves_other_connections():
    scores = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 4, 6)
    scores[0, 1] = -scores[0, 1]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[..., 0, 5] = False
    original = mask.clone()
    groups, rows = ((0, 1, 2), (3, 4)), (1, 3)
    result = ref.apply_ref_mask(mask, scores, (groups, rows), 0.5)
    # Independent scalar oracle: ceil(3/2)=2 and ceil(2/2)=1, separately per image.
    expected = original.clone()
    for head in range(2):
        for row in rows:
            for group, count in zip(groups, (2, 1)):
                ranked = sorted(group, key=lambda col: float(scores[0, head, row, col]), reverse=True)
                for col in group:
                    expected[0, head, row, col] = col in ranked[:count]
    assert torch.equal(result, expected)


def test_ratio_one_returns_original_mask():
    mask = torch.zeros(1, 1, 2, 3, dtype=torch.bool)
    assert ref.apply_ref_mask(mask, torch.ones_like(mask), (((0, 1),), (1,)), 1) is mask
    assert not mask.any()


@pytest.mark.parametrize("ratio", [0, -0.5, 1.1, float("nan")])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        ref.apply_ref_mask(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), ((), ()), ratio)


def test_ties_have_exact_cardinality_without_promising_indices():
    mask = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    ref.apply_ref_mask(mask, torch.zeros_like(mask, dtype=torch.float32), (((0, 1, 2),), (1,)), 0.5)
    assert mask[0, 0, 1, :3].sum() == 2
    assert mask[0, 0, 1, 3]
    assert mask[0, 0, 0].all()
