"""Experiment: exact image ownership and per-image/head/target-row selection."""

import math
from collections import Counter

import torch

COUNTS = Counter()


class RefDenseBlocks(set):
    def __init__(self, blocks, plan, keep_ratio):
        super().__init__(blocks)
        self.ref_plan = plan
        self.keep_ratio = keep_ratio


def build_ref_plan(permutation, image_ranges, target_ranges, block_size, protected, image_positions=()):
    n = len(permutation)
    assert sorted(permutation) == list(range(n)), 'permutation must be bijective'
    images = [set(range(a, b)) for a, b in image_ranges] if not image_positions else [set(p) for p in image_positions]
    seen = set()
    for owned in images:
        if not owned or min(owned) < 0 or max(owned) >= n or seen.intersection(owned):
            raise ValueError('invalid or overlapping image ownership')
        seen.update(owned)
    targets = set()
    for a, b in target_ranges:
        if not 0 <= a < b <= n:
            raise ValueError('invalid target range')
        targets.update(range(a, b))
    if seen.intersection(targets):
        raise ValueError('image overlaps target')
    groups = [[] for _ in images]
    rows = []
    for block in range(n // block_size):
        tokens = set(permutation[block * block_size : (block + 1) * block_size])
        for group, owned in zip(groups, images):
            if tokens.issubset(owned):
                group.append(block)
        if block not in protected and tokens.issubset(targets):
            rows.append(block)
    return tuple(tuple(g) for g in groups), tuple(rows)


def apply_ref_mask(mask, scores, plan, keep_ratio):
    if not 0 < keep_ratio <= 1:
        raise ValueError('invalid ref_keep_ratio')
    groups, rows = plan
    if keep_ratio == 1 or not rows:
        return mask
    row_idx = torch.tensor(rows, device=mask.device, dtype=torch.long)
    COUNTS['calls'] += 1
    COUNTS['target_rows'] += len(rows)
    for group in groups:
        COUNTS['image_groups'] += 1
        COUNTS['candidate_columns'] += len(group)
        keep = max(1, math.ceil(len(group) * keep_ratio))
        if keep >= len(group):
            continue
        columns = torch.tensor(group, device=mask.device, dtype=torch.long)
        values = scores.index_select(-2, row_idx).index_select(-1, columns)
        selected = values.topk(keep, dim=-1).indices
        keep_mask = torch.zeros_like(values, dtype=torch.bool).scatter_(-1, selected, True)
        mask[..., row_idx[:, None], columns[None, :]] = keep_mask
        COUNTS['groups_reduced'] += 1
        COUNTS['selected_columns'] += keep
    return mask
