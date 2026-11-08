# Reference-image block selection: fork integration candidate

## Status / 状态

This fork candidate is based on `dev` commit
`8701dabb0aba4ba52d3cf81d8798c73abaa187df`. It ports the Python changes used by the
MiniMax H3 reference-block-sparse experiment. It is not an upstream release or a
validated replacement for the complete experimental software environment.

本分支是配套复现候选：迁入参考图像块选择的 Python 改动，尚未完成 NPU
构建、安装和端到端验证。历史模型实验不自动证明本分支及其全部依赖通过验证。
发布时须记录本分支最终完整 commit SHA，并与调用方 Omni commit 成对固定。

## Behavior / 行为

- `build_ref_plan` tracks exact image ownership after sequence permutation.
  Only complete, pure image key blocks and eligible target query blocks participate.
- `apply_ref_mask` keeps `max(1, ceil(N * ratio))` blocks separately for each image,
  head and target query row, ranked by current scores. It replaces eligible image
  connections in the original mask; other entries are unchanged.
- A ratio of 1 disables reference selection. The mechanism does not shrink KV storage.
- Dynamic masks/scores are not cached. The full-query integration caches CPU geometry.
- Tied scores have an exact selection count; selected indices across devices are not guaranteed.

该实现按当前分数独立选择参考图像块，保护非目标行、非图像列、混合边界和不足一块的
尾部。ratio=1 关闭；ratio=0.5 不等于总耗时或 KV 容量减少一半。

## Caller contract / 调用方契约

For compatibility with the archived Omni implementation, full-query selection reads
`_ref_keep_ratio`, `_ref_images`, `_ref_image_positions`, and `_ref_targets` from
the first video span. The rectangular caller imports `build_ref_plan` and
`apply_ref_mask` directly. `RefDenseBlocks` and `COUNTS` remain for that integration.
These are experimental contracts, not a new stable public API.

现阶段保留实验私有字段、计数及几何日志，以兼容已有 Omni。后续正式接口整理需两仓
同步，并在固定候选版本上重验。Omni 的 `RESEARCH_FIXED_RATIO` 双入口问题不在本仓修复。
本分支不提供完整 Omni 优化；仅安装此分支不会自动启用 LoRA、参考缓存或 RBS。

## Build and verify / 构建与验证

Use a separate Linux Ascend environment with the matching CANN, driver, PyTorch and
torch_npu versions. Follow the repository's [build guide](docs/en/developer_guide/build_guide.md)
and [installation guide](docs/en/installation.md). Do not substitute the local CPU
test environment for the target deployment environment.

发布后在独立昇腾环境按最终完整 SHA checkout，再按
[构建指南](docs/zh/developer_guide/build_guide.md)和
[安装指南](docs/zh/installation.md)构建。构建入口如下；本轮未执行：

```bash
python -m build --wheel --no-isolation
```

Install the specific resulting wheel in the isolated environment. Record its SHA256,
`git rev-parse HEAD`, package versions, and `mindiesd.__file__`; verify that both
modified Python files are loaded from the installed candidate. Required NPU checks:
ratio=1 regression, full/rectangular same-mask tests, multi-rank integration, then
approved end-to-end workloads. No profiling collection is required by this document.

安装时指定本次构建的 wheel 文件，保存 wheel SHA256、代码 SHA、依赖版本及实际导入路径。
后续验证关闭回归、完整/矩形同 mask 对拍、多卡接入，再执行已获授权的业务用例。
不得沿用历史性能数字作为新构建结果，不自动追加模型请求或 msProf。

CPU-only helper tests (no NPU plugin loading):

```bash
python -m pytest --noconftest tests/UT/test_ref_sparse_mask.py -q
```

The 14 helper tests passed locally on Python 3.12.10 / PyTorch 2.6.0+cpu.
The initial test collection failed because the module was absent on the base revision.
The NumPy-not-installed warning did not affect these tensor-only tests.
This command intentionally bypasses repository conftest; it is not the full CPU suite.
The complete CPU suite could not collect tests with this local PyTorch version:
the repository mock imports the unavailable `torch._inductor.runtime.triton_compat`.
Full repository gate completion, Linux wheel construction and NPU tests remain pending.
The full pre-commit attempt was stopped during dependency initialization; it did not
complete. Targeted helper lint, three Python format checks, Markdown lint and diff
whitespace checks passed. Six pre-existing rf_v2 lint findings remain.

## Provenance and publication / 来源与发布

The two Python changes originate from the archived reference-block-sparse integration
patch, with formatting/import-order cleanup. Original copyright notices are retained.
The fork owner developed the original RBS patch on another computer and confirmed
its authorship before publication. The source archive and hashes are recorded in
the preparation report. Existing upstream copyright notices are retained.

仓库所有者已确认原始 RBS 补丁由其在另一台电脑开发。本分支不改版本号，不创建
MindIE SD 上游 PR。匹配 Omni commit 需在调用方冻结后补齐。
