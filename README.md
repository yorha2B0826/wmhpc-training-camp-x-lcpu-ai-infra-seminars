# wmhpc-training-camp-x-lcpu-ai-infra-seminars —— 个人作业实现

本仓库是 **北京大学未名超算队（The Radiance of Weiming）× 北京大学学生 Linux 俱乐部（LCPU）暑期 AI Infra 系列活动**作业仓库的 **个人 fork**：

- 上游：<https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars>
- 本 fork：<https://github.com/yorha2B0826/wmhpc-training-camp-x-lcpu-ai-infra-seminars>

仓库中的**练习解答**为本人独立完成，只提交到本 fork（`origin`），**不会合入上游**。`upstream` 仅用于 `git fetch` 同步新发放的作业。

> **交流与纠错**：解答若有错误，或你有不同思路，欢迎邮件联系 **<mjzheng26@mail.ustc.edu.cn>**。

上游仓库的原始说明（活动介绍、时间表、AI 使用政策）见 upstream 的 `README.md` / `CLAUDE.md`，本 fork 不再重复。

## 已完成的解答

### assignment01 · CUDA（`assignment01/cuda`）

| 文件 | 内容 |
| --- | --- |
| `m0_env/02_device_query.cu` | 5 个空：`multiProcessorCount` / `warpSize` / `sharedMemPerBlock` / `maxThreadsPerMultiProcessor` / `totalGlobalMem` |
| `m2_first_kernel/01_vector_add.cu` | 一维 vector add |
| `m2_first_kernel/02_vector_add_um.cu` | Unified Memory 版 |
| `m2_first_kernel/03_bug_launch.cu` | 找 bug 题 |
| `m2_first_kernel/04_matrix_add.cu` | 二维索引 + 边界保护 |
| `m2_first_kernel/05_grid_stride.cu` | grid-stride loop |
| `m2_first_kernel/saxpy.cu` | **新增文件**，问题 2.9 压轴题，可用自带 `judge_saxpy.sh` 对拍 |
| `m3_simt/02_sync_matters.cu` | 问题 3.3 实验：按题面要求注释掉 `__syncthreads()`，运行应观察到 `MISMATCH` / `FAIL`——这正是该实验要观察的现象，不是实现错误 |
| `m3_simt/03_reduce.cu` | 三版 block 归约：`interleaved` / `contiguous` / `shfl_down` |
| `m4_memory/01_stencil.cu` | 静态 + 动态 shared memory 的 halo tile |
| `m4_memory/02_constant_coeff.cu` | `__constant__ float COEF[8]` + `cudaMemcpyToSymbol` |
| `m4_memory/03_histogram.cu` | 全局 `atomicAdd` 直方图 |
| `m4_memory/04_histogram_priv.cu` | shared memory 私有化直方图 |

### assignment01 · Triton / TileLang（`assignment01/kernels`）

| 文件 | 内容 |
| --- | --- |
| `vector_add.py` | `tl.program_id` / `tl.arange` / mask |
| `fused_op.py` | `relu(a * x + b)`，`a`、`b` 以 `tl.constexpr` 传入 |
| `simt_sim.py` | SIMT 模拟器：`add` / `mul` / `if_lt`，含分支发散后的活跃 mask 与 reconverge |
| `tilelang_scale_add.py` | 二维 CTA grid（`T.ceildiv`）+ `T.Parallel` |
| `tilelang_copy2d.py` | `T.copy` 经 shared memory 往返 |
| `tilelang_matmul.py` | shared tile + `alloc_fragment` 累加器 + `T.Pipelined` + `T.gemm` |
| `softmax.py` | 问题 7.8（选做）：Triton 行 softmax，一个 program 一行，`tl.max`/`tl.sum` 归约，越界 lane 以 `-inf` 填充 |
| `tilelang_softmax.py` | 问题 7.7：TileLang 行 softmax，一个 block 一行，`T.reduce_max`/`T.reduce_sum` + `T.Parallel`/`T.exp`，宽度取 `next_power_of_2(N)` 并以 `-T.infinity` 补位；wrapper 按 `(M, N)` 缓存编译结果 |

### 其他改动

- `assignment01/uv.lock`、`assignment02/uv.lock`：`uv sync` 生成的锁文件，一并提交以便复现环境。
- `assignment02/pyproject.toml`：仅删除文件末尾多余空行。

### 尚未实现

- `assignment02` 的练习（`kernels/block_scale_sim.py`、`kernels/quant_outlier.py` 等）。

除上表列出的文件外，其余文件与上游 `main` 完全一致；上游自带的题面、handout、脚手架与判测代码均未改动。

## 验证记录

本机环境：RTX 4060 Laptop（`sm_89`）、CUDA 13.4、torch 2.14.0+cu130、triton 3.8.0、tilelang 0.1.12。

CUDA（`cd assignment01/cuda`，默认 `ARCH=native`）：

```
m0_env/02_device_query             exit 0（SM 24 / warp 32 / shared 49152 / threads 1536）
m2_first_kernel/01_vector_add      PASS
m2_first_kernel/02_vector_add_um   PASS
m2_first_kernel/03_bug_launch      PASS
m2_first_kernel/04_matrix_add      PASS
m2_first_kernel/05_grid_stride     PASS
m3_simt/02_sync_matters            FAIL —— 问题 3.3 规定的实验现象
m3_simt/03_reduce                  interleaved / contiguous / shuffle 三版均 PASS
m4_memory/01_stencil               static PASS / dynamic PASS
m4_memory/02_constant_coeff        global PASS / constant PASS
m4_memory/03_histogram             PASS
m4_memory/04_histogram_priv        naive PASS / priv PASS（约 66x）
```

压轴题：`cd assignment01/cuda && ./m2_first_kernel/judge_saxpy.sh ./m2_first_kernel/saxpy.cu` —— 7 组规模（含 0、1、非整除边界）全部 `PASS`。

Python（`cd assignment01`）：

```bash
uv sync --extra tilelang && uv run pytest tests/
# 24 passed, 11 warnings —— 全部通过。
# 11 个 warning 均为 tilelang 0.1.12 对题面脚手架里 T.Buffer 的 DeprecationWarning，
# 不是实现问题（详见下）。
```

分文件结果：

```
tests/test_fused_op.py            3 passed
tests/test_matmul_triton.py       2 passed
tests/test_simt_sim.py            5 passed
tests/test_softmax.py             4 passed
tests/test_tilelang.py            3 passed
tests/test_tilelang_softmax.py    4 passed
tests/test_vector_add.py          3 passed
```

`test_softmax.py` / `test_tilelang_softmax.py` 另以 `(M, N) ∈ {(1,1), (5,1), (33,127), (8,1000), (4,256), (7,4096)}` 做了额外对拍（含 `N=4096` 上界与 `N=1` 退化行），两者均与 `torch.softmax` 一致；`randn * 1000` 的数值稳定性用例也通过。

### 关于 tilelang 的 DeprecationWarning

`tests/test_tilelang.py` 运行时的 11 条 `T.Buffer(...) is deprecated, use T.Tensor(...) instead` 全部来自**题面脚手架**（`kernels/tilelang_scale_add.py`、`tilelang_copy2d.py`、`tilelang_matmul.py` 里的 `X: T.Buffer(...)` 等注解，每处每次调用触发一条），逐字是上游原文，未作改动。自行新写的 `kernels/tilelang_softmax.py` 已直接使用 `T.Tensor`，不产生该警告。这些警告不影响判测（判测只比数值），保留上游写法以保证与题面一致。

## 同步上游

```bash
git fetch upstream
git rebase upstream/main    # README.md 与已作答文件可能有冲突，需手工处理
git push origin main        # 只推 fork
```

remote 约定：`origin` = 本 fork（可 push），`upstream` = `lcpu-club/...`（只 fetch，不 push）。
