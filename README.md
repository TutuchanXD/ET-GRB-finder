# ET-GRB-finder

`ET-GRB-finder` 是一个面向星上低缓存场景的 GRB 候选搜索流水线。实现位于 `grbfinder/`，`scripts/` 下只保留短入口脚本，并在每个脚本顶部列出完整可编辑的默认参数表 `SCRIPT_DEFAULTS`。

 `1x1` 入口在：

```bash
conda run -n etbase python scripts/grbfind.py
```

## 默认流水线流程

流程以 `scripts/grbfind.py` 的默认配置为准。默认输入是实际注入 GRB 的全帧仿真结果：

```text
input_run = /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529
```

输入目录需要包含：

```text
<input_run>/frames/frame_000000.npy
<input_run>/frames/frame_000001.npy
...
```

当前实际数据为 120 帧，全帧尺寸 `(9120, 8900)`，单帧 dtype 为 `uint16`。

### 1. 读取脚本默认参数

`scripts/grbfind.py` 顶部的 `SCRIPT_DEFAULTS` 是本次运行的默认配置来源。最重要的默认值是：

```text
spatial_bin       = 1x1
window_size       = 12
stride            = 12
max_windows       = 2
template_strategy = rolling-previous
```

因此默认只纳入两个原始时间窗口：

```text
0-11   -> 种子模板窗口
12-23  -> 唯一检测窗口
```

### 2. 构造时间窗口

流水线用半开区间组织帧窗口：

```text
[0, 12), [12, 24), [24, 36), ...
```

CSV 输出中使用闭区间端点显示，因此同一个窗口会写成：

```text
0-11, 12-23, 24-35, ...
```

### 3. 选择模板窗口

默认 `template_strategy = rolling-previous`。在没有指定 `template_run` 时，每个检测窗口都减去最近的完整前序窗口：

```text
template 0-11   -> detect 12-23
template 12-23  -> detect 24-35
template 24-35  -> detect 36-47
```

如果显式设置 `template_strategy = first-window`，则所有后续检测窗口都减第一个窗口 `0-11`。如果指定 `template_run`，则进入 paired-template 验证模式，模板来自外部模板 run 的同帧号窗口。

### 4. 按 tile 流式处理全帧

全帧不会一次性构造成完整 12 帧 cube。流水线按 core tile 遍历检测网格，并在 tile 外侧加 `halo`，用于 cutout 和连通域边界安全处理。

默认 `spatial_bin = 1x1`，检测网格就是原图网格。`grbfind-bin2.py` 和 `grbfind-bin3.py` 分别默认使用 `2x2` 和 `3x3` 非重叠空间 bin。

### 5. 求当前窗口和模板窗口的 12 帧和

对每个 tile，先分别计算：

```text
current_sum  = sum(input frames in detection window)
template_sum = sum(input/template frames in template window)
```

求和 dtype 使用 `uint32`，避免 12 帧 `uint16` 相加溢出。

### 6. Residual-first 检测

候选不是先从原始 12 帧和图中找星源，而是先做差分：

```text
residual = current_sum - template_sum
```

然后在 residual 图上用局部背景和 sigma 阈值找正残差连通域。这样可以大幅减少静态星源造成的候选洪泛。

初筛主要检查：

```text
residual_min_npix <= residual_npix <= residual_max_npix
residual_peak_value >= min_residual_peak_value
residual_flux >= min_residual_flux
residual_flux / residual_peak_value >= min_flux_peak_ratio
```

### 7. 局部 cutout 测量

如果 `local_shape_check = True`，流水线会围绕候选位置截取局部 cutout，并重新计算局部背景、局部连通域面积和局部超额通量。

输出字段包括：

```text
local_excess_flux
peak_npix
peak_excess_value
peak_pixel_snr
```

这些字段默认用于诊断。`peak_npix` 不再单独作为 final 放行条件；它只在显式开启 `peak_pixel_snr_check` 时参与局部峰值显著性检查。

### 8. 最终候选选择

默认不启用 `keep_all_residual_candidates`，也不启用 `peak_pixel_snr_check`。因此通过 residual 连通域检测和 residual flux/peak 初筛的候选会直接进入 final 表。

如果启用 `peak_pixel_snr_check = True`，final gate 会同时要求局部 footprint 和局部峰值显著性：

```text
peak_npix >= effective_npix_threshold
and peak_pixel_snr >= peak_pixel_snr_threshold
```

`peak_pixel_snr` 是局部残差峰值除以局部 robust sigma 的显著性指标，不是物理 SNR。当前三份入口脚本默认关闭该 gate。

如果 `temporal_check = True`，时间支持和宇宙线字段仍会写入结果，并用于候选优先级标注；它们不是当前 final 硬门槛。如果 `keep_all_residual_candidates = True`，则绕过 residual flux/peak 初筛和 final gate，把 residual 连通域候选写入 final 表。

### 9. Truth match 仅用于地面验证

如果输入 run 下存在 `events.csv`，流水线会自动读取它并做 truth matching。相关字段只用于地面验证，不属于星上依赖：

```text
truth_match_flag
truth_event_id
truth_dist_px
```

## 入口脚本

当前只保留 3 个短入口：

```text
scripts/grbfind.py       -> 1x1, rolling-previous, max_windows=2
scripts/grbfind-bin2.py  -> 2x2, rolling-previous, max_windows=2
scripts/grbfind-bin3.py  -> 3x3, rolling-previous, max_windows=2
```

每个入口脚本顶部都有完整 `SCRIPT_DEFAULTS`。常用方式是直接编辑脚本中的默认值，然后运行脚本。

仍然可以用命令行覆盖参数，例如：

```bash
conda run -n etbase python scripts/grbfind.py \
  --input-run /path/to/run \
  --output-dir /path/to/output \
  --max-windows 3
```

## 输出文件

每次运行会写出：

```text
streaming_sum_candidates_after_measurement.csv
streaming_sum_transient_candidates.csv
streaming_sum_summary.csv
manifest.json
```

如果启用 `template_match_sources = True`，还会写出：

```text
template_sources.csv
```

`manifest.json` 记录运行配置和总计数；`streaming_sum_summary.csv` 是每个检测窗口的摘要；`streaming_sum_transient_candidates.csv` 是最终候选表。

## 实际数据 1x1 单块烟测结果

最近一次实际 injected 全帧数据烟测使用 `scripts/grbfind.py` 默认 `max_windows = 2`，未显式传 `--max-windows` 或 `--template-strategy`。

输出目录：

```text
/home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_1x1_default_oneblock_smoke_20260530_temporal_off
```

关键结果：

```text
template_strategy = rolling-previous
template_window = [0, 11]
detection window = 12-23
spatial_bin = 1x1
windows_processed = 1
all_windows_including_template = 2
candidates_after_measurement = 1
final_candidates = 1
truth_matched_final_candidates = 1
```

命中的 truth：

```text
truth_event_id = 6
x = 942
y = 7441
truth_dist_px = 0.49592883990681813
residual_flux = 3611813
```

## 参数说明

以下参数都在三个脚本的 `SCRIPT_DEFAULTS` 中显式列出。

| 参数                                | `grbfind.py` 默认值  | 作用                                                                             |
| ----------------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `input_run`                       | `DEFAULT_INPUT_RUN`  | 输入 run 目录，内部应包含 `frames/frame_*.npy`。                               |
| `template_run`                    | `None`               | 外部模板 run。为 `None` 时使用输入 run 自身按 `template_strategy` 生成模板。 |
| `output_dir`                      | `DEFAULT_OUTPUT_DIR` | 输出目录。目录已存在且 `overwrite=False` 时会报错。                            |
| `truth_events_csv`                | `None`               | truth 事件表路径。为 `None` 时自动尝试读取 `<input_run>/events.csv`。        |
| `truth_match_radius_px`           | `12.0`               | truth matching 的最大距离，单位为原图像素。                                      |
| `spatial_bin`                     | `"1x1"`              | 空间非重叠 bin 设置。可写成 `"N"` 或 `"RxC"`。                               |
| `window_size`                     | `12`                 | 每个检测窗口求和的帧数。                                                         |
| `stride`                          | `12`                 | 相邻窗口起点的帧步长。                                                           |
| `max_windows`                     | `2`                  | 最多纳入的原始窗口数，包含模板种子窗口。默认只检测一个块。                       |
| `template_strategy`               | `"rolling-previous"` | 无外部模板时的模板策略。可选 `"rolling-previous"` 或 `"first-window"`。      |
| `tile_size`                       | `1024`               | 流式处理时的 core tile 尺寸，单位为检测网格像素。                                |
| `halo`                            | `12`                 | tile 外扩边界，用于 cutout 和连通域边界安全处理。                                |
| `input_bit_depth`                 | `16`                 | 输入整数帧的位深检查。                                                           |
| `max_filter_size`                 | `7`                  | 模板星源诊断检测中的局部极大值滤波尺寸。                                         |
| `source_threshold_sigma`          | `4.0`                | 模板星源诊断检测阈值，只在模板星源 catalog 启用时使用。                          |
| `residual_threshold_sigma`        | `3.0`                | residual 图中像素进入候选连通域前需超过背景的 sigma 数。                         |
| `residual_min_npix`               | `50`                 | residual 连通域最小像素数。                                                      |
| `residual_max_npix`               | `400`                | residual 连通域最大像素数。                                                      |
| `min_residual_peak_value`         | `50000`              | residual 初筛所需最小峰值。                                                      |
| `min_residual_flux`               | `0`                  | residual 初筛所需最小总通量。                                                    |
| `min_flux_peak_ratio`             | `3.0`                | residual 总通量与峰值的最小比值，用于排除过尖候选。                              |
| `match_radius_px`                 | `0.75`               | 模板星源匹配半径，用于标注候选是否靠近模板源。                                   |
| `cut_half`                        | `9`                  | 局部形态 cutout 半宽；实际 cutout 尺寸约为 `2 * cut_half + 1`。                |
| `annulus_r_in`                    | `6.0`                | 局部背景环内半径。                                                               |
| `annulus_r_out`                   | `10.0`               | 局部背景环外半径。                                                               |
| `local_threshold_sigma`           | `3.0`                | 局部 residual 连通域阈值。                                                       |
| `seed_radius`                     | `1.5`                | 在局部 cutout 中选择候选对应连通域的种子半径。                                   |
| `peak_pixel_snr_check`            | `False`              | 是否启用局部 peak SNR final gate。关闭时 residual 前序通过的候选直接进入 final。 |
| `effective_npix_threshold`        | `4`                  | `peak_pixel_snr_check=True` 时，局部连通域所需的最小像素数。                    |
| `peak_pixel_snr_threshold`        | `5.0`                | `peak_pixel_snr_check=True` 时的局部残差峰值显著性阈值。                         |
| `temporal_cut_half`               | `5`                  | 时间支持 cutout 半宽。                                                           |
| `temporal_aperture_radius`        | `3.0`                | 时间序列 aperture flux 半径。                                                    |
| `temporal_annulus_r_in`           | `5.0`                | 时间序列背景环内半径。                                                           |
| `temporal_annulus_r_out`          | `8.0`                | 时间序列背景环外半径。                                                           |
| `temporal_sigma`                  | `3.0`                | 判定某帧 temporal flux active 的 sigma 阈值。                                    |
| `temporal_min_active_frames`      | `2`                  | 候选优先级标注所需的 active 帧数。                                               |
| `cosmic_single_frame_fraction`    | `0.80`               | 单帧 flux 占比超过该值时更像宇宙线。                                             |
| `cosmic_max_active_frames`        | `1`                  | active 帧数不超过该值且单帧占比过高时标记为 `likely_cosmic_ray`。              |
| `previous_match_radius_px`        | `2.0`                | 与上一检测块 final 候选做位置关联的半径。                                        |
| `template_match_sources`          | `False`              | 是否构建并输出模板星源 catalog，以及标注候选最近模板源。                         |
| `local_shape_check`               | `True`               | 是否执行局部形态重测。                                                           |
| `temporal_check`                  | `False`              | 是否执行逐帧时间支持测量。                                                       |
| `keep_all_residual_candidates`    | `False`              | 是否跳过 residual flux/peak 初筛和 final gate，把 residual 连通域候选写入 final 表。 |
| `max_final_candidates_per_window` | `5000`               | 每个检测窗口最多保留的 final 候选数。`<=0` 表示不限制。                        |
| `overwrite`                       | `False`              | 输出目录存在时是否允许覆盖。                                                     |

`grbfind-bin2.py` 和 `grbfind-bin3.py` 的参数表相同，但针对空间 bin 后的 residual 尺度有不同默认阈值：

```text
grbfind-bin2.py: spatial_bin = "2x2", output_dir = DEFAULT_OUTPUT_DIR + "_bin2"
grbfind-bin3.py: spatial_bin = "3x3", output_dir = DEFAULT_OUTPUT_DIR + "_bin3"
```

```text
grbfind-bin2.py:
  residual_min_npix = 12
  min_residual_peak_value = 250000
  min_residual_flux = 0
  min_flux_peak_ratio = 2.0

grbfind-bin3.py:
  residual_min_npix = 6
  min_residual_peak_value = 250000
  min_residual_flux = 1000000
  min_flux_peak_ratio = 1.5
```
