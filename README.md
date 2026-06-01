# ET-GRB-finder 

当前算法是对连续全帧图像做 12 帧求和差分搜索，在完整检测网格上直接完成 residual 背景估计、阈值分割、连通域识别和候选筛选，输出可下传或进一步处理的 GRB 候选位置和时间窗口。

## 在轨识别算法流程

![图 3 在轨识别算法流程](docs/fig3_algorithm_flow.png)

图 3 在轨识别算法流程

## 当前入口

### 1x1 链路

```bash
conda run -n etbase python scripts/grbfind.py
```

### 2x2 空间 bin 链路

```bash
conda run -n etbase python scripts/grbfind-bin2.py
```

### 3x3 空间 bin 链路

```bash
conda run -n etbase python scripts/grbfind-bin3.py
```

三个入口共享同一套 `grbfinder/` 实现，区别只在脚本顶部的 `SCRIPT_DEFAULTS` 参数。

## 当前关键配置

| 配置项                              |       `grbfind.py` | `grbfind-bin2.py` | `grbfind-bin3.py` | 作用                             |
| ----------------------------------- | ------------------: | ----------------: | ----------------: | -------------------------------- |
| `spatial_bin`                     |              `1x1` |             `2x2` |             `3x3` | 检测网格空间 bin                 |
| `window_size`                     |               `12` |              `12` |              `12` | 每个检测窗口求和帧数             |
| `stride`                          |               `12` |              `12` |              `12` | 相邻窗口起点间隔                 |
| `max_windows`                     |             `None` |            `None` |            `None` | 不截断，使用输入 run 的全部窗口  |
| `template_strategy`               | `rolling-previous` | `rolling-previous` | `rolling-previous` | 当前窗口减最近完整前序窗口       |
| `use_tiles`                       |            `False` |           `False` |           `False` | 每个时间窗口一次处理完整检测网格 |
| `input_bit_depth`                 |               `16` |              `16` |              `16` | 输入整数帧范围检查               |
| `residual_threshold_sigma`        |              `3.0` |             `3.0` |             `3.0` | residual 像素阈值的 sigma 系数   |
| `residual_min_npix`               |               `50` |              `12` |               `6` | residual 连通域最小像素数        |
| `residual_max_npix`               |              `400` |             `400` |             `400` | residual 连通域最大像素数        |
| `min_residual_peak_value`         |            `50000` |          `250000` |          `250000` | residual 连通域最小峰值          |
| `min_residual_flux`               |                `0` |               `0` |         `1000000` | residual 连通域最小总通量        |
| `min_flux_peak_ratio`             |              `3.0` |             `2.0` |             `1.5` | 总通量与峰值的最小比值           |
| `previous_block_match_check`      |            `False` |          `False` |          `False` | 是否启用跨窗口候选关联           |
| `previous_match_radius_px`        |              `2.0` |             `2.0` |             `2.0` | 与上一检测窗口候选关联的半径     |
| `max_final_candidates_per_window` |             `5000` |            `5000` |            `5000` | 每个检测窗口最多输出候选数       |

默认输入 run：

```text
/home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529
```

输入帧路径格式：

```text
<input_run>/frames/frame_000000.npy
<input_run>/frames/frame_000001.npy
...
```

当前实际输入为：

```text
n_frames = 120
frame_shape = (9120, 8900)  # row, col
frame_dtype = uint16
```

## 输出文件

每次运行会在 `output_dir` 下写出：

```text
streaming_sum_candidates_after_measurement.csv
streaming_sum_transient_candidates.csv
streaming_sum_summary.csv
manifest.json
```

### `streaming_sum_candidates_after_measurement.csv`

记录 residual 初筛后、写 final 前的候选。当前配置下，它与 final 表的行数通常一致。

字段：

| 字段                             | 含义                                    |
| -------------------------------- | --------------------------------------- |
| `frame_start`                  | 检测窗口起始帧，闭区间                  |
| `frame_end`                    | 检测窗口结束帧，闭区间                  |
| `window_frame_count`           | 当前窗口帧数，通常为 12                 |
| `x`                            | 探测器坐标 x                            |
| `y`                            | 探测器坐标 y                            |
| `bin_x`                        | 检测网格 x                              |
| `bin_y`                        | 检测网格 y                              |
| `source_peak_value`            | 当前窗口求和图在候选峰值位置的值        |
| `residual_peak_value`          | residual 连通域峰值                     |
| `residual_flux`                | 扣除 residual 背景后的连通域总通量      |
| `residual_npix`                | residual 连通域像素数                   |
| `flux_peak_ratio`              | `residual_flux / residual_peak_value` |
| `previous_block_match_check_enabled` | 是否启用跨窗口关联开关             |
| `previous_block_match_flag`    | 是否与上一检测窗口候选关联              |
| `previous_block_match_dist_px` | 与上一检测窗口候选的距离                |
| `track_length`                 | 跨窗口连续关联长度                      |
| `candidate_priority`           | 当前候选优先级标记                      |
| `pass_single_stack`            | 是否进入 final 表                       |

### `streaming_sum_transient_candidates.csv`

记录最终输出候选。当前交付配置下，final 候选由 residual 前序链路决定。

### `streaming_sum_summary.csv`

每个检测窗口一行，记录该窗口候选数量：

| 字段                         | 含义                     |
| ---------------------------- | ------------------------ |
| `frame_start`              | 检测窗口起始帧           |
| `frame_end`                | 检测窗口结束帧           |
| `window_frame_count`       | 窗口帧数                 |
| `template_frame_start`     | 模板窗口起始帧           |
| `template_frame_end`       | 模板窗口结束帧           |
| `initial_sources`          | residual 连通域候选数    |
| `after_residual_prefilter` | residual 初筛后候选数    |
| `previous_block_matches`   | 与上一窗口关联的候选数   |
| `after_measurement`        | 写入 measured 表的候选数 |
| `final_candidates`         | 写入 final 表的候选数    |

### `manifest.json`

记录运行配置和总计数。

```text
input_shape
detection_shape
cropped_input_shape
spatial_bin_rows
spatial_bin_cols
template_strategy
window_size
stride
residual_threshold_sigma
residual_min_npix
residual_max_npix
min_residual_peak_value
min_residual_flux
min_flux_peak_ratio
previous_block_match_check
previous_match_radius_px
max_final_candidates_per_window
windows_processed
all_windows_including_template
candidates_after_measurement
final_candidates
```

