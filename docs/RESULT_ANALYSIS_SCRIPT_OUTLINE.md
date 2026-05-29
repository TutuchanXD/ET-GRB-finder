# GRB Finder 结果分析脚本大纲

## 目标

新增一个独立于搜索执行脚本的结果分析脚本，用于对 `grbfinder` 已生成的搜索结果做具体、可复查的评估。该脚本不重新跑 finder，不修改搜索结果，只读取输出目录、truth 表和必要的辅助数据，生成诊断报告。

建议脚本路径：

```text
tools/analyze_grbfinder_results.py
```

不建议放进 `scripts/grbfind*.py` 系列，因为该脚本不是搜索入口，而是离线评估工具。

### 模块边界

结果分析诊断是 ET-GRB-finder 仓库的一部分，但不属于星上运行时 `grbfinder/` 模块。后续 `grbfinder/` 可能重构到星上环境，而星上不需要 truth 评估、CSV 报告、cosmic truth 读取或模板星源离线缓存。

因此实现时应遵守以下边界：

1. 不向现有 `grbfinder/` 包内加入分析逻辑、truth 逻辑或报告逻辑。
2. 不让 `grbfinder/` 依赖分析工具；依赖方向只能是分析工具读取 finder 输出。
3. 分析入口保持在 `tools/analyze_grbfinder_results.py`。
4. 如果脚本变大，可新建独立的 `tools/grbfinder_analysis/` 辅助模块，但仍不放入 `grbfinder/`。
5. 分析工具通过 CSV、JSON、NPY 等文件接口消费 finder 产物，不调用 finder 内部私有函数。

## 核心问题

分析脚本需要直接回答以下问题：

1. 每个检测块内实际有几个 GRB 事件。
2. 这些 GRB 事件中有几个被 finder 命中。
3. 每个检测块和整个结果的 GRB 命中率是多少。
4. 被命中的 GRB 如果按 SNR 排序，在候选列表中排第几。
5. 未命中的 GRB 的性质，说明是什么原因没有候选。
6. 误检候选中哪些更像宇宙线。
7. 误检候选中哪些更像模板星点或静态星残差。
8. 其它误检候选按什么原因归类，以及哪些需要人工检查。

## 输入

### 必需输入

```text
--result-dir
```

finder 输出目录，至少包含：

```text
manifest.json
streaming_sum_summary.csv
streaming_sum_transient_candidates.csv
streaming_sum_candidates_after_measurement.csv
```

```text
--events-csv
```

GRB 注入 truth 表。若未显式传入，默认从 `manifest.json` 的 `truth_events_csv` 读取；如果 manifest 中没有，则尝试 `<input_run>/events.csv`。

### 可选输入

```text
--input-run
```

输入 run 目录。若 manifest 中有 `input_run`，可自动读取。用于在需要时重新构建模板星源 catalog 或读取辅助 truth。

```text
--template-sources-csv
```

模板星源 catalog。若 finder 运行时启用了 `template_match_sources`，可直接读取 `template_sources.csv`。

```text
--cosmic-truth
```

宇宙线 truth 路径。若未显式传入，默认按以下顺序自动发现：

```text
<input-run>/copied_source_aux/cosmic_events/
<input-run>/copied_source_aux/frame_summaries/
```

当前 `main_rd_g17_120x10s_grb_seed20260529` 的 GRB 注入 run 中，`copied_source_aux/cosmic_events/` 已包含逐帧 cosmic truth：

```text
frame_000000_events.npy
frame_000001_events.npy
...
```

每个 `.npy` 是结构化数组，已确认包含以下关键字段：

```text
frame_index
x0
y0
x1
y1
total_adu
peak_adu
clipped_by_frame
```

其中 `x0/y0/x1/y1` 描述 cosmic-ray artifact 在全帧图上的足迹框，`frame_summaries/frame_XXXXXX.json` 可提供 `actual_cosmic_events`、`cosmic_mask_pixels` 等逐帧汇总。若未来 run 保存了 artifact mask，也可作为额外输入用于 footprint overlap 复核。

因此宇宙线诊断分为两级：

- finder advisory：基于 `likely_cosmic_ray`、`temporal_active_frames`、`temporal_max_single_frame_fraction`。
- truth-level cosmic：默认读取 `copied_source_aux/cosmic_events/`，将误检候选与真实 cosmic-ray footprint 匹配。

## 主要输出

分析脚本建议生成一个输出目录：

```text
<result-dir>/analysis/
```

其中包含：

```text
analysis_summary.json
analysis_report.md
window_grb_recall.csv
truth_event_ranks.csv
false_positive_diagnostics.csv
false_positive_groups.csv
top_candidates_by_snr.csv
template_sources.csv
```

### `analysis_report.md`

给人工阅读的 Markdown 报告。需要包含：

- 输入结果目录和 finder manifest 摘要。
- 每个检测块的 GRB 数、命中数、命中率。
- 全局 GRB 命中率。
- 命中 GRB 的多指标排名表，并写清楚每个排名的定义。
- 未命中 GRB 列表和细分原因。
- 误检候选分类统计。
- Top-N 高 SNR 误检候选列表。
- 数据缺失或诊断降级说明。

### `analysis_summary.json`

机器可读总览，至少包含：

```json
{
  "result_dir": "...",
  "input_run": "...",
  "template_strategy": "rolling-previous",
  "spatial_bin": "1x1",
  "windows": 1,
  "truth_events_in_windows": 1,
  "truth_events_detected": 1,
  "truth_recall": 1.0,
  "final_candidates": 1203,
  "false_candidates": 1202,
  "false_likely_cosmic": 0,
  "false_template_star_like": 0,
  "false_unknown": 1202
}
```

字段值只是结构示例，实际由脚本计算。

## 检测块内 GRB 统计

### 检测块定义

以 `streaming_sum_summary.csv` 为准。每一行表示一个检测块：

```text
frame_start
frame_end
template_frame_start
template_frame_end
```

例如默认单块烟测：

```text
template: 0-11
detect:   12-23
```

### GRB 是否属于检测块

从 `events.csv` 读取每个事件：

```text
event_id
first_visible_frame
last_visible_frame
requested_detector_xpix
requested_detector_ypix
stamp_center_col
stamp_center_row
peak_global_frame
peak_mag
brightness_scale
```

一个 GRB 属于某检测块，当且仅当：

```text
event.first_visible_frame <= block.frame_end
and event.last_visible_frame >= block.frame_start
```

脚本还应计算该事件在检测块内的可见帧数：

```text
overlap_start = max(first_visible_frame, frame_start)
overlap_end   = min(last_visible_frame, frame_end)
visible_frames_in_block = overlap_end - overlap_start + 1
```

## GRB 命中判定

### 优先判定方式

优先使用 finder 已写出的字段：

```text
truth_match_flag == 1
truth_event_id
truth_dist_px
```

对每个 block 内 GRB，若 final candidates 中存在同 `truth_event_id` 的候选，则认为该事件被命中。

### 重新计算判定

如果 finder 输出中没有 truth 字段，或用户希望用不同半径复核，则脚本应支持重新匹配：

```text
--truth-match-radius-px 12.0
```

匹配逻辑：

1. 只在时间窗口重叠的候选和 truth 之间匹配。
2. 候选坐标使用 `x`, `y`。
3. truth 坐标优先使用 `requested_detector_xpix`, `requested_detector_ypix`；如需整数中心，可同时输出到 `stamp_center_col`, `stamp_center_row` 的距离供检查。
4. 每个 truth event 选择距离最近的候选。
5. 每个候选最多归属一个 truth event，避免重复计数。

### 输出表：`window_grb_recall.csv`

每行一个检测块：

```text
frame_start
frame_end
template_frame_start
template_frame_end
truth_events_in_block
truth_events_detected
truth_recall
final_candidates
false_candidates
```

### 输出表：`truth_event_ranks.csv`

每行一个 block 内 truth event：

```text
event_id
frame_start
frame_end
first_visible_frame
last_visible_frame
visible_frames_in_block
requested_detector_xpix
requested_detector_ypix
detected
matched_candidate_x
matched_candidate_y
truth_dist_px
snr_rank
residual_flux_rank
residual_peak_rank
peak_pixel_snr
residual_flux
residual_peak_value
temporal_active_frames
likely_cosmic_ray
miss_reason
```

## SNR 排名

报告需要同时给出多个排名，而不是只报告单一 SNR 排名。默认主排序指标仍使用 finder 已输出的：

```text
peak_pixel_snr
```

排序规则：

1. 每个检测块内单独排序。
2. 只对 `streaming_sum_transient_candidates.csv` 中 final candidates 排序。
3. `peak_pixel_snr` 越大排名越靠前。
4. `nan` 或空值放到最后。
5. 若 SNR 相同，用 `residual_flux` 降序打破平局。

脚本同时输出以下排名：

```text
snr_rank
residual_flux_rank
residual_peak_rank
```

定义如下：

- `snr_rank`：按 `peak_pixel_snr` 降序排序。
- `residual_flux_rank`：按 `residual_flux` 降序排序。
- `residual_peak_rank`：按 `residual_peak_value` 降序排序。

原因是某些候选可能存在局部背景 sigma 异常，单看 `peak_pixel_snr` 可能不稳定。`analysis_report.md` 必须在“命中 GRB 的候选排名”章节明确说明这三个排名的物理含义和排序规则。

## 未命中 GRB 原因

未命中 GRB 需要细分原因，不能只输出 `missed`。第一版建议给出以下主原因：

```text
outside_detection_window
too_few_visible_frames
present_in_template_window
below_candidate_threshold
candidate_generated_but_filtered
merged_with_nearby_candidate
edge_or_stamp_clipped
candidate_ranked_but_not_truth_matched
unknown
```

判定建议：

1. 若 truth 与检测块无时间重叠，标记 `outside_detection_window`。
2. 若 truth 靠近图像边缘或注入 stamp 被裁剪，标记 `edge_or_stamp_clipped`。
3. 若 truth 在模板窗口已经可见，标记 `present_in_template_window`；这类事件可能被模板相减抵消，不能简单解释为阈值不足。
4. 若在检测块内可见帧数太少，标记 `too_few_visible_frames`。
5. 若 final candidates 没命中，但 measured candidates 或中间候选中存在近邻，标记 `candidate_generated_but_filtered`。
6. 若候选近邻存在但 truth id 未匹配，标记 `candidate_ranked_but_not_truth_matched`。
7. 若无法从现有输出判断，标记 `unknown`，并在报告中列入人工检查。

## 误检定义

候选若不满足以下条件，则视为误检候选：

```text
truth_match_flag == 1
```

或在重新匹配模式下，没有匹配到时间重叠且距离足够近的 GRB truth。

误检候选进入 `false_positive_diagnostics.csv`。

## 误检分类

分类建议按优先级执行。一个候选只给一个主分类，同时可给多个辅助 tag。

### 1. `false_likely_cosmic`

满足 finder advisory：

```text
likely_cosmic_ray == 1
```

同时输出诊断字段：

```text
temporal_active_frames
temporal_consecutive_active_frames
temporal_max_single_frame_fraction
temporal_flux_series
```

若成功读取 cosmic truth，或用户显式提供 artifact mask，则额外输出：

```text
cosmic_truth_match_flag
cosmic_truth_dist_px
cosmic_truth_frame
cosmic_truth_x0
cosmic_truth_y0
cosmic_truth_x1
cosmic_truth_y1
cosmic_truth_total_adu
cosmic_truth_peak_adu
cosmic_truth_same_frame_flag
```

对当前 injected GRB run，cosmic truth 默认可从 `copied_source_aux/cosmic_events/` 读取，因此第一版应支持 truth-level cosmic 归因。

匹配建议：

1. 对每个误检候选，只加载其检测块覆盖的 raw frames。例如默认单块烟测加载 `frame_000012_events.npy` 到 `frame_000023_events.npy`。
2. 候选坐标使用 `x`, `y`，cosmic truth 使用 footprint 框 `x0/y0/x1/y1`。
3. 若候选落在任一 cosmic footprint 内，或到 footprint 的最近距离小于 `--cosmic-match-radius-px`，则标记为 `cosmic_truth_match_flag=1`。
4. 如果 `temporal_flux_series` 可解析出候选最强 raw frame，并且同一 raw frame 上也匹配 cosmic footprint，则额外标记 `cosmic_truth_same_frame_flag=1`。
5. 如果无法解析最强 raw frame，仍保留检测块级 `cosmic_truth_match_flag`，但 `cosmic_truth_same_frame_flag` 置空或 `0`，并在报告中说明诊断强度较弱。
6. 输出最近 cosmic truth 的 frame、footprint、距离、`total_adu` 和 `peak_adu`。

新增建议参数：

```text
--cosmic-match-radius-px      默认 2.0
--cosmic-truth                显式指定 cosmic_events 目录；默认自动发现
--cosmic-frame-summaries      显式指定 frame_summaries 目录；默认自动发现
```

### 2. `false_template_star_like`

用于诊断星点或静态星残差。

优先使用 finder 输出：

```text
template_match_flag == 1
candidate_channel == template_source_brightening
nearest_template_dist_px
nearest_template_x
nearest_template_y
```

但当前默认 finder 不写 `template_sources.csv`，且不启用 `template_match_sources` 时这些字段会是空或 `nan`。因此分析脚本应支持两种补救：

1. 如果 `<result-dir>/analysis/template_sources.csv` 存在，直接读取并 KD-tree 匹配。
2. 如果不存在，但 `<input-run>/copied_source_aux/cache/stars_*.npz` 存在，则优先用仿真前星表 cache 生成 `template_sources.csv`。这是全帧 run 的默认高效路径。
3. 如果不存在星表 cache，但有 `input_run` 和模板窗口信息，则按模板窗口图像临时构建模板星源 catalog，作为慢速回退路径。

建议可调参数：

```text
--star-match-radius-px 1.5
--star-source-threshold-sigma 4.0
```

输出字段：

```text
nearest_template_star_dist_px
nearest_template_star_x
nearest_template_star_y
template_star_peak_value
```

模板星源 catalog 生成后应写入分析输出，作为后续结果分析可复用的缓存表：

```text
<result-dir>/analysis/template_sources.csv
```

如果用户显式提供 `--template-sources-csv`，则直接读取该表；否则从模板窗口临时构建并写出。后续分析同一输入 run 时，可以把第一次生成的 `template_sources.csv` 作为参数传入，避免重复构建。

### 3. `false_repeated_previous_block`

用于多检测块结果。若候选已经被上一块 final candidate 匹配：

```text
previous_block_match_flag == 1
```

则分类为跨块重复候选。默认单块烟测不会出现这一类。

### 4. `false_saturated_or_bright_residual`

用于标记可能与饱和亮源或强残差相关的候选。第一版可以用启发式：

```text
source_peak_value 接近或超过 12 * 65535
或 residual_peak_value 很高但 truth 不匹配
```

该分类只能作为诊断线索，不能直接等价为星点。

### 5. `false_unknown_residual`

不满足以上分类的误检候选进入未知残差类。报告中需要列出按 SNR 或 residual flux 排名前 N 的未知误检，供人工检查。

## 输出表：`false_positive_diagnostics.csv`

每行一个误检候选：

```text
frame_start
frame_end
x
y
bin_x
bin_y
false_positive_class
diagnostic_tags
peak_pixel_snr
residual_flux
residual_peak_value
residual_npix
peak_npix
local_bkg_sigma
temporal_active_frames
temporal_max_single_frame_fraction
likely_cosmic_ray
nearest_template_star_dist_px
previous_block_match_flag
candidate_priority
```

## 输出表：`false_positive_groups.csv`

按检测块和分类汇总：

```text
frame_start
frame_end
false_positive_class
count
fraction_of_false_candidates
max_peak_pixel_snr
max_residual_flux
median_residual_flux
```

## 推荐报告结构

`analysis_report.md` 建议使用以下结构：

```text
# GRB Finder 结果分析报告

## 1. 输入和运行配置
## 2. 检测块摘要
## 3. GRB truth 覆盖和命中率
## 4. 命中 GRB 的候选排名
## 5. 未命中 GRB 诊断
## 6. 误检总体统计
## 7. 宇宙线类误检
## 8. 星点/模板源类误检
## 9. 其它高 SNR 误检
## 10. 数据缺失和诊断限制
```

## CLI 草案

建议命令形式：

```bash
conda run -n etbase python tools/analyze_grbfinder_results.py \
  --result-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_1x1_default_oneblock_smoke_20260529_222112 \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_1x1_default_oneblock_smoke_20260529_222112/analysis \
  --rank-by peak_pixel_snr \
  --top-n 50
```

常用参数：

```text
--result-dir                  finder 输出目录
--events-csv                  显式指定 GRB truth 表
--input-run                   显式指定输入 run
--template-sources-csv        显式指定模板星源 catalog
--cosmic-truth                显式指定 cosmic_events 目录，默认从 input-run 自动发现
--cosmic-frame-summaries      显式指定逐帧 cosmic summary 目录
--truth-match-radius-px       重新计算 GRB 命中时使用的半径
--cosmic-match-radius-px      cosmic footprint 匹配半径
--star-match-radius-px        星点/模板源误检诊断半径
--rank-by                     默认 peak_pixel_snr，可选 residual_flux/residual_peak_value
--top-n                       报告中列出的高排名候选数
--write-csv                   是否写出中间 CSV，默认 true
```

## 第一版实现范围

第一版建议实现：

1. 读取 manifest、summary、final candidates、measured candidates、events.csv。
2. 按检测块统计 truth GRB 数、命中数和命中率。
3. 输出每个 truth event 的命中状态，以及 `snr_rank`、`residual_flux_rank`、`residual_peak_rank`。
4. 自动发现并读取 `copied_source_aux/cosmic_events/frame_XXXXXX_events.npy`。
5. 用真实 cosmic footprint 对误检候选做 truth-level cosmic 归因。
6. 同时输出检测块级 `cosmic_truth_match_flag` 和同帧强证据 `cosmic_truth_same_frame_flag`。
7. 同时保留 `likely_cosmic_ray` 作为 finder advisory，不把它和 truth-level cosmic 混为一类。
8. 从模板窗口生成 `template_sources.csv`，并用它诊断星点类误检。
9. 对未命中 GRB 输出细分 `miss_reason`。
10. 写出 `analysis_summary.json`、`analysis_report.md`、`window_grb_recall.csv`、`truth_event_ranks.csv`、`false_positive_diagnostics.csv`、`template_sources.csv`。

## 第二版增强

第二版再考虑：

1. 若 future run 保存了 artifact mask，支持 mask overlap 级别的 cosmic 复核。
2. 输出高 SNR 误检 cutout 坐标清单，方便后续可视化。
3. 支持跨多个 result-dir 对比不同阈值或不同 binning 的 recall/false-positive tradeoff。

## 已闭环决策

1. 排名同时报告 `snr_rank`、`residual_flux_rank`、`residual_peak_rank`，并在 `analysis_report.md` 中解释。
2. cosmic truth 使用双层判定：检测块级 footprint 匹配作为主判据，同帧匹配作为强证据。
3. 星点误检需要生成模板星源 catalog，并写出为可复用的 `template_sources.csv`。
4. 未命中 GRB 原因需要细分；其中模板窗口已可见的事件使用 `present_in_template_window`。
5. 现阶段只输出 Markdown 和 CSV，不生成图表或 cutout 图片。
6. 分析诊断属于仓库工具，但必须与星上 `grbfinder/` 运行时模块隔离。
