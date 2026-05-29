# 当前 GRB 星上 12 帧求和筛选算法设计

本文档固化当前脚本的算法设计：

```text
/home/cxgao/ET/GRB/GRB_find/GRB_from_fullframe_uint16_sum_template_match_noplot.py
```

当前算法的目标不是在星上确认一个候选一定是真实 GRB。它的目标是：在星上内存和算力受限的条件下，以较高召回率找出可疑的位置和时间窗口，使星上缓存的全幅图可以按候选位置切星并下传。真正的 GRB 判定、精细测光、PSF 拟合、伪迹排查应放在地面完成。

## 1. 输入数据布局

脚本读取一个 run 目录，要求帧文件放在：

```text
<run>/frames/frame_000000.npy
<run>/frames/frame_000001.npy
...
```

每一帧必须是二维整数全幅图。当前已经验证的数据形态是：

```text
shape = (9120, 8900)
dtype = uint16
range = [0, 65535]
```

脚本默认输入和输出路径使用本地 `~/Results`：

```text
input_run  = /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529
output_dir = /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_sum12_streaming
```

模板 run 可通过 `--template-run` 指定。如果不指定模板 run，脚本会使用输入 run 的第一个窗口作为模板。这个 fallback 更接近星上占位逻辑，但有明显风险：如果第一个窗口中已经有 GRB，模板会被污染，事件可能被减掉。

## 2. 帧窗口划分

算法按连续帧窗口处理数据。默认参数为：

```text
window_size = 12
stride      = 12
```

对 120 帧数据，这会产生 10 个不重叠窗口：

```text
0-11, 12-23, ..., 108-119
```

代码内部窗口使用半开区间：

```text
[frame_start, frame_end)
```

CSV 输出中的 `frame_end` 是闭区间的最后一帧。

参数含义：

- `window_size`：一个检测窗口中参与求和的帧数。当前每帧是 10 s 曝光，因此 12 帧对应 120 s。
- `stride`：相邻检测窗口之间的帧步长。
- `--max-windows`：调试或验证时限制处理窗口数量，不属于星上核心逻辑。

设计理由：

- 12 帧求和可以提高弱暂现源的可探测性。
- 不重叠窗口使计算量更稳定。
- 若未来需要更细的时间定位，可以设置 `stride < window_size`，但会增加计算量和重复候选。

## 3. Tile 流式处理

算法不在内存中保存整幅 12 帧 cube，而是按 tile 流式处理全幅图。

默认参数：

```text
tile_size = 1024
halo      = 12
```

对每个 core tile，实际读取带 halo 的扩展 tile：

```text
expanded rows = [row0 - halo, row1 + halo)
expanded cols = [col0 - halo, col1 + halo)
```

候选只在 core tile 内保留。如果某个候选只出现在 halo 区域，则由相邻 core tile 负责保留。这可以避免 tile 边界重复报点。

每个 tile 的主要内存对象为：

- `current_sum`：当前窗口的 12 帧 tile 求和图，`uint32`。
- `template_sum`：模板窗口的 12 帧 tile 求和图，`uint32`。
- 局部 residual/diff 数组，在需要差分时使用 `int64`。

这是当前算法最重要的星上内存策略：只保存 tile 级中间结果，不保存整幅多帧 cube。

## 4. 当前窗口求和

对某个 tile 和某个帧窗口，脚本计算：

```text
current_sum = sum(input_frame[i][tile] for i in frame_start:frame_end)
```

求和数组类型为：

```text
SUM_DTYPE = uint32
```

原因：

- 单帧是 `uint16`，最大值为 65535。
- 12 帧求和最大可到 `12 * 65535 = 786420`，超过 `uint16`。
- `uint32` 对当前窗口长度足够。

如果指定了 `--template-run`：

```text
template_sum = sum(template_frame[i][tile] for i in same frame_start:frame_end)
```

如果没有指定 `--template-run`：

```text
template_sum = sum(input_frame[i][tile] for i in 0:min(window_size, n_frames))
```

这个无模板 fallback 只是当前脚本支持的模式，不应视为最终星上模板策略。

## 5. Residual-First 候选检测

当前版本最关键的设计是 residual-first：先检测差分残差图，而不是先在原始求和图上找所有星斑。

每个 tile 先计算：

```text
residual = current_sum - template_sum
```

残差数组类型为：

```text
LOCAL_DIFF_DTYPE = int64
```

原因是模板相减后可能出现负值，不能用无符号类型。

候选像素 mask 为：

```text
residual > residual_threshold
and residual > 0
```

残差背景使用 median/MAD 估计：

```text
residual_bkg_median = median(residual)
residual_bkg_sigma  = 1.4826 * median(abs(residual - residual_bkg_median))
```

代码使用整数近似：

```text
sigma = (MAD * 14826 + 5000) // 10000
```

默认残差阈值为：

```text
residual_threshold_sigma = 3.0
residual_threshold = residual_bkg_median + 3.0 * residual_bkg_sigma
```

如果 `residual_bkg_sigma <= 0`，阈值改为：

```text
residual_threshold = max(residual_bkg_median, 0)
```

参数含义：

- `residual_threshold_sigma`：残差像素需要高出残差背景多少 sigma 才能进入连通域检测。
- 阈值越低，召回率越高，但假阳性更多。
- 阈值越高，候选更少，但可能漏掉较弱 GRB。

这一步的必要性：

- 原始 12 帧求和图中，普通恒星本身就是 PSF 星斑。
- 如果先在原始求和图上找 local maximum，静态恒星会产生约 62 万个候选。
- residual-first 会先减掉静态星源，只保留真正变亮的局部残差。

## 6. 残差连通域筛选

算法对正残差 mask 做 8 连通域标记：

```text
structure = ones((3, 3))
```

每个残差连通域会计算：

- `residual_npix`：连通域像素数。
- `residual_peak_value`：连通域内最大残差值。
- `residual_flux`：连通域残差总和，扣除残差背景。

默认连通域大小阈值：

```text
residual_min_npix = 2
residual_max_npix = 400
```

含义：

- `residual_min_npix = 2`：去掉单像素正残差。单像素残差更像宇宙线或坏点，不像 PSF 星斑。
- `residual_max_npix = 400`：去掉过大的正残差区域。这类区域更可能来自背景失配、大面积坏区、严重饱和或模板问题。

这一步是星上友好的简单形态筛选。它不是最终天体物理分类。

## 7. 候选位置选择

每个通过筛选的残差连通域输出一个候选位置。

候选中心选择规则：

1. 找到连通域内残差峰值最大的像素。
2. 如果多个像素有相同峰值，计算这些峰值像素的平均位置。
3. 选择距离该平均位置最近的峰值像素作为代表点。

输出坐标为探测器像元坐标：

```text
x = column index
y = row index
```

tile 内局部坐标会加回 tile origin。只有位于 core tile 内的候选会保留，halo 区域候选会被丢弃，避免相邻 tile 重复报点。

## 8. 可选模板星源标注

模板星源标注默认关闭。需要地面诊断时可显式开启：

```text
--template-match-sources
```

开启后，脚本会在模板求和图上用原始 local maximum 方式构建静态星源表，并用 KD-tree 查找每个 residual 候选最近的模板星源。

相关阈值：

```text
source_threshold_sigma = 4.0
max_filter_size        = 7
match_radius_px        = 0.75
```

含义：

- `source_threshold_sigma`：模板求和图上的星源峰值需超过 `median + 4 sigma`。
- `max_filter_size`：原始星源检测时的局部极大值滤波窗口大小，单位为像素。
- `match_radius_px`：候选距离模板星源小于等于 0.75 px 时，认为它位于已有星源附近。

开启后会输出：

- `nearest_template_dist_px`
- `nearest_template_x`
- `nearest_template_y`
- `template_match_flag`
- `candidate_channel`

`candidate_channel` 取值：

- `new_source`：附近没有模板星源。
- `template_source_brightening`：候选靠近模板星源，可能是已有星源上的变亮事件。

星上注意事项：

- 模板星源表可能达到约 62 万行。
- 当前默认不写 `template_sources.csv`。
- 该功能主要用于地面诊断，不应作为星上必需输出。

## 9. 局部残差 Cutout 测量

每个 residual 候选会进行一次局部 cutout 重测量。

默认参数：

```text
cut_half              = 9
cutout size           = 19 x 19
annulus_r_in          = 6.0 px
annulus_r_out         = 10.0 px
local_threshold_sigma = 3.0
seed_radius           = 1.5 px
```

局部差分为：

```text
local_diff = current_sum_cutout - template_sum_cutout
```

局部背景从环形区域估计：

```text
annulus_r_in <= radius <= annulus_r_out
```

若环形区域 sigma 为 0，则改用整个 cutout 估计。若仍为 0，则阈值使用：

```text
max(local_bkg_median, 0)
```

局部 signal mask 为：

```text
local_diff > local_threshold
```

其中：

```text
local_threshold = local_bkg_median + local_threshold_sigma * local_bkg_sigma
```

连通域选择规则：

- 首先尝试选择接触 `seed_radius = 1.5 px` 内种子区域的连通域。
- 如果找不到，则使用 fallback seed radius，即 `2.5 px`。
- 如果仍找不到，则局部测量返回 0。

输出字段：

- `local_excess_flux`：扣除局部背景后的连通域总残差。
- `peak_npix`：被选中局部连通域的像素数。
- `peak_excess_value`：局部连通域最大残差，扣除局部背景。
- `local_bkg_median`
- `local_bkg_sigma`
- `peak_pixel_snr`

阈值含义：

- `local_threshold_sigma = 3.0`：局部残差连通域检测阈值。
- `effective_npix_threshold = 4`：最终 pass 逻辑中，局部连通域像素数达到 4 就可通过空间形态条件。

## 10. 时间支持测量

只有局部残差测量为正时，才进行逐帧时间支持测量：

```text
local_excess_flux > 0 or peak_excess_value > 0
```

对窗口中的每一帧，算法读取候选附近的小 cutout。

默认参数：

```text
temporal_cut_half          = 5
temporal cutout size       = 11 x 11
temporal_aperture_radius   = 3.0 px
temporal_annulus_r_in      = 5.0 px
temporal_annulus_r_out     = 8.0 px
temporal_sigma             = 3.0
temporal_min_active_frames = 2
```

逐帧 aperture flux 为：

```text
aperture_sum - annulus_median * aperture_npix
```

如果提供了 paired template，逐帧 cutout 会先减去对应模板帧 cutout。

时间维 active 阈值为：

```text
median(flux_series) + temporal_sigma * MAD_sigma(flux_series)
```

并且最小阈值为 1。

输出字段：

- `temporal_active_frames`：逐帧 flux 高于时间阈值的帧数。
- `temporal_consecutive_active_frames`：最长连续 active 帧数。
- `temporal_max_single_frame_fraction`：最大单帧 flux 占总 flux 的比例。
- `temporal_flux_series`：逐帧 aperture flux 的 JSON 列表。

## 11. 宇宙线 Advisory Flag

当前宇宙线判断只是 advisory flag，不作为硬拒绝条件。

默认参数：

```text
cosmic_max_active_frames     = 1
cosmic_single_frame_fraction = 0.80
```

当同时满足以下条件时，候选被标记为疑似宇宙线：

```text
temporal_active_frames <= 1
and temporal_max_single_frame_fraction >= 0.80
```

含义：

- 如果信号主要集中在单帧，且该单帧占总 flux 的 80% 以上，则更像宇宙线。
- 当前算法仍保留该候选，因为星上策略优先保证不漏检。
- 后续可将该标记用于候选排序或地面优先级判断。

## 12. 最终候选 Pass 逻辑

当前最终 pass 条件为三者取或：

```text
pass_single_stack =
    peak_npix >= effective_npix_threshold
    or temporal_active_frames >= temporal_min_active_frames
    or peak_pixel_snr >= 5.0
```

默认值：

```text
effective_npix_threshold  = 4
temporal_min_active_frames = 2
peak_pixel_snr threshold  = 5.0
```

含义：

- `peak_npix >= 4`：局部残差 footprint 足够扩展，更像星斑而不是单像素噪声。
- `temporal_active_frames >= 2`：信号至少在两帧中有时间支持，有助于排除单帧宇宙线。
- `peak_pixel_snr >= 5.0`：局部峰值足够强时，即使其他条件弱，也允许通过。

该 pass 逻辑故意偏宽松。星上只负责生成可疑候选，不负责最终确认。

## 12.1 Residual 之后的显式检查开关

Residual 候选检测之后的检查可以显式关闭，用于评估星上算力和缓存开销。

```text
--no-local-shape-check
--no-temporal-check
--keep-all-residual-candidates
```

含义：

- `--no-local-shape-check`：跳过 19x19 局部 residual 形态重测。输出字段仍保留，但 `local_excess_flux`、`peak_npix`、`peak_excess_value` 直接使用 residual 连通域的 `residual_flux`、`residual_npix`、`residual_peak_value` 填充。
- `--no-temporal-check`：跳过逐帧 11x11 cutout 时间测量，也不计算宇宙线 advisory flag。时间字段填 0 或空序列。
- `--keep-all-residual-candidates`：跳过最终 `peak_npix` / `temporal_active_frames` / `peak_pixel_snr` pass 判断，让所有 residual 候选进入 `streaming_sum_transient_candidates.csv`。

这三个开关默认都不启用，因此默认行为仍然是执行局部形态检查、执行时间检查、再应用最终 pass 逻辑。

这些开关的设计目的不是改变 residual-first 检测本身，而是把 residual 之后的确认性检查变成可裁剪模块。若星上下传带宽足够、但算力或缓存更紧张，可以先关闭这些检查，只下传 residual 候选。

## 13. Truth Matching 验证逻辑

Truth matching 只用于注入实验验证，不属于星上检测流程。

脚本可读取：

```text
events.csv
```

默认 truth 路径为：

```text
<input-run>/events.csv
```

默认 truth 匹配半径：

```text
truth_match_radius_px = 12.0
```

候选满足以下条件时认为匹配 truth：

```text
distance(candidate_xy, truth_xy) <= truth_match_radius_px
and candidate window overlaps truth visible frame range
```

输出字段：

- `truth_match_flag`
- `truth_event_id`
- `truth_dist_px`

这些字段只能用于地面验证，不能作为星上运行依赖。

## 14. 输出文件

默认输出：

```text
streaming_sum_candidates_after_measurement.csv
streaming_sum_transient_candidates.csv
streaming_sum_summary.csv
manifest.json
```

可选诊断输出：

```text
template_sources.csv
```

只有显式设置 `--template-match-sources` 时才写 `template_sources.csv`。

### 14.1 候选 CSV 字段

窗口和坐标字段：

- `frame_start`
- `frame_end`
- `window_frame_count`
- `x`
- `y`

Residual-first 字段：

- `residual_peak_value`
- `residual_flux`
- `residual_npix`

局部测量字段：

- `local_excess_flux`
- `peak_npix`
- `peak_excess_value`
- `local_bkg_median`
- `local_bkg_sigma`
- `peak_pixel_snr`
- `local_shape_check_enabled`

时间支持字段：

- `temporal_check_enabled`
- `temporal_active_frames`
- `temporal_consecutive_active_frames`
- `temporal_max_single_frame_fraction`
- `likely_cosmic_ray`
- `temporal_flux_series`

验证字段：

- `truth_match_flag`
- `truth_event_id`
- `truth_dist_px`

### 14.2 Summary CSV 字段

每个窗口输出：

- `frame_start`
- `frame_end`
- `window_frame_count`
- `template_source_count`
- `initial_sources`
- `template_matched_sources_kept`
- `after_measurement`
- `final_candidates`

当前 residual-first 版本中：

- `initial_sources` 表示 residual 候选数，不再表示原始星源数。
- `template_source_count` 默认是 0，除非启用 `--template-match-sources`。

## 15. 当前验证快照

当前完整全幅 paired-template 验证命令：

```text
PYTHONPATH=/home/cxgao/ET/GRB conda run -n etbase python \
  /home/cxgao/ET/GRB/GRB_find/GRB_from_fullframe_uint16_sum_template_match_noplot.py \
  --input-run /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529 \
  --template-run /home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_residual_full_paired_no_template_catalog \
  --truth-events-csv /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529/events.csv \
  --tile-size 2048 --overwrite
```

验证结果：

```text
windows_processed = 10
candidates_after_measurement = 43
final_candidates = 43
truth_matched_final_candidates = 43
unique_truth_event_ids_matched = 20 / 20
template_source_catalog_written = false
output_size = about 36 KB
```

关闭 residual 之后检查的验证命令：

```text
PYTHONPATH=/home/cxgao/ET/GRB conda run -n etbase python \
  /home/cxgao/ET/GRB/GRB_find/GRB_from_fullframe_uint16_sum_template_match_noplot.py \
  --input-run /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529 \
  --template-run /home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_residual_full_no_post_checks \
  --truth-events-csv /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529/events.csv \
  --tile-size 2048 --no-local-shape-check --no-temporal-check \
  --keep-all-residual-candidates --overwrite
```

结果：

```text
windows_processed = 10
candidates_after_measurement = 43
final_candidates = 43
truth_matched_final_candidates = 43
unique_truth_event_ids_matched = 20 / 20
local_shape_check = false
temporal_check = false
keep_all_residual_candidates = true
output_size = about 28 KB
```

在当前注入数据上，关闭 residual 后续检查后候选数仍为 43。这说明当前候选数量主要由 residual-first 检测决定；后续检查主要提供形态、时间和宇宙线诊断字段，并未进一步压低候选数量。

对比历史 raw-sum 检测：

- raw-sum 单个 12 帧全幅窗口约产生 620k 个 measured candidates。
- residual-first 完整 120 帧只产生 43 个 measured candidates。
- 当前结果覆盖 20 个注入事件。

## 16. 已知限制

### 16.1 模板污染

如果模板中包含 GRB 事件，该事件可能被模板相减抵消。未提供 `--template-run`、使用输入 run 第一窗口做模板时，这个问题尤其严重。

### 16.2 模板配准和光度漂移

Residual-first 假设静态星源可以被模板干净减掉。如果指向、PSF、增益、背景、饱和行为在当前图和模板图之间不一致，静态星会留下残差，候选数可能重新上升。

### 16.3 饱和

强 GRB 或亮星可能饱和。当前 flux 和形态测量可用于检测，但不适合最终测光解释。

### 16.4 宇宙线

`residual_min_npix = 2` 可以拒绝单像素宇宙线，但多像素宇宙线仍可能通过。当前宇宙线逻辑只是标记，不硬拒绝。

### 16.5 跨窗口重复候选

脚本按窗口输出候选。持续时间较长或衰减较慢的事件可能在多个窗口中出现。地面流程需要按位置和时间合并候选。

### 16.6 星上模板策略尚未最终确定

当前脚本支持 paired-template 验证和第一窗口 fallback。真正星上部署仍需要明确模板策略，例如：

- 预先上传或预先计算静态天区模板。
- 延迟滚动模板。
- 滚动 median 或低分位背景模板。
- 模板相减前做姿态和光度归一化。

## 17. 参数总表

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `window_size` | 12 | 每个检测窗口求和的帧数。 |
| `stride` | 12 | 相邻检测窗口之间的帧步长。 |
| `tile_size` | 1024 | 流式处理时的 core tile 尺寸。 |
| `halo` | 12 | tile 外扩边界，用于边缘 cutout 和连通域安全处理。 |
| `input_bit_depth` | 16 | 输入帧的无符号整数位深检查。 |
| `max_filter_size` | 7 | 原始模板星源诊断检测中的局部极大值滤波尺寸。 |
| `source_threshold_sigma` | 4.0 | 原始模板星源检测阈值，只用于模板星源诊断。 |
| `residual_threshold_sigma` | 3.0 | residual 像素进入连通域前需要超过背景的 sigma 数。 |
| `residual_min_npix` | 2 | residual 连通域最小像素数。 |
| `residual_max_npix` | 400 | residual 连通域最大像素数。 |
| `match_radius_px` | 0.75 | 启用模板星源匹配时的最近邻匹配半径。 |
| `cut_half` | 9 | 局部 residual 测量 cutout 半宽。 |
| `annulus_r_in` | 6.0 | 局部背景环内半径。 |
| `annulus_r_out` | 10.0 | 局部背景环外半径。 |
| `local_threshold_sigma` | 3.0 | 局部 residual 连通域阈值。 |
| `seed_radius` | 1.5 | 选择局部连通域时的候选种子半径。 |
| `effective_npix_threshold` | 4 | 候选通过空间 footprint 条件的像素数阈值。 |
| `temporal_cut_half` | 5 | 逐帧时间 cutout 半宽。 |
| `temporal_aperture_radius` | 3.0 | 逐帧 aperture flux 半径。 |
| `temporal_annulus_r_in` | 5.0 | 逐帧背景环内半径。 |
| `temporal_annulus_r_out` | 8.0 | 逐帧背景环外半径。 |
| `temporal_sigma` | 3.0 | 时间序列 active 阈值的 sigma 数。 |
| `temporal_min_active_frames` | 2 | 候选通过时间支持条件所需 active 帧数。 |
| `cosmic_single_frame_fraction` | 0.80 | 宇宙线 advisory flag 的单帧占比阈值。 |
| `cosmic_max_active_frames` | 1 | 宇宙线 advisory flag 的 active 帧数阈值。 |
| `--no-local-shape-check` | false | 显式关闭局部形态重测，节省每候选 19x19 cutout 检查。 |
| `--no-temporal-check` | false | 显式关闭逐帧时间 cutout 检查，节省每候选多帧小 cutout 读取。 |
| `--keep-all-residual-candidates` | false | 跳过最终 pass 判断，所有 residual 候选均进入最终候选表。 |
