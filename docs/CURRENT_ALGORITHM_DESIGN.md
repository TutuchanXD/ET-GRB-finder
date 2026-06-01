# Current GRB Onboard Sum-Screener Algorithm

This document freezes the current design implemented in:

```text
/home/cxgao/ET/GRB/ET-GRB-finder/grbfinder/
```

The preferred direct-run wrapper is:

```text
/home/cxgao/ET/GRB/ET-GRB-finder/scripts/grbfind.py
```

Only short entry-point script names are kept under `scripts/`.

The goal is not final GRB confirmation. The goal is onboard, low-memory, high-recall detection of suspicious positions and time windows, so buffered full-frame data can be cut into small stamps and downlinked for ground processing.

## 1. Input Product Layout

The script reads a run directory with this frame layout:

```text
<run>/frames/frame_000000.npy
<run>/frames/frame_000001.npy
...
```

Each frame must be a 2-D integer full-frame image. The current validated product is:

```text
shape = (9120, 8900)
dtype = uint16
range = [0, 65535]
```

The default input and output paths are local `~/Results` paths:

```text
input_run  = /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529
output_dir = /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_sum12_streaming
```

The template can be provided with `--template-run`. If no template run is provided, `--template-strategy` controls onboard-style template selection:

```text
--template-strategy rolling-previous   # script default; each detection window subtracts the latest complete prior input window
--template-strategy first-window       # legacy fallback; all detections subtract the first input window
```

The first input window is still template-only in both onboard-style modes. A GRB in that seed window can contaminate the template and be missed.

## 1.1 Spatial Binning And Entry Points

The implementation uses a two-dimensional spatial bin setting:

```text
--spatial-bin N    # NxN
--spatial-bin RxC  # row bin R, column bin C
```

Examples:

```bash
python scripts/grbfind.py --spatial-bin 3
python scripts/grbfind.py --spatial-bin 3x4
```

The shorter wrappers are preferred:

```text
scripts/grbfind.py      -> 1x1 default
scripts/grbfind-bin2.py -> 2x2 default
scripts/grbfind-bin3.py -> 3x3 default
```

All three wrappers default to `--template-strategy rolling-previous`.
Each wrapper also contains a complete editable `SCRIPT_DEFAULTS` dictionary, so
thresholds and other run parameters can be changed directly in the script before
launching a run.
The wrappers also default to `max_windows = 2`, which keeps the first `0-11`
window as the seed template and runs only the `12-23` detection window.

Each detection pixel is the sum of the corresponding non-overlapping input
block. Candidate output contains both `bin_x`/`bin_y` detection-grid
coordinates and `x`/`y` detector coordinates derived from the block center.

## 2. Frame Windowing

The algorithm processes consecutive frame windows. Defaults:

```text
window_size = 12
stride      = 12
```

For 120 frames, this gives 10 non-overlapping windows:

```text
0-11, 12-23, ..., 108-119
```

The frame window is half-open internally, `[frame_start, frame_end)`, but CSV outputs use inclusive `frame_end`.

Meaning:

- `window_size`: number of 10 s exposures summed for one detection window.
- `stride`: frame offset between detection windows.
- `--max-windows`: optional validation/debug limit.

Design reason:

- Summing 12 frames improves weak transient detectability.
- Non-overlap keeps onboard compute predictable.
- The current implementation can be generalized to overlapping windows by setting `stride < window_size`, but that increases compute and duplicate detections.

## 3. Tile Streaming

The full frame is not materialized as a full-frame stack. Each window is processed tile by tile.

Defaults:

```text
tile_size = 1024
halo      = 12
```

For each core tile, the algorithm reads an expanded tile:

```text
expanded rows = [row0 - halo, row1 + halo)
expanded cols = [col0 - halo, col1 + halo)
```

The halo provides enough margin for cutout, local background, and connected-component measurement near tile boundaries. Candidate centers are retained only when they fall inside the core tile. This avoids duplicate detections across neighboring tiles.

Memory objects per tile are mainly:

- `current_sum`: `uint32` sum of current window frames.
- `template_sum`: `uint32` sum of template window frames.
- local residual/difference arrays in `int64` where needed.

This is the main onboard memory strategy: use tile-level products, not a full 12-frame cube.

## 4. Window Sum Construction

For a tile and a frame window, the script computes:

```text
current_sum = sum(input_frame[i][tile] for i in frame_start:frame_end)
```

The sum dtype is:

```text
SUM_DTYPE = uint32
```

Reason:

- A 12-frame sum of `uint16` can exceed `65535`.
- `uint32` is enough for the current 12-frame window: `12 * 65535 = 786420`.

If `--template-run` is provided:

```text
template_sum = sum(template_frame[i][tile] for i in same frame_start:frame_end)
```

If `--template-run` is not provided:

```text
--template-strategy first-window:
template_sum = sum(input_frame[i][tile] for i in 0:min(window_size, n_frames))

--template-strategy rolling-previous:
template_sum = sum(input_frame[i][tile] for i in previous complete window)
```

The first input window is template-only in both onboard-style modes.

## 5. Residual-First Candidate Detection

The current key design is residual-first detection. Instead of finding all star-like local maxima in the raw 12-frame sum, the script first computes:

```text
residual = current_sum - template_sum
```

The residual dtype is:

```text
LOCAL_DIFF_DTYPE = int64
```

Using a signed type is necessary because template subtraction can produce negative residuals.

The candidate mask is:

```text
residual > residual_threshold
and residual > 0
```

The residual background is estimated using median and MAD:

```text
residual_bkg_median = median(residual)
residual_bkg_sigma  = 1.4826 * median(abs(residual - residual_bkg_median))
```

The integer implementation uses:

```text
sigma = (MAD * 14826 + 5000) // 10000
```

Default threshold:

```text
residual_threshold_sigma = 3.0
residual_threshold = residual_bkg_median + 3.0 * residual_bkg_sigma
```

If `residual_bkg_sigma <= 0`, the threshold becomes:

```text
residual_threshold = max(residual_bkg_median, 0)
```

Meaning:

- `residual_threshold_sigma`: how far above the residual background a pixel must be to seed a residual component.
- A lower value increases recall and false positives.
- A higher value lowers candidate count but may miss faint GRB events.

Why this step matters:

- Raw-sum detection produced about 620k measured candidates per 12-frame full-frame window because every static star is a PSF-like local maximum.
- Residual-first detection suppresses static stars before candidate measurement.

## 6. Residual Connected Components

The positive residual mask is labeled with 8-connected components:

```text
structure = ones((3, 3))
```

For each connected residual component, the algorithm computes:

- `residual_npix`: number of pixels in the component.
- `residual_peak_value`: max residual value inside the component.
- `residual_flux`: sum of component residual values after subtracting residual background.

Default component-size thresholds:

```text
residual_min_npix = 2
residual_max_npix = 400
```

Meaning:

- `residual_min_npix = 2` rejects isolated single-pixel positive residuals, which are common cosmic-ray-like artifacts.
- `residual_max_npix = 400` rejects very large positive regions, such as broad artifacts, severe background mismatch, or large saturated areas.

These are intentionally simple onboard morphology cuts. They are not final astrophysical classification.

## 7. Candidate Position Selection

Each accepted residual component is represented by one candidate center.

The center is the pixel with maximum residual value. If multiple pixels share the same peak residual, the algorithm chooses the peak pixel closest to the average position of the peak plateau.

The output coordinates are detector pixel coordinates:

```text
x = column index
y = row index
```

Tile origin is added back after local tile processing. Candidates outside the core tile are discarded, even if they appear in the halo, so neighboring tiles do not duplicate them.

## 8. Optional Template-Source Annotation

Template-source matching is optional and off by default.

Enable it with:

```text
--template-match-sources
```

When enabled, the algorithm builds a static source catalog from the template sum using raw-sum local maxima. It then finds the nearest template source with a KD-tree.

Thresholds for template-source detection:

```text
source_threshold_sigma = 4.0
max_filter_size        = 7
match_radius_px        = 0.75
```

Meanings:

- `source_threshold_sigma`: raw template-sum local maxima must exceed `median + 4 sigma`.
- `max_filter_size`: local maximum filter size, in pixels.
- `match_radius_px`: nearest template source is considered matched if within 0.75 px.

Outputs when enabled:

- `nearest_template_dist_px`
- `nearest_template_x`
- `nearest_template_y`
- `template_match_flag`
- `candidate_channel`

`candidate_channel` is:

- `new_source` when no nearby template source is found.
- `template_source_brightening` when a residual candidate is near a template source.

Important onboard note:

- The static template-source catalog can contain about 620k rows for one full-frame 12-frame window.
- It is not written by default.
- It should be used for ground diagnostics, not as a required onboard product.

## 9. Local Residual Cutout Measurement

For each residual candidate, the script re-measures a local cutout around the candidate.

Default cutout parameters:

```text
cut_half         = 9
cutout size      = 19 x 19
annulus_r_in     = 6.0 px
annulus_r_out    = 10.0 px
local_threshold_sigma = 3.0
seed_radius      = 1.5 px
```

The local difference is:

```text
local_diff = current_sum_cutout - template_sum_cutout
```

The local background is estimated from the annulus using median/MAD. If annulus sigma is zero, the whole cutout is used. If sigma is still zero, the threshold is floored at:

```text
max(local_bkg_median, 0)
```

The local signal mask is:

```text
local_diff > local_threshold
```

Connected components are labeled. The component used for measurement is the component touching the candidate seed region:

```text
seed_radius = 1.5 px
fallback seed radius = 2.5 px
```

Measured fields:

- `local_excess_flux`: component sum after subtracting local background.
- `peak_npix`: number of pixels in the selected component.
- `peak_excess_value`: maximum component residual above local background.
- `local_bkg_median`
- `local_bkg_sigma`
- `peak_pixel_snr`

Meaning of thresholds:

- `local_threshold_sigma = 3.0`: local connected-component threshold relative to annulus residual noise.
- `effective_npix_threshold = 4`: minimum local component area used only when the optional local peak SNR final gate is enabled.

## 10. Temporal Support Measurement

Temporal support is measured only when the local residual measurement is positive:

```text
local_excess_flux > 0 or peak_excess_value > 0
```

For each frame in the window, the algorithm reads a small cutout around the candidate.

Defaults:

```text
temporal_cut_half          = 5
temporal cutout size       = 11 x 11
temporal_aperture_radius   = 3.0 px
temporal_annulus_r_in      = 5.0 px
temporal_annulus_r_out     = 8.0 px
temporal_sigma             = 3.0
temporal_min_active_frames = 2
```

Per-frame aperture flux is:

```text
aperture_sum - annulus_median * aperture_npix
```

If a paired template is available, the template frame cutout is subtracted before aperture measurement.

The temporal active threshold is:

```text
median(flux_series) + temporal_sigma * MAD_sigma(flux_series)
```

with a minimum threshold of 1.

Measured temporal fields:

- `temporal_active_frames`: number of frames above the temporal threshold.
- `temporal_consecutive_active_frames`: longest consecutive run of active frames.
- `temporal_max_single_frame_fraction`: max single-frame flux divided by total flux.
- `temporal_flux_series`: JSON list of per-frame aperture fluxes.

These fields are diagnostics and priority signals. They are not part of the current final hard gate.

## 11. Cosmic-Ray Advisory Flag

The cosmic-ray flag is advisory only. It is not a hard rejection.

Defaults:

```text
cosmic_max_active_frames        = 1
cosmic_single_frame_fraction    = 0.80
```

The candidate is marked as likely cosmic-ray-like when:

```text
temporal_active_frames <= 1
and temporal_max_single_frame_fraction >= 0.80
```

Meaning:

- A signal dominated by one frame is suspicious for a cosmic ray.
- The flag should help ground-side triage or future onboard ranking.
- It should not currently suppress candidates, because the onboard policy favors recall over purity.

## 12. Candidate Pass Logic

The final gate runs after residual connected-component detection, residual peak/flux prefiltering, template annotation, local cutout measurement, and optional temporal diagnostics.

Default behavior is intentionally simple:

```text
if peak_pixel_snr_check is false:
    pass_single_stack = true
```

In other words, a candidate that survived the residual-first chain enters the final table by default. Local and temporal measurements are still written as diagnostic fields, but they do not reject candidates.

When the optional local peak SNR gate is enabled, the gate requires both local footprint and local peak significance:

```text
if peak_pixel_snr_check is true:
    pass_single_stack =
        peak_npix >= effective_npix_threshold
        and peak_pixel_snr >= peak_pixel_snr_threshold
```

Defaults:

```text
peak_pixel_snr_check = false
effective_npix_threshold = 4
peak_pixel_snr_threshold = 5.0
```

Meaning:

- `peak_pixel_snr_check = false`: keep residual-prefiltered candidates by default.
- `peak_npix >= 4`: when the optional gate is enabled, require the local residual component to have at least four pixels.
- `peak_pixel_snr >= 5.0`: when the optional gate is enabled, require the local residual peak to be significant relative to the local robust sigma.
- `peak_pixel_snr` is a local residual significance metric, not physical source SNR.

Previous-window association is controlled by `previous_block_match_check` and is disabled by default. When enabled, if a current candidate matches a previous final candidate within `previous_match_radius_px`, it is kept and labeled as `confirmed_previous_block`.

## 12.1 Explicit Post-Residual Check Switches

Checks after residual candidate detection can be explicitly disabled to evaluate onboard compute and buffering cost.

```text
--no-local-shape-check
--no-temporal-check
--previous-block-match-check / --no-previous-block-match-check
--keep-all-residual-candidates
```

Meanings:

- `--no-local-shape-check`: skip 19x19 local residual morphology remeasurement. Output fields are still populated, but `local_excess_flux`, `peak_npix`, and `peak_excess_value` are filled directly from `residual_flux`, `residual_npix`, and `residual_peak_value`.
- `--no-temporal-check`: skip per-frame 11x11 temporal cutout measurement and cosmic-ray advisory calculation. Temporal fields are filled with zeros or an empty series.
- `--keep-all-residual-candidates`: bypass residual peak/flux prefiltering and the final peak-SNR gate, then write every residual connected-component candidate to `streaming_sum_transient_candidates.csv`.

In the current script wrappers, local cutout measurement is enabled, temporal measurement is disabled, and `keep_all_residual_candidates` is disabled. The default output therefore keeps residual-prefiltered candidates unless the optional local peak SNR gate is explicitly enabled.

These switches do not change residual-first detection. They make the post-residual confirmation checks optional. If downlink budget is sufficient but onboard compute or cache is tighter, the flight configuration can keep only residual candidates and defer confirmation to the ground.

## 13. Truth Matching For Validation

Truth matching is optional and for validation only. It is not part of onboard detection.

Truth event metadata is read from:

```text
events.csv
```

The default truth path is:

```text
<input-run>/events.csv
```

The truth match radius default is:

```text
truth_match_radius_px = 12.0
```

A candidate is truth-matched when:

```text
distance(candidate_xy, truth_xy) <= truth_match_radius_px
and candidate window overlaps truth visible frame range
```

Output fields:

- `truth_match_flag`
- `truth_event_id`
- `truth_dist_px`

These fields must not be required by a flight pipeline.

## 14. Output Files

The script writes:

```text
streaming_sum_candidates_after_measurement.csv
streaming_sum_transient_candidates.csv
streaming_sum_summary.csv
manifest.json
```

Optional diagnostic output:

```text
template_sources.csv
```

`template_sources.csv` is written only when `--template-match-sources` is set.

### Candidate CSV Fields

Key coordinate and window fields:

- `frame_start`
- `frame_end`
- `window_frame_count`
- `x`
- `y`
- `bin_x`
- `bin_y`

Residual-first fields:

- `residual_peak_value`
- `residual_flux`
- `residual_npix`

Local measurement fields:

- `local_excess_flux`
- `peak_npix`
- `peak_excess_value`
- `local_bkg_median`
- `local_bkg_sigma`
- `peak_pixel_snr`
- `local_shape_check_enabled`

Temporal fields:

- `temporal_check_enabled`
- `temporal_active_frames`
- `temporal_consecutive_active_frames`
- `temporal_max_single_frame_fraction`
- `likely_cosmic_ray`
- `temporal_flux_series`

Validation fields:

- `truth_match_flag`
- `truth_event_id`
- `truth_dist_px`

### Summary CSV Fields

Per-window fields:

- `frame_start`
- `frame_end`
- `window_frame_count`
- `template_frame_start`
- `template_frame_end`
- `template_source_count`
- `initial_sources`
- `template_matched_sources_kept`
- `after_measurement`
- `final_candidates`

Current residual-first behavior makes `initial_sources` the number of residual candidates, not the number of raw stars.

The manifest records the spatial binning as:

- `spatial_bin_size`: legacy square-bin value, or `null` for non-square bins.
- `spatial_bin_rows`
- `spatial_bin_cols`

## 14.3 Development Smoke Policy

Any code change that is not specifically about spatial binning should include a `1x1` smoke run through `scripts/grbfind.py`. Spatial-binning-only changes can use targeted bin smoke tests, with `1x1` added when shared pipeline behavior is touched.

## 15. Current Validation Snapshot

The current full-frame paired-template validation command was:

```text
conda run -n etbase python \
  /home/cxgao/ET/GRB/ET-GRB-finder/scripts/grbfind.py \
  --input-run /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529 \
  --template-run /home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_residual_full_paired_no_template_catalog \
  --truth-events-csv /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529/events.csv \
  --tile-size 2048 --overwrite
```

Observed result:

```text
windows_processed = 10
candidates_after_measurement = 43
final_candidates = 43
truth_matched_final_candidates = 43
unique_truth_event_ids_matched = 20 / 20
template_source_catalog_written = false
output_size = about 36 KB
```

Validation command with post-residual checks disabled:

```text
conda run -n etbase python \
  /home/cxgao/ET/GRB/ET-GRB-finder/scripts/grbfind.py \
  --input-run /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529 \
  --template-run /home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_residual_full_no_post_checks \
  --truth-events-csv /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529/events.csv \
  --tile-size 2048 --no-local-shape-check --no-temporal-check \
  --keep-all-residual-candidates --overwrite
```

Observed result:

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

On the current injected data, disabling post-residual checks leaves the candidate count at 43. This means candidate count is currently controlled by residual-first detection; the post-residual checks mainly add morphology, timing, and cosmic-ray diagnostic fields instead of reducing candidate count.

Historical raw-sum candidate detection produced about 620k candidates for a single 12-frame full-frame window. The residual-first design is the change that made candidate count compatible with onboard downlink selection.

## 16. Known Limitations

### 16.1 Template Contamination

If the template contains a GRB event, that event can be subtracted away. This is especially important when no `--template-run` is provided and the first input window is used as the template.

### 16.2 Template Alignment And Photometric Drift

The residual-first design assumes that static stars subtract cleanly. If pointing, PSF, gain, background, or saturation behavior differs between current and template frames, static stars can leave residuals and increase candidate count.

### 16.3 Saturation

Saturated GRB cores and saturated stars can distort flux and morphology. Current measurements are suitable for detection but not final photometry.

### 16.4 Cosmic Rays

Single-pixel residual components are rejected by `residual_min_npix = 2`. Multi-pixel cosmic rays can still pass. The current cosmic-ray logic only marks suspicious temporal behavior.

### 16.5 Candidate Duplicates Across Windows

The script reports candidates per window. A long or slowly decaying event can appear in multiple windows. Ground processing should merge candidates by event position and overlapping time.

### 16.6 Flight Template Policy Still Needs Flight Validation

The current script supports paired-template validation, fixed first-window fallback, and rolling previous-window templates. A flight version still needs validation against realistic pointing, background, and long-duration transient cases. Possible later policies include:

- pre-uploaded or precomputed static sky template,
- robust rolling median/low-percentile background,
- separate attitude/photometric normalization before subtraction.

## 17. Parameter Summary

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `window_size` | 12 | Number of frames summed per detection window. |
| `stride` | 12 | Frame step between detection windows. |
| `max_windows` | 2 in scripts | Maximum number of raw windows considered, including the seed template window. With the script default, only one detection block is processed. |
| `--template-strategy` | `rolling-previous` in scripts | Input-run template strategy when no `--template-run` is supplied: `first-window` or `rolling-previous`. |
| `tile_size` | 1024 | Core tile size for streaming full-frame processing. |
| `halo` | 12 | Extra tile margin for boundary-safe cutouts. |
| `input_bit_depth` | 16 | Expected unsigned input range validation. |
| `max_filter_size` | 7 | Local maximum filter size for raw template-source diagnostics. |
| `source_threshold_sigma` | 4.0 | Raw template-source detection threshold, used only for template-source diagnostics. |
| `residual_threshold_sigma` | 3.0 | Residual pixel threshold above residual background. |
| `residual_min_npix` | 2 | Minimum residual connected-component size. |
| `residual_max_npix` | 400 | Maximum residual connected-component size. |
| `match_radius_px` | 0.75 | Nearest template-source match radius when template matching is enabled. |
| `cut_half` | 9 | Half-size of local residual measurement cutout. |
| `annulus_r_in` | 6.0 | Inner radius of local background annulus. |
| `annulus_r_out` | 10.0 | Outer radius of local background annulus. |
| `local_threshold_sigma` | 3.0 | Local residual component threshold above local background. |
| `seed_radius` | 1.5 | Candidate seed radius for selecting the local component. |
| `peak_pixel_snr_check` | false | Enable the optional local peak SNR final gate. When false, residual-prefiltered candidates pass final by default. |
| `effective_npix_threshold` | 4 | Local component area required when `peak_pixel_snr_check=true`. |
| `peak_pixel_snr_threshold` | 5.0 | Local residual peak-significance threshold used when `peak_pixel_snr_check=true`. |
| `temporal_cut_half` | 5 | Half-size of per-frame temporal cutout. |
| `temporal_aperture_radius` | 3.0 | Aperture radius for per-frame flux series. |
| `temporal_annulus_r_in` | 5.0 | Inner radius of temporal background annulus. |
| `temporal_annulus_r_out` | 8.0 | Outer radius of temporal background annulus. |
| `temporal_sigma` | 3.0 | Per-frame active threshold above temporal median/MAD. |
| `temporal_min_active_frames` | 2 | Number of active frames used for temporal priority labeling. |
| `cosmic_single_frame_fraction` | 0.80 | Single-frame dominance threshold for cosmic-ray advisory flag. |
| `cosmic_max_active_frames` | 1 | Active-frame count threshold for cosmic-ray advisory flag. |
| `--no-local-shape-check` | false | Explicitly disable local morphology remeasurement and avoid each candidate's 19x19 cutout check. |
| `--no-temporal-check` | false | Explicitly disable per-frame temporal cutout checks. |
| `--keep-all-residual-candidates` | false | Bypass residual peak/flux prefiltering and final peak-SNR gate. |
