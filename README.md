# Onboard 12-Frame Sum GRB Screener Notes

This directory tracks known algorithm issues and decisions for the onboard-style GRB transient screener.

The current adapted script is:

```text
/home/cxgao/ET/GRB/ET-GRB-finder/scripts/GRB_from_fullframe_uint16_sum_template_match_noplot.py
```

The original 501x501 prototype is left untouched:

```text
/home/cxgao/ET/GRB/ET-GRB-finder/wenxiong-version/GRB_from_integer20_npy_template_match_sum_noplot.py
```

## Current Purpose

The onboard algorithm should find suspicious locations and time windows, not prove a source is a real GRB. The expected onboard product is a compact candidate list so that buffered full-frame data can be cut out and downlinked. Ground processing can then use heavier photometry, PSF modeling, cross-frame checks, and artifact rejection.

## Main Design Choices

- Keep the 12-frame sum strategy because onboard memory is limited.
- Process the full frame in tiles with halo instead of materializing a full-frame difference cube.
- Detect candidates on the 12-frame residual sum, `current_sum - template_sum`, before any candidate measurement.
- Do not drop candidates simply because they match a template source. If template-source matching is explicitly enabled, matched sources are kept as `template_source_brightening`.
- Lower hard morphology threshold from the old `peak_npix >= 10` to default `peak_npix >= 4`.
- Lower local excess threshold from old `5 sigma` to default `3 sigma`.
- Lower template match radius from old `1.5 px` to default `0.75 px`.
- Remove hard-coded target coordinates. Optional truth matching is driven by `events.csv` only during ground validation.
- Add an advisory cosmic-ray flag based on small-cutout temporal persistence. It is not a hard rejection by default.
- Do not write the static template-source catalog by default. Use `--template-match-sources` only for ground diagnostics.

## Known Issues To Track

1. **First-window template contamination**
   - If a GRB occurs in the first 12-frame window, a first-window template contains the event.
   - This is unavoidable without a pre-existing sky template or a delayed/rolling template.
   - Current behavior: first-window events may be missed in onboard mode.

2. **Brightening on top of existing stars**
   - The original script dropped candidates within the template match radius.
   - Current behavior: matched sources are retained as `template_source_brightening`.
   - Remaining risk: if the local excess is small compared with source noise or saturation, detection can still fail.

3. **Cosmic rays**
   - A 12-frame sum can make a single-frame cosmic ray look like a transient.
   - Current behavior: read only small candidate cutouts across the 12 source frames and compute temporal support fields:
     - `temporal_active_frames`
     - `temporal_consecutive_active_frames`
     - `temporal_max_single_frame_fraction`
     - `likely_cosmic_ray`
   - This avoids storing a frame cube, but still gives a low-memory cosmic-ray warning.

4. **Accuracy vs recall**
   - Onboard selection should favor recall over purity.
   - Current behavior: `likely_cosmic_ray` does not reject candidates; it only marks them.
   - Ground processing should make the final astrophysical decision.

5. **Saturation**
   - Strong GRB injections and bright stars can saturate at `65535`.
   - Local excess flux is then a lower limit.
   - Candidate detection can still find saturated peaks, but photometric interpretation must happen on ground.

6. **Tile boundaries**
   - Tile processing uses a halo around each core tile.
   - Detections are kept only inside the core tile to avoid duplicates.
   - If future PSF/cutout sizes grow, the halo should grow with them.

7. **Threshold tuning**
   - Current defaults are intentionally permissive:
     - `source_threshold_sigma = 4`
     - `local_threshold_sigma = 3`
     - `effective_npix_threshold = 4`
     - `match_radius_px = 0.75`
   - These should be validated against injected truth and expected downlink budget.

8. **Ground validation path**
   - The adapted script can optionally read injected `events.csv` and annotate `truth_event_id`.
   - This is for validation only and must not be required for onboard deployment.

9. **Sparse paired-template residuals**
   - In paired-template validation, real GRB residual cutouts can have a zero-MAD background because nearly every residual pixel is exactly zero.
   - Previous behavior in the adapted copy returned zero flux whenever `local_bkg_sigma <= 0`.
   - Current behavior: if the residual sigma is zero, use a zero-or-median floor threshold and keep positive connected residuals.

10. **Candidate flood from full-frame source finding**
   - Previous raw-sum detection produced about 620k measured source candidates for one 12-frame full-frame window.
   - Current behavior: detect on `current_sum - template_sum`; on the same local paired-template smoke this reduced measured candidates from 620,217 to 6.
   - Remaining risk: if a flight template is contaminated, stale, or not aligned photometrically, residuals around static sources can reappear.
   - If future data still has candidate floods, add an explicit per-tile top-N residual-score budget.

## Local Smoke Test, 2026-05-29

The SSHFS run was too slow for repeated full-frame tile reads, so both full-frame directories were copied locally:

```text
/home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529
/home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s
```

Local copy checks:

- injected frames: 120
- template frames: 120
- checked first and last frames in both runs: `(9120, 8900)`, `uint16`

Residual paired-template smoke command processed all 10 full-frame windows:

```text
PYTHONPATH=/home/cxgao/ET/GRB conda run -n etbase python \
  /home/cxgao/ET/GRB/ET-GRB-finder/scripts/GRB_from_fullframe_uint16_sum_template_match_noplot.py \
  --input-run /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529 \
  --template-run /home/cxgao/Results/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s \
  --output-dir /home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_residual_full_paired_no_template_catalog \
  --truth-events-csv /home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529/events.csv \
  --tile-size 2048 --overwrite
```

Result:

- windows processed: 10
- measured candidates: 43
- final candidates: 43
- truth-matched final candidates: 43
- unique truth event ids matched: 20 / 20
- template source catalog written: no
- output size: about 36 KB

Historical raw-sum one-window smoke, before residual-first candidate detection:

- measured candidates: 620,217
- final candidates: 8
- truth-matched final candidates: 8
- output size: about 95 MB
