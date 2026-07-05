# Additions (ported from the second reviewed project)

This file documents two things added to this project as-is, from the other
"drowsiness-detection" project that was compared against it. **Nothing else in this
repository was changed.**

## 1. `src/low_light.py`

A standalone low-light enhancement module: gamma correction -> CLAHE -> histogram
equalization, stacked. It is a more aggressive 3-stage alternative to the CLAHE-only
step already used elsewhere in this project (`src/preprocessing.py`, untouched).

It is **not** automatically wired into the pipeline. To use it, import it yourself, e.g.
in `feature_extraction.py` or `realtime_infer.py`:

```python
from low_light import enhance_low_light
frame = enhance_low_light(frame)   # before passing frame to the feature extractor
```

## 2. `dashboard_streamlit.py`

A read-only Streamlit dashboard for reviewing your recorded sessions and your
existing-vs-proposed model comparison, useful for your project demo/review. It reads
files this project already produces:

- `data/processed/features.csv` (from `src/feature_extraction.py`)
- `docs/results/comparison_summary.csv` (from `src/compare_models.py`)
- `models/train_summary.json` (from `src/train.py`)

It also has an optional tab to visually demo `low_light.py` (upload a frame, see
before/after).

Run it with:

```bash
pip install -r requirements.txt -r requirements-additions.txt
streamlit run dashboard_streamlit.py
```

It does not touch the webcam, the LSTM, or the alert system -- `src/realtime_infer.py`
still owns real-time detection exactly as before.

## 3. `src/realtime_infer_live.py` + dashboard "Live Monitor" tab

A new script, separate from `src/realtime_infer.py` (which is untouched and still
works with no logging if that's all you want). `realtime_infer_live.py` reuses the
exact same existing modules (`feature_extraction.py`, `model.py`, `utils.py`,
`alert_system.py`, `calibration.py` -- none of them modified) but additionally
appends one row per processed frame to `logs/live_features.csv`
(timestamp, EAR_left/right/avg, MAR, pitch/yaw/roll, alert level).

The dashboard's new **Live Monitor** tab tails that CSV and redraws EAR/MAR/pose/
status charts roughly once a second while the checkbox "Start live monitoring" is
ticked -- this is what gives you the "graphs updating in the browser while the
detector runs" experience, the same way the second reviewed project's
detector + Streamlit dashboard worked together.

To use it, run these in two separate terminals:

```bash
# Terminal 1: webcam + LSTM + alerts + live CSV logging
python src/realtime_infer_live.py --source 0 --checkpoint models/best_model.pt

# Terminal 2: dashboard
pip install -r requirements.txt -r requirements-additions.txt
streamlit run dashboard_streamlit.py
# -> open http://localhost:8501, go to the "Live Monitor" tab, check
#    "Start live monitoring"
```
