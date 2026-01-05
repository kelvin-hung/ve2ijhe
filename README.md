# VE2Plume Streamlit Decision Tool

## What this is
A lightweight Streamlit dashboard wrapping your VE2Plume Python model (hysteresis-aware VE gravity-current plume simulator).
It supports:
- Single-case simulation (obs vs pred maps and time-series metrics)
- Optional local calibration (`refine_best`) for one case
- Monte Carlo UQ around calibrated parameters: P10/P50/P90 maps, uncertainty width, exceedance probability maps
- RTI/STI time-series plots if your VE module provides them (in `diag`)

## Files
- `app.py` : Streamlit app
- `requirements.txt` : dependencies
- `ve2plume_full_rti_sti.py` : your VE model (copy your version here, or set the path in the UI)

## Run locally (Windows)
1) Open Anaconda Prompt / terminal in this folder
2) Install dependencies:
   `pip install -r requirements.txt`
3) Run:
   `streamlit run app.py`

## Point the app to your data
In the sidebar:
- Set `Dataset folder` to your LMDB dataset directory (e.g., `C:\...\dataset`)
- Set `sim_id`
- Set `Nt` and `tidx`

## Deployment options
- **Lab server / workstation**: run Streamlit on an internal host and share the URL.
- **Streamlit Community Cloud**: works if you can package the dataset (often too big). Better: keep dataset on the server.
- **Docker**: easy to containerize for reproducible deployment.

If you want, I can also provide a `Dockerfile` + `docker-compose.yml` for a clean one-command deployment.
