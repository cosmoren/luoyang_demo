# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
`luoyang_demo` is a small Python (3.12) proof-of-concept for renewable-energy power forecasting. There is no web server, database, or Docker — just scripts and Tkinter desktop GUIs. Stub models live in `models/models.py`.

### Services / entry points
- Offline inference simulator (primary E2E demo): `inference/infer_offline.py` — Tkinter GUI, click "Start" to replay bundled historical weather and plot 4h/48h forecasts vs ground truth.
- Online inference GUI (shell only, no data wiring yet): `inference/infer_online.py`.
- Open-Meteo live fetcher (optional utility, needs network): `external/openmeteo.py`.
- NWP downloader (optional batch job): `external/dowload_nwp.py`.

### Non-obvious gotchas
- The GUIs require Tkinter (`python3-tk` system package) plus a display. In this environment a display is available at `DISPLAY=:1` — set it explicitly when launching (e.g. `DISPLAY=:1 python3 infer_offline.py`).
- `infer_offline.py`'s `__main__` passes a CSV path relative to the `inference/` directory (`../datasets/...`). Run it from inside `inference/` (`cd inference && python3 infer_offline.py`), otherwise it raises `FileNotFoundError`. (Calling `load_historical_data()` with no arg resolves the dataset via an absolute path.)
- After launching a GUI, if the window's bottom controls appear off-screen, reposition it with `xdotool` (installed), e.g. `xdotool search --name "Inference Offline" windowmove <wid> 40 40`.
- `external/dowload_nwp.py` writes to absolute paths under `/home/cosmo/workspace/data/...` (from `config/conf.yaml`) and pulls from Huawei Cloud OBS; it is not integrated into the inference GUIs. Edit `config/conf.yaml` paths before running locally.

### Dependencies
Python packages are listed in `requirements.txt` and installed by the startup update script. `python3-tk` is a system (apt) package required for the GUIs.
