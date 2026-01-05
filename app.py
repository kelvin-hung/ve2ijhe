# app.py
# Streamlit decision-support dashboard for the VE2Plume hydrogen plume model
# Usage (local):
#   pip install -r requirements.txt
#   streamlit run app.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import json
import time
import zipfile
import shutil
import hashlib
from pathlib import Path
import importlib.util

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_ve_module(ve_py_path: str):
    ve_py_path = str(Path(ve_py_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("ve2plume", ve_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import VE module from: {ve_py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False)
def load_case_from_lmdb(ve_py_path: str, dataset_dir: str, sim_id: int, nt_max_guess: int = 8000):
    ve = load_ve_module(ve_py_path)
    phi, k, p_list, sg_obs = ve.load_sim_to_memory(dataset_dir, int(sim_id), Nt_max=nt_max_guess)
    p = np.asarray(p_list, dtype=np.float32)
    sg = np.asarray(sg_obs, dtype=np.float32)
    return phi.astype(np.float32), k.astype(np.float32), p, sg


def _normalize_k(ve, k: np.ndarray):
    k_eff = float(ve.geom_mean(k))
    return (k / (k_eff + 1e-12)).astype(np.float32), k_eff


def _ensure_well(ve, phi, p_series):
    well = ve.detect_well_from_dp(p_series)
    if well is None:
        nx, ny = phi.shape
        well = (nx // 2, ny // 2)
    return well


def _plot_map(M, title, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_lines(x, ys, labels, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if labels:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def _dict_input(title, d, keys, help_text=None):
    st.subheader(title)
    if help_text:
        st.caption(help_text)
    out = dict(d)
    cols = st.columns(3)
    for i, k in enumerate(keys):
        col = cols[i % 3]
        v0 = float(out.get(k, 0.0))
        out[k] = float(col.number_input(k, value=v0, format="%.6g"))
    return out


def _area_t(mask, dx=1.0, dy=1.0):
    return mask.sum(axis=(1, 2)).astype(np.float64) * float(dx) * float(dy)


def _radius_eq_t(area):
    return np.sqrt(area / np.pi)


# -----------------------------
# NEW: Uploaded dataset support
# -----------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _prepare_upload_root() -> Path:
    root = (Path(__file__).parent / "uploaded_datasets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _looks_like_lmdb_folder(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    names = {x.name.lower() for x in p.iterdir() if x.is_file()}
    # LMDB usually has data.mdb + lock.mdb
    if "data.mdb" in names and "lock.mdb" in names:
        return True
    # Sometimes users upload only data.mdb (still can work if lock is created by LMDB)
    if "data.mdb" in names:
        return True
    return False


def _write_uploaded_dataset(uploaded_files, uploaded_zip) -> str | None:
    """
    Returns a folder path that contains LMDB files, or None if nothing uploaded.
    Supported:
      - ZIP containing a folder with data.mdb/lock.mdb (or just data.mdb)
      - Multiple files upload containing data.mdb (+ optional lock.mdb)
    """
    root = _prepare_upload_root()

    # Case A: ZIP upload
    if uploaded_zip is not None:
        zbytes = uploaded_zip.getvalue()
        tag = _hash_bytes(zbytes)
        outdir = root / f"zip_{tag}"
        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
                zf.extractall(outdir)

        # Find LMDB folder inside extracted tree
        # Prefer: a directory that directly contains data.mdb
        candidates = []
        for p in outdir.rglob("*"):
            if p.is_dir() and _looks_like_lmdb_folder(p):
                candidates.append(p)
        if len(candidates) == 0:
            return None
        # Choose the shallowest candidate
        candidates.sort(key=lambda p: len(p.parts))
        return str(candidates[0].resolve())

    # Case B: multiple files upload
    if uploaded_files:
        # Concatenate bytes for stable hash
        blob = b"".join([f.getvalue() for f in uploaded_files])
        tag = _hash_bytes(blob)
        outdir = root / f"files_{tag}"
        outdir.mkdir(parents=True, exist_ok=True)

        for f in uploaded_files:
            name = Path(f.name).name
            # If user uploads "data" (no extension), keep it but also mirror to data.mdb for robustness
            out_path = outdir / name
            out_path.write_bytes(f.getvalue())
            if name.lower() == "data":
                (outdir / "data.mdb").write_bytes(f.getvalue())
            if name.lower() == "lock":
                (outdir / "lock.mdb").write_bytes(f.getvalue())

        # If we only have data.mdb, LMDB can often create lock.mdb on access;
        # but some environments need it present. We'll allow either.
        if _looks_like_lmdb_folder(outdir):
            return str(outdir.resolve())

    return None


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="VE2Plume – H2 plume decision tool", layout="wide")
st.title("VE2Plume – Hysteresis-aware VE gravity-current model (decision dashboard)")

with st.sidebar:
    st.header("1) Paths")

    ve_py_path = st.text_input(
        "VE model .py path",
        value=str(Path(__file__).with_name("ve2plume_full_rti_sti.py")),
        help="Point this to your VE python file. Default assumes it sits next to app.py."
    )

    st.markdown("**Dataset input (choose one):**")

    # Upload option (ZIP or LMDB files)
    uploaded_zip = st.file_uploader(
        "Upload dataset as ZIP (recommended)",
        type=["zip"],
        accept_multiple_files=False,
        help="ZIP should contain an LMDB folder with data.mdb and (optionally) lock.mdb."
    )

    uploaded_files = st.file_uploader(
        "…or upload LMDB files (data.mdb + lock.mdb)",
        type=["mdb", "lmdb", "dat", ""],
        accept_multiple_files=True,
        help="Upload data.mdb (required) and lock.mdb (optional). If your file is named 'data' with no extension, upload it here."
    )

    # Fallback path option (original behavior)
    default_dataset = str((Path(__file__).parent / "dataset").resolve())
    dataset_dir_text = st.text_input(
        "Dataset folder (fallback path)",
        value=default_dataset,
        help="Used only if you don't upload files above. Folder containing LMDB simulations (data.mdb/lock.mdb)."
    )

    out_dir = st.text_input("Output folder", value="streamlit_outputs", help="Where to write figures/json exports.")

    # Resolve dataset_dir: uploads override textbox
    dataset_dir_uploaded = None
    if uploaded_zip is not None or (uploaded_files is not None and len(uploaded_files) > 0):
        try:
            dataset_dir_uploaded = _write_uploaded_dataset(uploaded_files, uploaded_zip)
        except Exception as e:
            st.error(f"Failed to prepare uploaded dataset: {e}")
            dataset_dir_uploaded = None

    dataset_dir = dataset_dir_uploaded or dataset_dir_text
    st.caption(f"Resolved dataset_dir: {Path(dataset_dir).expanduser().resolve()}")

    st.header("2) Case selection")
    sim_id = st.number_input("sim_id", min_value=0, max_value=10_000_000, value=0, step=1)
    nt = st.number_input("Nt (time steps to use)", min_value=2, max_value=10_000, value=61, step=1)

    st.header("3) Controls")
    tidx = st.number_input("Map time index (tidx)", min_value=0, max_value=10_000, value=60, step=1)
    thr = st.number_input("Threshold (Sg > thr)", min_value=0.0, max_value=1.0, value=0.02, step=0.005, format="%.4f")

    st.header("4) Model switches")
    fit_pressure = st.checkbox("Fit pressure (slower)", value=False)
    use_pred_p_for_vel = st.checkbox("Use predicted pressure for velocity (advanced)", value=False)

    st.header("5) UQ")
    uq_on = st.checkbox("Enable Monte Carlo UQ", value=False)
    B = st.number_input("MC samples (B)", min_value=10, max_value=5000, value=200, step=10)
    uq_mode = st.selectbox("Sampling", ["lognormal", "gaussian"], index=0)
    uq_scale = st.slider("Noise scale", min_value=0.01, max_value=0.60, value=0.20, step=0.01)
    uq_seed = st.number_input("Seed", min_value=0, max_value=2_000_000_000, value=123, step=1)

    st.header("6) Calibration")
    do_calib = st.checkbox("Run local calibration around START_PARAMS_DEFAULT", value=False)
    n_iter = st.number_input("Calibration iterations", min_value=5, max_value=500, value=60, step=5)
    step_scale = st.slider("Calibration step scale", min_value=0.02, max_value=1.0, value=0.25, step=0.01)
    calib_seed = st.number_input("Calibration seed", min_value=0, max_value=2_000_000_000, value=7, step=1)


# Load VE module and data
try:
    ve = load_ve_module(ve_py_path)
except Exception as e:
    st.error(f"Failed to import VE module: {e}")
    st.stop()

p_dataset = Path(dataset_dir).expanduser()
if not p_dataset.exists():
    st.warning("Dataset folder not found. Upload a ZIP / LMDB files in the sidebar, or set the full path.")
    st.info(r"Windows example: C:\Users\<you>\Downloads\ve2plume_streamlit_tool\dataset")
    st.stop()
if p_dataset.is_file():
    st.warning("You selected a file. Please select the *folder* that contains LMDB files (data.mdb/lock.mdb), or upload ZIP/files.")
    st.stop()

# Quick diagnostics to help users validate the folder
try:
    files = sorted([x.name for x in p_dataset.iterdir()])
    preview = ", ".join(files[:12]) + (" ..." if len(files) > 12 else "")
    st.sidebar.info(f"Dataset contents ({len(files)} files): {preview}")
except Exception:
    pass

with st.spinner("Loading LMDB case into memory..."):
    phi, k, p_series, sg_obs = load_case_from_lmdb(
        ve_py_path, dataset_dir, int(sim_id),
        nt_max_guess=getattr(ve, "NT_MAX_GUESS", 8000)
    )

nx, ny = phi.shape
nt = int(min(nt, len(p_series), len(sg_obs)))
tidx = int(min(max(0, tidx), nt - 1))

k_norm, k_eff = _normalize_k(ve, k)
well = _ensure_well(ve, phi, p_series[:nt])

# Layout
colA, colB = st.columns([1.05, 1.0], gap="large")

with colA:
    st.subheader("Case overview")
    st.write(f"- Grid: **{nx}×{ny}**  |  Nt used: **{nt}**  |  Well (i,j): **{well}**  |  k_eff (geom mean): **{k_eff:.3g}**")
    figp = _plot_lines(np.arange(nt), [p_series[:nt]], ["p_obs"], "Observed pressure control", "pressure (arb.)")
    st.pyplot(figp, use_container_width=True)

with colB:
    st.subheader("Static fields")
    st.pyplot(_plot_map(np.log10(np.maximum(k, 1e-12)), "log10 permeability"), use_container_width=True)
    st.pyplot(_plot_map(phi, "porosity"), use_container_width=True)

# Parameters (base)
base_params = dict(getattr(ve, "START_PARAMS_DEFAULT", {}))
if not base_params:
    st.error("Your VE module does not expose START_PARAMS_DEFAULT. Add it to the VE file.")
    st.stop()

param_keys = ["D0", "alpha_p", "src_amp", "prod_frac", "Swr", "Sgr_max", "hyst_rate",
              "mob_exp", "beta_sol", "thr_contact", "p_ref_bar", "sti_clip"]
param_keys = [k for k in param_keys if k in base_params]

st.markdown("---")
base_params = _dict_input(
    "Model parameters (baseline)",
    base_params,
    param_keys,
    help_text="Tip: run calibration first, then run UQ around the calibrated parameters."
)

# ---- The rest of your app continues unchanged below ----
# (calibration, forward run, UQ plots, exports, etc.)
st.info("✅ Dataset upload support enabled. Continue with your existing calibration/UQ/run sections below.")
