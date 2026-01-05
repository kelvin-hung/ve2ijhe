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

import json
import time
import io
import zipfile
import importlib.util
from pathlib import Path

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
    # lists -> arrays
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


def _mc_sample_params(base_params: dict, n: int, mode: str, scale: float, seed: int):
    """
    Simple uncertainty around a calibrated parameter set.
    - 'lognormal': multiplies positive params by lognormal noise.
    - 'gaussian' : additive Gaussian noise.
    """
    rng = np.random.default_rng(int(seed))
    keys = list(base_params.keys())
    base = np.array([float(base_params[k]) for k in keys], dtype=np.float64)
    samples = []
    for _ in range(int(n)):
        if mode == "lognormal":
            # multiplicative, keeps positivity
            eps = rng.normal(0.0, scale, size=base.shape)
            s = base * np.exp(eps)
        else:
            eps = rng.normal(0.0, scale, size=base.shape)
            s = base + eps * (np.abs(base) + 1e-12)
        samp = {k: float(v) for k, v in zip(keys, s)}
        samples.append(samp)
    return samples


def _run_forward(ve, phi, k_norm, well, p_series, params, nt, fit_pressure, use_pred_p_for_vel):
    sg_pred, p_pred, sg_mob, sg_res, diag = ve.simulate_ve(
        p_series[:nt], k_norm, well, params,
        fit_pressure=bool(fit_pressure),
        use_pred_p_for_vel=bool(use_pred_p_for_vel)
    )
    sg_pred = np.asarray(sg_pred, dtype=np.float32)
    return sg_pred, diag


def _compute_rmse_map(pred, obs):
    # per time RMSE over grid
    diff = pred - obs
    rmse_t = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
    return rmse_t


def _compute_iou_t(pred, obs, thr=0.02):
    # IoU(t) over binary exceedance masks
    P = pred > thr
    O = obs > thr
    inter = np.sum(P & O, axis=(1, 2)).astype(np.float64)
    union = np.sum(P | O, axis=(1, 2)).astype(np.float64) + 1e-12
    return (inter / union).astype(np.float32)


def _area_t(mask, dx=1.0, dy=1.0):
    return mask.sum(axis=(1, 2)).astype(np.float64) * float(dx) * float(dy)


def _radius_eq_t(area):
    return np.sqrt(area / np.pi)


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
    dataset_dir = st.text_input("Dataset folder", value="dataset", help="Folder containing LMDB simulations.")
    out_dir = st.text_input("Output folder", value="streamlit_outputs", help="Where to write figures/json exports.")

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

if not Path(dataset_dir).exists():
    st.warning("Dataset folder not found yet. Update the sidebar path.")
    st.stop()

with st.spinner("Loading LMDB case into memory..."):
    phi, k, p_series, sg_obs = load_case_from_lmdb(ve_py_path, dataset_dir, int(sim_id), nt_max_guess=getattr(ve, "NT_MAX_GUESS", 8000))

nx, ny = phi.shape
nt = int(min(nt, len(p_series), len(sg_obs)))
tidx = int(min(max(0, tidx), nt - 1))

k_norm, k_eff = _normalize_k(ve, k)
well = _ensure_well(ve, phi, p_series[:nt])

# Layout
colA, colB = st.columns([1.05, 1.0], gap="large")

with colA:
    st.subheader("Case overview")
    st.write(
        f"- Grid: **{nx}×{ny}**  |  Nt used: **{nt}**  |  Well (i,j): **{well}**  |  k_eff (geom mean): **{k_eff:.3g}**"
    )
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

param_keys = ["D0", "alpha_p", "src_amp", "prod_frac", "Swr", "Sgr_max", "hyst_rate", "mob_exp", "beta_sol", "thr_contact", "p_ref_bar", "sti_clip"]
param_keys = [k for k in param_keys if k in base_params]

st.markdown("---")
base_params = _dict_input(
    "Model parameters (baseline)",
    base_params,
    param_keys,
    help_text="Tip: run calibration first, then run UQ around the calibrated parameters."
)

# Optional calibration
calib_params = base_params
calib_info = None
if do_calib:
    st.subheader("Local calibration (single case)")
    st.caption("Runs a local random search around the current parameters to reduce mismatch to the observation fields.")
    if st.button("Run calibration now"):
        with st.spinner("Calibrating..."):
            t0 = time.time()
            # refine_best is implemented in your VE module
            calib_params, calib_info = ve.refine_best(
                phi, k_norm, well,
                p_series[:nt], sg_obs[:nt],
                start_params=base_params,
                n_iter=int(n_iter),
                step_scale=float(step_scale),
                seed=int(calib_seed),
                fit_pressure=bool(fit_pressure),
                use_pred_p_for_vel=bool(use_pred_p_for_vel),
            )
            st.success(f"Calibration finished in {time.time()-t0:.1f}s")
    if calib_info is not None:
        st.json({"best_params": calib_params, "best_obj": calib_info.get("best_obj", None)})

# Run forward
st.markdown("---")
st.subheader("Baseline simulation (single realization)")
run_now = st.button("Run baseline forward simulation")
if run_now:
    with st.spinner("Running VE forward model..."):
        t0 = time.time()
        sg_pred, diag = _run_forward(
            ve, phi, k_norm, well, p_series, calib_params, nt,
            fit_pressure=fit_pressure, use_pred_p_for_vel=use_pred_p_for_vel
        )
        st.session_state["baseline"] = {"sg_pred": sg_pred, "diag": diag, "params": calib_params}
        st.success(f"Baseline run done in {time.time()-t0:.1f}s")

baseline = st.session_state.get("baseline", None)
if baseline is not None:
    sg_pred = baseline["sg_pred"]
    diag = baseline["diag"]

    # Maps
    vmin = float(min(sg_obs[tidx].min(), sg_pred[tidx].min()))
    vmax = float(max(sg_obs[tidx].max(), sg_pred[tidx].max()))
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.pyplot(_plot_map(sg_obs[tidx], f"Obs Sg (t={tidx})", vmin=vmin, vmax=vmax), use_container_width=True)
    with c2:
        st.pyplot(_plot_map(sg_pred[tidx], f"Pred Sg (t={tidx})", vmin=vmin, vmax=vmax), use_container_width=True)
    with c3:
        st.pyplot(_plot_map(np.abs(sg_pred[tidx] - sg_obs[tidx]), f"|Error| (t={tidx})"), use_container_width=True)

    # Time series metrics
    rmse_t = _compute_rmse_map(sg_pred[:nt], sg_obs[:nt])
    iou_t = _compute_iou_t(sg_pred[:nt], sg_obs[:nt], thr=float(thr))
    area_obs = _area_t(sg_obs[:nt] > float(thr), dx=getattr(ve, "DX_DEFAULT", 1.0), dy=getattr(ve, "DY_DEFAULT", 1.0))
    area_pred = _area_t(sg_pred[:nt] > float(thr), dx=getattr(ve, "DX_DEFAULT", 1.0), dy=getattr(ve, "DY_DEFAULT", 1.0))
    req_obs = _radius_eq_t(area_obs)
    req_pred = _radius_eq_t(area_pred)

    st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [rmse_t], ["RMSE"], "RMSE(t)", "RMSE"), use_container_width=True)
    st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [iou_t], ["IoU"], f"IoU(t) @ thr={thr:g}", "IoU"), use_container_width=True)
    st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [area_obs, area_pred], ["obs", "pred"], f"Area(t) @ thr={thr:g}", "area"), use_container_width=True)
    st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [req_obs, req_pred], ["obs", "pred"], f"Equivalent radius(t) @ thr={thr:g}", "radius"), use_container_width=True)

    # Trapping indices if present
    if isinstance(diag, dict) and ("RTI" in diag and "STI" in diag):
        rti = np.asarray(diag["RTI"][:nt], dtype=np.float32)
        sti = np.asarray(diag["STI"][:nt], dtype=np.float32)
        st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [rti], ["RTI"], "Residual trapping index (RTI)", "RTI"), use_container_width=True)
        st.columns(1)[0].pyplot(_plot_lines(np.arange(nt), [sti], ["STI"], "Solubility trapping index (STI)", "STI"), use_container_width=True)

# UQ
st.markdown("---")
st.subheader("Uncertainty quantification (Monte Carlo around calibrated parameters)")
if uq_on:
    if st.button("Run Monte Carlo UQ"):
        if baseline is None:
            st.warning("Run baseline first (or at least load parameters). We'll proceed using current parameters.")
        with st.spinner("Sampling and running Monte Carlo... (this can take a while)"):
            t0 = time.time()
            params_list = _mc_sample_params(calib_params, int(B), uq_mode, float(uq_scale), int(uq_seed))

            # Run MC
            stack = []
            rti_stack = []
            sti_stack = []
            prog = st.progress(0)
            for b, pset in enumerate(params_list, start=1):
                sg_b, diag_b = _run_forward(
                    ve, phi, k_norm, well, p_series, pset, nt,
                    fit_pressure=fit_pressure, use_pred_p_for_vel=use_pred_p_for_vel
                )
                stack.append(sg_b[tidx])
                if isinstance(diag_b, dict) and ("RTI" in diag_b and "STI" in diag_b):
                    rti_stack.append(np.asarray(diag_b["RTI"][:nt], dtype=np.float32))
                    sti_stack.append(np.asarray(diag_b["STI"][:nt], dtype=np.float32))
                prog.progress(int(100 * b / max(1, len(params_list))))

            stack = np.asarray(stack, dtype=np.float32)  # (B,nx,ny)

            q10 = np.quantile(stack, 0.10, axis=0)
            q50 = np.quantile(stack, 0.50, axis=0)
            q90 = np.quantile(stack, 0.90, axis=0)
            width = q90 - q10
            prob = (stack > float(thr)).mean(axis=0)

            st.session_state["uq"] = {
                "tidx": int(tidx),
                "thr": float(thr),
                "q10": q10, "q50": q50, "q90": q90,
                "width": width,
                "prob": prob,
                "params_list": params_list,
            }
            if rti_stack and sti_stack:
                st.session_state["uq"]["rti_stack"] = np.asarray(rti_stack, dtype=np.float32)
                st.session_state["uq"]["sti_stack"] = np.asarray(sti_stack, dtype=np.float32)

            st.success(f"UQ done in {time.time()-t0:.1f}s (B={len(params_list)})")

uq = st.session_state.get("uq", None)
if uq is not None:
    st.caption(f"Showing UQ maps at t={uq['tidx']} and thr={uq['thr']}.")
    vmin = float(min(sg_obs[tidx].min(), uq["q10"].min(), uq["q90"].min()))
    vmax = float(max(sg_obs[tidx].max(), uq["q10"].max(), uq["q90"].max()))
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.pyplot(_plot_map(uq["q10"], "P10 Sg", vmin=vmin, vmax=vmax), use_container_width=True)
    with c2:
        st.pyplot(_plot_map(uq["q50"], "P50 Sg", vmin=vmin, vmax=vmax), use_container_width=True)
    with c3:
        st.pyplot(_plot_map(uq["q90"], "P90 Sg", vmin=vmin, vmax=vmax), use_container_width=True)

    c4, c5 = st.columns(2, gap="medium")
    with c4:
        st.pyplot(_plot_map(uq["width"], "Uncertainty width (P90−P10)"), use_container_width=True)
    with c5:
        st.pyplot(_plot_map(uq["prob"], f"P(Sg>{thr:g})", vmin=0.0, vmax=1.0), use_container_width=True)

    # RTI/STI bands if available
    if "rti_stack" in uq and "sti_stack" in uq:
        rti_s = uq["rti_stack"]  # (B,nt)
        sti_s = uq["sti_stack"]
        q = lambda A, qq: np.quantile(A, qq, axis=0)
        r10, r50, r90 = q(rti_s, 0.10), q(rti_s, 0.50), q(rti_s, 0.90)
        s10, s50, s90 = q(sti_s, 0.10), q(sti_s, 0.50), q(sti_s, 0.90)

        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        t = np.arange(r50.shape[0])
        ax.plot(t, r50, label="RTI P50")
        ax.fill_between(t, r10, r90, alpha=0.25, label="RTI P10–P90")
        ax.set_title("RTI uncertainty band")
        ax.set_xlabel("time index")
        ax.set_ylabel("RTI")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(t, s50, label="STI P50")
        ax.fill_between(t, s10, s90, alpha=0.25, label="STI P10–P90")
        ax.set_title("STI uncertainty band")
        ax.set_xlabel("time index")
        ax.set_ylabel("STI")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

# Export
st.markdown("---")
st.subheader("Export (figures + JSON)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

if st.button("Export current baseline/UQ to out_dir"):
    payload = {"sim_id": int(sim_id), "nt": int(nt), "tidx": int(tidx), "thr": float(thr), "well": list(map(int, well))}
    # baseline
    if baseline is not None:
        payload["params"] = baseline["params"]
    # uq (maps can be large; save as npy)
    if uq is not None:
        np.save(Path(out_dir)/f"sim{sim_id}_q10_t{tidx}.npy", uq["q10"])
        np.save(Path(out_dir)/f"sim{sim_id}_q50_t{tidx}.npy", uq["q50"])
        np.save(Path(out_dir)/f"sim{sim_id}_q90_t{tidx}.npy", uq["q90"])
        np.save(Path(out_dir)/f"sim{sim_id}_width_t{tidx}.npy", uq["width"])
        np.save(Path(out_dir)/f"sim{sim_id}_prob_thr{thr:g}_t{tidx}.npy", uq["prob"])
        payload["uq"] = {"B": len(uq["params_list"]), "mode": uq_mode, "scale": float(uq_scale), "seed": int(uq_seed)}
    (Path(out_dir)/f"sim{sim_id}_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    st.success(f"Saved to {out_dir}")

# Optional zip download
def _zip_folder(folder: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder)))
    return bio.getvalue()

if Path(out_dir).exists():
    zbytes = _zip_folder(Path(out_dir))
    st.download_button("Download out_dir as ZIP", data=zbytes, file_name="ve2plume_outputs.zip", mime="application/zip")
