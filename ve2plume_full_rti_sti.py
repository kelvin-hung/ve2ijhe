import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import pickle
import lmdb
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# DEFAULTS (edit if you want)
# =========================
DATASET_DIR = r"C:\Users\hungv\Downloads\data assimilation\analytical H2 plume\dataset"
SIM = 238
OUT_ROOT = "sim_physics_outputs_v3"

NT_MAX_GUESS = 8000
THR = 0.02
DX_DEFAULT = 1.0
DY_DEFAULT = 1.0

# Optional: include pressure prediction in the objective
FIT_PRESSURE_DEFAULT = False
USE_PRED_P_FOR_VEL_DEFAULT = False   # set True only after pressure fit looks good


# Your current best (used as a great starting point)
START_PARAMS_DEFAULT = {
    "D0": 0.27802127542157334,
    "alpha_p": 0.3035718747536404,
    "src_amp": 289.9128164502606,
    "prod_frac": 0.21370220515124894,
    "Swr": 0.32291144862162,
    "Sgr_max": 0.07441779469199929,
    "C_L": 0.5969969351350207,
    "eps_h": 0.08728309474730092,
    "nu": 0.0010582605526077622,
    "m_spread": 2.731778941305009,
    "rad_w": 1,
    "dx": 1.0,
    "dy": 1.0,

    # NEW (capillary fringe + anisotropy)
    "hc": 0.10,         # thickness cutoff before mobile gas appears
    "mob_exp": 1.20,    # mapping exponent (front sharpness)
    "anisD": 1.00,      # D_x = D0*anisD, D_y = D0/anisD

    # OPTIONAL pressure model params (dimensionless calibration)
    "ap_diff": 0.03,    # pressure diffusivity strength
    "qp_amp": 1.0,      # pressure source amplitude

    # Trapping-index proxy parameters (STI). RTI comes from Land residual in the VE solver.
    "p_ref_bar": 100.0,     # reference pressure for solubility proxy (bar)
    "beta_sol": 0.02,       # dissolved-equivalent strength (dimensionless proxy)
    "sti_clip": 0.20,       # cap for dissolved-equivalent saturation
    "thr_contact": THR,     # contacted region threshold for STI proxy (use plume threshold)
}

# =========================
# LMDB helpers
# =========================
def pkey(s: str) -> bytes:
    return pickle.dumps(s)

def open_env(dataset_dir: str):
    dataset_dir = os.path.abspath(dataset_dir)
    if dataset_dir.lower().endswith("data.mdb"):
        dataset_dir = os.path.dirname(dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"LMDB folder not found: {dataset_dir}")
    if not os.path.exists(os.path.join(dataset_dir, "data.mdb")):
        raise FileNotFoundError(f"'data.mdb' not found in folder: {dataset_dir}")
    return lmdb.open(
        dataset_dir,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

def geom_mean(a: np.ndarray, eps=1e-30) -> float:
    a = np.asarray(a, dtype=float)
    a = np.clip(a, eps, None)
    return float(np.exp(np.mean(np.log(a))))

def find_Nt(txn, sim: int, Nt_max: int = 2000) -> int:
    for t in range(Nt_max):
        if txn.get(pkey(f"{sim}-{t}")) is None:
            return t
    return Nt_max

def load_sim_to_memory(dataset_dir: str, sim: int, Nt_max: int = 2000):
    env = open_env(dataset_dir)
    with env.begin(write=False) as txn:
        const_raw = txn.get(pkey(str(sim)))
        if const_raw is None:
            raise KeyError(f"Missing constants for sim={sim} (key='{sim}')")
        const = pickle.loads(const_raw)
        phi = np.array(const[0], dtype=np.float32)
        k = np.array(const[1], dtype=np.float32)

        Nt = find_Nt(txn, sim, Nt_max=Nt_max)
        if Nt <= 0:
            raise KeyError(f"No time keys found for sim={sim} (expected '{sim}-0')")

        p_list, sg_list = [], []
        for t in range(Nt):
            raw = txn.get(pkey(f"{sim}-{t}"))
            var = pickle.loads(raw)
            p_bar = np.array(var[0], dtype=np.float32)
            sg = np.array(var[1], dtype=np.float32)
            sg = np.clip(sg, 0.0, 1.0)
            p_list.append(p_bar)
            sg_list.append(sg)

    env.close()
    return phi, k, p_list, sg_list


# =========================
# Numerical operators
# =========================
def central_grad(a: np.ndarray, dx=1.0, dy=1.0):
    ap = np.pad(a, ((1,1),(1,1)), mode="edge")
    dax = (ap[1:-1, 2:] - ap[1:-1, :-2]) * (0.5 / dx)
    day = (ap[2:, 1:-1] - ap[:-2, 1:-1]) * (0.5 / dy)
    return dax, day

def laplacian(a: np.ndarray):
    ap = np.pad(a, ((1,1),(1,1)), mode="edge")
    return (ap[1:-1, 2:] + ap[1:-1, :-2] + ap[2:, 1:-1] + ap[:-2, 1:-1] - 4.0*ap[1:-1, 1:-1])

def upwind_advect(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, dt: float):
    hp = np.pad(h, ((1,1),(1,1)), mode="edge")
    hx_f = hp[1:-1, 2:] - hp[1:-1, 1:-1]
    hx_b = hp[1:-1, 1:-1] - hp[1:-1, :-2]
    hy_f = hp[2:, 1:-1] - hp[1:-1, 1:-1]
    hy_b = hp[1:-1, 1:-1] - hp[:-2, 1:-1]
    dhdx = np.where(ux >= 0, hx_b, hx_f)
    dhdy = np.where(uy >= 0, hy_b, hy_f)
    return h - dt * (ux*dhdx + uy*dhdy)

def k_spreading_power_aniso(h: np.ndarray, k_norm: np.ndarray, D0x: float, D0y: float, eps_h: float, m_spread: float, dt: float):
    """
    ∂h/∂t = ∂/∂x( D0x*k*(h+eps)^m * ∂h/∂x ) + ∂/∂y( D0y*k*(h+eps)^m * ∂h/∂y )
    """
    hp = np.pad(h, ((1,1),(1,1)), mode="edge")
    kp = np.pad(k_norm, ((1,1),(1,1)), mode="edge")

    h_c = hp[1:-1, 1:-1]
    h_e = hp[1:-1, 2:]
    h_w = hp[1:-1, :-2]
    h_n = hp[:-2, 1:-1]
    h_s = hp[2:, 1:-1]

    k_c = kp[1:-1, 1:-1]
    k_e = kp[1:-1, 2:]
    k_w = kp[1:-1, :-2]
    k_n = kp[:-2, 1:-1]
    k_s = kp[2:, 1:-1]

    def havg(a, b):
        return 2*a*b/(a+b+1e-12)

    ke = havg(k_c, k_e)
    kw = havg(k_c, k_w)
    kn = havg(k_c, k_n)
    ks = havg(k_c, k_s)

    he = 0.5*(h_c + h_e)
    hw = 0.5*(h_c + h_w)
    hn = 0.5*(h_c + h_n)
    hs = 0.5*(h_c + h_s)

    Ce = D0x * ke * np.power(np.maximum(he + eps_h, 0.0), m_spread)
    Cw = D0x * kw * np.power(np.maximum(hw + eps_h, 0.0), m_spread)
    Cn = D0y * kn * np.power(np.maximum(hn + eps_h, 0.0), m_spread)
    Cs = D0y * ks * np.power(np.maximum(hs + eps_h, 0.0), m_spread)

    Fe = Ce * (h_e - h_c)
    Fw = Cw * (h_c - h_w)
    Fn = Cn * (h_c - h_n)
    Fs = Cs * (h_s - h_c)

    divF = (Fe - Fw) + (Fs - Fn)
    return h + dt * divF


# =========================
# VE saturation + hysteresis
# =========================
def ve_mobile_sg_from_h(h: np.ndarray, Swr: float, hc: float, mob_exp: float):
    """
    Capillary/fringe mapping:
      effective thickness = clip((h - hc)/(1-hc), 0, 1)
      Sg_mob = (1-Swr) * eff^mob_exp
    """
    hc = float(np.clip(hc, 0.0, 0.9))
    eff = np.clip((h - hc) / (1.0 - hc + 1e-12), 0.0, 1.0)
    eff = np.power(eff, max(0.25, float(mob_exp)))
    return (1.0 - Swr) * eff

def land_residual(Sg_max: np.ndarray, Sgr_max: float, C_L: float):
    return Sgr_max * (Sg_max / (Sg_max + C_L + 1e-12))


# =========================
# Features / loss
# =========================
def plume_mask(sg, thr):
    return sg > thr

def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    uni = np.logical_or(mask_a, mask_b).sum()
    return float(inter / (uni + 1e-12))

def features(sg, thr=THR):
    m = plume_mask(sg, thr)
    area = float(m.sum())
    smax = float(sg.max())
    if area > 0:
        ii, jj = np.nonzero(m)
        cy = float(ii.mean())
        cx = float(jj.mean())
        req = float(np.sqrt(area / np.pi))
    else:
        cx = cy = req = 0.0
    return area, cx, cy, req, smax, m


# =========================
# Trapping indices (RTI/STI)
# =========================
def pore_volume_weights(phi: np.ndarray, dx: float, dy: float):
    """Pore-volume weights per cell for volume-weighted indices.
    Uses unit thickness (VE); thickness cancels in ratios."""
    return (phi.astype(np.float32) * (dx * dy)).astype(np.float32)

def solubility_proxy_sg(p_bar: np.ndarray,
                        sg_tot: np.ndarray,
                        sg_max_hist: np.ndarray,
                        thr_contact: float,
                        p_ref_bar: float,
                        beta_sol: float,
                        sti_clip: float):
    """A fast, screening-level dissolved-equivalent proxy for H2.
    This does NOT solve dissolved transport; it provides a monotone index:
    higher pressure + more brine + contacted region => larger dissolved-equivalent.
    """
    contacted = (sg_max_hist > thr_contact)
    brine_frac = np.clip(1.0 - sg_tot, 0.0, 1.0)
    p_scale = float(np.mean(p_bar) / (p_ref_bar + 1e-12))
    sg_diss_eq = beta_sol * p_scale * brine_frac * contacted.astype(np.float32)
    sg_diss_eq = np.clip(sg_diss_eq, 0.0, float(sti_clip)).astype(np.float32)
    return sg_diss_eq

def compute_rti_sti(phi: np.ndarray,
                    dx: float, dy: float,
                    sg_mob: np.ndarray,
                    sg_res: np.ndarray,
                    sg_tot: np.ndarray,
                    p_bar: np.ndarray,
                    sg_max_hist: np.ndarray,
                    thr_contact: float,
                    p_ref_bar: float,
                    beta_sol: float,
                    sti_clip: float):
    """Compute RTI and STI at a single time step.

    RTI(t) = V_res / (V_mob + V_res)
    STI(t) = V_diss_eq / (V_mob + V_res + V_diss_eq)   (proxy)

    Returns:
      rti, sti, V_mob, V_res, V_diss_eq
    """
    w = pore_volume_weights(phi, dx, dy)
    V_mob = float(np.sum(w * sg_mob))
    V_res = float(np.sum(w * sg_res))
    sg_diss_eq = solubility_proxy_sg(p_bar, sg_tot, sg_max_hist, thr_contact, p_ref_bar, beta_sol, sti_clip)
    V_diss_eq = float(np.sum(w * sg_diss_eq))

    rti = float(V_res / (V_mob + V_res + 1e-12))
    sti = float(V_diss_eq / (V_mob + V_res + V_diss_eq + 1e-12))
    return rti, sti, V_mob, V_res, V_diss_eq
def plume_loss_multi(sg_obs_list, sg_pred_list, picks):
    """
    Stronger loss: enforce shape at multiple thresholds.
    """
    thrs = [0.02, 0.05, 0.10]
    w_iou, w_area, w_cent, w_rmse, w_smax = 8.0, 4.0, 0.5, 2.0, 1.0

    L = 0.0
    for t in picks:
        obs = sg_obs_list[t]
        pred = sg_pred_list[t]

        # union rmse (base thr)
        _, _, _, _, _, mo = features(obs, thr=thrs[0])
        _, _, _, _, _, mp = features(pred, thr=thrs[0])
        um = np.logical_or(mo, mp)
        rmse = float(np.sqrt(np.mean((pred[um] - obs[um])**2))) if um.sum() > 0 else float(np.sqrt(np.mean((pred-obs)**2)))

        # multi-threshold shape terms
        Liou = 0.0
        Larea = 0.0
        Lcent = 0.0
        for thr in thrs:
            Ao, xo, yo, _, _, mo = features(obs, thr=thr)
            Ap, xp, yp, _, _, mp = features(pred, thr=thr)
            Liou += (1.0 - iou(mo, mp))
            Larea += abs(Ap - Ao) / (Ao + 1e-6)
            Lcent += np.sqrt((xp-xo)**2 + (yp-yo)**2)

        Liou /= len(thrs)
        Larea /= len(thrs)
        Lcent /= len(thrs)

        # peak saturation match
        so = float(obs.max())
        sp = float(pred.max())
        Lsmax = abs(sp - so)

        L += (w_iou*Liou + w_area*Larea + w_cent*Lcent + w_rmse*rmse + w_smax*Lsmax)

    return float(L / max(1, len(picks)))

def pressure_loss(p_obs_list, p_pred_list, picks, well_ij):
    wi, wj = well_ij
    p0 = p_obs_list[0]
    L = 0.0
    for t in picks:
        do = (p_obs_list[t] - p0)
        dp = (p_pred_list[t] - p_pred_list[0])
        denom = float(np.std(do) + 1e-6)
        nrmse = float(np.sqrt(np.mean((dp - do)**2)) / denom)
        werr = abs(float(p_pred_list[t][wi, wj] - p_obs_list[t][wi, wj])) / (abs(float(p_obs_list[t][wi, wj])) + 1e-6)
        L += (nrmse + 0.5*werr)
    return float(L / max(1, len(picks)))


# =========================
# Well detection + q(t) magnitude
# =========================
def detect_well_from_dp(p_list):
    if len(p_list) < 2:
        return None
    dp = (p_list[1] - p_list[0]).astype(np.float32)
    ij = np.unravel_index(np.argmax(dp), dp.shape)
    return (int(ij[0]), int(ij[1]))

def infer_q_sign_and_weight(p_list, wi, wj):
    pw = np.array([p[wi, wj] for p in p_list], dtype=float)
    dp = np.diff(pw, prepend=pw[0])

    dead = np.std(dp) * 0.25 + 1e-6
    sign = np.where(dp > dead,  1.0, np.where(dp < -dead, -1.0, 0.0))

    dp_pos = np.clip(dp, 0.0, None)
    dp_neg = np.clip(-dp, 0.0, None)

    pos_ref = np.mean(dp_pos[dp_pos > dead]) if np.any(dp_pos > dead) else 1.0
    neg_ref = np.mean(dp_neg[dp_neg > dead]) if np.any(dp_neg > dead) else 1.0

    w = np.where(sign > 0, dp_pos / (pos_ref + 1e-12),
                 np.where(sign < 0, dp_neg / (neg_ref + 1e-12), 0.0))
    w = np.clip(w, 0.0, 3.0)
    return sign.astype(np.float32), w.astype(np.float32)


# =========================
# Source/sink (distributed)
# =========================
def apply_well_source_sink(h, q_sign, q_w, src_amp, prod_frac, wi, wj, rad_w, dt):
    h2 = h.copy()
    i0, i1 = max(0, wi-rad_w), min(h.shape[0], wi+rad_w+1)
    j0, j1 = max(0, wj-rad_w), min(h.shape[1], wj+rad_w+1)
    n = (i1-i0)*(j1-j0)
    if q_sign > 0:
        h2[i0:i1, j0:j1] += (src_amp * q_w * dt) / max(1, n)
    elif q_sign < 0:
        h2[i0:i1, j0:j1] -= (prod_frac * src_amp * q_w * dt) / max(1, n)
    return h2


# =========================
# Optional pressure surrogate (diffusivity)
# =========================
def simulate_pressure(p_obs_list, k_norm, well_ij, q_sign, q_w, ap_diff, qp_amp, dx, dy):
    """
    Simple variable-coefficient pressure diffusion for Δp:
      ∂p/∂t = ap_diff * ∇·(k_norm ∇p) + qp_amp*q(t)*I_well
    """
    wi, wj = well_ij
    Nt = len(p_obs_list)
    p = p_obs_list[0].copy().astype(np.float32)
    p0 = p.copy()
    out = []

    # stability
    maxk = float(np.max(k_norm))
    dt_stable = 0.18 * (min(dx, dy)**2) / (ap_diff * maxk + 1e-12)
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 60))
    dt = 1.0 / nsub

    rad = 1
    for t in range(Nt):
        for _ in range(nsub):
            gx, gy = central_grad(p, dx=dx, dy=dy)
            # flux = k * grad(p)
            fx = (k_norm * gx).astype(np.float32)
            fy = (k_norm * gy).astype(np.float32)
            # div(k grad p) ~ laplacian(p) weighted approx
            div = laplacian(p)  # cheap fallback
            p = p + (ap_diff * dt) * (k_norm * div)

            # source term (sign + magnitude)
            if q_sign[t] != 0.0:
                i0, i1 = max(0, wi-rad), min(p.shape[0], wi+rad+1)
                j0, j1 = max(0, wj-rad), min(p.shape[1], wj+rad+1)
                n = (i1-i0)*(j1-j0)
                p[i0:i1, j0:j1] += (qp_amp * q_sign[t] * q_w[t] * dt) / max(1, n)

        out.append(p.copy())

    # shift baseline to match p0 exactly at t=0
    out[0] = p0
    return out


# =========================
# Main VE simulator
# =========================

def simulate_ve(p_obs_list, k_norm, well_ij, params,
                fit_pressure=False, use_pred_p_for_vel=False,
                phi_for_indices=None,
                thr_contact=None):
    """Run VE plume simulation.

    Returns:
      sg_pred: list of total gas saturation fields (mobile + residual)
      p_pred_list: optional pressure surrogate list (or None)
      sg_mob_list: list of mobile gas saturation fields
      sg_res_list: list of residual gas saturation fields (Land)
      diag: dict with time series for RTI/STI (if phi_for_indices is provided)
    """
    wi, wj = well_ij
    Nt = len(p_obs_list)

    dx        = float(params.get("dx", DX_DEFAULT))
    dy        = float(params.get("dy", DY_DEFAULT))
    D0        = float(params["D0"])
    alpha_p   = float(params["alpha_p"])
    src_amp   = float(params["src_amp"])
    prod_frac = float(params["prod_frac"])
    Swr       = float(params["Swr"])
    Sgr_max   = float(params["Sgr_max"])
    C_L       = float(params["C_L"])
    eps_h     = float(params["eps_h"])
    nu        = float(params["nu"])
    m_spread  = float(params["m_spread"])
    rad_w     = int(params["rad_w"])
    hc        = float(params.get("hc", 0.0))
    mob_exp   = float(params.get("mob_exp", 1.0))
    anisD     = float(params.get("anisD", 1.0))

    # STI proxy params (screening-level)
    p_ref_bar = float(params.get("p_ref_bar", 100.0))
    beta_sol  = float(params.get("beta_sol", 0.02))
    sti_clip  = float(params.get("sti_clip", 0.20))

    if thr_contact is None:
        thr_contact = float(params.get("thr_contact", THR))

    q_sign, q_w = infer_q_sign_and_weight(p_obs_list, wi, wj)

    # optional predicted pressure
    p_pred_list = None
    if fit_pressure or use_pred_p_for_vel:
        ap_diff = float(params.get("ap_diff", 0.03))
        qp_amp  = float(params.get("qp_amp", 1.0))
        p_pred_list = simulate_pressure(p_obs_list, k_norm, well_ij, q_sign, q_w, ap_diff, qp_amp, dx, dy)

    # substepping for stability
    maxk = float(np.max(k_norm))
    D0x = D0 * anisD
    D0y = D0 / (anisD + 1e-12)
    coefmax = (max(D0x, D0y) * maxk * (1.0 + eps_h)**max(1.0, m_spread) + nu)
    dt_stable = 0.18 * (min(dx, dy)**2) / (coefmax + 1e-12)
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 40))
    dt = 1.0 / nsub

    h = np.zeros_like(p_obs_list[0], dtype=np.float32)
    Sg_max_hist = np.zeros_like(h, dtype=np.float32)

    sg_pred = []
    sg_mob_list = []
    sg_res_list = []

    # diagnostics
    diag = {
        "RTI": [],
        "STI": [],
        "Vmob": [],
        "Vres": [],
        "Vdiss_eq": [],
    }

    for t in range(Nt):
        p_field = p_pred_list[t] if (use_pred_p_for_vel and p_pred_list is not None) else p_obs_list[t]
        gx, gy = central_grad(p_field, dx=dx, dy=dy)
        ux = (-alpha_p * k_norm * gx).astype(np.float32)
        uy = (-alpha_p * k_norm * gy).astype(np.float32)

        for _ in range(nsub):
            h = upwind_advect(h, ux, uy, dt=dt)
            h = k_spreading_power_aniso(h, k_norm, D0x=D0x, D0y=D0y, eps_h=eps_h, m_spread=m_spread, dt=dt)
            h = h + (nu * dt) * laplacian(h)
            h = apply_well_source_sink(h, q_sign[t], q_w[t], src_amp, prod_frac, wi, wj, rad_w, dt=dt)
            h = np.clip(h, 0.0, 1.0).astype(np.float32)

        sg_mob = ve_mobile_sg_from_h(h, Swr=Swr, hc=hc, mob_exp=mob_exp).astype(np.float32)
        Sg_max_hist = np.maximum(Sg_max_hist, sg_mob)
        sg_res = land_residual(Sg_max_hist, Sgr_max=Sgr_max, C_L=C_L).astype(np.float32)
        sg_tot = np.maximum(sg_mob, sg_res)
        sg_tot = np.clip(sg_tot, 0.0, 1.0).astype(np.float32)

        sg_pred.append(sg_tot)
        sg_mob_list.append(sg_mob)
        sg_res_list.append(sg_res)

        if phi_for_indices is not None:
            rti, sti, Vmob, Vres, Vdiss = compute_rti_sti(
                phi_for_indices, dx, dy,
                sg_mob, sg_res, sg_tot,
                p_bar=p_field, sg_max_hist=Sg_max_hist,
                thr_contact=thr_contact,
                p_ref_bar=p_ref_bar, beta_sol=beta_sol, sti_clip=sti_clip
            )
        else:
            rti = sti = Vmob = Vres = Vdiss = float("nan")

        diag["RTI"].append(rti)
        diag["STI"].append(sti)
        diag["Vmob"].append(Vmob)
        diag["Vres"].append(Vres)
        diag["Vdiss_eq"].append(Vdiss)

    # convert to arrays
    for k in list(diag.keys()):
        diag[k] = np.array(diag[k], dtype=float)

    return sg_pred, p_pred_list, sg_mob_list, sg_res_list, diag


# =========================
# Search: local refinement around best
# =========================
def clip_params(p):
    p = dict(p)
    # safe bounds
    p["D0"] = float(np.clip(p["D0"], 1e-4, 5.0))
    p["alpha_p"] = float(np.clip(p["alpha_p"], 1e-4, 5.0))
    p["src_amp"] = float(np.clip(p["src_amp"], 1e-3, 1e6))
    p["prod_frac"] = float(np.clip(p["prod_frac"], 0.05, 1.0))
    p["Swr"] = float(np.clip(p["Swr"], 0.0, 0.6))
    p["Sgr_max"] = float(np.clip(p["Sgr_max"], 0.0, 0.8))
    p["C_L"] = float(np.clip(p["C_L"], 1e-4, 10.0))
    p["eps_h"] = float(np.clip(p["eps_h"], 1e-4, 0.3))
    p["nu"] = float(np.clip(p["nu"], 1e-6, 0.5))
    p["m_spread"] = float(np.clip(p["m_spread"], 1.0, 5.0))
    p["rad_w"] = int(np.clip(int(p["rad_w"]), 1, 4))
    p["hc"] = float(np.clip(p.get("hc", 0.0), 0.0, 0.5))
    p["mob_exp"] = float(np.clip(p.get("mob_exp", 1.0), 0.4, 3.0))
    p["anisD"] = float(np.clip(p.get("anisD", 1.0), 0.5, 2.0))
    p["ap_diff"] = float(np.clip(p.get("ap_diff", 0.03), 1e-4, 1.0))
    p["qp_amp"] = float(np.clip(p.get("qp_amp", 1.0), 1e-3, 200.0))
    p["dx"] = float(p.get("dx", 1.0))
    p["dy"] = float(p.get("dy", 1.0))
    return p

def refine_best(p_list, sg_obs_list, k_norm, well_ij, picks, out_dir, base_params,
                n_iter=600, seed=0, fit_pressure=False, use_pred_p_for_vel=False):
    rng = np.random.default_rng(seed)
    best = clip_params(base_params)
    best_sg, best_p, *_ = simulate_ve(p_list, k_norm, well_ij, best,
                                  fit_pressure=fit_pressure, use_pred_p_for_vel=use_pred_p_for_vel)
    bestL = plume_loss_multi(sg_obs_list, best_sg, picks)
    if fit_pressure and best_p is not None:
        bestL += 0.8 * pressure_loss(p_list, best_p, picks, well_ij)

    (out_dir/"refine_start.txt").write_text(json.dumps({"loss": bestL, "params": best}, indent=2), encoding="utf-8")
    print(f"[REFINE] start loss={bestL:.4f}")

    # annealed step sizes
    for it in range(n_iter):
        frac = 1.0 - it / max(1, n_iter-1)
        mult = 0.25*frac + 0.03     # multiplicative perturbation scale
        add  = 0.10*frac + 0.01     # additive scale for bounded params

        cand = dict(best)

        # multiplicative (positive parameters)
        for key in ["D0", "alpha_p", "src_amp", "C_L", "eps_h", "nu", "ap_diff", "qp_amp"]:
            r = rng.normal(0.0, 1.0)
            cand[key] = cand[key] * float(np.exp(mult * r))

        # additive bounded
        cand["prod_frac"] = cand["prod_frac"] + add*rng.normal()
        cand["Swr"] = cand["Swr"] + 0.07*add*rng.normal()
        cand["Sgr_max"] = cand["Sgr_max"] + 0.07*add*rng.normal()
        cand["m_spread"] = cand["m_spread"] + 0.6*add*rng.normal()
        cand["hc"] = cand.get("hc", 0.1) + 0.25*add*rng.normal()
        cand["mob_exp"] = cand.get("mob_exp", 1.2) + 0.8*add*rng.normal()
        cand["anisD"] = cand.get("anisD", 1.0) * float(np.exp(0.4*mult*rng.normal()))

        # discrete
        if rng.random() < 0.15:
            cand["rad_w"] = int(cand["rad_w"] + rng.integers(-1, 2))

        cand = clip_params(cand)

        sg_pred, p_pred, *_ = simulate_ve(p_list, k_norm, well_ij, cand,
                                      fit_pressure=fit_pressure, use_pred_p_for_vel=use_pred_p_for_vel)
        L = plume_loss_multi(sg_obs_list, sg_pred, picks)
        if fit_pressure and p_pred is not None:
            L += 0.8 * pressure_loss(p_list, p_pred, picks, well_ij)

        if L < bestL:
            bestL = L
            best = cand
            (out_dir/"best_refined.txt").write_text(json.dumps({"it": it, "loss": bestL, "params": best}, indent=2), encoding="utf-8")
            print(f"[REFINE] it={it:04d} NEW best loss={bestL:.4f}")

    return {"loss": bestL, "params": best}


# =========================
# Plotting
# =========================
def save_compare_sg(out_path, obs, pred, title, thr=THR):
    """
    Saves a 3-panel figure: OBS, PRED, and ABS ERROR.
    Also annotates RMSE/MAE/IoU (computed on the union plume mask by default).
    """
    obs = np.asarray(obs, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    err = (pred - obs).astype(np.float32)
    abs_err = np.abs(err)

    # Metrics (focus on plume region = union mask)
    mo = obs > thr
    mp = pred > thr
    um = np.logical_or(mo, mp)

    if um.sum() > 0:
        e = err[um]
        ae = abs_err[um]
    else:
        e = err.ravel()
        ae = abs_err.ravel()

    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(ae))
    iou_val = iou(mo, mp)

    # Robust scaling for error map (avoid one outlier dominating)
    vmax = float(np.quantile(abs_err, 0.99))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        vmax = float(abs_err.max() if abs_err.max() > 0 else 1.0)

    plt.figure(figsize=(16.5, 4.6))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    im = plt.imshow(obs, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Sg OBS")

    plt.subplot(1, 3, 2)
    im = plt.imshow(pred, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Sg VE model")

    plt.subplot(1, 3, 3)
    im = plt.imshow(abs_err, origin="lower", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"|Pred-Obs| (Abs Error)\nRMSE={rmse:.4f}  MAE={mae:.4f}  IoU@{thr:g}={iou_val:.3f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_compare_p(out_path, p_obs, p_pred, title):
    plt.figure(figsize=(11.5, 4.4))
    plt.suptitle(title)
    plt.subplot(1,2,1)
    im = plt.imshow(p_obs, origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.title("P OBS")
    plt.subplot(1,2,2)
    im = plt.imshow(p_pred, origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.title("P predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET_DIR)
    ap.add_argument("--sim", type=int, default=SIM)
    ap.add_argument("--n_refine", type=int, default=700)
    ap.add_argument("--fit_pressure", type=int, default=int(FIT_PRESSURE_DEFAULT))
    ap.add_argument("--use_pred_p_for_vel", type=int, default=int(USE_PRED_P_FOR_VEL_DEFAULT))
    ap.add_argument("--start_params_json", type=str, default="")

    ap.add_argument("--beta_sol", type=float, default=None, help="Solubility proxy strength for STI (dimensionless). If set, overrides params.")
    ap.add_argument("--p_ref_bar", type=float, default=None, help="Reference pressure for STI proxy (bar). If set, overrides params.")
    ap.add_argument("--sti_clip", type=float, default=None, help="Upper cap for dissolved-equivalent saturation in STI proxy.")
    args = ap.parse_args()

    dataset_dir = args.dataset
    sim = args.sim
    fit_pressure = bool(args.fit_pressure)
    use_pred_p_for_vel = bool(args.use_pred_p_for_vel)

    out_dir = Path(__file__).resolve().parent / OUT_ROOT / f"sim_{sim:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "started.txt").write_text("started\n", encoding="utf-8")

    t0 = time.time()
    print("[INFO] DATASET_DIR:", os.path.abspath(dataset_dir))
    print("[INFO] fit_pressure:", fit_pressure, "| use_pred_p_for_vel:", use_pred_p_for_vel)

    phi, k, p_list, sg_obs_list = load_sim_to_memory(dataset_dir, sim, Nt_max=NT_MAX_GUESS)
    nx, ny = phi.shape
    phi_eff = float(phi.mean())
    k_eff = geom_mean(k)
    k_norm = (k / (k_eff + 1e-12)).astype(np.float32)

    well_auto = detect_well_from_dp(p_list)
    well_ij = well_auto if well_auto is not None else (nx//2, ny//2)

    Nt = len(p_list)
    picks = sorted(set([3, 6, 9, 12, 20, max(0, Nt-10), max(0, Nt-5), Nt-1]))

    print(f"[INFO] sim={sim} Nt={Nt} nx,ny={nx},{ny} phi_eff={phi_eff:.4f} k_eff≈{k_eff:.2f}")
    print("[INFO] well_ij:", well_ij)
    print("[INFO] picks:", picks)

    # starting params: from JSON if provided, else default best
    start_params = dict(START_PARAMS_DEFAULT)
    if args.start_params_json:
        start_params.update(json.loads(Path(args.start_params_json).read_text(encoding="utf-8")))

    
    # Optional overrides for STI proxy parameters (screening-level; no dissolved-transport PDE)
    if args.beta_sol is not None:
        start_params["beta_sol"] = float(args.beta_sol)
    if args.p_ref_bar is not None:
        start_params["p_ref_bar"] = float(args.p_ref_bar)
    if args.sti_clip is not None:
        start_params["sti_clip"] = float(args.sti_clip)

    start_params = clip_params(start_params)

    # refine around the starting best
    best = refine_best(
        p_list, sg_obs_list, k_norm, well_ij, picks,
        out_dir=out_dir,
        base_params=start_params,
        n_iter=args.n_refine,
        seed=0,
        fit_pressure=fit_pressure,
        use_pred_p_for_vel=use_pred_p_for_vel
    )
    (out_dir / "best_params_v3.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # final run with best
    best_params = best["params"]
    sg_pred, p_pred, sg_mob_list, sg_res_list, diag_pred = simulate_ve(
        p_list, k_norm, well_ij, best_params,
        fit_pressure=fit_pressure,
        use_pred_p_for_vel=use_pred_p_for_vel,
        phi_for_indices=phi,
        thr_contact=THR
    )

    
    # timeseries: plume + trapping diagnostics (threshold THR)
    # For observations, we construct a consistent residual-gas proxy by treating Sg_obs as "mobile"
    # and applying the same Land model to its running maximum.
    Sgmax_obs = np.zeros_like(sg_obs_list[0], dtype=np.float32)
    rti_obs_ts = []
    sti_obs_ts = []
    vmob_obs_ts = []
    vres_obs_ts = []
    vdiss_obs_ts = []

    # use the same STI proxy params as best_params (for apples-to-apples indices)
    p_ref_bar = float(best_params.get("p_ref_bar", 100.0))
    beta_sol  = float(best_params.get("beta_sol", 0.02))
    sti_clip  = float(best_params.get("sti_clip", 0.20))

    rows = []
    for t in range(Nt):
        # plume features
        Ao, _, _, ro, so, _ = features(sg_obs_list[t], thr=THR)
        Ap, _, _, rp, sp, _ = features(sg_pred[t], thr=THR)

        # observed indices (proxy)
        sg_mob_obs = np.clip(sg_obs_list[t], 0.0, 1.0).astype(np.float32)
        Sgmax_obs = np.maximum(Sgmax_obs, sg_mob_obs)
        sg_res_obs = land_residual(Sgmax_obs, Sgr_max=float(best_params["Sgr_max"]), C_L=float(best_params["C_L"])).astype(np.float32)
        sg_tot_obs = np.maximum(sg_mob_obs, sg_res_obs)

        rti_o, sti_o, Vmob_o, Vres_o, Vdiss_o = compute_rti_sti(
            phi, float(best_params.get("dx", DX_DEFAULT)), float(best_params.get("dy", DY_DEFAULT)),
            sg_mob_obs, sg_res_obs, sg_tot_obs,
            p_bar=p_list[t], sg_max_hist=Sgmax_obs,
            thr_contact=THR,
            p_ref_bar=p_ref_bar, beta_sol=beta_sol, sti_clip=sti_clip
        )

        rti_obs_ts.append(rti_o); sti_obs_ts.append(sti_o)
        vmob_obs_ts.append(Vmob_o); vres_obs_ts.append(Vres_o); vdiss_obs_ts.append(Vdiss_o)

        # predicted indices from simulator
        rti_p = float(diag_pred["RTI"][t])
        sti_p = float(diag_pred["STI"][t])
        Vmob_p = float(diag_pred["Vmob"][t])
        Vres_p = float(diag_pred["Vres"][t])
        Vdiss_p = float(diag_pred["Vdiss_eq"][t])

        rows.append([t, Ao, Ap, ro, rp, so, sp,
                     rti_o, rti_p,
                     sti_o, sti_p,
                     Vmob_o, Vmob_p,
                     Vres_o, Vres_p,
                     Vdiss_o, Vdiss_p])

    rows = np.array(rows, dtype=float)
    np.savetxt(
        out_dir / "timeseries_plume.csv", rows, delimiter=",",
        header="tidx,area_obs,area_pred,r_eq_obs,r_eq_pred,smax_obs,smax_pred,"
               "RTI_obs,RTI_pred,STI_obs,STI_pred,"
               "Vmob_obs,Vmob_pred,Vres_obs,Vres_pred,Vdiss_eq_obs,Vdiss_eq_pred",
        comments=""
    )

    
    # trapping indices plots
    plt.figure()
    plt.plot(rows[:,0], rows[:,7], label="RTI_obs")
    plt.plot(rows[:,0], rows[:,8], label="RTI_pred")
    plt.grid(True); plt.legend(); plt.xlabel("tidx"); plt.ylabel("RTI (-)")
    plt.tight_layout(); plt.savefig(out_dir / "rti_timeseries.png", dpi=200); plt.close()

    plt.figure()
    plt.plot(rows[:,0], rows[:,9], label="STI_obs (proxy)")
    plt.plot(rows[:,0], rows[:,10], label="STI_pred (proxy)")
    plt.grid(True); plt.legend(); plt.xlabel("tidx"); plt.ylabel("STI (-)")
    plt.tight_layout(); plt.savefig(out_dir / "sti_timeseries.png", dpi=200); plt.close()
    plt.figure()
    plt.plot(rows[:,0], rows[:,3], label="r_eq_obs")
    plt.plot(rows[:,0], rows[:,4], label="r_eq_pred")
    plt.grid(True); plt.legend(); plt.xlabel("tidx"); plt.ylabel("r_eq (cells)")
    plt.tight_layout(); plt.savefig(out_dir / "radius_timeseries.png", dpi=200); plt.close()

    plt.figure()
    plt.plot(rows[:,0], rows[:,1], label="area_obs")
    plt.plot(rows[:,0], rows[:,2], label="area_pred")
    plt.grid(True); plt.legend(); plt.xlabel("tidx"); plt.ylabel("area (cells)")
    plt.tight_layout(); plt.savefig(out_dir / "area_timeseries.png", dpi=200); plt.close()

    # compare maps
    for t in picks:
        save_compare_sg(out_dir / f"compare_sg_t{t:03d}.png",
                        sg_obs_list[t], sg_pred[t],
                        title=f"sim {sim} | tidx {t} | VE+Darcy+Land + (hc,mob_exp,anisD)")

        if p_pred is not None:
            save_compare_p(out_dir / f"compare_p_t{t:03d}.png",
                           p_list[t], p_pred[t],
                           title=f"sim {sim} | tidx {t} | Pressure surrogate")

    (out_dir / "done.txt").write_text("done\n", encoding="utf-8")
    print("[DONE] out_dir:", out_dir)
    print("[DONE] best loss:", best["loss"])
    print("[DONE] best params:", json.dumps(best_params, indent=2))
    print("[TIME] %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
