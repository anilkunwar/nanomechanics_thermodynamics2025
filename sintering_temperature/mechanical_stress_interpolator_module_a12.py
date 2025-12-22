# ======================================================================
# Stress Sunburst Visualizer — FINAL ROBUST VERSION
# ======================================================================

import os
import pickle
import json
import sqlite3
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn

from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import interp1d

# ======================================================================
# GLOBALS
# ======================================================================
CURRENT_MODE = "ISF"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")

os.makedirs(FIGURE_DIR, exist_ok=True)

# ======================================================================
# MATPLOTLIB STYLE
# ======================================================================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.linewidth': 2.0,
    'figure.dpi': 300
})

# ======================================================================
# DATABASE
# ======================================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sunburst_sessions (
            session_id TEXT PRIMARY KEY,
            parameters TEXT,
            von_mises_matrix BLOB,
            sigma_hydro_matrix BLOB,
            sigma_mag_matrix BLOB,
            times BLOB,
            theta_spokes BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_sunburst_data(session_id, params, vm, sh, sm, times, theta):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO sunburst_sessions
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        session_id,
        json.dumps(params),
        pickle.dumps(vm),
        pickle.dumps(sh),
        pickle.dumps(sm),
        pickle.dumps(times),
        pickle.dumps(theta)
    ))
    conn.commit()
    conn.close()


def load_sunburst_data(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT parameters, von_mises_matrix, sigma_hydro_matrix,
               sigma_mag_matrix, times, theta_spokes
        FROM sunburst_sessions WHERE session_id=?
    """, (session_id,))
    r = c.fetchone()
    conn.close()
    if r:
        p, vm, sh, sm, t, th = r
        return json.loads(p), pickle.loads(vm), pickle.loads(sh), pickle.loads(sm), pickle.loads(t), pickle.loads(th)
    return None


def get_recent_sessions(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_id, created_at
        FROM sunburst_sessions ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    r = c.fetchall()
    conn.close()
    return r

# ======================================================================
# HISTORY NORMALIZATION (CRITICAL FIX)
# ======================================================================
def normalize_history(history):
    """
    Convert ANY history format into:
    [(eta, {'von_mises': arr, 'sigma_hydro': arr, 'sigma_mag': arr}), ...]
    """
    normalized = []

    for entry in history:
        eta = 0.0
        stresses = {}

        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            eta, raw = entry
            stresses = raw if isinstance(raw, dict) else {}
        elif isinstance(entry, dict):
            stresses = entry
        else:
            continue

        def safe(key):
            v = stresses.get(key)
            return np.asarray(v) if v is not None else None

        vm = safe('von_mises')
        sh = safe('sigma_hydro')
        sm = safe('sigma_mag')

        if vm is None:
            continue

        shape = vm.shape
        stresses = {
            'von_mises': vm,
            'sigma_hydro': sh if sh is not None else np.zeros(shape),
            'sigma_mag': sm if sm is not None else np.zeros(shape)
        }

        normalized.append((eta, stresses))

    return normalized

# ======================================================================
# SOLUTION LOADER
# ======================================================================
@st.cache_data
def load_solutions(solution_dir):
    sols, params, logs = [], [], []

    for fname in os.listdir(solution_dir):
        if not fname.endswith(('.pkl', '.pt')):
            continue

        try:
            path = os.path.join(solution_dir, fname)
            sol = torch.load(path, map_location='cpu') if fname.endswith('.pt') else pickle.load(open(path, 'rb'))

            if 'history' not in sol or 'params' not in sol:
                raise ValueError("Missing keys")

            sol['history'] = normalize_history(sol['history'])

            if not sol['history']:
                raise ValueError("No valid stress data")

            p = sol['params']
            defect = p.get('defect_type', 'ISF')
            theta = float(p.get('theta', 0.0))

            sols.append(sol)
            params.append((defect, theta))
            logs.append(f"{fname}: OK ({defect}, {np.rad2deg(theta):.1f}°)")

        except Exception as e:
            logs.append(f"{fname}: FAILED → {e}")

    logs.append(f"Loaded {len(sols)} valid solutions.")
    return sols, params, logs

# ======================================================================
# INTERPOLATOR
# ======================================================================
class MultiParamAttentionInterpolator(nn.Module):

    def forward(self, solutions, params, defect_target, theta_target):

        history_len = min(len(s['history']) for s in solutions)

        shape = solutions[0]['history'][0][1]['von_mises'].shape

        out_hist = []

        for t in range(history_len):
            vm = np.zeros(shape)
            sh = np.zeros(shape)
            sm = np.zeros(shape)

            w = 1.0 / len(solutions)

            for sol in solutions:
                _, st = sol['history'][t]
                vm += w * st['von_mises']
                sh += w * st['sigma_hydro']
                sm += w * st['sigma_mag']

            out_hist.append((0.0, {
                'von_mises': vm,
                'sigma_hydro': sh,
                'sigma_mag': sm
            }))

        return {
            'params': {'defect_type': defect_target, 'theta': theta_target},
            'history': out_hist,
            'interpolated': True
        }

# ======================================================================
# CENTERLINE EXTRACTION
# ======================================================================
def get_center_stress(sol, stress_type, center_fraction):
    h = sol['history']
    shape = h[0][1][stress_type].shape
    ix = shape[0] // 2
    iy = int(shape[1] * center_fraction)
    return np.array([e[1][stress_type][ix, iy] for e in h])

# ======================================================================
# SUNBURST MATRIX BUILDER
# ======================================================================
def build_sunburst_matrices(sols, params, interpolator, defect, stress, frac, theta_spokes, log_time):

    N = 50
    out = np.zeros((N, len(theta_spokes)))
    times = np.logspace(-1, 2.3, N) if log_time else np.linspace(0, 200, N)

    sols_f = [s for s, p in zip(sols, params) if p[0] == defect]
    params_f = [p for p in params if p[0] == defect]

    for j, th in enumerate(theta_spokes):
        sol = interpolator(sols_f, params_f, defect, th)
        c = get_center_stress(sol, stress, frac)
        out[:, j] = np.pad(c[:N], (0, max(0, N - len(c))), mode='edge')

    return out, times

# ======================================================================
# MAIN
# ======================================================================
def main():

    st.set_page_config(layout="wide")
    st.title("Interpolated Stress Sunburst")

    init_database()

    sols, params, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Load log"):
        for l in logs:
            st.write(l)

    if not sols:
        st.stop()

    interpolator = MultiParamAttentionInterpolator()

    defect = st.sidebar.radio("Defect", ["ISF", "ESF", "Twin"])
    theta_step = st.sidebar.selectbox("Theta step (deg)", [5, 10, 15])
    theta_spokes = np.deg2rad(np.arange(0, 91, theta_step))
    center_fraction = 0.5
    log_time = st.sidebar.checkbox("Log time", True)

    if st.sidebar.button("Compute"):
        vm, times = build_sunburst_matrices(
            sols, params, interpolator,
            defect, 'von_mises',
            center_fraction, theta_spokes, log_time
        )
        st.write("Von Mises matrix:", vm.shape)


if __name__ == "__main__":
    main()
