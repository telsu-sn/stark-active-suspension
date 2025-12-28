import numpy as np
import pandas as pd
import os

# ============================================================
# 1. FILE CONFIGURATION
# ============================================================

FILE_PATH = "/kaggle/input/volatile-cargo/road_profiles.csv"
if not os.path.exists(FILE_PATH):
    FILE_PATH = "/kaggle/input/road-profiles/road_profiles.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError("road_profiles.csv not found")

print(f"Dataset loaded from: {FILE_PATH}")

# ============================================================
# 2. SYSTEM PARAMETERS (GIVEN)
# ============================================================

MS = 290.0      # sprung mass (kg)
MU = 59.0       # unsprung mass (kg)
KS = 16000.0    # suspension stiffness (N/m)
KT = 190000.0   # tire stiffness (N/m)

C_MIN = 800.0
C_MAX = 3500.0

DT = 0.005
DELAY_STEPS = 4

# ============================================================
# 3. FINAL TUNED CONTROLLER GAINS (LEADERBOARD)
# ============================================================

SKYHOOK_GAIN_LF = 4800.0   # low-frequency body control
SKYHOOK_GAIN_HF = 4200.0   # high-frequency jerk suppression
GROUND_GAIN    = 250.0    # wheel control
ACC_GAIN       = 100.0    # acceleration feedback

# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def soft_clip(x, xmin, xmax):
    mid = 0.5 * (xmin + xmax)
    span = 0.5 * (xmax - xmin)
    return mid + span * np.tanh((x - mid) / span)

# ============================================================
# 5. QUARTER-CAR SIMULATION
# ============================================================

def simulate_quarter_car(road):

    N = len(road)

    # States
    z_s = v_s = z_u = v_u = 0.0

    zs_hist = np.zeros(N)
    zu_hist = np.zeros(N)
    acc_s_hist = np.zeros(N)

    # Filters
    v_s_lf = 0.0
    v_u_lf = 0.0

    # Actuator delay buffer
    c_buffer = [C_MIN] * DELAY_STEPS

    prev_a_s = 0.0
    prev_a_u = 0.0

    for i in range(N):

        r = road[i]
        c_act = c_buffer.pop(0)

        # Forces
        f_spring = KS * (z_s - z_u)
        f_damper = c_act * (v_s - v_u)
        f_tire   = KT * (z_u - r)

        # Accelerations
        a_s = -(f_spring + f_damper) / MS
        a_u = (f_spring + f_damper - f_tire) / MU

        # Store histories
        zs_hist[i] = z_s
        zu_hist[i] = z_u
        acc_s_hist[i] = a_s

        # ====================================================
        # CONTROLLER (FREQUENCY-SELECTIVE SKYHOOK)
        # ====================================================

        v_s_lf = 0.05 * v_s + 0.95 * v_s_lf
        v_u_lf = 0.15 * v_u + 0.85 * v_u_lf

        v_s_hf = v_s - v_s_lf
        rel_vel = v_s - v_u

        c_target = C_MIN

        # Low-frequency skyhook
        if v_s_lf * rel_vel > 0:
            c_target += SKYHOOK_GAIN_LF * abs(v_s_lf)

        # High-frequency skyhook
        c_target += SKYHOOK_GAIN_HF * abs(v_s_hf)

        # Groundhook
        c_target += GROUND_GAIN * abs(v_u_lf)

        # Acceleration feedback
        c_target += ACC_GAIN * abs(a_s)

        # Saturation
        c_target = soft_clip(c_target, C_MIN, C_MAX)
        c_buffer.append(c_target)

        # ====================================================
        # INTEGRATION (TRAPEZOIDAL)
        # ====================================================

        v_s += 0.5 * (a_s + prev_a_s) * DT
        v_u += 0.5 * (a_u + prev_a_u) * DT

        z_s += v_s * DT
        z_u += v_u * DT

        prev_a_s = a_s
        prev_a_u = a_u

    return zs_hist, zu_hist, acc_s_hist

# ============================================================
# 6. METRICS (EXACT SPEC)
# ============================================================

def compute_metrics(zs, acc_s):

    zs_rel = zs - zs[0]

    rms_zs = rms(zs_rel)
    max_zs = np.max(np.abs(zs_rel))

    jerk = np.diff(acc_s) / DT
    jerk = np.append(jerk, 0.0)

    rms_jerk = rms(jerk)
    jerk_max = np.max(np.abs(jerk))

    comfort = (
        0.5 * rms_zs
        + max_zs
        + 0.5 * rms_jerk
        + jerk_max
    )

    return rms_zs, max_zs, rms_je_

