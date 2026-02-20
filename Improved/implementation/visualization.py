# =========================
# GIF visualization (one run only)
# =========================

import numpy as np
from benchmarks import *
from hho import *
from opfunu.cec_based import cec2022
import os

GIF_DIR = r"C:\Users\Acer\Videos\master\3\meta\presentation\hho impl\gifs"
os.makedirs(GIF_DIR, exist_ok=True)

POP = 30
ITERS = 20

MODES = {
    "EXPLORE_A": True,
    "EXPLORE_B": True,
    "SOFT": True,
    "HARD": True,
    "SOFT_DIVE": True,
    "HARD_DIVE": False,
}

VIS_FUNCS = {
    "F1": (F1, -100, 100, 2),
    "F2": (F2, -10, 10, 2),
    "F3": (F3, -100, 100, 2),
    "F4": (F4, -100, 100, 2),
    "F5": (F5, -30, 30, 2),
    "F6": (F6, -100, 100, 2),
    "F7": (F7, -1.28, 1.28, 2),
    "F8": (F8, -500, 500, 2),
    "F9": (F9, -5.12, 5.12, 2),
    "F10": (F10, -32, 32, 2),
    "F11": (F11, -600, 600, 2),
    "F12": (F12, -50, 50, 2),
    "F13": (F13, -50, 50, 2),
    "F14": (F14, -65, 65, 2),
    "F14": (F14, -65, 65, 2),
    "F16": (F16, -5, 5, 2),
    "F17": (F17, -5, 5, 2),
    "F18": (F18, -2, 2, 2),
}

for name, (f, lb, ub, dim) in VIS_FUNCS.items():
    lb = np.ones(dim) * lb
    ub = np.ones(dim) * ub

    best, history = HHO(
        obj=f,
        lb=lb,
        ub=ub,
        dim=dim,
        pop=POP,
        iters=ITERS,
        MODES=MODES
    )

    save_hho_gif(
        history=history,
        obj=f,
        lb=lb,
        ub=ub,
        ref=np.zeros(dim),
        filename=os.path.join(GIF_DIR, f"{name}.gif")
    )

    print(f"{name} GIF saved.")



CEC_FUNCS = {
    "CEC2022_F1": cec2022.F12022(ndim=2),
    "CEC2022_F2": cec2022.F22022(ndim=2),
    "CEC2022_F3": cec2022.F32022(ndim=2),
    "CEC2022_F4": cec2022.F42022(ndim=2),
    "CEC2022_F5": cec2022.F52022(ndim=2),
}

for name, f in CEC_FUNCS.items():
    print(f"Generating GIF for {name} ...")

    lb = f.lb
    ub = f.ub
    ref = f.x_global.copy()   # 🔥 FIX OTHER DIMENSIONS AT TRUE OPTIMUM

    best, history = HHO(
        obj=f.evaluate,
        lb=lb,
        ub=ub,
        dim=2,
        pop=POP,
        iters=ITERS,
        MODES=MODES
    )

    save_hho_gif(
        history=history,
        obj=f.evaluate,
        lb=lb,
        ub=ub,
        ref=ref,
        filename=os.path.join(GIF_DIR, f"{name}.gif")
    )

    print(f"{name} GIF saved.")
