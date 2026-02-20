import numpy as np
from benchmarks import *
from hho import HHO
from opfunu.cec_based import cec2022

POP = 30
ITERS = 500
RUNS = 30
DIM = 30

MODES = {
    "EXPLORE_A": True,
    "EXPLORE_B": True,
    "SOFT": True,
    "HARD": True,
    "SOFT_DIVE": True,
    "HARD_DIVE": True,
}

FUNCS = {
    "F1": (F1, -100, 100, DIM),
    "F2": (F2, -10, 10, DIM),
    "F3": (F3, -100, 100, DIM),
    "F4": (F4, -100, 100, DIM),
    "F5": (F5, -30, 30, DIM),
    "F6": (F6, -100, 100, DIM),
    "F7": (F7, -1.28, 1.28, DIM),
    "F8": (F8, -500, 500, DIM),
    "F9": (F9, -5.12, 5.12, DIM),
    "F10": (F10, -32, 32, DIM),
    "F11": (F11, -600, 600, DIM),
    "F12": (F12, -50, 50, DIM),
    "F13": (F13, -50, 50, DIM),
    "F14": (F14, -65, 65, 2),
    "F15": (F15, -5, 5, 4),
    "F16": (F16, -5, 5, 2),
    "F17": (F17, -5, 5, 2),
    "F18": (F18, -2, 2, 2),
    "F19": (F19, 0, 1, 3),
    "F20": (F20, 0, 1, 6),
    "F21": (F21, 0, 10, 4),
    "F22": (F22, 0, 10, 4),
    "F23": (F23, 0, 10, 4),
}

for name,(f,lb,ub,dim) in FUNCS.items():
    results=[]
    lb=np.ones(dim)*lb
    ub=np.ones(dim)*ub
    for r in range(RUNS):
        best, _ = HHO(f,lb,ub,dim,POP,ITERS,MODES)
        results.append(best)
    print(f"{name}: mean={np.mean(results):.6e}, std={np.std(results):.6e}")

# =========================
# CEC-2022 F1–F5
# =========================

CEC_FUNCS = {
    "CEC2022_F1": cec2022.F12022(ndim=20),
    "CEC2022_F2": cec2022.F22022(ndim=20),
    "CEC2022_F3": cec2022.F32022(ndim=20),
    "CEC2022_F4": cec2022.F42022(ndim=20),
    "CEC2022_F5": cec2022.F52022(ndim=20),
}

print("\n===== CEC-2022 F1–F5 Results =====\n")

for name, f in CEC_FUNCS.items():
    results = []
    lb = f.lb
    ub = f.ub
    dim = len(lb)

    for r in range(RUNS):
        best, _ = HHO(
            obj=f.evaluate,
            lb=lb,
            ub=ub,
            dim=dim,
            pop=POP,
            iters=ITERS,
            MODES=MODES
        )
        results.append(best)

    mean_val = np.mean(results)
    std_val = np.std(results)

    print(f"{name}: mean={mean_val:.6e}, std={std_val:.6e}, f*={f.f_global:.6e}")
