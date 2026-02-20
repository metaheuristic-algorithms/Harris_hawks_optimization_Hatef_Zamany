import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import gamma, sin, pi
import os

# ======================================================
# PATHS
# ======================================================
GIF_DIR = r"C:\Users\Acer\Videos\master\3\meta\presentation\hho impl\gifs"
os.makedirs(GIF_DIR, exist_ok=True)

# ======================================================
# LEVY FLIGHT
# ======================================================
def levy(dim, beta=1.5):
    sigma = (
        gamma(1 + beta) * sin(pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1 / beta)

# ======================================================
# MOVING PEAKS BENCHMARK (MINIMIZATION)
# ======================================================
class MovingPeaks:
    def __init__(self, n_peaks=5, dim=2, bounds=(-5, 5)):
        self.n_peaks = n_peaks
        self.dim = dim
        self.lb, self.ub = bounds
        self.pos = np.random.uniform(self.lb, self.ub, (n_peaks, dim))
        self.height = np.random.uniform(30, 70, n_peaks)
        self.width = np.random.uniform(1, 5, n_peaks)

    def update(self):
        self.pos += np.random.normal(0, 0.7, self.pos.shape)
        self.pos = np.clip(self.pos, self.lb, self.ub)
        self.height += np.random.normal(0, 1.0, self.n_peaks)
        self.width = np.clip(
            self.width + np.random.normal(0, 0.2, self.n_peaks),
            0.5, 6
        )

    def __call__(self, x):
        return min(
            -(self.height[i] - self.width[i] * np.linalg.norm(x - self.pos[i]))
            for i in range(self.n_peaks)
        )

    def true_optimum(self):
        return min(self(p) for p in self.pos)

# ======================================================
# HHO STEP (DYNAMIC, MEMORYLESS)
# ======================================================
def HHO_step(obj, X, fit, rabbit, rabbit_fit, lb, ub, dim, t, T):
    pop = X.shape[0]
    E1 = 2 * (1 - t / T)

    for i in range(pop):
        # E = E1 * (2 * np.random.rand() - 1)
        E = 2 * np.sin(np.pi * t / T) * (2*np.random.rand()-1)

        if abs(E) >= 1:
            q = np.random.rand()
            idx = np.random.randint(pop)
            if q < 0.5:
                X_new = X[idx] - np.random.rand(dim) * abs(
                    X[idx] - 2 * np.random.rand(dim) * X[i]
                )
            else:
                X_mean = np.mean(X, axis=0)
                X_new = (rabbit - X_mean) - np.random.rand(dim) * (
                    lb + np.random.rand(dim) * (ub - lb)
                )
        else:
            r = np.random.rand()
            J = 2 * (1 - np.random.rand())

            if r >= 0.5 and abs(E) >= 0.5:
                X_new = rabbit - E * abs(J * rabbit - X[i])
            elif r >= 0.5 and abs(E) < 0.5:
                X_new = rabbit - E * abs(rabbit - X[i])
            elif r < 0.5 and abs(E) >= 0.5:
                Y = rabbit - E * abs(J * rabbit - X[i])
                Z = Y + np.random.randn(dim) * levy(dim)
                X_new = Y if obj(Y) < fit[i] else Z
            else:
                Y = rabbit - E * abs(rabbit - np.mean(X, axis=0))
                Z = Y + np.random.randn(dim) * levy(dim)
                X_new = Y if obj(Y) < fit[i] else Z

        X_new = np.clip(X_new, lb, ub)
        f_new = obj(X_new)

        if f_new < fit[i]:
            X[i], fit[i] = X_new, f_new
            if f_new < rabbit_fit:
                rabbit, rabbit_fit = X_new.copy(), f_new

    return X, fit, rabbit, rabbit_fit

# ======================================================
# PSO STEP (DYNAMIC, MEMORY RESET)
# ======================================================
def PSO_step(obj, X, V, pbest, pbest_fit, gbest, gbest_fit,
             lb, ub, w=0.3, c1=1, c2=1):

    r1, r2 = np.random.rand(*X.shape), np.random.rand(*X.shape)
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    X = np.clip(X + V, lb, ub)

    fit = np.array([obj(x) for x in X])

    improved = fit < pbest_fit
    pbest[improved] = X[improved]
    pbest_fit[improved] = fit[improved]

    idx = np.argmin(pbest_fit)
    gbest, gbest_fit = pbest[idx].copy(), pbest_fit[idx]

    return X, V, pbest, pbest_fit, gbest, gbest_fit

# ======================================================
# EXPERIMENT SETUP
# ======================================================
np.random.seed(42)

lb = np.array([-5, -5])
ub = np.array([5, 5])
dim = 2
pop = 25
steps = 100
env_change = 10

mpb_hho = MovingPeaks()
mpb_pso = MovingPeaks()

X = np.random.uniform(lb, ub, (pop, dim))
fit = np.array([mpb_hho(x) for x in X])
rabbit = X[np.argmin(fit)].copy()
rabbit_fit = fit.min()

Xp = np.random.uniform(lb, ub, (pop, dim))
V = np.zeros_like(Xp)
pbest = Xp.copy()
pbest_fit = np.array([mpb_pso(x) for x in Xp])
gbest = pbest[np.argmin(pbest_fit)].copy()
gbest_fit = pbest_fit.min()

hho_frames, pso_frames = [], []
hho_peaks, pso_peaks = [], []
hho_err, pso_err = [], []

# ======================================================
# MAIN LOOP (TIME-CORRECT)
# ======================================================
for t in range(steps):

    # store environment at time t
    hho_peaks.append(mpb_hho.pos.copy())
    pso_peaks.append(mpb_pso.pos.copy())

    # run optimizers
    X, fit, rabbit, rabbit_fit = HHO_step(
        mpb_hho, X, fit, rabbit, rabbit_fit, lb, ub, dim, t, steps
    )

    Xp, V, pbest, pbest_fit, gbest, gbest_fit = PSO_step(
        mpb_pso, Xp, V, pbest, pbest_fit, gbest, gbest_fit, lb, ub
    )

    # store populations
    hho_frames.append(X.copy())
    pso_frames.append(Xp.copy())

    # metrics (current environment only)
    hho_err.append(rabbit_fit - mpb_hho.true_optimum())
    current_pso_best = np.min([mpb_pso(x) for x in Xp])
    pso_err.append(current_pso_best - mpb_pso.true_optimum())

    # update environment for next step
    if (t + 1) % env_change == 0:
        mpb_hho.update()
        mpb_pso.update()

        # reset PSO memory
        pbest = Xp.copy()
        pbest_fit = np.array([mpb_pso(x) for x in Xp])
        gbest = pbest[np.argmin(pbest_fit)].copy()
        gbest_fit = pbest_fit.min()

# ======================================================
# GIF FUNCTION
# ======================================================
def save_gif(frames, peaks, name):
    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        ax.clear()
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.scatter(frames[i][:,0], frames[i][:,1], c="blue", s=20)
        ax.scatter(peaks[i][:,0], peaks[i][:,1],
                   c="red", marker="*", s=200)
        ax.set_title(f"{name} – t={i}")

    ani = FuncAnimation(fig, animate, frames=len(frames), interval=300)
    ani.save(os.path.join(GIF_DIR, name + ".gif"),
             writer=PillowWriter(fps=6))
    plt.close()

save_gif(hho_frames, hho_peaks, "HHO_MPB")
save_gif(pso_frames, pso_peaks, "PSO_MPB")

# ======================================================
# RESULTS
# ======================================================
print("\n=== NUMERIC COMPARISON ===")
print(f"HHO Offline Error: {np.mean(hho_err):.4f}")
print(f"PSO Offline Error: {np.mean(pso_err):.4f}")
