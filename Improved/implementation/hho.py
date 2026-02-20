import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def levy(dim):
    beta = 1.5
    sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    return u / np.abs(v)**(1/beta)

def HHO(obj, lb, ub, dim, pop=30, iters=500, MODES=None):
    X = np.random.uniform(lb, ub, (pop, dim))
    fit = np.array([obj(x) for x in X])

    rabbit = X[np.argmin(fit)].copy()
    rabbit_fit = fit.min()

    history = []

    for t in range(iters):
        E1 = 2*(1 - t/iters)
        for i in range(pop):
            E = E1*(2*np.random.rand()-1)
            if abs(E) >= 1:
                q = np.random.rand()
                idx = np.random.randint(pop)
                if q < 0.5 and MODES["EXPLORE_A"]:
                    X_new = X[idx] - np.random.rand(dim)*abs(X[idx]-2*np.random.rand(dim)*X[i])
                elif q >= 0.5 and MODES["EXPLORE_B"]:
                    X_mean = np.mean(X,axis=0)
                    X_new = (rabbit-X_mean) - np.random.rand(dim)*(lb+np.random.rand(dim)*(ub-lb))
                else:
                    X_new = X[i]
            else:
                r = np.random.rand()
                J = 2*(1-np.random.rand())
                if r>=0.5 and abs(E)>=0.5 and MODES["SOFT"]:
                    X_new = rabbit - E*abs(J*rabbit-X[i])
                elif r>=0.5 and abs(E)<0.5 and MODES["HARD"]:
                    X_new = rabbit - E*abs(rabbit-X[i])
                elif r<0.5 and abs(E)>=0.5 and MODES["SOFT_DIVE"]:
                    Y = rabbit - E*abs(J*rabbit-X[i])
                    Z = Y + np.random.randn(dim)*levy(dim)
                    X_new = Y if obj(Y)<fit[i] else Z
                elif r<0.5 and abs(E)<0.5 and MODES["HARD_DIVE"]:
                    Y = rabbit - E*abs(rabbit-np.mean(X,axis=0))
                    Z = Y + np.random.randn(dim)*levy(dim)
                    X_new = Y if obj(Y)<fit[i] else Z
                else:
                    X_new = X[i]

            # Feasible reset (UNCHANGED)
            for d in range(dim):
                if X_new[d]<lb[d] or X_new[d]>ub[d]:
                    X_new[d]=np.random.uniform(lb[d],ub[d])

            f_new = obj(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < rabbit_fit:
                    rabbit, rabbit_fit = X_new.copy(), f_new

        history.append(X.copy())

    return rabbit_fit, history

def lift_x12(x1, x2, ref):
    x = ref.copy()
    x[0] = x1
    x[1] = x2
    return x

def contour_2d(obj, lb, ub, ref, n=200):
    xs = np.linspace(lb[0], ub[0], n)
    ys = np.linspace(lb[1], ub[1], n)
    Z = np.zeros((n, n))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[j, i] = obj(lift_x12(x, y, ref))

    return xs, ys, Z

def save_hho_gif(history, obj, lb, ub, ref, filename):
    fig, ax = plt.subplots(figsize=(6, 6))

    xs, ys, Z = contour_2d(obj, lb, ub, ref)
    ax.contourf(xs, ys, Z, levels=50, cmap="viridis")

    scat = ax.scatter([], [], c="red", s=25)

    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    def update(frame):
        X = history[frame]
        scat.set_offsets(X[:, :2])   # 🔥 FIRST TWO DIMENSIONS ONLY
        ax.set_title(f"Iteration {frame}")
        return scat,

    anim = FuncAnimation(
        fig, update, frames=len(history), interval=150
    )

    anim.save(filename, writer=PillowWriter(fps=5))
    plt.close()

