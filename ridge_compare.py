#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# -----------------------------
# Metrics / utils
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.sqrt(np.mean(d * d)))


def ridge_objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    r = A @ x - b
    return float(r @ r + lam * (x @ x))


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def parse_float_list(s: str) -> List[float]:
    xs = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        xs.append(float(t))
    return xs


# -----------------------------
# Right-end labels with auto "avoid overlap"
# -----------------------------
def label_lines_at_right_auto(
    ax,
    x: np.ndarray,
    ys_dict: Dict[str, List[float]],
    labels_dict: Dict[str, str],
    order: List[str],
    *,
    x_index: int = -1,
    fontsize: int = 8,
    xpad: float = 0.03,
    min_sep: float = 0.03,
):
    x0 = float(x[x_index])
    x_text = x0 * (10 ** xpad)

    items = []
    for m in order:
        y_arr = np.asarray(ys_dict[m], dtype=float)
        y0 = float(y_arr[x_index])
        items.append((m, y0))
    if len(items) == 0:
        return

    yscale = ax.get_yscale()
    y_min, y_max = map(float, ax.get_ylim())

    items.sort(key=lambda t: t[1])
    ys = np.array([t[1] for t in items], dtype=float)
    y_targets = ys.copy()

    if yscale == "log":
        lo = np.log(max(y_min, 1e-300))
        hi = np.log(max(y_max, 1e-300))
        log_range = max(hi - lo, 1e-12)
        dlog = min_sep * log_range

        for i in range(1, len(y_targets)):
            y_targets[i] = max(y_targets[i], y_targets[i - 1] * np.exp(dlog))
        for i in range(len(y_targets) - 2, -1, -1):
            y_targets[i] = min(y_targets[i], y_targets[i + 1] / np.exp(dlog))

        y_targets = np.clip(y_targets, max(y_min, 1e-300), max(y_max, 1e-300))
    else:
        y_range = max(y_max - y_min, 1e-12)
        dy = min_sep * y_range

        for i in range(1, len(y_targets)):
            y_targets[i] = max(y_targets[i], y_targets[i - 1] + dy)
        for i in range(len(y_targets) - 2, -1, -1):
            y_targets[i] = min(y_targets[i], y_targets[i + 1] - dy)

        y_targets = np.clip(y_targets, y_min, y_max)

    for (m, y0), ytxt in zip(items, y_targets):
        ax.annotate(
            labels_dict[m],
            xy=(x0, y0),
            xytext=(x_text, float(ytxt)),
            textcoords="data",
            fontsize=fontsize,
            ha="left",
            va="center",
            clip_on=True,
            arrowprops=dict(arrowstyle="-", lw=0.7, alpha=0.7),
        )


# -----------------------------
# Direct solvers
# -----------------------------
def solve_cholesky_normal_eq(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    n = A.shape[1]
    ATA = A.T @ A
    rhs = A.T @ b
    ATA = 0.5 * (ATA + ATA.T)
    M = ATA + lam * np.eye(n)

    jitter = 0.0
    for _ in range(8):
        try:
            L = np.linalg.cholesky(M + jitter * np.eye(n))
            y = np.linalg.solve(L, rhs)
            x = np.linalg.solve(L.T, y)
            return x
        except np.linalg.LinAlgError:
            if jitter == 0.0:
                jitter = 1e-12 * np.trace(M) / n
            else:
                jitter *= 10.0
    raise np.linalg.LinAlgError("Cholesky failed even after jitter stabilization.")


def solve_qr_augmented(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    n = A.shape[1]
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n, dtype=b.dtype)])
    Q, R = np.linalg.qr(A_aug, mode="reduced")
    x = np.linalg.solve(R, Q.T @ b_aug)
    return x


def solve_svd_reference(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Utb = U.T @ b
    filt = s / (s * s + lam)
    x = (Vt.T * filt) @ Utb
    return x


# -----------------------------
# CG on normal equations
# -----------------------------
@dataclass
class CGConfig:
    max_iter: int = 500
    tol: float = 1e-10


def cg_solve_spd(matvec, rhs: np.ndarray, cfg: CGConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    b = rhs
    n = b.size
    x = np.zeros(n, dtype=b.dtype)

    r = b - matvec(x)
    p = r.copy()
    rr_old = float(r @ r)
    bnorm = float(np.linalg.norm(b)) + 1e-30
    rel_res = np.sqrt(rr_old) / bnorm

    it = 0
    while it < cfg.max_iter and rel_res > cfg.tol:
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        alpha = rr_old / (pAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(r @ r)
        beta = rr_new / (rr_old + 1e-30)
        p = r + beta * p
        rr_old = rr_new
        it += 1
        rel_res = np.sqrt(rr_old) / bnorm

    t1 = time.perf_counter()
    return x, {"iters": it, "rel_res": float(rel_res), "time_sec": float(t1 - t0)}


def solve_cg_ridge(A: np.ndarray, b: np.ndarray, lam: float, cfg: CGConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    rhs = A.T @ b

    def matvec(v: np.ndarray) -> np.ndarray:
        return A.T @ (A @ v) + lam * v

    return cg_solve_spd(matvec, rhs, cfg)


# -----------------------------
# LSQR on augmented system with KKT-based stopping
# -----------------------------
@dataclass
class LSQRConfig:
    max_iter: int = 4000
    tol_grad: float = 1e-6           # KKT relative gradient tol
    check_every: int = 20            # compute grad_rel every k iters


def lsqr_core(
    matvec,
    rmatvec,
    b: np.ndarray,
    n: int,
    cfg: LSQRConfig,
    grad_rel_fn,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Minimal LSQR iterations; stopping via grad_rel_fn(x) <= tol_grad (checked periodically).
    """
    t0 = time.perf_counter()

    x = np.zeros(n, dtype=b.dtype)

    u = b.copy()
    beta = float(np.linalg.norm(u))
    if beta > 0:
        u /= beta

    v = rmatvec(u)
    alpha = float(np.linalg.norm(v))
    if alpha > 0:
        v /= alpha

    w = v.copy()
    phi_bar = beta
    rho_bar = alpha

    it = 0
    grad_rel = float("inf")

    while it < cfg.max_iter:
        # bidiagonalization
        u = matvec(v) - alpha * u
        beta = float(np.linalg.norm(u))
        if beta > 0:
            u /= beta

        v = rmatvec(u) - beta * v
        alpha = float(np.linalg.norm(v))
        if alpha > 0:
            v /= alpha

        # orthogonal transformation
        rho = np.sqrt(rho_bar * rho_bar + beta * beta)
        c = rho_bar / (rho + 1e-30)
        s = beta / (rho + 1e-30)
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # update
        x = x + (phi / (rho + 1e-30)) * w
        w = v - (theta / (rho + 1e-30)) * w

        it += 1

        if it == 1 or (it % cfg.check_every) == 0:
            grad_rel = float(grad_rel_fn(x))
            if grad_rel <= cfg.tol_grad:
                break

    t1 = time.perf_counter()
    return x, {"iters": it, "grad_rel": float(grad_rel), "time_sec": float(t1 - t0)}


def solve_lsqr_ridge(A: np.ndarray, b: np.ndarray, lam: float, cfg: LSQRConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve ridge via LSQR on augmented system:
        min || [A; sqrt(lam) I] x - [b; 0] ||
    but stop using KKT residual:
        g(x)=A^T(Ax-b)+lam x
    """
    m, n = A.shape
    sLam = float(np.sqrt(lam))

    # augmented operators
    def matvec(x: np.ndarray) -> np.ndarray:
        return np.concatenate([A @ x, sLam * x])

    def rmatvec(u: np.ndarray) -> np.ndarray:
        u1 = u[:m]
        u2 = u[m:]
        return A.T @ u1 + sLam * u2

    b_aug = np.concatenate([b, np.zeros(n, dtype=b.dtype)])

    # grad_rel function (KKT)
    ATb = A.T @ b
    ATb_norm = float(np.linalg.norm(ATb)) + 1e-30

    def grad_rel_fn(x: np.ndarray) -> float:
        r = A @ x - b
        g = A.T @ r + lam * x
        denom = ATb_norm + lam * (float(np.linalg.norm(x)) + 1e-30)
        return float(np.linalg.norm(g)) / denom

    x, st = lsqr_core(matvec, rmatvec, b_aug, n=n, cfg=cfg, grad_rel_fn=grad_rel_fn)
    return x, st


# -----------------------------
# Ill-conditioning transform
# -----------------------------
def make_ill_conditioned(
    A_tr: np.ndarray, A_va: np.ndarray, A_te: np.ndarray,
    seed: int,
    alpha: float = 12.0,
    dup: int = 8,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    _, d = A_tr.shape

    G = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(G)
    eigs = np.logspace(0.0, alpha, d)
    T = Q @ (eigs[:, None] * Q.T)

    A_tr2 = A_tr @ T
    A_va2 = A_va @ T
    A_te2 = A_te @ T

    if d >= 2:
        base_tr0, base_tr1 = A_tr2[:, [0]], A_tr2[:, [1]]
        base_va0, base_va1 = A_va2[:, [0]], A_va2[:, [1]]
        base_te0, base_te1 = A_te2[:, [0]], A_te2[:, [1]]
    else:
        base_tr0 = base_tr1 = A_tr2[:, [0]]
        base_va0 = base_va1 = A_va2[:, [0]]
        base_te0 = base_te1 = A_te2[:, [0]]

    for _ in range(dup):
        noise_tr = rng.normal(size=(A_tr2.shape[0], 1)) * eps
        noise_va = rng.normal(size=(A_va2.shape[0], 1)) * eps
        noise_te = rng.normal(size=(A_te2.shape[0], 1)) * eps

        new_tr = (1.0 + 1e-6) * base_tr0 + (1e-6) * base_tr1 + noise_tr
        new_va = (1.0 + 1e-6) * base_va0 + (1e-6) * base_va1 + noise_va
        new_te = (1.0 + 1e-6) * base_te0 + (1e-6) * base_te1 + noise_te

        A_tr2 = np.hstack([A_tr2, new_tr])
        A_va2 = np.hstack([A_va2, new_va])
        A_te2 = np.hstack([A_te2, new_te])

    d2 = A_tr2.shape[1]
    scales = np.ones(d2)
    scales[: max(1, d2 // 5)] *= 1e6
    scales[-max(1, d2 // 5):] *= 1e-6

    return A_tr2 * scales, A_va2 * scales, A_te2 * scales


# -----------------------------
# LSQR tol tuning (fairness): choose fastest tol that matches SVD RMSE
# -----------------------------
def tune_lsqr_tol(
    A_train: np.ndarray,
    A_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_mean: float,
    lambdas: np.ndarray,
    svd_val_rmse: List[float],
    max_iter: int,
    check_every: int,
    tol_list: List[float],
    rmse_eps: float,
) -> float:
    b_train = y_train - y_mean

    def val_rmse_from_x(x: np.ndarray) -> float:
        return rmse(y_val, (A_val @ x) + y_mean)

    best_tol = None
    best_avg_time = None

    print("\n[tune] Tuning LSQR tol_grad for fair comparison...")
    for tol in tol_list:
        cfg = LSQRConfig(max_iter=max_iter, tol_grad=tol, check_every=check_every)
        times = []
        diffs = []
        iters = []

        for i, lam in enumerate(lambdas):
            x, st = solve_lsqr_ridge(A_train, b_train, lam, cfg)
            times.append(float(st["time_sec"]))
            iters.append(int(st["iters"]))
            v = val_rmse_from_x(x)
            diffs.append(abs(v - svd_val_rmse[i]))

        max_diff = float(np.max(diffs))
        avg_time = float(np.mean(times))
        avg_it = float(np.mean(iters))
        ok = (max_diff <= rmse_eps)

        print(f"[tune] tol_grad={tol:.1e} | maxΔvalRMSE={max_diff:.2e} "
              f"| avg_time={avg_time*1000:.2f}ms | avg_iter={avg_it:.1f} | ok={ok}")

        if ok and (best_avg_time is None or avg_time < best_avg_time):
            best_avg_time = avg_time
            best_tol = tol

    if best_tol is None:
        best_tol = min(tol_list)
        print(f"[tune] No tol met maxΔvalRMSE<= {rmse_eps:.1e}. Fallback tol_grad={best_tol:.1e}")
    else:
        print(f"[tune] Selected tol_grad={best_tol:.1e} (fastest among accurate).")

    return best_tol


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # splits
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)

    # lambda grid
    parser.add_argument("--lambda_min", type=int, default=-6)
    parser.add_argument("--lambda_max", type=int, default=2)
    parser.add_argument("--lambda_num", type=int, default=17)

    # CG
    parser.add_argument("--cg_max_iter", type=int, default=500)
    parser.add_argument("--cg_tol", type=float, default=1e-10)

    # LSQR base
    parser.add_argument("--lsqr_max_iter", type=int, default=4000)
    parser.add_argument("--lsqr_tol_grad", type=float, default=1e-6)
    parser.add_argument("--lsqr_check_every", type=int, default=20)

    # LSQR tol tuning
    parser.add_argument("--tune_lsqr", action="store_true")
    parser.add_argument("--lsqr_tol_list", type=str, default="1e-2,1e-3,1e-4,1e-5,1e-6")
    parser.add_argument("--lsqr_rmse_eps", type=float, default=1e-4)

    # ill-conditioning
    parser.add_argument("--ill", action="store_true")
    parser.add_argument("--ill_alpha", type=float, default=12.0)
    parser.add_argument("--ill_dup", type=int, default=8)
    parser.add_argument("--ill_eps", type=float, default=1e-8)

    # plot
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_fig", type=str, default="",
                        help="if set, save figure to this path (e.g., out.png)")
    parser.add_argument("--no_show", action="store_true",
                        help="do not call plt.show(); useful in headless runs")

    args = parser.parse_args()

    # ----- load data -----
    data = fetch_california_housing()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_size, random_state=args.seed
    )

    mu, sd = standardize_fit(X_train)
    A_train = standardize_apply(X_train, mu, sd)
    A_val = standardize_apply(X_val, mu, sd)
    A_test = standardize_apply(X_test, mu, sd)

    if args.ill:
        A_train, A_val, A_test = make_ill_conditioned(
            A_train, A_val, A_test,
            seed=args.seed,
            alpha=args.ill_alpha,
            dup=args.ill_dup,
            eps=args.ill_eps,
        )
        print(f"[ill] enabled: alpha={args.ill_alpha}, dup={args.ill_dup}, eps={args.ill_eps}")
        try:
            condA = np.linalg.cond(A_train)
            condATA = np.linalg.cond(A_train.T @ A_train)
            print(f"[ill] cond(A_train)≈{condA:.2e}, cond(A^T A)≈{condATA:.2e}")
        except Exception as e:
            print(f"[ill] cond() failed: {e}")

    y_mean = y_train.mean()
    b_train = y_train - y_mean

    print(f"[data] train={A_train.shape}, val={A_val.shape}, test={A_test.shape}")
    print(f"[data] y centered by train mean={y_mean:.6f}")

    lambdas = np.logspace(args.lambda_min, args.lambda_max, args.lambda_num)

    # eval helper
    def eval_all(x: np.ndarray) -> Tuple[float, float, float]:
        ytr = (A_train @ x) + y_mean
        yva = (A_val @ x) + y_mean
        yte = (A_test @ x) + y_mean
        return rmse(y_train, ytr), rmse(y_val, yva), rmse(y_test, yte)

    # precompute SVD refs
    x_svd_list: List[np.ndarray] = []
    svd_val_rmse: List[float] = []
    svd_obj: List[float] = []

    for lam in lambdas:
        x_svd = solve_svd_reference(A_train, b_train, lam)
        x_svd_list.append(x_svd)
        _, v, _ = eval_all(x_svd)
        svd_val_rmse.append(v)
        svd_obj.append(ridge_objective(A_train, b_train, x_svd, lam))

    # tune LSQR tol_grad
    chosen_tol = args.lsqr_tol_grad
    if args.tune_lsqr:
        tol_list = parse_float_list(args.lsqr_tol_list)
        chosen_tol = tune_lsqr_tol(
            A_train=A_train,
            A_val=A_val,
            y_train=y_train,
            y_val=y_val,
            y_mean=y_mean,
            lambdas=lambdas,
            svd_val_rmse=svd_val_rmse,
            max_iter=args.lsqr_max_iter,
            check_every=args.lsqr_check_every,
            tol_list=tol_list,
            rmse_eps=args.lsqr_rmse_eps,
        )

    lsqr_cfg = LSQRConfig(
        max_iter=args.lsqr_max_iter,
        tol_grad=chosen_tol,
        check_every=args.lsqr_check_every,
    )
    print(f"[lsqr] using tol_grad={lsqr_cfg.tol_grad:.1e}, max_iter={lsqr_cfg.max_iter}, check_every={lsqr_cfg.check_every}")

    cg_cfg = CGConfig(max_iter=args.cg_max_iter, tol=args.cg_tol)

    methods = ["chol", "qr", "svd", "cg", "lsqr"]
    labels = {
        "chol": "NormalEq+Cholesky",
        "qr": "QR (augmented)",
        "svd": "SVD (reference)",
        "cg": "CG on NormalEq",
        "lsqr": "LSQR (augmented)",
    }

    curves: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    for m in methods:
        for k in ["train_rmse", "val_rmse", "test_rmse", "time_sec",
                  "rel_x_err", "rel_obj_gap", "x_norm", "iters", "aux"]:
            curves[m][k] = []

    print("[run] sweeping lambdas...")
    for i, lam in enumerate(lambdas):
        x_svd = x_svd_list[i]
        f_svd = svd_obj[i]

        for m in methods:
            if m == "svd":
                x = x_svd
                elapsed = 0.0
                iters = 0
                aux = 0.0
            elif m == "chol":
                ts = time.perf_counter()
                x = solve_cholesky_normal_eq(A_train, b_train, lam)
                elapsed = time.perf_counter() - ts
                iters = 0
                aux = 0.0
            elif m == "qr":
                ts = time.perf_counter()
                x = solve_qr_augmented(A_train, b_train, lam)
                elapsed = time.perf_counter() - ts
                iters = 0
                aux = 0.0
            elif m == "cg":
                ts = time.perf_counter()
                x, st = solve_cg_ridge(A_train, b_train, lam, cg_cfg)
                elapsed = time.perf_counter() - ts
                iters = int(st["iters"])
                aux = float(st["rel_res"])
            elif m == "lsqr":
                ts = time.perf_counter()
                x, st = solve_lsqr_ridge(A_train, b_train, lam, lsqr_cfg)
                elapsed = time.perf_counter() - ts
                iters = int(st["iters"])
                aux = float(st["grad_rel"])
            else:
                raise ValueError(m)

            tr_rmse, va_rmse, te_rmse = eval_all(x)

            f = ridge_objective(A_train, b_train, x, lam)
            rel_x_err = float(np.linalg.norm(x - x_svd) / (np.linalg.norm(x_svd) + 1e-30))
            rel_obj_gap = float((f - f_svd) / (abs(f_svd) + 1e-30))
            x_norm = float(np.linalg.norm(x))

            curves[m]["train_rmse"].append(tr_rmse)
            curves[m]["val_rmse"].append(va_rmse)
            curves[m]["test_rmse"].append(te_rmse)
            curves[m]["time_sec"].append(elapsed)
            curves[m]["rel_x_err"].append(rel_x_err)
            curves[m]["rel_obj_gap"].append(rel_obj_gap)
            curves[m]["x_norm"].append(x_norm)
            curves[m]["iters"].append(iters)
            curves[m]["aux"].append(aux)

        print(
            f"lam={lam:.2e} | valRMSE "
            f"chol={curves['chol']['val_rmse'][-1]:.4f} "
            f"qr={curves['qr']['val_rmse'][-1]:.4f} "
            f"svd={curves['svd']['val_rmse'][-1]:.4f} "
            f"cg={curves['cg']['val_rmse'][-1]:.4f} "
            f"lsqr={curves['lsqr']['val_rmse'][-1]:.4f} | "
            f"it_cg={curves['cg']['iters'][-1]} "
            f"it_lsqr={curves['lsqr']['iters'][-1]} "
            f"grad_rel(lsqr)={curves['lsqr']['aux'][-1]:.2e}"
        )

    print("\n=== Best lambda per method (by val RMSE) ===")
    for m in methods:
        val = np.array(curves[m]["val_rmse"])
        idx = int(np.argmin(val))
        print(
            f"{labels[m]:22s} lambda={lambdas[idx]:.2e} "
            f"valRMSE={curves[m]['val_rmse'][idx]:.4f} "
            f"testRMSE={curves[m]['test_rmse'][idx]:.4f} "
            f"time={curves[m]['time_sec'][idx]*1000:.2f}ms "
            f"rel_x_err={curves[m]['rel_x_err'][idx]:.2e} "
            f"rel_obj_gap={curves[m]['rel_obj_gap'][idx]:.2e} "
            f"iters={curves[m]['iters'][idx]}"
        )

    if args.plot:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        order = ["chol", "qr", "svd", "cg", "lsqr"]

        for ax in axes:
            ax.margins(x=0.18)

        # 1) val RMSE
        ax = axes[0]
        for m in order:
            ax.semilogx(lambdas, curves[m]["val_rmse"], marker="o")
        ax.set_title("Validation RMSE vs λ")
        ax.set_xlabel("λ")
        ax.set_ylabel("RMSE")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(ax, lambdas, {m: curves[m]["val_rmse"] for m in order}, labels, order, x_index=-2, min_sep=0.05)

        # 2) test RMSE
        ax = axes[1]
        for m in order:
            ax.semilogx(lambdas, curves[m]["test_rmse"], marker="o")
        ax.set_title("Test RMSE vs λ")
        ax.set_xlabel("λ")
        ax.set_ylabel("RMSE")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(ax, lambdas, {m: curves[m]["test_rmse"] for m in order}, labels, order, x_index=-2, min_sep=0.05)

        # 3) runtime
        ax = axes[2]
        for m in order:
            ax.semilogx(lambdas, curves[m]["time_sec"], marker="o")
        ax.set_title("Runtime vs λ")
        ax.set_xlabel("λ")
        ax.set_ylabel("seconds")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(ax, lambdas, {m: curves[m]["time_sec"] for m in order}, labels, order, x_index=-1, min_sep=0.06)

        # 4) rel x err
        ax = axes[3]
        for m in order:
            ax.semilogx(lambdas, curves[m]["rel_x_err"], marker="o")
        ax.set_title(r"Relative solution error $\|x-x_{svd}\|/\|x_{svd}\|$")
        ax.set_xlabel("λ")
        ax.set_ylabel("relative error")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(ax, lambdas, {m: curves[m]["rel_x_err"] for m in order}, labels, order, x_index=-1, min_sep=0.08)

        # 5) ||x||
        ax = axes[4]
        for m in order:
            ax.semilogx(lambdas, curves[m]["x_norm"], marker="o")
        ax.set_title(r"Solution norm $\|x\|_2$ vs λ")
        ax.set_xlabel("λ")
        ax.set_ylabel(r"$\|x\|_2$")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(ax, lambdas, {m: curves[m]["x_norm"] for m in order}, labels, order, x_index=-1, min_sep=0.07)

        # 6) iterations (CG & LSQR)
        ax = axes[5]
        ax.semilogx(lambdas, curves["cg"]["iters"], marker="o")
        ax.semilogx(lambdas, curves["lsqr"]["iters"], marker="o")
        ax.set_title("Iterations vs λ (CG / LSQR)")
        ax.set_xlabel("λ")
        ax.set_ylabel("iterations")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        label_lines_at_right_auto(
            ax, lambdas,
            {"cg": curves["cg"]["iters"], "lsqr": curves["lsqr"]["iters"]},
            {"cg": "CG iters", "lsqr": "LSQR iters"},
            ["cg", "lsqr"],
            x_index=-1,
            min_sep=0.10,
        )

        title = "Ridge Regression: Solver Comparison on California Housing"
        if args.ill:
            title += " (Ill-conditioned features)"
        if args.tune_lsqr:
            title += f" [LSQR tuned tol_grad={chosen_tol:.0e}]"
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        # always save if asked
        if args.save_fig:
            plt.savefig(args.save_fig, dpi=200)
            print(f"[plot] saved figure to: {args.save_fig}")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()