"""Counting metrics and paired statistics (no new measurement, only reporting)."""
import numpy as np


def mae(pred, gt):
    return float(np.abs(np.asarray(pred, float) - np.asarray(gt, float)).mean())


def rmse(pred, gt):
    return float(np.sqrt(((np.asarray(pred, float) - np.asarray(gt, float)) ** 2).mean()))


def report(pred, gt):
    return {"mae": mae(pred, gt), "rmse": rmse(pred, gt), "n": int(len(gt))}


def bootstrap_ci(diff, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap of the mean of paired differences."""
    d = np.asarray(diff, float)
    rng = np.random.RandomState(seed)
    n = len(d)
    means = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_test(pred, ref, gt, n_boot=10000, seed=0):
    """Paired absolute-error difference: mean, bootstrap CI, win rate, Wilcoxon p."""
    e_new = np.abs(np.asarray(pred, float) - np.asarray(gt, float))
    e_ref = np.abs(np.asarray(ref, float) - np.asarray(gt, float))
    d = e_new - e_ref
    lo, hi = bootstrap_ci(d, n_boot, seed)
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = float("nan")
    return {"dmae": float(d.mean()), "ci": [lo, hi],
            "win": float((d < 0).mean()), "p": p}
