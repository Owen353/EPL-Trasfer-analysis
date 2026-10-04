"""Does transfer spending explain Premier League results? Phase 1 analysis.

Run build_dataset.py first. Figures go to figures/, a text summary is printed.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from models import (Standardizer, add_bias, linear_regression_direct, linear_regression_gd,
                    linear_predict, logistic_predict, logistic_regression_gd, r_squared)

warnings.filterwarnings("ignore", message="Setting penalty=None")  # sklearn >= 1.8 notice about C=np.inf
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
df = pd.read_csv("data/club_seasons.csv")

FEATURES = {"spend": "Gross spend (EUR m)",
            "net_spend": "Net spend (EUR m)",
            "spend_pct": "Share of league spending (%)"}


def unscale(theta, scaler):
    """Convert (w, b) fitted on standardized x back to the original units of x."""
    w, b = theta[0] / scaler.std[0], theta[1]
    return np.array([w, b - w * scaler.mean[0]])


# ============ 1. Linear regression: points ~ spending ============
print("=" * 60, "\n1. LINEAR REGRESSION: points ~ spending\n" + "=" * 60)
y = df["points"].to_numpy(float)
linear_fits = {}
for col, label in FEATURES.items():
    scaler = Standardizer().fit(df[col])
    X_tilde = add_bias(scaler.transform(df[col]))

    theta_gd, steps, _ = linear_regression_gd(X_tilde, y)
    theta_direct = linear_regression_direct(X_tilde, y)
    sk = LinearRegression(fit_intercept=False).fit(X_tilde, y)

    assert np.allclose(theta_gd, theta_direct, atol=1e-4) and np.allclose(theta_direct, sk.coef_)
    w, b = unscale(theta_direct, scaler)
    r2 = r_squared(y, linear_predict(X_tilde, theta_direct))
    linear_fits[col] = (w, b, r2)
    print(f"{label:32s} R^2 = {r2:.3f}   slope = {w:.3f} pts/unit   "
          f"(GD converged in {steps} steps, matches direct & sklearn)")

# Scatter + fitted line for each feature
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (col, label) in zip(axes, FEATURES.items()):
    w, b, r2 = linear_fits[col]
    ax.scatter(df[col], y, s=12, alpha=0.5, label="club-seasons")
    xs = np.linspace(df[col].min(), df[col].max(), 50)
    ax.plot(xs, w * xs + b, color="red", label=f"fit, $R^2$={r2:.2f}")
    ax.set_xlabel(label); ax.set_ylabel("Points"); ax.legend()
fig.suptitle("Premier League 2010-11 to 2024-25: points vs transfer spending")
fig.tight_layout(); fig.savefig(FIG / "linear_fits.png", dpi=150)

# Why feature scaling matters: GD on raw vs standardized spend
X_raw = add_bias(df["spend"])
X_std = add_bias(Standardizer().fit(df["spend"]).transform(df["spend"]))
_, _, hist_raw = linear_regression_gd(X_raw, y, lr=1e-5, num_steps=2000, eps=0)   # larger lr diverges
_, _, hist_std = linear_regression_gd(X_std, y, lr=0.1, num_steps=2000, eps=0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist_raw, label="raw spend (lr=1e-5)")
ax.plot(hist_std, label="standardized spend (lr=0.1)")
ax.set_yscale("log"); ax.set_xlabel("Gradient descent step"); ax.set_ylabel("MSE")
ax.set_title("Feature scaling and convergence"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "scaling.png", dpi=150)
print(f"\nAfter 2000 steps: MSE raw = {hist_raw[-1]:.1f}, standardized = {hist_std[-1]:.1f}")

# Over/under-performers relative to the spending-share model
w, b, _ = linear_fits["spend_pct"]
df["expected_points"] = w * df["spend_pct"] + b
df["residual"] = df["points"] - df["expected_points"]
show = ["season", "club", "position", "points", "spend_pct", "expected_points", "residual"]
fmt = lambda d: d[show].round(1).to_string(index=False)
print("\nBiggest over-performers (points above what spending predicts):")
print(fmt(df.nlargest(5, "residual")))
print("\nBiggest under-performers:")
print(fmt(df.nsmallest(5, "residual")))


# ============ 2. Logistic regression: P(top 4), P(relegation) ~ spending share ============
print("\n" + "=" * 60, "\n2. LOGISTIC REGRESSION: outcome ~ share of spending\n" + "=" * 60)
scaler = Standardizer().fit(df["spend_pct"])
X_tilde = add_bias(scaler.transform(df["spend_pct"]))
xs = np.linspace(0, df["spend_pct"].max(), 200)
X_plot = add_bias(scaler.transform(xs))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, target, title in zip(axes, ["top4", "relegated"], ["Finish top 4", "Relegated"]):
    t = df[target].to_numpy(float)
    theta, steps, _ = logistic_regression_gd(X_tilde, t)
    sk = LogisticRegression(fit_intercept=False, C=np.inf).fit(X_tilde, t)

    acc = np.mean((logistic_predict(X_tilde, theta) > 0.5) == t)
    w, b = unscale(theta, scaler)
    boundary = -b / w                                  # where P = 0.5
    print(f"{title:14s} GD theta = {np.round(theta, 3)}, sklearn = {np.round(sk.coef_[0], 3)}, "
          f"steps = {steps}")
    print(f"{'':14s} P = 0.5 at {boundary:.1f}% of league spending; accuracy = {acc:.3f} "
          f"(always guessing 'no' = {1 - t.mean():.3f})")

    ax.scatter(df["spend_pct"], t, s=14, facecolors="none", edgecolors="blue", alpha=0.5,
               label="club-seasons")
    ax.plot(xs, logistic_predict(X_plot, theta), color="red", label="P(outcome)")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Share of league spending (%)"); ax.set_ylabel("Probability")
    ax.set_title(title); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "logistic_fits.png", dpi=150)

print(f"\nFigures saved to {FIG}/")
