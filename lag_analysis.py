"""Phase 3: does spending pay off with a delay?

Compares linear models that use current, past, and cumulative spending, plus last season's
points as a baseline. Models are fit on 2010-11..2021-22 and evaluated on held-out seasons
2022-23..2024-25, so the test R^2 measures prediction on seasons the model never saw.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import Standardizer, add_bias, linear_regression_direct, linear_regression_gd, r_squared

TEST_FROM = 2022
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

df = pd.read_csv("data/club_seasons.csv")
# Lag features exist only for clubs that were in the Premier League the previous two seasons.
# Promoted clubs drop out here, so results describe established PL clubs.
needed = ["spend_pct_lag1", "spend_pct_lag2", "points_prev"]
data = df.dropna(subset=needed)
print(f"Using {len(data)} of {len(df)} club-seasons "
      f"(dropped {len(df) - len(data)} without two prior PL seasons)\n")

train, test = data[data.season < TEST_FROM], data[data.season >= TEST_FROM]

MODELS = {
    "Spend this season":            ["spend_pct"],
    "Spend last season":            ["spend_pct_lag1"],
    "Spend: this + last 2 seasons": ["spend_pct", "spend_pct_lag1", "spend_pct_lag2"],
    "Spend: 3-season total":        ["spend_pct_3yr"],
    "Last season's points":         ["points_prev"],
    "Last season's points + 3-season spend": ["points_prev", "spend_pct_3yr"],
}

rows, coefs = [], {}
for name, cols in MODELS.items():
    # Fit the scaler on training data only, so nothing about the test seasons leaks in.
    scaler = Standardizer().fit(train[cols])
    X_train = add_bias(scaler.transform(train[cols]))
    X_test = add_bias(scaler.transform(test[cols]))
    y_train, y_test = train["points"].to_numpy(float), test["points"].to_numpy(float)

    theta = linear_regression_direct(X_train, y_train)
    theta_gd, _, _ = linear_regression_gd(X_train, y_train)
    assert np.allclose(theta, theta_gd, atol=1e-4)

    rows.append({"model": name,
                 "train R^2": r_squared(y_train, X_train @ theta),
                 "test R^2": r_squared(y_test, X_test @ theta)})
    coefs[name] = dict(zip(cols, theta[:-1]))   # standardized: points per 1 std of each feature

results = pd.DataFrame(rows).set_index("model")
print(results.round(3).to_string(), "\n")

print("Standardized coefficients (points per 1 standard deviation of the feature):")
for name in ["Spend: this + last 2 seasons", "Last season's points + 3-season spend"]:
    print(f"  {name}: " + ", ".join(f"{k} = {v:+.1f}" for k, v in coefs[name].items()))

# ----- Figures -----
fig, ax = plt.subplots(figsize=(9, 4.5))
idx = np.arange(len(results))
ax.barh(idx + 0.2, results["train R^2"], height=0.4, label=f"train (to {TEST_FROM - 1}-{TEST_FROM % 100:02d})")
ax.barh(idx - 0.2, results["test R^2"], height=0.4, label=f"test ({TEST_FROM}-{(TEST_FROM + 1) % 100:02d} on)")
ax.set_yticks(idx, results.index); ax.invert_yaxis()
ax.set_xlabel("$R^2$ (share of variation in points explained)")
ax.set_title("Which spending measure best explains league points?")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "lag_r2.png", dpi=150)

c = coefs["Spend: this + last 2 seasons"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["this season", "1 season ago", "2 seasons ago"], list(c.values()), color="tab:blue")
ax.set_ylabel("Points per 1 std of spending share")
ax.set_title("How much does spending in each season matter?")
fig.tight_layout(); fig.savefig(FIG / "lag_coefficients.png", dpi=150)
print(f"\nFigures saved to {FIG}/")

# ----- Sensitivity check: one club's spending spree in the test seasons -----
no_chelsea = test[test.club != "Chelsea"]
print(f"\nTest R^2 without Chelsea ({len(no_chelsea)} rows):")
for name in ["Spend this season", "Spend: 3-season total"]:
    cols = MODELS[name]
    scaler = Standardizer().fit(train[cols])
    theta = linear_regression_direct(add_bias(scaler.transform(train[cols])), train["points"].to_numpy(float))
    y_hat = add_bias(scaler.transform(no_chelsea[cols])) @ theta
    print(f"  {name:28s} {r_squared(no_chelsea['points'].to_numpy(float), y_hat):.3f}")
