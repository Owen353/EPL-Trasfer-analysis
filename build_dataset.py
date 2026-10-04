"""Build a club-season table of Premier League transfer spending and league results.

Sources (downloaded from GitHub, no scraping needed):
  - Transfers: github.com/eordo/transfermarkt-data (scraped from Transfermarkt, fees in EUR)
  - Results:   github.com/openfootball/football.json (every match, 2010-11 onward)

Output: data/club_seasons.csv, one row per (season, club).
"""
import json
from pathlib import Path

import pandas as pd
import requests

FIRST_SEASON, LAST_SEASON = 2010, 2024   # season = year it starts; 2024 means 2024-25
N_LAGS = 2                               # also load transfers for 2 earlier seasons, for lag features

# Points deducted by the league, which match results alone can't show.
POINT_DEDUCTIONS = {
    (2023, "Everton"): 8,            # two PSR breaches in 2023-24 (-6 after appeal, then -2)
    (2023, "Nottingham Forest"): 4,  # PSR breach in 2023-24
}
DATA = Path("data")
TM_URL = "https://raw.githubusercontent.com/eordo/transfermarkt-data/master/premier_league/{}.csv"
OF_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{}/en.1.json"


def fetch(url, path):
    """Download url to path once; later runs reuse the local copy."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def normalize(name):
    """The two sources disagree on ' FC' suffixes (e.g. 'Leeds United' vs 'Leeds United FC')."""
    return name.removesuffix(" FC").strip()


def load_transfers():
    frames = []
    for season in range(FIRST_SEASON - N_LAGS, LAST_SEASON + 1):
        path = fetch(TM_URL.format(season), DATA / "raw" / "transfers" / f"{season}.csv")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df["club"] = df["club"].map(normalize)
    # ~7% of fees are missing (undisclosed). Treat as 0 and keep a count so we can check the effect.
    df["fee_missing"] = df["fee"].isna()
    df["fee"] = df["fee"].fillna(0)
    return df


def summarize_transfers(df):
    """One row per (season, club): spending in, income out, net spend, in millions of EUR."""
    g = df.groupby(["season", "club", "movement"])["fee"].sum().unstack(fill_value=0)
    out = pd.DataFrame({
        "spend": g.get("in", 0) / 1e6,
        "income": g.get("out", 0) / 1e6,
    })
    out["net_spend"] = out["spend"] - out["income"]
    # Spending relative to the whole league that season, to remove transfer-market inflation
    out["spend_pct"] = 100 * out["spend"] / out.groupby("season")["spend"].transform("sum")
    out["n_signings"] = df[df.movement == "in"].groupby(["season", "club"]).size()
    out["n_fee_missing"] = df[df.fee_missing].groupby(["season", "club"]).size()
    return out.fillna({"n_signings": 0, "n_fee_missing": 0}).reset_index()


def load_table(season):
    """Compute the final league table from match results."""
    label = f"{season}-{(season + 1) % 100:02d}"
    matches = json.loads(fetch(OF_URL.format(label), DATA / "raw" / "results" / f"{label}.json").read_text())["matches"]
    rows = []
    for m in matches:
        h, a = normalize(m["team1"]), normalize(m["team2"])
        hg, ag = m["score"]["ft"]
        hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
        rows += [(h, hp, hg, ag), (a, ap, ag, hg)]
    t = pd.DataFrame(rows, columns=["club", "points", "gf", "ga"]).groupby("club").sum()
    t["gd"] = t["gf"] - t["ga"]
    for (s, club), pts in POINT_DEDUCTIONS.items():
        if s == season:
            t.loc[club, "points"] -= pts
    t = t.sort_values(["points", "gd", "gf"], ascending=False)
    t["position"] = range(1, len(t) + 1)
    t["season"] = season
    return t.reset_index()


def main():
    transfers = summarize_transfers(load_transfers())
    tables = pd.concat([load_table(s) for s in range(FIRST_SEASON, LAST_SEASON + 1)])

    df = tables.merge(transfers, on=["season", "club"], how="left")
    unmatched = df[df["spend"].isna()]
    if len(unmatched):
        print("WARNING: no transfer data for:\n", unmatched[["season", "club"]])
    df = df.fillna({"spend": 0, "income": 0, "net_spend": 0, "spend_pct": 0,
                    "n_signings": 0, "n_fee_missing": 0})

    # Lag features: the club's spending share 1 and 2 seasons earlier. NaN means the club
    # wasn't in the Premier League that season (e.g. newly promoted), so we have no data.
    for k in range(1, N_LAGS + 1):
        lag = transfers[["season", "club", "spend_pct"]].assign(season=lambda d: d.season + k)
        df = df.merge(lag.rename(columns={"spend_pct": f"spend_pct_lag{k}"}),
                      on=["season", "club"], how="left")
    df["spend_pct_3yr"] = df[["spend_pct", "spend_pct_lag1", "spend_pct_lag2"]].sum(axis=1, skipna=False)

    # Last season's points, a strong baseline: good teams tend to stay good.
    prev = tables[["season", "club", "points"]].assign(season=lambda d: d.season + 1)
    df = df.merge(prev.rename(columns={"points": "points_prev"}), on=["season", "club"], how="left")

    # Labels for logistic regression
    df["top4"] = (df["position"] <= 4).astype(int)
    df["relegated"] = (df["position"] >= 18).astype(int)

    cols = ["season", "club", "position", "points", "gf", "ga", "gd",
            "spend", "income", "net_spend", "spend_pct", "spend_pct_lag1", "spend_pct_lag2",
            "spend_pct_3yr", "points_prev", "n_signings", "n_fee_missing", "top4", "relegated"]
    df = df[cols].sort_values(["season", "position"])
    df.to_csv(DATA / "club_seasons.csv", index=False)
    print(f"Saved {len(df)} rows ({df.season.nunique()} seasons) to {DATA / 'club_seasons.csv'}")


if __name__ == "__main__":
    main()
