# icu_dashboard_v2.py
# Dash (Bootstrap) ICU Bed Utilization – Decision Support
#
# NEW FEATURES ADDED ✅
# 1) Action Board (next 72 hours): Expected beds + risk + confidence + action (OK/WATCH/ESCALATE)
# 2) Confidence + Model Disagreement: shows how much models differ (trust builder)
# 3) Scenario Presets: one-click scenario buttons (meeting-friendly)
# 4) Export buttons: download forecast / risk / metrics as CSV
#
# IMPORTANT NOTE:
# - "Actual - Prediction error" cannot be shown for FUTURE dates in outlook table.
#   (No actual available yet.) We show backtest metrics instead.
#
# Run:
#   cd C:\ICU_CLEAN_V2
#   python icu_dashboard_v2.py
# Open:
#   http://127.0.0.1:8050

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from decimal import Decimal, ROUND_HALF_UP
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update, callback_context
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

# =========================
# CONFIG (EDIT PATHS)
# =========================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent   # folder where .exe is
else:
    BASE_DIR = Path(__file__).resolve().parent  
DATA_PATH = str(BASE_DIR / "icu_beds.csv")  # ✅ UPDATED: expects icu_beds.csv in same folder
ARTIFACTS_DIR = BASE_DIR / "artifacts"      # ✅ UPDATED
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"

HORIZONS = [7, 14, 28]

TARGET_COLS = [
    "available_adult_icu_beds",
    "total_adult_icu_patients",
    "available_ped_icu_beds",
    "total_ped_icu_patients",
]
TOTAL_BEDS_COLS = {"adult": "total_adult_icu_beds", "ped": "total_ped_icu_beds"}

BASE_THRESHOLDS = {"adult": 200, "ped": 10}

KNOWN_MODELS = [
    "deepar", "tft", "patchtst",
    "ensemble", "ensemblemean", "ens",
    "mean", "avg", "average", "combined", "blend",
]

# =========================
# ✅ SLIDER MARKS (COARSE) — prevents 0 10 20 ... spam
# =========================
SLIDER_MARKS_ADULT = {i: str(i) for i in range(0, 1001, 100)}
SLIDER_MARKS_PED = {i: str(i) for i in range(0, 201, 25)}
SLIDER_MARKS_BUFFER = {i: str(i) for i in range(0, 1001, 100)}


# =========================
# SMALL UTILITIES
# =========================
def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _parse_date_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["date", "ds", "timestamp", "time", "datetime"])


def _find_outputs_file(stem: str) -> Optional[Path]:
    """
    ✅ UPDATED: supports both old and new forecast filenames.
    - old: latest_forecast_{h}d.csv
    - new: test_forecast_quantiles_{h}d.csv
    """
    for ext in [".csv", ".xlsx", ".xls"]:
        p = OUTPUTS_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    p0 = OUTPUTS_DIR / stem
    if p0.exists():
        return p0
    return None


def _read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv" or path.suffix == "":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


@lru_cache(maxsize=64)
def _read_tabular_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return _read_tabular(Path(path_str))


def read_outputs_df(stem: str) -> pd.DataFrame:
    p = _find_outputs_file(stem)
    if not p:
        raise FileNotFoundError(f"Missing outputs file: {stem} (searched in {OUTPUTS_DIR})")
    return _read_tabular_cached(str(p), p.stat().st_mtime).copy()


def normal_cdf(x: np.ndarray) -> np.ndarray:
    import math
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt_int(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return str(int(Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
    except Exception:
        return ""


def pct(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x)*100:.1f}%"
    except Exception:
        return ""


def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")


def _snap_int(v, step: int, vmin: int, vmax: int) -> int:
    """
    Snap to step using HALF_UP rounding, then clamp.
    """
    try:
        d = Decimal(str(v))
    except Exception:
        d = Decimal("0")
    step_d = Decimal(str(step))
    q = (d / step_d).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step_d
    out = int(q)
    out = max(vmin, min(vmax, out))
    return out


# =========================
# HISTORY LOADER
# =========================
def load_wide_csv(path: str, clean_negatives: bool = True) -> pd.DataFrame:
    """
    ✅ UPDATED (safe): if clean_negatives=True, treat negatives as missing (NaN) then interpolate.
    This prevents impossible negative beds/patients from breaking dashboard risk math.
    Data Quality tab can still show raw negatives by loading with clean_negatives=False.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")

    needed = set(TARGET_COLS) | {"date"} | set(TOTAL_BEDS_COLS.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    wide = df.set_index("date")

    cols_to_fix = list(needed - {"date"})
    for c in cols_to_fix:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
        if clean_negatives:
            wide[c] = wide[c].mask(wide[c] < 0, np.nan)

    if wide[cols_to_fix].isna().any().any():
        wide[cols_to_fix] = wide[cols_to_fix].interpolate(limit_direction="both")
        wide[cols_to_fix] = wide[cols_to_fix].ffill().bfill()

    return wide


# =========================
# FORECAST OUTPUT PARSER
# =========================
def _find_quantile_cols(df: pd.DataFrame, model: str) -> dict:
    cols = list(df.columns)
    norm_map = {_norm_key(c): c for c in cols}

    m = _norm_key(model)
    out = {}
    for q in ["p10", "p50", "p90"]:
        key1 = _norm_key(f"{model}_{q}")
        key2 = _norm_key(f"{model}{q}")
        key3 = _norm_key(f"{m}_{q}")
        key4 = _norm_key(f"{m}{q}")

        col = norm_map.get(key1) or norm_map.get(key2) or norm_map.get(key3) or norm_map.get(key4)
        out[q] = col
    return out


def _standardize_forecast_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = _parse_date_col(df) or _first_existing_col(df, ["Date"])
    if date_col is None:
        raise ValueError(f"Forecast file has no date column. Columns={list(df.columns)[:40]}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    series_col = _first_existing_col(df, ["series", "target", "item_id", "item", "variable", "col", "unit"])
    if series_col is None:
        raise ValueError(f"Forecast file has no series column. Columns={list(df.columns)[:40]}")

    possible_models = ["DeepAR", "TFT", "PatchTST", "EnsembleMean"]
    found_any = False
    model_cols_map = {}
    for m in possible_models:
        qcols = _find_quantile_cols(df, m)
        if qcols.get("p50") is not None:
            found_any = True
            model_cols_map[m] = qcols

    if found_any:
        rows = []
        for m, qcols in model_cols_map.items():
            p10c = qcols.get("p10")
            p50c = qcols.get("p50")
            p90c = qcols.get("p90")

            sub = pd.DataFrame({
                "date": df[date_col],
                "series": df[series_col].astype(str),
                "model": m,
                "p50": pd.to_numeric(df[p50c], errors="coerce") if p50c else np.nan,
                "p10": pd.to_numeric(df[p10c], errors="coerce") if p10c else np.nan,
                "p90": pd.to_numeric(df[p90c], errors="coerce") if p90c else np.nan,
            })
            rows.append(sub)

        out = pd.concat(rows, ignore_index=True).dropna(subset=["p50"])

        if "EnsembleMean" not in out["model"].unique():
            indiv = out[out["model"] != "EnsembleMean"].copy()
            if not indiv.empty:
                ens = indiv.groupby(["date", "series"], as_index=False)[["p10", "p50", "p90"]].mean(numeric_only=True)
                ens["model"] = "EnsembleMean"
                out = pd.concat([out, ens], ignore_index=True)

        return out[["date", "series", "model", "p10", "p50", "p90"]]

    model_col = _first_existing_col(df, ["model"])
    p50_col = _first_existing_col(df, ["p50", "median", "mean", "yhat", "prediction", "forecast", "expected"])
    p10_col = _first_existing_col(df, ["p10", "q0.1", "q10", "lower", "lo", "low"])
    p90_col = _first_existing_col(df, ["p90", "q0.9", "q90", "upper", "hi", "high"])

    if series_col and p50_col and model_col:
        out = pd.DataFrame({
            "date": df[date_col],
            "series": df[series_col].astype(str),
            "model": df[model_col].astype(str),
            "p50": pd.to_numeric(df[p50_col], errors="coerce"),
            "p10": pd.to_numeric(df[p10_col], errors="coerce") if p10_col else np.nan,
            "p90": pd.to_numeric(df[p90_col], errors="coerce") if p90_col else np.nan,
        }).dropna(subset=["p50"])
        return out

    if series_col and p50_col and not model_col:
        out = pd.DataFrame({
            "date": df[date_col],
            "series": df[series_col].astype(str),
            "model": "EnsembleMean",
            "p50": pd.to_numeric(df[p50_col], errors="coerce"),
            "p10": pd.to_numeric(df[p10_col], errors="coerce") if p10_col else np.nan,
            "p90": pd.to_numeric(df[p90_col], errors="coerce") if p90_col else np.nan,
        }).dropna(subset=["p50"])
        return out

    if series_col and (model_col is None) and (p50_col is None):
        id_cols = [date_col, series_col]
        candidate_model_cols = [c for c in df.columns if c not in id_cols]

        def _norm(s: str) -> str:
            return str(s).strip().lower().replace(" ", "").replace("_", "")

        name_map = {
            "deepar": "DeepAR",
            "tft": "TFT",
            "patchtst": "PatchTST",
            "ensemblemean": "EnsembleMean",
            "ensemble": "EnsembleMean",
            "ens": "EnsembleMean",
        }

        model_cols = []
        for c in candidate_model_cols:
            nc = _norm(c)
            if any(k in nc for k in ["deepar", "tft", "patchtst", "ensemble", "ens"]):
                model_cols.append(c)
        if not model_cols and candidate_model_cols:
            model_cols = candidate_model_cols

        wide_pred = df[id_cols + model_cols].copy()
        for c in model_cols:
            wide_pred[c] = pd.to_numeric(wide_pred[c], errors="coerce")

        env_cols = []
        for c in model_cols:
            if name_map.get(_norm(c), str(c)) != "EnsembleMean":
                env_cols.append(c)
        if not env_cols:
            env_cols = model_cols

        wide_pred["p10_env"] = wide_pred[env_cols].min(axis=1)
        wide_pred["p90_env"] = wide_pred[env_cols].max(axis=1)

        ens_col = None
        for c in model_cols:
            if name_map.get(_norm(c), str(c)) == "EnsembleMean":
                ens_col = c
                break
        if ens_col is not None:
            wide_pred["p50_ens"] = wide_pred[ens_col]
        else:
            wide_pred["p50_ens"] = wide_pred[env_cols].mean(axis=1)

        melted = wide_pred[id_cols + env_cols].melt(
            id_vars=id_cols,
            value_vars=env_cols,
            var_name="model",
            value_name="p50",
        )
        melted["date"] = pd.to_datetime(melted[date_col])
        melted["series"] = melted[series_col].astype(str)
        melted["model"] = melted["model"].map(lambda x: name_map.get(_norm(x), str(x)))
        melted["p50"] = pd.to_numeric(melted["p50"], errors="coerce")
        melted["p10"] = np.nan
        melted["p90"] = np.nan

        ens_rows = pd.DataFrame({
            "date": pd.to_datetime(wide_pred[date_col]),
            "series": wide_pred[series_col].astype(str),
            "model": "EnsembleMean",
            "p10": pd.to_numeric(wide_pred["p10_env"], errors="coerce"),
            "p50": pd.to_numeric(wide_pred["p50_ens"], errors="coerce"),
            "p90": pd.to_numeric(wide_pred["p90_env"], errors="coerce"),
        })

        out = pd.concat(
            [melted[["date", "series", "model", "p10", "p50", "p90"]], ens_rows],
            ignore_index=True
        ).dropna(subset=["p50"])
        return out

    raise ValueError("Could not parse forecast file into long format. "
                     f"Columns={list(df.columns)[:40]}")


# =========================
# RISK + LABELS
# =========================
def risk_level(p: float) -> str:
    if not np.isfinite(p):
        return "UNKNOWN"
    if p < 0.05:
        return "LOW"
    if p < 0.20:
        return "MEDIUM"
    return "HIGH"


def action_label(p: float) -> str:
    lvl = risk_level(p)
    if lvl == "LOW":
        return "OK"
    if lvl == "MEDIUM":
        return "WATCH"
    if lvl == "HIGH":
        return "ESCALATE"
    return "N/A"


def badge_for_action(action: str) -> dbc.Badge:
    if action == "OK":
        return dbc.Badge("OK", color="success", pill=True)
    if action == "WATCH":
        return dbc.Badge("WATCH", color="warning", pill=True)
    if action == "ESCALATE":
        return dbc.Badge("ESCALATE", color="danger", pill=True)
    return dbc.Badge("N/A", color="secondary", pill=True)


def badge_for_conf(conf: str) -> dbc.Badge:
    if conf == "HIGH":
        return dbc.Badge("High", color="success", pill=True)
    if conf == "MEDIUM":
        return dbc.Badge("Medium", color="warning", pill=True)
    if conf == "LOW":
        return dbc.Badge("Low", color="danger", pill=True)
    return dbc.Badge("N/A", color="secondary", pill=True)


# =========================
# FORECAST LOADERS
# =========================
def read_forecast_long(horizon: int) -> pd.DataFrame:
    stems_to_try = [
        f"test_forecast_quantiles_{horizon}d",
        f"latest_forecast_{horizon}d",
    ]
    last_err = None
    for stem in stems_to_try:
        try:
            df = read_outputs_df(stem)
            long_df = _standardize_forecast_long(df)
            long_df = long_df[long_df["series"].isin(["available_adult_icu_beds", "available_ped_icu_beds"])].copy()
            return long_df.sort_values(["date", "series", "model"])
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Could not load forecast for horizon={horizon}. Tried: {stems_to_try}. Last error: {last_err}")


def available_models(f_long: pd.DataFrame) -> List[str]:
    models = sorted(f_long["model"].dropna().unique().tolist())
    if "EnsembleMean" in models:
        models = ["EnsembleMean"] + [m for m in models if m != "EnsembleMean"]
    return models


# =========================
# RISK TABLE FROM FORECAST
# =========================
def make_risk_table_from_forecast(
    f_long: pd.DataFrame,
    series_key: str,
    model: str,
    threshold: float,
) -> pd.DataFrame:
    f = f_long[(f_long["series"] == series_key) & (f_long["model"] == model)].copy().sort_values("date")
    if f.empty:
        return pd.DataFrame(columns=["Date", "Expected", "Likely low–high", "Shortage chance", "Level"])

    p10 = f["p10"].to_numpy(dtype=float)
    p50 = f["p50"].to_numpy(dtype=float)
    p90 = f["p90"].to_numpy(dtype=float)

    denom = 2.0 * 1.281551565545
    sigma = (p90 - p10) / np.maximum(1e-6, denom)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, np.nan)
    fallback = np.nanmedian(np.abs(np.diff(p50))) if len(p50) > 2 else 5.0
    sigma = np.where(np.isfinite(sigma), sigma, max(1.0, float(fallback)))

    z = (threshold - p50) / sigma
    prob = normal_cdf(z)

    rows = []
    for d, e, lo, hi, p in zip(f["date"], p50, p10, p90, prob):
        band = ""
        if np.isfinite(lo) and np.isfinite(hi):
            band = f"{int(round(lo))}–{int(round(hi))}"
        rows.append({
            "Date": pd.to_datetime(d).date().isoformat(),
            "Expected": float(e),
            "Likely low–high": band,
            "Shortage chance": float(p),
            "Level": risk_level(float(p)),
        })
    return pd.DataFrame(rows)


# =========================
# CONFIDENCE + DISAGREEMENT
# =========================
def compute_disagreement_and_confidence(
    f_long: pd.DataFrame,
    series_key: str,
) -> pd.DataFrame:
    df = f_long[f_long["series"] == series_key].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "disagree_abs", "disagree_ratio", "band_abs", "band_ratio", "confidence"])

    indiv = df[df["model"].astype(str).str.lower() != "ensemblemean"].copy()
    if not indiv.empty:
        g = indiv.groupby("date")["p50"].agg(["min", "max", "mean"]).reset_index()
        g["disagree_abs"] = (g["max"] - g["min"]).astype(float)
        g["disagree_ratio"] = g["disagree_abs"] / np.maximum(1.0, np.abs(g["mean"]))
        disagree = g[["date", "disagree_abs", "disagree_ratio"]].copy()
    else:
        disagree = pd.DataFrame({"date": df["date"].unique(), "disagree_abs": np.nan, "disagree_ratio": np.nan})

    ens = df[df["model"].astype(str).str.lower() == "ensemblemean"].copy()
    if not ens.empty:
        ens = ens.groupby("date")[["p10", "p50", "p90"]].mean(numeric_only=True).reset_index()
        ens["band_abs"] = (ens["p90"] - ens["p10"]).astype(float)
        ens["band_ratio"] = ens["band_abs"] / np.maximum(1.0, np.abs(ens["p50"]))
        band = ens[["date", "band_abs", "band_ratio"]].copy()
    else:
        band = pd.DataFrame({"date": df["date"].unique(), "band_abs": np.nan, "band_ratio": np.nan})

    out = pd.merge(disagree, band, on="date", how="outer").sort_values("date")

    def conf_row(r) -> str:
        dr = r.get("disagree_ratio", np.nan)
        br = r.get("band_ratio", np.nan)
        score = np.nanmax([dr, br])
        if not np.isfinite(score):
            return "N/A"
        if score <= 0.05:
            return "HIGH"
        if score <= 0.15:
            return "MEDIUM"
        return "LOW"

    out["confidence"] = out.apply(conf_row, axis=1)
    return out


# =========================
# TABLE COMPONENT (ONLY LEVEL COLORED)
# =========================
def risk_table_component(df: pd.DataFrame, focus_date: str) -> dash_table.DataTable:
    show = df.copy()
    show["Expected"] = show["Expected"].map(fmt_int)

    show["Shortage chance %"] = (pd.to_numeric(show["Shortage chance"], errors="coerce") * 100).map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else ""
    )

    show["_is_focus"] = (show["Date"].astype(str) == str(focus_date)).astype(int)
    show = show.sort_values(["_is_focus", "Date"], ascending=[False, True])

    cols = ["Date", "Expected", "Likely low–high", "Shortage chance %", "Level"]

    style_rules = [
        {"if": {"filter_query": "{_is_focus} = 1"}, "border": "2px solid #4C6FFF"},
        {"if": {"filter_query": "{_is_focus} = 1", "column_id": "Date"}, "fontWeight": "800"},

        {"if": {"filter_query": '{Level} = "LOW"', "column_id": "Level"},
         "backgroundColor": "#E8F5EE", "color": "#0F5132", "fontWeight": "900"},
        {"if": {"filter_query": '{Level} = "MEDIUM"', "column_id": "Level"},
         "backgroundColor": "#FFF4E1", "color": "#7A4B00", "fontWeight": "900"},
        {"if": {"filter_query": '{Level} = "HIGH"', "column_id": "Level"},
         "backgroundColor": "#FCE8E8", "color": "#842029", "fontWeight": "900"},
    ]

    return dash_table.DataTable(
        columns=[{"name": c.upper(), "id": c, "hideable": True} for c in cols],
        data=show[cols + ["_is_focus"]].to_dict("records"),
        hidden_columns=["_is_focus"],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "10px", "fontFamily": "system-ui", "fontSize": "13px", "whiteSpace": "normal"},
        style_header={"fontWeight": "900", "backgroundColor": "#f3f5f7"},
        style_data_conditional=style_rules,
        page_size=8,
    )


# =========================
# METRICS LOOKUP (for pills)
# =========================
def lookup_backtest_mae_rmse(h: int, model: str) -> tuple[Optional[float], Optional[float], str]:
    try:
        m = read_outputs_df("backtest_metrics")
    except Exception:
        return None, None, "n/a"

    col_h = _first_existing_col(m, ["horizon_days", "horizon"]) or "horizon_days"
    col_m = _first_existing_col(m, ["model"]) or "model"
    col_mae = _first_existing_col(m, ["mae_mean_over_series", "mae"]) or "mae_mean_over_series"
    col_rmse = _first_existing_col(m, ["rmse_mean_over_series", "rmse"]) or "rmse_mean_over_series"

    if col_h not in m.columns or col_m not in m.columns or col_mae not in m.columns or col_rmse not in m.columns:
        return None, None, "n/a"

    df = m[[col_h, col_m, col_mae, col_rmse]].copy()
    df.columns = ["horizon_days", "model", "mae", "rmse"]
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    want = _norm_key(model)

    sub = df[df["horizon_days"] == float(h)].copy()
    if sub.empty:
        return None, None, "n/a"

    hit = sub[sub["model"].map(_norm_key) == want]
    if hit.empty:
        hit = sub[sub["model"].map(_norm_key) == _norm_key("EnsembleMean")]
        if hit.empty:
            return None, None, "n/a"

    r = hit.iloc[0]
    return (float(r["mae"]) if pd.notna(r["mae"]) else None,
            float(r["rmse"]) if pd.notna(r["rmse"]) else None,
            str(r["model"]))


# =========================
# PLOT
# =========================
def plot_unit(
    unit_title: str,
    series_key: str,
    hist_df: pd.DataFrame,
    f_long: pd.DataFrame,
    horizon: int,
    selected_model: str,
    show_band: bool,
    threshold: float,
    focus_date: str,
) -> go.Figure:
    fig = go.Figure()

    y_hist = hist_df[series_key].to_numpy()
    x_hist = hist_df.index

    fig.add_trace(go.Scatter(
        x=x_hist, y=y_hist,
        mode="lines",
        name="Observed (past)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Observed: %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x_hist[-1]], y=[y_hist[-1]],
        mode="markers",
        name="Observed (today)",
        marker=dict(size=9),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Observed(today): %{y:.0f}<extra></extra>",
    ))

    f_unit = f_long[f_long["series"] == series_key].copy()
    if f_unit.empty:
        fig.update_layout(template="plotly_white", title=f"{unit_title} (no forecast loaded)", height=360,
                          margin=dict(l=16, r=16, t=56, b=16))
        return fig

    fig.add_hline(
        y=threshold,
        line_dash="dot",
        annotation_text=f"Threshold: {int(round(threshold))}",
        annotation_position="bottom right"
    )

    first_fc = f_unit["date"].min()
    fig.add_vline(x=first_fc, line_dash="dot", opacity=0.6)

    if focus_date and focus_date != "OVERVIEW":
        try:
            fd = pd.to_datetime(focus_date)
            fig.add_vline(x=fd, line_dash="dash", opacity=0.9)
        except Exception:
            pass

    if selected_model not in f_unit["model"].unique():
        selected_model = "EnsembleMean" if "EnsembleMean" in f_unit["model"].unique() else f_unit["model"].iloc[0]

    f_m = f_unit[f_unit["model"] == selected_model].copy().sort_values("date")
    x_fc = f_m["date"].to_numpy()
    y50 = f_m["p50"].to_numpy(dtype=float)

    fig.add_trace(go.Scatter(
        x=x_fc, y=y50,
        mode="lines+markers",
        name=f"Expected ({selected_model})",
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}"
            f"<br>Horizon: {horizon}d"
            f"<br>Model: {selected_model}"
            "<br>Expected: %{y:.0f}"
            "<extra></extra>"
        ),
    ))

    if show_band and np.isfinite(f_m["p10"].to_numpy()).any() and np.isfinite(f_m["p90"].to_numpy()).any():
        y10 = f_m["p10"].to_numpy(dtype=float)
        y90 = f_m["p90"].to_numpy(dtype=float)

        fig.add_trace(go.Scatter(x=x_fc, y=y90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x_fc, y=y10,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Likely range (p10–p90)",
            hoverinfo="skip",
        ))

    if focus_date and focus_date != "OVERVIEW":
        try:
            fd = pd.to_datetime(focus_date).date()
            hit = f_m[pd.to_datetime(f_m["date"]).dt.date == fd]
            if not hit.empty:
                fig.add_trace(go.Scatter(
                    x=hit["date"], y=hit["p50"],
                    mode="markers",
                    name="Focus date",
                    marker=dict(size=14, symbol="diamond"),
                    hovertemplate="Focus date: %{x|%Y-%m-%d}<br>Expected: %{y:.0f}<extra></extra>",
                ))
        except Exception:
            pass

    fig.update_layout(
        template="plotly_white",
        title=f"{unit_title} available beds (past → today → forecast)",
        height=360,
        margin=dict(l=16, r=16, t=56, b=16),
        legend_orientation="h",
    )
    return fig


# =========================
# ACTION BOARD (compact + details)
# =========================
def _merge_action_board(risk_tbl: pd.DataFrame, conf_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if risk_tbl.empty:
        return pd.DataFrame()

    merged = risk_tbl.copy()
    merged["date"] = pd.to_datetime(merged["Date"])
    c = conf_df.copy()
    merged = merged.merge(c, on="date", how="left")
    return merged.sort_values("date").head(top_n)


def action_board_compact_card(
    unit_name: str,
    risk_tbl: pd.DataFrame,
    conf_df: pd.DataFrame,
    top_n: int = 3,
) -> dbc.Card:
    merged = _merge_action_board(risk_tbl, conf_df, top_n=top_n)
    if merged.empty:
        return dbc.Card(dbc.CardBody(html.Div("No data", className="muted tiny")))

    rows = []
    for _, r in merged.iterrows():
        p = float(r["Shortage chance"]) if pd.notna(r["Shortage chance"]) else np.nan
        conf = r.get("confidence", "N/A")
        act = action_label(p) if np.isfinite(p) else "N/A"

        rows.append(html.Tr([
            html.Td(pd.to_datetime(r["date"]).date().isoformat()),
            html.Td(fmt_int(r["Expected"])),
            html.Td(pct(p)),
            html.Td(badge_for_conf(conf)),
            html.Td(badge_for_action(act)),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Date"),
                html.Th("Exp"),
                html.Th("Risk"),
                html.Th("Conf"),
                html.Th("Act"),
            ])),
            html.Tbody(rows),
        ],
        bordered=False,
        hover=False,
        responsive=True,
        size="sm",
        className="ab-mini mb-0",
    )

    return dbc.Card(
        dbc.CardBody([
            html.Div(f"{unit_name} (72h)", className="ab-card-title"),
            table
        ]),
        className="shadow-soft",
    )


def action_board_details_tables(
    adult_tbl: pd.DataFrame,
    ped_tbl: pd.DataFrame,
    conf_adult: pd.DataFrame,
    conf_ped: pd.DataFrame,
    top_n: int = 3,
) -> html.Div:
    def _detail_table(unit_name: str, risk_tbl: pd.DataFrame, conf_df: pd.DataFrame) -> dbc.Table:
        merged = _merge_action_board(risk_tbl, conf_df, top_n=top_n)
        if merged.empty:
            return dbc.Table([], bordered=False)

        rows = []
        for _, r in merged.iterrows():
            p = float(r["Shortage chance"]) if pd.notna(r["Shortage chance"]) else np.nan
            act = action_label(p) if np.isfinite(p) else "N/A"
            conf = r.get("confidence", "N/A")
            dis = r.get("disagree_abs", np.nan)
            band = r.get("band_abs", np.nan)

            rows.append(html.Tr([
                html.Td(pd.to_datetime(r["date"]).date().isoformat()),
                html.Td(fmt_int(r["Expected"])),
                html.Td(pct(p)),
                html.Td(badge_for_conf(conf)),
                html.Td(f"{int(round(dis))}" if pd.notna(dis) else "n/a"),
                html.Td(f"{int(round(band))}" if pd.notna(band) else "n/a"),
                html.Td(badge_for_action(act)),
            ]))

        return dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th(f"{unit_name} (72h)"),
                    html.Th("Exp"),
                    html.Th("Risk"),
                    html.Th("Conf"),
                    html.Th("Δ"),
                    html.Th("Band"),
                    html.Th("Act"),
                ])),
                html.Tbody(rows),
            ],
            bordered=False,
            hover=True,
            responsive=True,
            size="sm",
            className="ab-mini mb-0",
        )

    return html.Div([
        dbc.Row(className="g-3 mt-2", children=[
            dbc.Col(_detail_table("Adult ICU", adult_tbl, conf_adult), md=6),
            dbc.Col(_detail_table("Pediatric ICU", ped_tbl, conf_ped), md=6),
        ])
    ])


# =========================
# LOAD HISTORY ONCE (cleaned for dashboard math)
# =========================
wide = load_wide_csv(DATA_PATH, clean_negatives=True)
last = wide.iloc[-1]
data_through = wide.index.max().date()
staleness_days = (date.today() - data_through).days


# =========================
# DASH APP + STYLE
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.title = "ICU Bed Utilization – Decision Support"

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { background: #f6f8fb; }
      .card { border-radius: 16px !important; }
      .shadow-soft { box-shadow: 0 10px 30px rgba(16, 24, 40, 0.08); }
      .pill-nav .btn { border-radius: 999px !important; padding: 10px 16px; font-weight: 700; }
      .muted { color: #667085; }
      .kpi-value { font-size: 2.25rem; font-weight: 900; letter-spacing: -0.02em; }
      .section-title { font-weight: 900; letter-spacing: -0.01em; }
      .compact-label { font-size: .9rem; font-weight: 800; color: #344054; }
      .tiny { font-size: 0.86rem; }
      .btn-soft { border-radius: 999px !important; font-weight: 800; }

      /* --- Action Board compact --- */
      .ab-mini th, .ab-mini td { padding: .35rem .45rem !important; font-size: .86rem; vertical-align: middle; }
      .ab-mini .badge { font-size: .75rem; }
      .ab-card-title { font-weight: 900; font-size: .95rem; margin-bottom: .35rem; }

      /* --- Pills (metrics) --- */
      .pill-metric {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid #D0D5DD;
        background: #F9FAFB;
        font-weight: 800;
        color: #101828;
        font-size: .86rem;
      }

      /* --- Downloads compact --- */
      .downloads-compact { display: flex; flex-wrap: wrap; gap: 8px; }
      .downloads-compact .btn { padding: 6px 10px !important; font-size: .82rem !important; }

      /* --- ✅ Slider marks: keep them readable (coarse marks only) --- */
      .rc-slider-mark-text { font-size: 11px; color: #98A2B3; white-space: nowrap; }
      .scenario-num { max-width: 92px; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""


def kpi_card(title: str, value: str, subtitle: str, badge: dbc.Badge):
    return dbc.Card(
        dbc.CardBody([
            html.Div([html.Div(title, className="muted"), badge],
                     className="d-flex align-items-center justify-content-between"),
            html.Div(value, className="kpi-value"),
            html.Div(subtitle, className="muted tiny"),
        ]),
        className="shadow-soft",
    )


def section_card(header: str, children):
    return dbc.Card(
        [dbc.CardHeader(header, className="section-title"), dbc.CardBody(children)],
        className="shadow-soft",
    )


tab_buttons = dbc.ButtonGroup(
    [
        dbc.Button("Dashboard", id="btn-tab-dashboard", color="dark", outline=False, className="me-2"),
        dbc.Button("Model & Metrics", id="btn-tab-metrics", color="dark", outline=True, className="me-2"),
        dbc.Button("Data Quality", id="btn-tab-quality", color="dark", outline=True),
    ],
    className="pill-nav",
)

howto_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("How to use")),
        dbc.ModalBody([
            html.Ul([
                html.Li("Select Horizon (7/14/28) to load forecast output."),
                html.Li("Line dropdown shows ONE model at a time (EnsembleMean or a single model)."),
                html.Li("Action Board summarizes the next 72 hours: Risk + Confidence + Disagreement + Action."),
                html.Li("Scenario presets help in meetings (Add 50/100 beds, buffer +20)."),
                html.Li("Download buttons export forecast/risk/metrics."),
            ])
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", color="secondary")),
    ],
    id="howto-modal",
    is_open=False,
)

downloads = html.Div([
    dcc.Download(id="dl-forecast"),
    dcc.Download(id="dl-risk"),
    dcc.Download(id="dl-metrics"),
])

app.layout = dbc.Container(
    fluid=True,
    children=[
        downloads,
        dcc.Store(id="active-tab", data="dashboard"),

        dbc.Row(
            className="align-items-center mt-3",
            children=[
                dbc.Col(
                    html.Div([
                        html.H2("ICU Bed Utilization – Decision Support", className="mb-0"),
                        html.Div(
                            f"Data through {data_through} • Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            className="muted"
                        ),
                    ]),
                    md=7
                ),
                dbc.Col(
                    html.Div(
                        [tab_buttons, dbc.Button("HOW TO USE", id="howto-open", color="light", className="ms-3"), howto_modal],
                        className="d-flex justify-content-end align-items-center"
                    ),
                    md=5
                ),
            ],
        ),

        dbc.Alert(
            f"Data freshness warning: data is stale by {staleness_days} days. Interpret forecasts cautiously."
            if staleness_days > 30 else "Data freshness: OK",
            color="warning" if staleness_days > 30 else "success",
            className="mt-3",
        ),

        # =========================
        # DASHBOARD PAGE
        # =========================
        html.Div(
            id="page-dashboard",
            children=[
                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(html.Div(id="kpi-adult"), md=3),
                    dbc.Col(html.Div(id="kpi-ped"), md=3),
                    dbc.Col(section_card("Forecast summary", html.Div(id="forecast-summary")), md=6),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Action Board (next 72 hours)", html.Div(id="action-board")), md=12),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Controls", [
                        dbc.Row(className="g-2", children=[
                            dbc.Col([
                                html.Div("Horizon", className="compact-label"),
                                dcc.Dropdown(
                                    options=[{"label": str(h), "value": h} for h in HORIZONS],
                                    value=14, clearable=False, id="horizon"
                                ),
                            ], md=2),

                            dbc.Col([
                                html.Div("Focus date", className="compact-label"),
                                dcc.Dropdown(id="focus-date", clearable=False),
                            ], md=4),

                            dbc.Col([
                                html.Div("Line (one model only)", className="compact-label"),
                                dcc.Dropdown(id="lines", multi=False, clearable=False),
                            ], md=6),
                        ]),

                        dbc.Row(className="g-2 mt-2", children=[
                            dbc.Col([
                                dbc.Checklist(
                                    options=[{"label": "Show uncertainty band (p10–p90)", "value": 1}],
                                    value=[1], id="show-band", switch=True,
                                ),
                            ], md=4),
                            dbc.Col([
                                dbc.Checklist(
                                    options=[{"label": "Enable scenario (what-if)", "value": 1}],
                                    value=[], id="scenario-on", switch=True,
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Div("Downloads", className="compact-label"),
                                html.Div([
                                    dbc.Button("Forecast CSV", id="btn-dl-forecast", color="dark", outline=True, className="btn-soft", size="sm"),
                                    dbc.Button("Risk CSV", id="btn-dl-risk", color="dark", outline=True, className="btn-soft", size="sm"),
                                    dbc.Button("Metrics CSV", id="btn-dl-metrics", color="dark", outline=True, className="btn-soft", size="sm"),
                                ], className="downloads-compact")
                            ], md=4),
                        ]),
                    ]), md=12),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Adult ICU", dcc.Graph(id="chart-adult", config={"displayModeBar": True})), md=6),
                    dbc.Col(section_card("Pediatric ICU", dcc.Graph(id="chart-ped", config={"displayModeBar": True})), md=6),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Scenario builder", [
                        html.Div("Scenario presets", className="compact-label"),
                        html.Div([
                            dbc.Button("+50 Adult beds", id="preset-a50", color="secondary", outline=True, className="btn-soft me-2"),
                            dbc.Button("+100 Adult beds", id="preset-a100", color="secondary", outline=True, className="btn-soft me-2"),
                            dbc.Button("+20 Buffer", id="preset-b20", color="secondary", outline=True, className="btn-soft me-2"),
                            dbc.Button("Baseline", id="preset-baseline", color="secondary", outline=True, className="btn-soft"),
                        ], className="mt-1"),

                        html.Hr(),

                        # =========================
                        # ✅ UPDATED: sliders + right-side value boxes + coarse marks
                        # =========================
                        html.Div("Extra beds added (+) — Adult", className="compact-label"),
                        dbc.Row(className="g-2 align-items-center", children=[
                            dbc.Col(
                                dcc.Slider(
                                    min=0, max=1000, step=10, value=0, id="extra-adult",
                                    marks=SLIDER_MARKS_ADULT,
                                    included=True,
                                    updatemode="drag",
                                ),
                                md=10
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="extra-adult-box",
                                    type="number",
                                    min=0, max=1000, step=10, value=0,
                                    size="sm",
                                    className="scenario-num"
                                ),
                                md=2
                            ),
                        ]),

                        html.Div("Extra beds added (+) — Pediatric", className="compact-label mt-3"),
                        dbc.Row(className="g-2 align-items-center", children=[
                            dbc.Col(
                                dcc.Slider(
                                    min=0, max=200, step=5, value=0, id="extra-ped",
                                    marks=SLIDER_MARKS_PED,
                                    included=True,
                                    updatemode="drag",
                                ),
                                md=10
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="extra-ped-box",
                                    type="number",
                                    min=0, max=200, step=5, value=0,
                                    size="sm",
                                    className="scenario-num"
                                ),
                                md=2
                            ),
                        ]),

                        html.Div("Safety buffer (min available beds) +", className="compact-label mt-3"),
                        dbc.Row(className="g-2 align-items-center", children=[
                            dbc.Col(
                                dcc.Slider(
                                    min=0, max=1000, step=10, value=0, id="safety-buffer",
                                    marks=SLIDER_MARKS_BUFFER,
                                    included=True,
                                    updatemode="drag",
                                ),
                                md=10
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="safety-buffer-box",
                                    type="number",
                                    min=0, max=1000, step=10, value=0,
                                    size="sm",
                                    className="scenario-num"
                                ),
                                md=2
                            ),
                        ]),

                        dbc.Button("Reset scenario", id="reset-btn", color="secondary", className="mt-3"),
                        html.Div("Tip: Risk is P(available < threshold).", className="muted mt-2 tiny"),
                    ]), md=6),
                    dbc.Col(section_card("Scenario impact", html.Div(id="impact-box")), md=6),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Adult ICU shortage outlook", html.Div(id="table-adult")), md=6),
                    dbc.Col(section_card("Pediatric ICU shortage outlook", html.Div(id="table-ped")), md=6),
                ]),

                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Recommendations / perspective", html.Div(id="reco-box")), md=12),
                ]),

                html.Div("Disclaimer: decision support only. Confirm with local policy + real-time context.", className="muted mt-3 tiny"),
            ],
        ),

        # =========================
        # METRICS PAGE
        # =========================
        html.Div(
            id="page-metrics",
            style={"display": "none"},
            children=[
                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Model performance summary", html.Div(id="metrics-summary")), md=12),
                ]),
                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Metrics Table", html.Div(id="metrics-table")), md=12),
                ]),
            ],
        ),

        # =========================
        # DATA QUALITY PAGE
        # =========================
        html.Div(
            id="page-quality",
            style={"display": "none"},
            children=[
                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Dataset summary", html.Div(id="quality-summary")), md=6),
                    dbc.Col(section_card("Sanity checks (negative counts)", html.Div(id="quality-neg")), md=6),
                ]),
                dbc.Row(className="g-3 mt-2", children=[
                    dbc.Col(section_card("Missing values", html.Div(id="quality-missing")), md=12),
                ]),
            ],
        ),
    ],
)


# =========================
# TAB SWITCH + HOWTO MODAL
# =========================
@app.callback(
    Output("active-tab", "data"),
    Input("btn-tab-dashboard", "n_clicks"),
    Input("btn-tab-metrics", "n_clicks"),
    Input("btn-tab-quality", "n_clicks"),
    prevent_initial_call=True,
)
def set_active_tab(n1, n2, n3):
    if not callback_context.triggered:
        return "dashboard"
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "btn-tab-metrics":
        return "metrics"
    if trig == "btn-tab-quality":
        return "quality"
    return "dashboard"


@app.callback(
    Output("page-dashboard", "style"),
    Output("page-metrics", "style"),
    Output("page-quality", "style"),
    Output("btn-tab-dashboard", "outline"),
    Output("btn-tab-metrics", "outline"),
    Output("btn-tab-quality", "outline"),
    Input("active-tab", "data"),
)
def show_page(tab):
    dash_style = {"display": "block"} if tab == "dashboard" else {"display": "none"}
    met_style = {"display": "block"} if tab == "metrics" else {"display": "none"}
    qua_style = {"display": "block"} if tab == "quality" else {"display": "none"}
    return dash_style, met_style, qua_style, (tab != "dashboard"), (tab != "metrics"), (tab != "quality")


@app.callback(
    Output("howto-modal", "is_open"),
    Input("howto-open", "n_clicks"),
    Input("howto-close", "n_clicks"),
    State("howto-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_howto(open_clicks, close_clicks, is_open):
    if not callback_context.triggered:
        return is_open
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "howto-open":
        return True
    if trig == "howto-close":
        return False
    return is_open


# =========================
# DASHBOARD CONTROL OPTIONS
# =========================
@app.callback(
    Output("focus-date", "options"),
    Output("focus-date", "value"),
    Output("lines", "options"),
    Output("lines", "value"),
    Input("horizon", "value"),
)
def update_controls(h):
    h = int(h)
    try:
        f_long = read_forecast_long(h)

        dates = sorted(pd.to_datetime(f_long["date"]).dt.date.unique().tolist())
        focus_opts = [{"label": f"Overview (next {h} days)", "value": "OVERVIEW"}]
        focus_opts += [{"label": f"{pd.Timestamp(d).strftime('%a')} — {d.isoformat()}", "value": d.isoformat()} for d in dates]
        focus_val = "OVERVIEW"

        models = available_models(f_long)
        line_opts = [{"label": m, "value": m} for m in models]
        line_val = "EnsembleMean" if "EnsembleMean" in models else (models[0] if models else "EnsembleMean")

        return focus_opts, focus_val, line_opts, line_val
    except Exception:
        return (
            [{"label": f"Overview (next {h} days)", "value": "OVERVIEW"}],
            "OVERVIEW",
            [{"label": "EnsembleMean", "value": "EnsembleMean"}],
            "EnsembleMean",
        )


# =========================
# ✅ SCENARIO CONTROLLER (PRESETS + SLIDER/BOX SYNC) — single callback, no conflicts
# =========================
@app.callback(
    Output("extra-adult", "value"),
    Output("extra-adult-box", "value"),
    Output("extra-ped", "value"),
    Output("extra-ped-box", "value"),
    Output("safety-buffer", "value"),
    Output("safety-buffer-box", "value"),
    Output("scenario-on", "value"),
    Input("preset-a50", "n_clicks"),
    Input("preset-a100", "n_clicks"),
    Input("preset-b20", "n_clicks"),
    Input("preset-baseline", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    Input("extra-adult", "value"),
    Input("extra-adult-box", "value"),
    Input("extra-ped", "value"),
    Input("extra-ped-box", "value"),
    Input("safety-buffer", "value"),
    Input("safety-buffer-box", "value"),
    prevent_initial_call=True,
)
def scenario_controller(n1, n2, n3, n4, nreset, a_s, a_b, p_s, p_b, b_s, b_b):
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]

    # current values (prefer slider if present, else box)
    a = int(a_s if a_s is not None else (a_b or 0))
    p = int(p_s if p_s is not None else (p_b or 0))
    b = int(b_s if b_s is not None else (b_b or 0))

    # presets / reset
    if trig == "preset-a50":
        a, p, b = 50, p, b
        scen = [1]
        return a, a, p, p, b, b, scen

    if trig == "preset-a100":
        a, p, b = 100, p, b
        scen = [1]
        return a, a, p, p, b, b, scen

    if trig == "preset-b20":
        a, p, b = a, p, 20
        scen = [1]
        return a, a, p, p, b, b, scen

    if trig in ("preset-baseline", "reset-btn"):
        return 0, 0, 0, 0, 0, 0, []

    # manual edit: snap + clamp, and keep slider + box synced
    if trig in ("extra-adult", "extra-adult-box"):
        new_a = a_s if trig == "extra-adult" else a_b
        a = _snap_int(new_a, step=10, vmin=0, vmax=1000)

    if trig in ("extra-ped", "extra-ped-box"):
        new_p = p_s if trig == "extra-ped" else p_b
        p = _snap_int(new_p, step=5, vmin=0, vmax=200)

    if trig in ("safety-buffer", "safety-buffer-box"):
        new_b = b_s if trig == "safety-buffer" else b_b
        b = _snap_int(new_b, step=10, vmin=0, vmax=1000)

    # auto-enable scenario if any non-zero
    scen = [1] if (a > 0 or p > 0 or b > 0) else []

    return a, a, p, p, b, b, scen


# =========================
# MAIN DASHBOARD UPDATE
# =========================
@app.callback(
    Output("kpi-adult", "children"),
    Output("kpi-ped", "children"),
    Output("forecast-summary", "children"),
    Output("action-board", "children"),
    Output("chart-adult", "figure"),
    Output("chart-ped", "figure"),
    Output("table-adult", "children"),
    Output("table-ped", "children"),
    Output("impact-box", "children"),
    Output("reco-box", "children"),
    Input("horizon", "value"),
    Input("focus-date", "value"),
    Input("lines", "value"),
    Input("show-band", "value"),
    Input("scenario-on", "value"),
    Input("extra-adult", "value"),
    Input("extra-ped", "value"),
    Input("safety-buffer", "value"),
)
def update_dashboard(h, focus, line_model, show_band_val, scenario_on_val, extra_a, extra_p, safety):
    h = int(h)
    selected_model = line_model or "EnsembleMean"
    show_band = bool(show_band_val and 1 in show_band_val)
    scenario_on = bool(scenario_on_val and 1 in scenario_on_val)

    adult_avail = safe_num(last["available_adult_icu_beds"])
    ped_avail = safe_num(last["available_ped_icu_beds"])
    adult_beds = safe_num(last[TOTAL_BEDS_COLS["adult"]])
    ped_beds = safe_num(last[TOTAL_BEDS_COLS["ped"]])
    adult_pat = safe_num(last["total_adult_icu_patients"])
    ped_pat = safe_num(last["total_ped_icu_patients"])
    adult_occ = 100.0 * adult_pat / max(1.0, adult_beds)
    ped_occ = 100.0 * ped_pat / max(1.0, ped_beds)

    try:
        f_long = read_forecast_long(h)
    except Exception as e:
        err = dbc.Alert(f"Cannot load forecast for horizon={h}: {e}", color="danger")
        empty_fig = go.Figure().update_layout(template="plotly_white", height=360)
        return (
            kpi_card("Adult ICU — available beds (today)", f"{int(round(adult_avail))}", f"Occupancy: {adult_occ:.1f}%", dbc.Badge("n/a", color="secondary", pill=True)),
            kpi_card("Pediatric ICU — available beds (today)", f"{int(round(ped_avail))}", f"Occupancy: {ped_occ:.1f}%", dbc.Badge("n/a", color="secondary", pill=True)),
            err,
            err,
            empty_fig, empty_fig,
            err, err,
            err,
            err
        )

    thr_adult = float(BASE_THRESHOLDS["adult"]) + (float(safety) if scenario_on else 0.0)
    thr_ped = float(BASE_THRESHOLDS["ped"]) + (float(safety) if scenario_on else 0.0)

    shift_adult = float(extra_a) if scenario_on else 0.0
    shift_ped = float(extra_p) if scenario_on else 0.0

    f_disp = f_long.copy()
    if scenario_on:
        for series_key, shift in [("available_adult_icu_beds", shift_adult), ("available_ped_icu_beds", shift_ped)]:
            mask = (f_disp["series"] == series_key) & (f_disp["model"].astype(str).str.lower() == "ensemblemean")
            f_disp.loc[mask, "p10"] = f_disp.loc[mask, "p10"] + shift
            f_disp.loc[mask, "p50"] = f_disp.loc[mask, "p50"] + shift
            f_disp.loc[mask, "p90"] = f_disp.loc[mask, "p90"] + shift

    adult_tbl = make_risk_table_from_forecast(f_disp, "available_adult_icu_beds", "EnsembleMean", thr_adult)
    ped_tbl = make_risk_table_from_forecast(f_disp, "available_ped_icu_beds", "EnsembleMean", thr_ped)

    adult_max = pd.to_numeric(adult_tbl["Shortage chance"], errors="coerce").max()
    ped_max = pd.to_numeric(ped_tbl["Shortage chance"], errors="coerce").max()
    adult_lvl = risk_level(float(adult_max)) if pd.notna(adult_max) else "UNKNOWN"
    ped_lvl = risk_level(float(ped_max)) if pd.notna(ped_max) else "UNKNOWN"

    kpiA = kpi_card(
        "Adult ICU — available beds (today)",
        f"{int(round(adult_avail))}",
        f"Occupancy: {adult_occ:.1f}% (Beds {int(round(adult_beds))}, Patients {int(round(adult_pat))})",
        dbc.Badge(adult_lvl, color=("success" if adult_lvl == "LOW" else "warning" if adult_lvl == "MEDIUM" else "danger" if adult_lvl == "HIGH" else "secondary"), pill=True),
    )
    kpiP = kpi_card(
        "Pediatric ICU — available beds (today)",
        f"{int(round(ped_avail))}",
        f"Occupancy: {ped_occ:.1f}% (Beds {int(round(ped_beds))}, Patients {int(round(ped_pat))})",
        dbc.Badge(ped_lvl, color=("success" if ped_lvl == "LOW" else "warning" if ped_lvl == "MEDIUM" else "danger" if ped_lvl == "HIGH" else "secondary"), pill=True),
    )

    f_dates = sorted(pd.to_datetime(f_disp["date"]).dt.date.unique().tolist())
    start_fc = f_dates[0] if f_dates else wide.index.max().date()
    end_fc = f_dates[-1] if f_dates else start_fc

    focus_date = "OVERVIEW"
    if focus and focus != "OVERVIEW":
        focus_date = str(focus)

    mae, rmse, ref_model = lookup_backtest_mae_rmse(h, selected_model)
    mae_txt = fmt_int(mae) if mae is not None else "n/a"
    rmse_txt = fmt_int(rmse) if rmse is not None else "n/a"

    pills = html.Div([
        html.Span(f"Backtest MAE ({h}d): {mae_txt}", className="pill-metric me-2"),
        html.Span(f"Backtest RMSE ({h}d): {rmse_txt}", className="pill-metric me-2"),
        html.Span(f"Ref model: {ref_model}", className="pill-metric"),
    ], className="mt-1")

    summary = html.Div([
        pills,
        html.Div([
            dbc.Badge("Scenario active" if scenario_on else "Baseline", color="primary", pill=True),
        ], className="mt-2"),
        html.Div([html.B(f"{h}-day forecast:"), f" {start_fc} → {end_fc}"], className="mt-2"),
        html.Div([html.B("Displayed line:"), f" {selected_model}"], className="mt-1"),
        html.Div([html.B("Thresholds:"), f" Adult {int(BASE_THRESHOLDS['adult'])} → {int(round(thr_adult))} | Peds {int(BASE_THRESHOLDS['ped'])} → {int(round(thr_ped))}"], className="mt-1"),
        html.Div([html.B("Scenario shift:"), f" +{int(shift_adult)} adult beds, +{int(shift_ped)} peds beds, buffer +{int(safety) if scenario_on else 0}"], className="mt-1"),
    ])

    conf_adult = compute_disagreement_and_confidence(f_disp, "available_adult_icu_beds")
    conf_ped = compute_disagreement_and_confidence(f_disp, "available_ped_icu_beds")

    compact = dbc.Row(className="g-3", children=[
        dbc.Col(action_board_compact_card("Adult ICU", adult_tbl, conf_adult, top_n=3), md=6),
        dbc.Col(action_board_compact_card("Pediatric ICU", ped_tbl, conf_ped, top_n=3), md=6),
    ])

    details = html.Details([
        html.Summary("Show details (Δ disagreement + band)", className="muted tiny mt-2"),
        action_board_details_tables(adult_tbl, ped_tbl, conf_adult, conf_ped, top_n=3),
    ], open=False)

    action_board = html.Div([
        compact,
        details,
        html.Div("Interpretation: High model disagreement or wide uncertainty band = lower confidence.", className="muted tiny mt-2"),
    ])

    hist_window = wide.iloc[-180:]
    figA = plot_unit(
        unit_title="Adult ICU",
        series_key="available_adult_icu_beds",
        hist_df=hist_window,
        f_long=f_disp,
        horizon=h,
        selected_model=selected_model,
        show_band=show_band,
        threshold=thr_adult,
        focus_date=focus_date,
    )
    figP = plot_unit(
        unit_title="Pediatric ICU",
        series_key="available_ped_icu_beds",
        hist_df=hist_window,
        f_long=f_disp,
        horizon=h,
        selected_model=selected_model,
        show_band=show_band,
        threshold=thr_ped,
        focus_date=focus_date,
    )

    tAdult = risk_table_component(adult_tbl, focus_date)
    tPed = risk_table_component(ped_tbl, focus_date)

    if not scenario_on:
        impact = dbc.Alert("Scenario is not active. Enable it or use presets to compare vs baseline.", color="info")
    else:
        baseA = make_risk_table_from_forecast(f_long, "available_adult_icu_beds", "EnsembleMean", float(BASE_THRESHOLDS["adult"]))
        baseP = make_risk_table_from_forecast(f_long, "available_ped_icu_beds", "EnsembleMean", float(BASE_THRESHOLDS["ped"]))
        abm = pd.to_numeric(baseA["Shortage chance"], errors="coerce").max()
        pbm = pd.to_numeric(baseP["Shortage chance"], errors="coerce").max()

        impact = html.Div([
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Unit"), html.Th("Baseline max risk"), html.Th("Scenario max risk"), html.Th("Δ")])),
                    html.Tbody([
                        html.Tr([
                            html.Td("Adult ICU"),
                            html.Td(pct(abm)),
                            html.Td(pct(adult_max)),
                            html.Td(f"{((adult_max-abm)*100):+.1f} pp" if (pd.notna(abm) and pd.notna(adult_max)) else "n/a"),
                        ]),
                        html.Tr([
                            html.Td("Pediatric ICU"),
                            html.Td(pct(pbm)),
                            html.Td(pct(ped_max)),
                            html.Td(f"{((ped_max-pbm)*100):+.1f} pp" if (pd.notna(pbm) and pd.notna(ped_max)) else "n/a"),
                        ]),
                    ])
                ],
                bordered=False, hover=True, responsive=True, className="mt-2",
            )
        ])

    reco_items = []
    if staleness_days > 30:
        reco_items.append(html.Li("Data is stale → retrain models with latest data before using for real operations."))
    reco_items.append(html.Li("If confidence is LOW (high disagreement or wide band), use the forecast as ‘directional’ only."))
    if adult_lvl == "HIGH" or ped_lvl == "HIGH":
        reco_items.append(html.Li("High risk detected → prepare buffer capacity, staffing readiness, and monitor real-time admissions."))
    elif adult_lvl == "MEDIUM" or ped_lvl == "MEDIUM":
        reco_items.append(html.Li("Medium risk → monitor daily trend; keep buffer and track sudden admission spikes."))
    reco_items.append(html.Li("Downloads: share the risk CSV + forecast CSV with managers for approval/traceability."))

    reco = html.Div([html.Ul(reco_items)])

    return kpiA, kpiP, summary, action_board, figA, figP, tAdult, tPed, impact, reco


# =========================
# METRICS TAB
# =========================
@app.callback(
    Output("metrics-summary", "children"),
    Output("metrics-table", "children"),
    Input("active-tab", "data"),
)
def update_metrics_tab(tab):
    if tab != "metrics":
        return no_update, no_update

    try:
        m = read_outputs_df("backtest_metrics")
    except Exception as e:
        err = dbc.Alert(f"Cannot load backtest_metrics*: {e}", color="danger")
        return err, err

    model_col = _first_existing_col(m, ["model"])
    horizon_col = _first_existing_col(m, ["horizon_days", "horizon"])
    mae_col = _first_existing_col(m, ["mae_mean_over_series", "mae"])
    rmse_col = _first_existing_col(m, ["rmse_mean_over_series", "rmse"])

    if not (model_col and horizon_col and mae_col and rmse_col):
        return dbc.Alert("backtest_metrics missing required columns (model/horizon/mae/rmse).", color="warning"), \
               dash_table.DataTable(data=m.to_dict("records"), columns=[{"name": c, "id": c} for c in m.columns])

    df = m[[horizon_col, model_col, mae_col, rmse_col]].copy()
    df.columns = ["horizon_days", "model", "mae_mean_over_series", "rmse_mean_over_series"]

    indiv = df[df["model"].astype(str).str.lower() != "ensemblemean"].copy()
    ens = df[df["model"].astype(str).str.lower() == "ensemblemean"].copy()

    score = indiv.groupby("model")["mae_mean_over_series"].mean(numeric_only=True).sort_values()
    best_model = score.index[0] if len(score) else "n/a"

    note = dbc.Alert(
        [
            html.B(f"Best model (lowest average MAE): {best_model}. "),
            html.Span("EnsembleMean improves stability by averaging, but weak models can drag the average down."),
        ],
        color="info"
    )

    def metrics_table_component(m: pd.DataFrame, highlight_best: bool, title: str) -> html.Div:
        d = m.copy()
        col_h = _first_existing_col(d, ["horizon_days", "horizon"]) or "horizon_days"
        col_m = _first_existing_col(d, ["model"]) or "model"
        col_mae = _first_existing_col(d, ["mae_mean_over_series", "mae"]) or "mae_mean_over_series"
        col_rmse = _first_existing_col(d, ["rmse_mean_over_series", "rmse"]) or "rmse_mean_over_series"

        d = d[[col_h, col_m, col_mae, col_rmse]].copy()
        d.columns = ["HORIZON_DAYS", "MODEL", "MAE", "RMSE"]

        d["MAE"] = pd.to_numeric(d["MAE"], errors="coerce")
        d["RMSE"] = pd.to_numeric(d["RMSE"], errors="coerce")

        d["_best_mae"] = 0
        d["_best_rmse"] = 0
        if highlight_best and d["HORIZON_DAYS"].notna().any():
            for hh in sorted(d["HORIZON_DAYS"].dropna().unique().tolist()):
                sub = d[d["HORIZON_DAYS"] == hh]
                if len(sub) > 0:
                    mae_min = sub["MAE"].min()
                    rmse_min = sub["RMSE"].min()
                    d.loc[(d["HORIZON_DAYS"] == hh) & (d["MAE"] == mae_min), "_best_mae"] = 1
                    d.loc[(d["HORIZON_DAYS"] == hh) & (d["RMSE"] == rmse_min), "_best_rmse"] = 1

        d["MAE"] = d["MAE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        d["RMSE"] = d["RMSE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        style = [
            {"if": {"filter_query": "{_best_mae} = 1", "column_id": "MAE"}, "backgroundColor": "#E8F5EE", "fontWeight": "900"},
            {"if": {"filter_query": "{_best_rmse} = 1", "column_id": "RMSE"}, "backgroundColor": "#E8F5EE", "fontWeight": "900"},
        ]

        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["HORIZON_DAYS", "MODEL", "MAE", "RMSE"]],
            data=d.to_dict("records"),
            hidden_columns=["_best_mae", "_best_rmse"],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "10px", "fontFamily": "system-ui", "fontSize": "13px"},
            style_header={"fontWeight": "900", "backgroundColor": "#f3f5f7"},
            style_data_conditional=style,
            page_size=20,
        )
        return html.Div([html.H5(title, className="mt-2"), table])

    tables = html.Div([
        metrics_table_component(indiv, highlight_best=True, title="Individual models (best values highlighted per horizon)"),
        html.Hr(),
        metrics_table_component(ens, highlight_best=True, title="EnsembleMean (separate)"),
    ])

    return note, tables


# =========================
# DATA QUALITY TAB
# =========================
@app.callback(
    Output("quality-summary", "children"),
    Output("quality-neg", "children"),
    Output("quality-missing", "children"),
    Input("active-tab", "data"),
)
def update_quality_tab(tab):
    if tab != "quality":
        return no_update, no_update, no_update

    raw = load_wide_csv(DATA_PATH, clean_negatives=False)

    cols = TARGET_COLS + list(TOTAL_BEDS_COLS.values())
    na = raw[cols].isna().sum().sort_values(ascending=False).reset_index()
    na.columns = ["column", "missing_count"]

    neg = (raw[cols] < 0).sum().reset_index()
    neg.columns = ["column", "negative_count"]

    summary = html.Div([
        html.Div(f"Rows (days): {len(raw)}"),
        html.Div(f"Date range: {raw.index.min().date()} → {raw.index.max().date()}"),
        html.Div(f"Staleness: {staleness_days} days"),
        html.Div("Note: dashboard computations treat negatives as missing and interpolate (safe).", className="muted tiny mt-1"),
    ])

    neg_tbl = dash_table.DataTable(
        columns=[{"name": c.upper(), "id": c} for c in neg.columns],
        data=neg.to_dict("records"),
        style_cell={"padding": "10px", "fontFamily": "system-ui", "fontSize": "13px"},
        style_header={"fontWeight": "900", "backgroundColor": "#f3f5f7"},
        page_size=12
    )

    na_tbl = dash_table.DataTable(
        columns=[{"name": c.upper(), "id": c} for c in na.columns],
        data=na.to_dict("records"),
        style_cell={"padding": "10px", "fontFamily": "system-ui", "fontSize": "13px"},
        style_header={"fontWeight": "900", "backgroundColor": "#f3f5f7"},
        page_size=12
    )

    return summary, neg_tbl, na_tbl


# =========================
# EXPORT BUTTONS (CSV)
# =========================
@app.callback(
    Output("dl-forecast", "data"),
    Input("btn-dl-forecast", "n_clicks"),
    State("horizon", "value"),
    State("scenario-on", "value"),
    State("extra-adult", "value"),
    State("extra-ped", "value"),
    prevent_initial_call=True,
)
def download_forecast(n, h, scenario_on_val, extra_a, extra_p):
    h = int(h)
    scenario_on = bool(scenario_on_val and 1 in scenario_on_val)
    f_long = read_forecast_long(h)

    if scenario_on:
        shift_adult = float(extra_a or 0)
        shift_ped = float(extra_p or 0)
        for series_key, shift in [("available_adult_icu_beds", shift_adult), ("available_ped_icu_beds", shift_ped)]:
            mask = (f_long["series"] == series_key) & (f_long["model"].astype(str).str.lower() == "ensemblemean")
            f_long.loc[mask, "p10"] = f_long.loc[mask, "p10"] + shift
            f_long.loc[mask, "p50"] = f_long.loc[mask, "p50"] + shift
            f_long.loc[mask, "p90"] = f_long.loc[mask, "p90"] + shift

    out = f_long.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    filename = f"forecast_{h}d_current.csv"
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@app.callback(
    Output("dl-risk", "data"),
    Input("btn-dl-risk", "n_clicks"),
    State("horizon", "value"),
    State("scenario-on", "value"),
    State("extra-adult", "value"),
    State("extra-ped", "value"),
    State("safety-buffer", "value"),
    prevent_initial_call=True,
)
def download_risk(n, h, scenario_on_val, extra_a, extra_p, safety):
    h = int(h)
    scenario_on = bool(scenario_on_val and 1 in scenario_on_val)
    safety = float(safety or 0)

    f_long = read_forecast_long(h)
    thr_adult = float(BASE_THRESHOLDS["adult"]) + (safety if scenario_on else 0)
    thr_ped = float(BASE_THRESHOLDS["ped"]) + (safety if scenario_on else 0)

    if scenario_on:
        shift_adult = float(extra_a or 0)
        shift_ped = float(extra_p or 0)
        for series_key, shift in [("available_adult_icu_beds", shift_adult), ("available_ped_icu_beds", shift_ped)]:
            mask = (f_long["series"] == series_key) & (f_long["model"].astype(str).str.lower() == "ensemblemean")
            f_long.loc[mask, "p10"] = f_long.loc[mask, "p10"] + shift
            f_long.loc[mask, "p50"] = f_long.loc[mask, "p50"] + shift
            f_long.loc[mask, "p90"] = f_long.loc[mask, "p90"] + shift

    a_tbl = make_risk_table_from_forecast(f_long, "available_adult_icu_beds", "EnsembleMean", thr_adult)
    p_tbl = make_risk_table_from_forecast(f_long, "available_ped_icu_beds", "EnsembleMean", thr_ped)
    a_tbl["Unit"] = "Adult"
    p_tbl["Unit"] = "Pediatric"
    out = pd.concat([a_tbl, p_tbl], ignore_index=True)
    filename = f"risk_{h}d_current.csv"
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@app.callback(
    Output("dl-metrics", "data"),
    Input("btn-dl-metrics", "n_clicks"),
    prevent_initial_call=True,
)
def download_metrics(n):
    m = read_outputs_df("backtest_metrics")
    return dcc.send_data_frame(m.to_csv, "backtest_metrics.csv", index=False)


if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050")

    Timer(1.0, open_browser).start()
    app.run(debug=False, port=8050)

