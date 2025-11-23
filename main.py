"""Simple regression skeleton and ML evaluator for fantasy win record.

This script compares PF, PA, and (PF-PA) using simple least-squares
regression (existing behavior) and provides an optional ML evaluation
mode (flag `--ml`) that runs several scikit-learn regressors using the
two-feature input `[PF, PA]`.

Run examples (use `uv` as your environment runner):

    uv run ./main.py --csv ./data/mscott/2025.csv
    uv run ./main.py --all ./data/ --ml

Dependencies: numpy, matplotlib, scikit-learn
"""

from typing import Sequence, Tuple
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# scikit-learn imports (required for --ml mode)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from scipy.stats import randint, uniform
except Exception:
    randint = None
    uniform = None


def _check_lengths(*arrays: Sequence) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All input arrays must have the same length, got lengths={lengths}"
        )


def fit_linear_regression(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Fit linear regression via least squares (adds intercept).

    Returns (coefficients, intercept, y_pred, r_squared).
    """
    # Add intercept column
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    # Solve least squares
    coef_all, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = float(coef_all[0])
    coefs = np.asarray(coef_all[1:], dtype=float)
    y_pred = X_design @ coef_all
    # R^2 calculation
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, intercept, y_pred, r2


def evaluate_models(
    pf: Sequence[float],
    pa: Sequence[float],
    wins: Sequence[float],
    source_name: str = "data",
) -> None:
    _check_lengths(pf, pa, wins)
    pf_arr = np.asarray(pf, dtype=float)
    pa_arr = np.asarray(pa, dtype=float)
    wins_arr = np.asarray(wins, dtype=float)

    # Feature sets
    X_pf = pf_arr.reshape(-1, 1)
    X_pa = pa_arr.reshape(-1, 1)
    # Use `diff` to represent the point differential (PF - PA)
    diff = pf_arr - pa_arr
    X_diff = diff.reshape(-1, 1)
    # You could also try multi-feature regression with both PF and PA:
    X_both = np.column_stack([pf_arr, pa_arr])

    models = [
        ("PF", X_pf),
        ("PA", X_pa),
        ("(PF-PA)", X_diff),
        ("[PF, PA]", X_both),
    ]

    outdir = Path.cwd() / "plots"
    outdir.mkdir(exist_ok=True)

    print("Model comparison (higher R^2 indicates better fit):")
    for name, X in models:
        coefs, intercept, ypred, r2 = fit_linear_regression(X, wins_arr)
        coef_str = ", ".join(f"{c:.4f}" for c in np.atleast_1d(coefs))
        print(f"- {name}: R^2={r2:.4f}, intercept={intercept:.4f}, coefs=[{coef_str}]")
        try:
            plot_model(
                name, X, wins_arr, ypred, coefs, intercept, r2, outdir, source_name
            )
        except Exception as exc:
            print(f"Warning: failed to create plot for {name}: {exc}")


def plot_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    ypred: np.ndarray,
    coefs: np.ndarray,
    intercept: float,
    r2: float,
    outdir: Path,
    source_name: str = "data",
    ml_mode: bool = False,
) -> Path:
    """Create and save a plot for the given model. Returns the saved file path."""
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    src = str(source_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    fname = outdir / f"model_{safe}_{src}.png"

    plt.close("all")
    if ml_mode or (X.ndim > 1 and X.shape[1] > 1):
        # Predicted vs actual + residuals
        resid = y - np.asarray(ypred).ravel()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(ypred, y, alpha=0.8)
        axes[0].plot(
            [y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--"
        )
        axes[0].set_xlabel("Predicted WinRecord")
        axes[0].set_ylabel("Actual WinRecord")
        axes[0].set_title(f"Predicted vs Actual — R²={r2:.3f}")

        axes[1].hist(resid, bins=20, color="gray", edgecolor="black")
        axes[1].set_title("Residuals")
        axes[1].set_xlabel("Residual (actual - predicted)")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
    else:
        # Single-feature scatter + regression line
        x = np.asarray(X).ravel()
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, label="data", alpha=0.8)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = intercept + coefs.ravel()[0] * xs
        plt.plot(xs, ys, color="red", label="fit")
        plt.xlabel(name)
        plt.ylabel("WinRecord")
        plt.title(f"{name} vs WinRecord — R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname, dpi=150)

    print(f"Saved plot: {fname}")
    return fname


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def evaluate_models_ml(
    pf: Sequence[float],
    pa: Sequence[float],
    wins: Sequence[float],
    source_name: str = "data",
    outdir: Path | None = None,
    cv_splits: int = 5,
    top_k: int = 3,
) -> dict:
    """Evaluate several sklearn regressors using `[PF, PA]` as the input.

    - All models use the two-feature input `[PF, PA]`.
    - Prints a concise summary of each model's metrics.
    - Saves diagnostic plots only for the top `top_k` models by R².

    Returns:
        A dict mapping model_name -> metrics dict (r2, rmse, mae).
    """
    pf_arr = np.asarray(pf, dtype=float)
    pa_arr = np.asarray(pa, dtype=float)
    y = np.asarray(wins, dtype=float)

    if not (pf_arr.size and pa_arr.size and y.size):
        raise ValueError("Empty input arrays provided")
    if not (pf_arr.shape[0] == pa_arr.shape[0] == y.shape[0]):
        raise ValueError("Input arrays must have the same length")

    # Use both PF and PA together for every model
    X = np.column_stack([pf_arr, pa_arr])

    models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.1, max_iter=5000)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42)),
        (
            "GradientBoosting",
            GradientBoostingRegressor(n_estimators=200, random_state=42),
        ),
        ("SVR", Pipeline([("scale", StandardScaler()), ("svr", SVR(C=1.0))])),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
    ]

    n_samples = y.shape[0]
    cv = min(max(2, cv_splits), n_samples)
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    if outdir is None:
        outdir = Path.cwd() / "plots"
    outdir.mkdir(exist_ok=True)

    results: dict = {}
    preds: dict = {}

    # Evaluate each model (all use X)
    for model_name, model in models:
        key = f"{model_name}"
        try:
            y_pred = cross_val_predict(model, X, y, cv=cv_splitter, n_jobs=1)
        except Exception:
            try:
                model.fit(X, y)
                y_pred = model.predict(X)
            except Exception as exc2:
                print(f"Skipping {key} because it failed: {exc2}")
                continue

        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        results[key] = {"r2": r2, "rmse": rmse, "mae": mae}
        preds[key] = y_pred

    # Print summary table (sorted by R² descending)
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]["r2"], reverse=True)
    print("\nModel evaluation summary (sorted by R²):")
    for name, metrics in sorted_models:
        print(
            f"- {name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}"
        )

    # Save plots only for the top_k models
    top_k = max(1, int(top_k))
    to_save = [name for name, _ in sorted_models[:top_k]]
    for name in to_save:
        y_pred = preds.get(name)
        if y_pred is None:
            continue
        try:
            # plot_model uses ml_mode=True to draw predicted vs actual
            plot_model(
                name,
                X,
                y,
                np.asarray(y_pred),
                np.array([0.0]),
                0.0,
                results[name]["r2"],
                outdir,
                source_name,
                ml_mode=True,
            )
        except Exception as exc:
            print(f"Warning: failed to save plot for {name}: {exc}")

    return results


def evaluate_models_ml_with_tuning(
    pf: Sequence[float],
    pa: Sequence[float],
    wins: Sequence[float],
    source_name: str = "data",
    outdir: Path | None = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_iter: int = 60,
    top_k: int = 3,
) -> dict:
    """Evaluate sklearn regressors with nested CV + RandomizedSearchCV tuning.

    Uses `[PF, PA]` as input for all models. Returns summary dict mapping
    model_name -> metrics (mean/std for r2, rmse, mae).
    """
    pf_arr = np.asarray(pf, dtype=float)
    pa_arr = np.asarray(pa, dtype=float)
    y = np.asarray(wins, dtype=float)

    if not (pf_arr.size and pa_arr.size and y.size):
        raise ValueError("Empty input arrays provided")
    if not (pf_arr.shape[0] == pa_arr.shape[0] == y.shape[0]):
        raise ValueError("Input arrays must have the same length")

    X = np.column_stack([pf_arr, pa_arr])

    n_samples = y.shape[0]
    outer_cv_splits = min(max(2, outer_folds), n_samples)
    inner_cv_splits = min(max(2, inner_folds), max(2, n_samples // 2))

    outer_cv = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=1)

    if outdir is None:
        outdir = Path.cwd() / "plots"
    outdir.mkdir(exist_ok=True)

    # Scoring dict used for cross_validate
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }

    # Helper to cap n_iter reasonably
    n_iter = max(5, int(n_iter))

    # Build estimators and parameter distributions
    models_and_params = []

    # LinearRegression (no hyperparams besides fit_intercept)
    models_and_params.append(
        (
            "LinearRegression",
            Pipeline([("scale", StandardScaler()), ("est", LinearRegression())]),
            {"est__fit_intercept": [True, False]},
        )
    )

    # ElasticNet (needs scaling)
    models_and_params.append(
        (
            "ElasticNet",
            Pipeline([("scale", StandardScaler()), ("est", ElasticNet(max_iter=5000))]),
            {
                "est__alpha": (
                    uniform(1e-4, 1.0)
                    if uniform is not None
                    else [1e-4, 0.01, 0.1, 1.0]
                ),
                "est__l1_ratio": (
                    uniform(0.0, 1.0) if uniform is not None else [0.1, 0.5, 0.9]
                ),
            },
        )
    )

    # RandomForest
    models_and_params.append(
        (
            "RandomForest",
            RandomForestRegressor(n_estimators=200, random_state=42),
            {
                "max_depth": randint(1, 12) if randint is not None else [3, 5, 7, None],
                "max_features": ["sqrt", "log2", None],
            },
        )
    )

    # Gradient Boosting
    models_and_params.append(
        (
            "GradientBoosting",
            GradientBoostingRegressor(n_estimators=200, random_state=42),
            {
                "learning_rate": (
                    uniform(0.01, 0.5) if uniform is not None else [0.01, 0.05, 0.1]
                ),
                "max_depth": randint(1, 6) if randint is not None else [1, 3, 5],
            },
        )
    )

    # SVR (needs scaling)
    models_and_params.append(
        (
            "SVR",
            Pipeline([("scale", StandardScaler()), ("est", SVR())]),
            {
                "est__C": (
                    uniform(0.1, 20.0) if uniform is not None else [0.1, 1.0, 10.0]
                ),
                "est__gamma": ["scale", "auto"],
            },
        )
    )

    # KNN (needs scaling)
    max_knn = max(1, min(20, n_samples - 1))
    models_and_params.append(
        (
            "KNN",
            Pipeline([("scale", StandardScaler()), ("est", KNeighborsRegressor())]),
            {
                "est__n_neighbors": (
                    randint(1, max_knn)
                    if randint is not None
                    else list(range(1, max_knn + 1))
                )
            },
        )
    )

    summary: dict = {}

    for name, estimator, param_dist in models_and_params:
        print(f"Tuning and evaluating: {name}")
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=min(n_iter, 80),
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )

        # outer CV evaluation of the whole search (nested CV)
        try:
            cv_res = cross_validate(
                search,
                X,
                y,
                cv=outer_cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )
        except Exception as exc:
            print(f"Skipping {name} because nested CV failed: {exc}")
            continue

        test_mse = -cv_res["test_neg_mse"]
        rmse = np.sqrt(test_mse)

        summary[name] = {
            "r2_mean": float(np.mean(cv_res["test_r2"])),
            "r2_std": float(np.std(cv_res["test_r2"])),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_std": float(np.std(rmse)),
            "mae_mean": float(np.mean(-cv_res["test_neg_mae"])),
            "mae_std": float(np.std(-cv_res["test_neg_mae"])),
        }

        print(
            f"- {name}: R2={summary[name]['r2_mean']:.3f}±{summary[name]['r2_std']:.3f}, RMSE={summary[name]['rmse_mean']:.3f}±{summary[name]['rmse_std']:.3f}"
        )

        # Save predicted vs actual using cross_val_predict which will run the search inside each fold
        try:
            y_pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=-1)
            plot_model(
                name,
                X,
                y,
                np.asarray(y_pred),
                np.array([0.0]),
                0.0,
                summary[name]["r2_mean"],
                outdir,
                source_name,
                ml_mode=True,
            )
        except Exception as exc:
            print(f"Warning: failed to save plot for {name}: {exc}")

    # Save a small JSON-like summary to disk for convenience
    try:
        import json

        with (outdir / f"model_summary_{str(source_name).replace(' ', '_')}.json").open(
            "w"
        ) as fh:
            json.dump(summary, fh, indent=2)
    except Exception:
        pass

    return summary


def load_from_csv(path: str) -> tuple[list[float], list[float], list[float]]:
    """Load CSV with columns PF, PA, WinRecord.

    Supports files with headers (containing strings like 'PF' or 'Points For')
    or headerless CSVs in the exact order PF,PA,WinRecord.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with p.open(newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("CSV is empty")

    # Detect header: if first row contains any non-numeric token
    first = rows[0]
    has_header = any(not _is_number(cell) for cell in first)

    data_rows = rows[1:] if has_header else rows

    pf_list: list[float] = []
    pa_list: list[float] = []
    win_list: list[float] = []

    if has_header:
        # Map header names to indices
        header = [c.strip().lower() for c in first]

        # find best matches
        def _find(name_candidates: list[str]) -> int:
            for cand in name_candidates:
                if cand in header:
                    return header.index(cand)
            return -1

        i_pf = _find(["pf", "points for", "points_for"])
        i_pa = _find(["pa", "points against", "points_against"])
        i_win = _find(
            [
                "winrecord",
                "wins",
                "win_record",
                "win pct",
                "winpercentage",
                "win_percentage",
            ]
        )

        if i_pf == -1 or i_pa == -1 or i_win == -1:
            # Fall back to expecting first three columns
            i_pf, i_pa, i_win = 0, 1, 2

        for row in data_rows:
            if len(row) <= max(i_pf, i_pa, i_win):
                continue
            try:
                pf_list.append(float(row[i_pf]))
                pa_list.append(float(row[i_pa]))
                win_list.append(float(row[i_win]))
            except Exception:
                # skip malformed rows
                continue
    else:
        # headerless: assume order PF,PA,WinRecord
        for row in data_rows:
            if len(row) < 3:
                continue
            try:
                pf_list.append(float(row[0]))
                pa_list.append(float(row[1]))
                win_list.append(float(row[2]))
            except Exception:
                continue

    if not (pf_list and pa_list and win_list):
        raise ValueError("No valid numeric rows parsed from CSV")
    if not (len(pf_list) == len(pa_list) == len(win_list)):
        raise ValueError("Parsed columns have mismatched lengths")

    return pf_list, pa_list, win_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PF, PA, and uv (PF-PA) as predictors of WinRecord"
    )
    parser.add_argument(
        "--all",
        help="Use all CSVs within a directory",
    )
    parser.add_argument(
        "--csv",
        help="Path to CSV file with columns PF,PA,WinRecord (header optional)",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Run ML model evaluations using scikit-learn",
    )
    args = parser.parse_args()

    source_base = "data"

    pf_data = []
    pa_data = []
    win_data = []
    if args.csv:
        pf_data, pa_data, win_data = load_from_csv(args.csv)
        source_base = Path(args.csv).stem
    elif args.all:
        # combine all the CSVs in the directory
        # then run the loading
        dir_path = Path(args.all)
        if not dir_path.is_dir():
            parser.error(f"--all argument must be a directory, got: {args.all}")
            sys.exit(1)
        combined_pf = []
        combined_pa = []
        combined_win = []
        for csv_file in dir_path.glob("**/*.csv"):
            try:
                pf, pa, win = load_from_csv(str(csv_file))
                combined_pf.extend(pf)
                combined_pa.extend(pa)
                combined_win.extend(win)
            except Exception as exc:
                print(f"Warning: failed to load {csv_file}: {exc}")
        pf_data = combined_pf
        pa_data = combined_pa
        win_data = combined_win
        source_base = f"all_from_{dir_path.name}"
    else:
        # what the heck happened, error out
        parser.error("Either --csv or --all must be specified")
        sys.exit(1)

    if not (pf_data and pa_data and win_data):
        print(
            "No data available. Fill the `PF`, `PA`, and `WinRecord` lists at the top of this file, or pass `--csv <file>`."
        )
        return

    try:
        if args.ml:
            # Use nested CV + RandomizedSearch tuning for ML mode
            evaluate_models_ml_with_tuning(pf_data, pa_data, win_data, source_base)
        else:
            evaluate_models(pf_data, pa_data, win_data, source_base)
    except Exception as exc:
        print(f"Error evaluating models: {exc}")


if __name__ == "__main__":
    main()
