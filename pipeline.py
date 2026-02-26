"""Cybersecurity Attacks Dataset - Exploratory Data Analysis.

Orchestrates the full EDA pipeline by calling modular components from the src/ package.
Independent pipeline stages are executed in parallel using multiprocessing.

Usage:
    python script.py                      # Full run with figures
    python script.py --no-figures         # Skip all figure rendering (faster)
    python script.py --model-only         # Show only model metrics figures
    python script.py --sequential         # Run all stages sequentially (lower RAM usage)
    python script.py --monitor            # Show live memory usage TUI
    python script.py --mem-limit 16       # Cap memory usage (8, 16, 32, 64 GB)
"""

import copy
import gc
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import src.utilities.config as config
from src.utilities.config import load_data
from src.utilities.diagrams import paracat_diag, sankey_diag
from src.eda_pipeline import run_eda
from src.modelling import modelling, ports_modelling
from src.utilities.statistical_analysis import run_chi_square_test, run_mcc_analysis
from src.utilities.visualization import (
    plot_geo_attack_types,
    plot_interpretation_histograms,
    plot_strip_anomaly_scores,
    plot_strip_packet_length,
)


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _worker_visualizations(df, show_figures):
    """Geographic and strip-plot visualizations."""
    config.SHOW_FIGURES = show_figures
    df_attype = {}
    attypes = ["Malware", "Intrusion", "DDoS"]
    for attype in attypes:
        df_attype[attype] = df[df["Attack Type"] == attype]
    plot_geo_attack_types(df, df_attype, attypes)
    plot_strip_packet_length(df_attype, attypes)
    plot_strip_anomaly_scores(df_attype, attypes)
    return "visualizations done"


def _worker_sankey(df, crosstabs_x_AttackType, show_figures):
    """Sankey diagram."""
    config.SHOW_FIGURES = show_figures
    sankey_diag(
        df,
        crosstabs_x_AttackType,
        cols_bully=[
            False, False, False, False, False, False,
            True, True,
            False, False, False, False, False, False, False,
            False, False, False, False, False, False, False,
            False, False, False, False, False,
        ],
    )
    return "sankey done"


def _worker_paracat(df, show_figures):
    """Parallel categories diagram."""
    config.SHOW_FIGURES = show_figures
    paracat_diag(
        df,
        cols_bully=[
            False, False, False, False,
            True,
            False, False, False, False, False, False, False,
            False, False, False, False,
            True,
            False, False, False, False, False, False, False,
        ],
        colorvar="Attack Type",
    )
    return "paracat done"


def _worker_statistics(df):
    """MCC and chi-square analysis."""
    run_mcc_analysis(df)
    run_chi_square_test(df)
    return "statistics done"


def _worker_interpretation(df, show_figures):
    """Interpretation histograms."""
    config.SHOW_FIGURES = show_figures
    attypes = df["Attack Type"].unique()
    plot_interpretation_histograms(df, attypes)
    return "interpretation done"


def _worker_export(df, base_dir):
    """Export DataFrame to parquet."""
    out_path = base_dir / "data" / "df.parquet"
    # Coerce object columns to str for pyarrow compatibility
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    df.to_parquet(out_path)
    return "export done"




def main():
    """Run the full cybersecurity EDA pipeline.

    Independent stages are dispatched to separate processes via ProcessPoolExecutor.
    """
    show_figures = "--no-figures" not in sys.argv and "--model-only" not in sys.argv
    model_only = "--model-only" in sys.argv
    sequential = "--sequential" in sys.argv
    monitor = "--monitor" in sys.argv
    config.SHOW_FIGURES = show_figures

    # Parse --mem-limit N
    from src.utilities.mem_monitor import (
        VALID_MEM_LIMITS, MemoryMonitor, NullMonitor, mem_profile,
    )
    limit_gb = None
    if "--mem-limit" in sys.argv:
        idx = sys.argv.index("--mem-limit")
        if idx + 1 >= len(sys.argv):
            print(f"Error: --mem-limit requires a value: {VALID_MEM_LIMITS}")
            sys.exit(1)
        try:
            limit_gb = int(sys.argv[idx + 1])
        except ValueError:
            print(f"Error: --mem-limit requires an integer: {VALID_MEM_LIMITS}")
            sys.exit(1)
        if limit_gb not in VALID_MEM_LIMITS:
            print(f"Error: --mem-limit must be one of {VALID_MEM_LIMITS}, got {limit_gb}")
            sys.exit(1)

    # Build memory profile and apply concurrency settings
    profile = mem_profile(limit_gb)
    from src.utilities import data_preparation

    if sequential or profile.sequential:
        data_preparation.SEQUENTIAL_MODE = True
    data_preparation.MAX_WORKERS = profile.max_workers

    if limit_gb:
        print(profile.describe())
    elif sequential:
        print("Running in sequential mode (reduced memory usage).")

    # Optional memory monitor (context manager is a no-op when disabled)
    mon = MemoryMonitor(limit_gb=limit_gb) if (monitor or limit_gb) else NullMonitor()

    with mon:
        _run_pipeline(show_figures, model_only, sequential or profile.sequential, profile, mon)
        _run_ports_pipeline(mon)

def _run_pipeline(show_figures, model_only, sequential, profile, mon):
    """Core pipeline logic, separated so the monitor can wrap it."""
    # Cache paths for pre-model checkpoint
    cache_dir = Path("data")
    cache_df = cache_dir / "pre_model_df.parquet"
    cache_ct = cache_dir / "pre_model_crosstabs.pkl"

    if cache_df.exists() and cache_ct.exists():
        mon.stage = "loading cache"
        print("Loading cached pre-model data...")
        df = pd.read_parquet(cache_df)
        with open(cache_ct, "rb") as f:
            crosstabs_x_AttackType = pickle.load(f)
    else:
        # Data loading
        mon.stage = "loading data"
        df_original, df = load_data()

        # EDA pipeline (sequential — mutates df)
        mon.stage = "EDA"
        df, crosstabs_x_AttackType = run_eda(df)

        # --- Stage 1: visualizations, diagrams, statistics --------------------
        workers = [
            ("visualizations", _worker_visualizations, (df, show_figures)),
            ("sankey", _worker_sankey, (df, crosstabs_x_AttackType, show_figures)),
            ("paracat", _worker_paracat, (df, show_figures)),
            ("statistics", _worker_statistics, (df,)),
        ]
        if sequential:
            print("Running EDA visualizations & statistics sequentially...")
            for name, func, args in workers:
                mon.stage = name
                try:
                    result = func(*args)
                    print(f"  [{name}] {result}")
                except Exception as exc:
                    print(f"  [{name}] failed: {exc}")
        else:
            mon.stage = "viz + stats"
            print(f"Running EDA visualizations & statistics ({profile.pipeline_workers} workers)...")
            with ThreadPoolExecutor(max_workers=profile.pipeline_workers) as pool:
                futures = {
                    pool.submit(func, *args): name
                    for name, func, args in workers
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        print(f"  [{name}] {result}")
                    except Exception as exc:
                        print(f"  [{name}] failed: {exc}")

        # Drop EDA-only "max n d" columns created by build_daily_aggregates()
        # for the Paracat diagram.  Not needed for modelling.
        mon.stage = "trim columns"
        agg_cols = [c for c in df.columns if " max n d" in c]
        if agg_cols:
            print(f"Dropping {len(agg_cols)} daily-aggregate columns to save memory...")
            df.drop(columns=agg_cols, inplace=True)
            gc.collect()

        # Save pre-model checkpoint
        mon.stage = "saving cache"
        print("Saving pre-model cache...")
        # Coerce mixed-type object columns so pyarrow can serialize them
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        df.to_parquet(cache_df)
        with open(cache_ct, "wb") as f:
            pickle.dump(crosstabs_x_AttackType, f)

    # Modelling (sequential — needs full df, returns modified df)
    features_mask = [
        True,   # "Source Port & Destination Port"
        False,  # "Source IP latitude/longitude combination"
        False,  # "Source IP Country, Destination IP country combination"
        True,   # "date dd"
        True,   # "Network & Traffic + device info"
        True,   # "Security Response + device info"
        True,   # "time"
        True,   # "Anomaly Scores"
        True,   # "Packet Length"
    ]
    threshold_floor = 37.5
    nobs_floor = 10

    if model_only:
        config.SHOW_FIGURES = True

    modelling_kwargs = dict(
        testp=0.1,
        features_mask=features_mask,
        threshold_floor=threshold_floor,
        contvar_nobs_b_class=15,
        dynamic_threshold_pars=[5, 100, nobs_floor, threshold_floor],
        split_before_training=False,
        n_jobs=profile.max_workers,
    )

    # --- Logistic Regression ---
    mon.stage = "logistic reg."
    y_pred_logit, df = modelling(
        df,
        crosstabs_x_AttackType,
        model_type="logit",
        **modelling_kwargs,
    )
    gc.collect()

    # --- Random Forest ---
    # modelling() clears crosstabs internally, so pass a fresh copy
    mon.stage = "random forest"
    y_pred_rf, df = modelling(
        df,
        copy.deepcopy(crosstabs_x_AttackType),
        model_type="randomforrest",
        **modelling_kwargs,
    )
    gc.collect()

    if model_only:
        config.SHOW_FIGURES = False

    # --- Stage 2: post-model tasks -------------------------------------------
    post_workers = [
        ("interpretation", _worker_interpretation, (df, show_figures)),
        ("export", _worker_export, (df, Path(__file__).resolve().parent)),
    ]
    if sequential:
        print("Running post-model tasks sequentially...")
        for name, func, args in post_workers:
            mon.stage = name
            try:
                result = func(*args)
                print(f"  [{name}] {result}")
            except Exception as exc:
                print(f"  [{name}] failed: {exc}")
    else:
        mon.stage = "post-model"
        print(f"Running post-model tasks ({profile.pipeline_workers} workers)...")
        with ThreadPoolExecutor(max_workers=profile.pipeline_workers) as pool:
            futures = {
                pool.submit(func, *args): name
                for name, func, args in post_workers
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    print(f"  [{name}] {result}")
                except Exception as exc:
                    print(f"  [{name}] failed: {exc}")

# TODO: Check parameters to add for multi threading
def _run_ports_pipeline(mon):
    # Data loading - we have to do this again since the cached data will be modified for the other model
    mon.stage = "loading data"
    df_original, _ = load_data()

    mon.stage = "extra trees"
    y_pred_et, df = ports_modelling(df_original)

    # mon.stage = "post-model"

if __name__ == "__main__":
    main()


