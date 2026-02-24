"""Cybersecurity Attacks Dataset - Exploratory Data Analysis.

Orchestrates the full EDA pipeline by calling modular components from the src/ package.
Independent pipeline stages are executed in parallel using multiprocessing.

Usage:
    python script.py              # Full run with figures
    python script.py --no-figures # Skip all figure rendering (faster)
    python script.py --model-only # Show only model metrics figures
"""

import copy
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import src.utilities.config as config
from src.utilities.config import load_data
from src.utilities.diagrams import paracat_diag, sankey_diag
from src.eda_pipeline import run_eda
from src.modelling import modelling
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
    config.SHOW_FIGURES = show_figures

    # Cache paths for pre-model checkpoint
    cache_dir = Path("data")
    cache_df = cache_dir / "pre_model_df.parquet"
    cache_ct = cache_dir / "pre_model_crosstabs.pkl"

    if cache_df.exists() and cache_ct.exists():
        print("Loading cached pre-model data...")
        df = pd.read_parquet(cache_df)
        with open(cache_ct, "rb") as f:
            crosstabs_x_AttackType = pickle.load(f)
    else:
        # Data loading
        df_original, df = load_data()

        # EDA pipeline (sequential — mutates df)
        df, crosstabs_x_AttackType, df_n = run_eda(df)

        # --- Parallel stage 1: visualizations, diagrams, statistics -----------
        print("Running EDA visualizations & statistics in parallel...")
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(_worker_visualizations, df, show_figures): "visualizations",
                pool.submit(_worker_sankey, df, crosstabs_x_AttackType, show_figures): "sankey",
                pool.submit(_worker_paracat, df, show_figures): "paracat",
                pool.submit(_worker_statistics, df): "statistics",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    print(f"  [{name}] {result}")
                except Exception as exc:
                    print(f"  [{name}] failed: {exc}")

        # Save pre-model checkpoint
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
    )

    # --- Logistic Regression ---
    y_pred_logit, df = modelling(
        df,
        crosstabs_x_AttackType,
        model_type="logit",
        **modelling_kwargs,
    )

    # --- Random Forest ---
    # modelling() clears crosstabs internally, so pass a fresh copy
    y_pred_rf, df = modelling(
        df,
        copy.deepcopy(crosstabs_x_AttackType),
        model_type="randomforrest",
        **modelling_kwargs,
    )

    if model_only:
        config.SHOW_FIGURES = False

    # --- Parallel stage 2: post-model tasks --------------------------------
    print("Running post-model tasks in parallel...")
    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(_worker_interpretation, df, show_figures): "interpretation",
            pool.submit(_worker_export, df, Path(__file__).resolve().parent): "export",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                print(f"  [{name}] {result}")
            except Exception as exc:
                print(f"  [{name}] failed: {exc}")


if __name__ == "__main__":
    main()


