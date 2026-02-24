"""Cybersecurity Attacks Dataset - Exploratory Data Analysis.

Orchestrates the full EDA pipeline by calling modular components from the src/ package.

Usage:
    python script.py              # Full run with figures
    python script.py --no-figures # Skip figure rendering (faster)
"""

import sys

import pandas as pd

import src.config as config
from src.config import load_data
from src.diagrams import paracat_diag, sankey_diag
from src.eda_pipeline import run_eda
from src.modelling import modelling
from src.statistical_analysis import run_chi_square_test, run_mcc_analysis
from src.time_series import plot_acf_pacf, plot_daily_attack_counts, plot_ts_by_variable
from src.visualization import (
    plot_geo_attack_types,
    plot_interpretation_histograms,
    plot_strip_anomaly_scores,
    plot_strip_packet_length,
)


def main():
    """Run the full cybersecurity EDA pipeline."""
    if "--no-figures" in sys.argv:
        config.SHOW_FIGURES = False

    # Data loading
    df_original, df = load_data()

    # EDA pipeline: column transforms, encodings, geolocation, device parsing, daily aggregates
    df, crosstabs_x_AttackType, df_n = run_eda(df)

    # Attack type segmentation
    df_attype = {}
    attypes = ["Malware", "Intrusion", "DDoS"]
    for attype in attypes:
        df_attype[attype] = df[df["Attack Type"] == attype]

    # Geographic visualization
    plot_geo_attack_types(df, df_attype, attypes)

    # Attack characteristics comparison
    plot_strip_packet_length(df_attype, attypes)
    plot_strip_anomaly_scores(df_attype, attypes)

    # Multi-variable Sankey diagram
    sankey_ct = sankey_diag(
        df,
        crosstabs_x_AttackType,
        cols_bully=[
            False,  # "Source IP country"
            False,  # "Destination IP country"
            False,  # "Source Port ephemeral"
            False,  # "Destination Port ephemeral"
            False,  # "Protocol"
            False,  # "Packet Type Control"
            True,   # "Traffic Type"
            True,   # "Malware Indicators"
            False,  # "Alert Trigger"
            False,  # "Attack Signature patA"
            False,  # "Action Taken"
            False,  # "Severity Level"
            False,  # "Browser family"
            False,  # "Browser major"
            False,  # "Browser minor"
            False,  # "OS family"
            False,  # "OS major"
            False,  # "OS minor"
            False,  # "OS patch"
            False,  # "Device family"
            False,  # "Device brand"
            False,  # "Device type"
            False,  # "Device bot"
            False,  # "Network Segment"
            False,  # "Firewall Logs"
            False,  # "IDS/IPS Alerts"
            False,  # "Log Source Firewall"
        ],
    )

    # Parallel categories diagram
    paracat_diag(
        df,
        cols_bully=[
            False,  # "Source IP country"
            False,  # "Destination IP country"
            False,  # "Source Port ephemeral"
            False,  # "Destination Port ephemeral"
            True,   # "Protocol"
            False,  # "Packet Type Control"
            False,  # "Traffic Type"
            False,  # "Malware Indicators"
            False,  # "Alert Trigger"
            False,  # "Attack Signature patA"
            False,  # "Action Taken"
            False,  # "Severity Level"
            False,  # "Network Segment"
            False,  # "Firewall Logs"
            False,  # "IDS/IPS Alerts"
            False,  # "Log Source Firewall"
            True,   # "Attack Type max n d"
            False,  # "Protocol max n d"
            False,  # "Traffic Type max n d"
            False,  # "Alert Trigger max n d"
            False,  # "Firewall Logs max n d"
            False,  # "Attack Signature patA max n d"
            False,  # "Action Taken max n d"
            False,  # "Browser family max n d"
        ],
        colorvar="Attack Type",
    )

    # Statistical analysis
    run_mcc_analysis(df)
    run_chi_square_test(df)

    # Modelling
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
    y_pred, df = modelling(
        df,
        crosstabs_x_AttackType,
        testp=0.1,
        features_mask=features_mask,
        threshold_floor=threshold_floor,
        contvar_nobs_b_class=15,
        dynamic_threshold_pars=[5, 100, nobs_floor, threshold_floor],
        model_type="logit",
        split_before_training=False,
    )

    # Interpretation
    attypes = df["Attack Type"].unique()
    plot_interpretation_histograms(df, attypes)

    # Export
    df.to_csv("data/df.csv", sep="|")

    # Time series analysis
    Attacks_pday = df.copy(deep=True)
    Attacks_pday["date_dd"] = Attacks_pday["date"].dt.floor("D")
    Attacks_pday = (
        Attacks_pday.groupby(["date_dd", "Attack Type"])
        .size()
        .unstack()
        .iloc[1:-1,]
    )
    plot_daily_attack_counts(Attacks_pday)
    plot_acf_pacf(Attacks_pday, nlags=100)
    plot_ts_by_variable(df, target_col="Network Segment", nlags=100)


if __name__ == "__main__":
    main()


