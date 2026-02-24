"""EDA pipeline: orchestrates all data transformation and exploration steps."""

import warnings

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots as subp

from src.config import show_fig
from src.data_preparation import (
    extract_ipv4_prefix,
    geolocation_data_parallel,
    ip_to_coords_parallel,
    parse_device_info_parallel,
    MAX_WORKERS,
)
from src.diagrams import sankey_diag_IPs
from src.feature_engineering import build_daily_aggregates, crosstab_col
from src.utils import catvar_mapping, piechart_col


def run_eda(df):
    """Run the full EDA pipeline: column transforms, encodings, geolocation, and device parsing.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset (mutated in place via catvar_mapping returns).

    Returns
    -------
    df : pd.DataFrame
        Transformed DataFrame.
    crosstabs_x_AttackType : dict
        Crosstab dictionary populated during EDA.
    df_n : dict
        Daily aggregate pivot tables.
    """
    crosstabs_x_AttackType = {}

    # Suppress fragmentation warning — we defragment with df.copy() at checkpoints
    warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

    # =========================================================================
    # Column renaming
    # =========================================================================
    df = df.rename(
        columns={
            "Timestamp": "date",
            "Alerts/Warnings": "Alert Trigger",
        }
    )
    df = df.drop(["User Information", "Payload Data"], axis=1, errors="ignore")

    # print("Dataset summary statistics", "-" * 60, df.describe())

    # Missing value analysis
    # print("Missing value analysis", "-" * 60)
    # df_s0 = df.shape[0]
    # for col in df.columns:
    #     NA_n = sum(df[col].isna())
    #     if NA_n > 0:
    #         print(f"number of NAs in {col} = {NA_n} / {df_s0} = {NA_n / df_s0}")

    # =========================================================================
    # Attack Type
    # =========================================================================
    col_name = "Attack Type"
    # print(
    #     "Target variable : Attack Type distribution",
    #     "-" * 60,
    #     df[col_name].value_counts(),
    # )
    piechart_col(df, col_name)

    # =========================================================================
    # Date features
    # =========================================================================
    col_name = "date"
    df[col_name] = pd.to_datetime(df[col_name])
    date_end = max(df[col_name])
    date_start = min(df[col_name])

    # print(
    #     "Temporal distribution",
    #     "-" * 60,
    #     f"dataset time range : from {date_start} to {date_end}",
    # )
    fig = px.histogram(
        df,
        x=col_name,
        title="Distribution of cyberattack types over time",
        labels={"date": "date", "count": "number of attacks"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Number of Attacks", showlegend=False
    )
    show_fig(fig)

    col_pos = df.columns.get_loc(col_name)
    col_insert = [
        "date dd", "date MW", "date MW1", "date MW2", "date MW3",
        "date WD", "date WD1", "date WD2", "date WD3", "date WD4",
        "date WD5", "date WD6", "date H", "date H1", "date H2",
        "date H3", "date M", "date M1", "date M2", "date M3",
        "date M4", "date M5", "date M6", "date M7", "date M8",
        "date M9", "date M10", "date M11",
    ]
    na_cols = {"date dd", "date MW", "date WD", "date H", "date M"}
    new_cols = pd.DataFrame(
        {c: pd.NA if c in na_cols else 0 for c in col_insert},
        index=df.index,
    )
    df = pd.concat(
        [df.iloc[:, :col_pos + 1], new_cols, df.iloc[:, col_pos + 1:]],
        axis=1,
    )

    df["date dd"] = df["date"].dt.floor("D")
    sched = [0, 7, 14, 22]
    for day_filter in range(0, len(sched) - 1):
        day_bully = (df["date"].dt.day > sched[day_filter]) & (
            df["date"].dt.day <= sched[day_filter + 1]
        )
        df.loc[day_bully, f"date MW{day_filter + 1}"] = 1
        df.loc[day_bully, "date MW"] = f"MW{day_filter + 1}"
    df["date MW"] = df["date MW"].fillna("MW4")
    for day_filter in range(0, 6):
        day_bully = df["date"].dt.weekday == day_filter
        df.loc[day_bully, f"date WD{day_filter + 1}"] = 1
        df.loc[day_bully, "date WD"] = f"WD{day_filter + 1}"
    df["date WD"] = df["date WD"].fillna("WD7")
    sched = [3, 9, 15, 21]
    for day_filter in range(0, len(sched) - 1):
        day_bully = (df["date"].dt.hour >= sched[day_filter]) & (
            df["date"].dt.hour < sched[day_filter + 1]
        )
        df.loc[day_bully, f"date H{day_filter + 1}"] = 1
        df.loc[day_bully, "date H"] = f"H{day_filter + 1}"
    df["date H"] = df["date H"].fillna("H4")
    for day_filter in range(1, 12):
        day_bully = df["date"].dt.month == day_filter
        df.loc[day_bully, f"date M{day_filter}"] = 1
        df.loc[day_bully, "date M"] = f"M{day_filter}"
    df["date M"] = df["date M"].fillna("M12")

    # Cross tables for date categories
    for col_name in ["date MW", "date WD", "date H", "date M"]:
        crosstab_col(df, [col_name], crosstabs_x_AttackType)

    # =========================================================================
    # IP address geolocation
    # =========================================================================
    for destsource in ["Source", "Destination"]:
        col_pos = df.columns.get_loc(f"{destsource} IP Address")
        col_insert = [
            f"{destsource} IP 3octet",
            f"{destsource} IP 2octet",
            f"{destsource} IP 1octet",
            f"{destsource} IP latitude",
            f"{destsource} IP longitude",
            f"{destsource} IP country",
            f"{destsource} IP city",
        ]
        new_cols = pd.DataFrame(
            {c: pd.NA for c in col_insert},
            index=df.index,
        )
        df = pd.concat(
            [df.iloc[:, :col_pos + 1], new_cols, df.iloc[:, col_pos + 1:]],
            axis=1,
        )
        # print(f"  Geolocating {destsource} IPs using {MAX_WORKERS} threads...")
        geo_result = ip_to_coords_parallel(df[f"{destsource} IP Address"])
        df[f"{destsource} IP latitude"] = geo_result["latitude"]
        df[f"{destsource} IP longitude"] = geo_result["longitude"]
        df[f"{destsource} IP country"] = geo_result["country"]
        df[f"{destsource} IP city"] = geo_result["city"]
        for i in range(1, 4):
            df[f"{destsource} IP {i}octet"] = extract_ipv4_prefix(df, destsource, i)

    # Global IP distribution map
    fig = subp(
        rows=1,
        cols=2,
        specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
        subplot_titles=("Source IP locations", "Destination IP locations"),
    )
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scattergeo(
            lat=df["Source IP latitude"],
            lon=df["Source IP longitude"],
            mode="markers",
            marker={"size": 5, "color": "blue", "opacity": 0.6},
            name="Source IPs",
            hovertemplate="<b>Attack Origin</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergeo(
            lat=df["Destination IP latitude"],
            lon=df["Destination IP longitude"],
            mode="markers",
            marker={"size": 5, "color": "blue", "opacity": 0.6},
            name="Destination IPs",
            hovertemplate="<b>Attack Target</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_geos(showcountries=True, showland=True)
    fig.update_layout(
        height=750,
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        title_text="Global Distribution of Cyber Attack Traffic",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    show_fig(fig)

    # =========================================================================
    # Proxy Information
    # =========================================================================
    col_name = "Proxy Information"
    col_pos = df.columns.get_loc(col_name)
    new_cols = pd.DataFrame(
        {
            "Proxy usage": 0,
            "Proxy latitude": pd.NA,
            "Proxy longitude": pd.NA,
            "Proxy country": pd.NA,
            "Proxy city": pd.NA,
        },
        index=df.index,
    )
    df = pd.concat(
        [df.iloc[:, :col_pos + 1], new_cols, df.iloc[:, col_pos + 1:]],
        axis=1,
    )
    df.loc[~df["Proxy Information"].isna(), "Proxy usage"] = 1
    # print(f"  Geolocating Proxy IPs using {MAX_WORKERS} threads...")
    geo_result = ip_to_coords_parallel(df["Proxy Information"])
    df["Proxy latitude"] = geo_result["latitude"]
    df["Proxy longitude"] = geo_result["longitude"]
    df["Proxy country"] = geo_result["country"]
    df["Proxy city"] = geo_result["city"]

    # Defragment after IP/Proxy column additions
    df = df.copy()

    # print("Traffic flow analysis", "-" * 60)
    sankey_diag_IPs(df, 10)

    # =========================================================================
    # Source Port & Destination Port
    # =========================================================================
    for col_name in ["Source Port", "Destination Port"]:
        # print(df[col_name].value_counts())
        fig = px.histogram(
            df,
            x=col_name,
            title=f"Distribution of {col_name} values",
            labels={"date": "date", "count": "number of attacks"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(
            xaxis_title=col_name,
            yaxis_title="Number of attacks",
            showlegend=False,
        )
        show_fig(fig)

    # =========================================================================
    # Protocol
    # =========================================================================
    col_name = "Protocol"
    # print(df[col_name].value_counts())
    piechart_col(df, col_name)
    df = catvar_mapping(df, col_name, ["UDP", "ICMP"])
    crosstab_col(df, [col_name], crosstabs_x_AttackType)

    # Packet Length
    fig = px.histogram(
        df,
        "Packet Length",
        title="Distribution of network pack sizes",
        labels={"Packet Length": "Packet Length ( bytes )", "count": "Frequency"},
        color_discrete_sequence=["#00CC96"],
    )
    fig.update_layout(
        xaxis_title="Packet length ( bytes )",
        yaxis_title="Number of packets",
        showlegend=False,
    )
    show_fig(fig)

    # Packet Type
    col_name = "Packet Type"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Control"], ["Control"])
    piechart_col(df, "Packet Type Control")

    # Traffic Type
    col_name = "Traffic Type"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["DNS", "HTTP"])
    piechart_col(df, col_name)

    # Malware Indicators
    col_name = "Malware Indicators"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["IoC Detected"], ["/"])
    piechart_col(df, col_name)

    # Anomaly Scores
    fig = px.histogram(
        df,
        x="Anomaly Scores",
        title="Distribution of Anomaly Scores",
        labels={"Anomaly Scores": "Anomaly Score", "count": "Frequency"},
        color_discrete_sequence=["#EF553B"],
    )
    fig.update_layout(
        xaxis_title="Anomaly Score ( higher = more suspicious )",
        yaxis_title="Number of Records",
        showlegend=False,
    )
    show_fig(fig)

    # Alert Trigger
    col_name = "Alert Trigger"
    df = catvar_mapping(df, col_name, ["Alert Triggered"], ["/"])
    piechart_col(df, col_name, names=["Alert triggered", "Alert not triggered"])

    # Attack Signature
    col_name = "Attack Signature"
    # print(df["Attack Signature"].value_counts())
    df = catvar_mapping(df, col_name, ["Known Pattern A"], ["patA"])
    piechart_col(df, "Attack Signature patA", ["Pattern A", "Pattern B"])

    # Action Taken
    col_name = "Action Taken"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Logged", "Blocked"])
    piechart_col(df, col_name)

    # Severity Level
    col_name = "Severity Level"
    # print(df[col_name].value_counts())
    df.loc[df[col_name] == "Low", col_name] = -1
    df.loc[df[col_name] == "Medium", col_name] = 0
    df.loc[df[col_name] == "High", col_name] = +1
    piechart_col(df, col_name, ["Low", "Medium", "High"])

    # =========================================================================
    # Device Information
    # =========================================================================
    col_name = "Device Information"
    # print(df[col_name].value_counts())
    col_pos = df.columns.get_loc(col_name)
    col_insert = [
        "Browser family", "Browser major", "Browser minor",
        "OS family", "OS major", "OS minor", "OS patch",
        "Device family", "Device brand", "Device type", "Device bot",
    ]
    new_cols = pd.DataFrame(
        {c: pd.NA for c in col_insert},
        index=df.index,
    )
    df = pd.concat(
        [df.iloc[:, :col_pos + 1], new_cols, df.iloc[:, col_pos + 1:]],
        axis=1,
    )
    # print(f"  Parsing User-Agent strings using {MAX_WORKERS} threads...")
    ua_result = parse_device_info_parallel(df[col_name])
    for ua_col in ua_result.columns:
        df[ua_col] = ua_result[ua_col]

    # Defragment after Device Information column additions
    df = df.copy()

    col_name = "Browser family"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Logged", "Blocked"])  # !!!!! TO BE REVIEWED !!!!!!
    piechart_col(df, col_name)

    # =========================================================================
    # Network Segment
    # =========================================================================
    col_name = "Network Segment"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Segment A", "Segment B"], ["segA", "segB"])
    piechart_col(df, col_name)

    # Geo-location Data
    col_name = "Geo-location Data"
    # print(df[col_name].value_counts())
    col_pos = df.columns.get_loc(col_name)
    new_cols = pd.DataFrame(
        {"Geo-location City": pd.NA, "Geo-location State": pd.NA},
        index=df.index,
    )
    df = pd.concat(
        [df.iloc[:, :col_pos + 1], new_cols, df.iloc[:, col_pos + 1:]],
        axis=1,
    )
    geo_loc = geolocation_data_parallel(df["Geo-location Data"])
    df["Geo-location City"] = geo_loc["city"]
    df["Geo-location State"] = geo_loc["state"]

    # Firewall Logs
    col_name = "Firewall Logs"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Log Data"], ["/"])

    # IDS/IPS Alerts
    col_name = "IDS/IPS Alerts"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Alert Data"], ["/"])

    # Log Source
    col_name = "Log Source"
    # print(df[col_name].value_counts())
    df = catvar_mapping(df, col_name, ["Firewall"], ["Firewall"])

    # =========================================================================
    # Feature Engineering: Aggregates by day
    # =========================================================================
    target_cols = [
        "Attack Type", "Protocol", "Source Port", "Traffic Type",
        "Alert Trigger", "Firewall Logs", "Attack Signature patA",
        "Action Taken", "Browser family",
    ]
    df_n = build_daily_aggregates(df, target_cols)

    # Defragment after all column insertions and catvar_mapping copies
    df = df.copy()

    return df, crosstabs_x_AttackType, df_n
