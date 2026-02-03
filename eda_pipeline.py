##################################################################################################################
###
### Modular version of EDA pipeline. Uses modules contributed from the group!
###
##################################################################################################################

# =============================================================================
# LIBRARY IMPORTS AND CONFIGURATION
# =============================================================================
# This cell imports all required libraries and configures the environment:
# - pandas: Data manipulation and analysis
# - prince: Multiple Correspondence Analysis (MCA) for categorical data
# - plotly: Interactive visualizations
# - geoip2/django: IP geolocation services
# - sklearn: Machine learning metrics (Matthews correlation)
# - user_agents: Browser/device detection from User-Agent strings

from dotenv import load_dotenv

import pandas as pd
import prince
import plotly.io
import os
import numpy as np

import geoip2.database


import plotly.io
import plotly.express as px
import plotly.graph_objects as go

import random

import src.utilities.pipeline_utility_functions as pipeline_utils
import src.utilities.django_settings as django_settings
import src.utilities.data_preparation as data_prep

# from src.utilities.data_visualization import *


from plotly.subplots import make_subplots as subp

from scipy.stats import chi2_contingency


# Configure Plotly to render charts in the default browser
plotly.io.renderers.default = "browser"

# Load environment variables from .env file
DOTENV_PATH = os.path.join(".env")
UA_CACHE_FILE = "data/ua_cache.json"

# Initialize empty dictionary to store cross-tabulation results
crosstables = {}

load_dotenv(DOTENV_PATH)

# =============================================================================
# DATA PREPARATION
# =============================================================================
# Rename columns for better readability and consistency
# - "Timestamp" -> "date" for temporal analysis
# - Port columns renamed to indicate ephemeral port analysis
# - Alerts/Warnings -> Alert Trigger for binary encoding
df = pd.read_csv(os.path.join(os.getenv("DATA_PATH"), "cybersecurity_attacks.csv"))

df = data_prep.rename_columns(df)

# Display summary statistics for numerical columns
print("=" * 60)
print("DATASET SUMMARY STATISTICS")
print("=" * 60)
print(df.describe())


# =============================================================================
# EDA PIPELINE
# =============================================================================

# -----------------------------------------------------------------------------
# TARGET VARIABLE ANALYSIS: Attack Type
# This is the primary classification target with 3 classes:
# - Malware: Malicious software attacks
# - Intrusion: Unauthorized access attempts
# - DDoS: Distributed Denial of Service attacks
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TARGET VARIABLE: Attack Type Distribution")
print("=" * 60)
col_name = "Attack Type"
print(df[col_name].value_counts())


pipeline_utils.piechart_col(df, col_name)

# -----------------------------------------------------------------------------
# TEMPORAL ANALYSIS: Attack Distribution Over Time
# Analyze when attacks occurred to identify patterns or trends
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEMPORAL DISTRIBUTION")
print("=" * 60)
col_name = "date"
df[col_name] = pd.to_datetime(df[col_name])
date_end = max(df[col_name])
date_start = min(df[col_name])
print(f"Dataset time range: {date_start} to {date_end}")
print(f"Duration: {(date_end - date_start).days} days")

fig = px.histogram(
    df,
    x=col_name,
    title="Distribution of Cyber Attacks Over Time",
    labels={"date": "Date", "count": "Number of Attacks"},
    color_discrete_sequence=["#636EFA"],
)
fig.update_layout(xaxis_title="Date", yaxis_title="Number of Attacks", showlegend=False)
fig.show()

# =============================================================================

# =============================================================================
# IP ADDRESS GEOLOCATION EXTRACTION
# =============================================================================
# Convert IP addresses to geographic coordinates using MaxMind GeoIP2
# This enables geographic visualization and analysis of attack patterns

print("=" * 60)
print("IP ADDRESS GEOLOCATION PROCESSING")
print("=" * 60)

df = pipeline_utils.ip_geolocation_processing(df)

print("\n\n")
print("Geolocation extraction completed.")
print("\n\n")
print("=" * 60)

# -----------------------------------------------------------------------------
# PROXY INFORMATION COLUMNS
# Prepare columns for proxy IP geolocation data
# Proxies may be used to mask the true origin of attacks
# -----------------------------------------------------------------------------
print("\n\n")
print("=" * 60)

df = pipeline_utils.proxy_info_columns(df)

# Generate IP traffic flow Sankey diagram with top 10 countries

print("\n" + "=" * 60)
print("IP TRAFFIC FLOW ANALYSIS")
print("\n\n")
print("=" * 60)

aggregate_IPs = pipeline_utils.sankey_diag_IPs(df, ntop=10)

# =============================================================================
# USER-AGENT PARSING AND DEVICE CLASSIFICATION
# =============================================================================
# The "Device Information" column contains User-Agent strings that reveal:
# - Browser: Chrome, Firefox, Safari, Edge, IE, etc.
# - Operating System: Windows, macOS, Linux, iOS, Android
# - Device Type: Desktop PC, Mobile phone, Tablet
# - Bot Detection: Automated crawlers vs human users

print("\n" + "=" * 60)
print("DEVICE INFORMATION ANALYSIS")
print("=" * 60)

col_name = "Device Information"

print(f"Unique User-Agent strings: {df[col_name].nunique()}")
print("\nTop 10 User-Agent strings:")
print(df[col_name].value_counts().head(10))

# Parse User-Agent strings (skip if already processed)
if "Browser family" not in df.columns:
    print("\nParsing User-Agent strings...")
    col = df.columns.get_loc(col_name)

    # Create columns for parsed device information
    device_info_cols = [
        "Browser family",
        "Browser major",
        "Browser minor",
        "OS family",
        "OS major",
        "OS minor",
        "OS patch",
        "Device family",
        "Device brand",
        "Device type",
        "Device bot",
    ]
    for i, new_col in enumerate(device_info_cols, start=1):
        if new_col not in df.columns:
            df.insert(col + i, new_col, value=pd.NA)

    # Claude: Optimized User-Agent parsing - parse once per unique UA string and cache results with mods by Eugenio

    # Load existing cache
    ua_cache = pipeline_utils.load_cache(UA_CACHE_FILE)
    print(f"Loaded {len(ua_cache)} cached User-Agent entries")

    # Get unique UA strings to parse
    unique_uas = df[col_name].dropna().unique()
    new_uas = [ua for ua in unique_uas if ua not in ua_cache]
    print(
        f"Parsing {len(new_uas)} new User-Agent strings ({len(unique_uas) - len(new_uas)} cached)"
    )

    # Parse new UA strings and add to cache
    for i, ua_string in enumerate(new_uas):
        if i % 1000 == 0 and i > 0:
            print(f"  Parsed {i}/{len(new_uas)} User-Agent strings...")
        result = pipeline_utils.parse_ua_string(ua_string)
        ua_cache[ua_string] = result

    # Save updated cache
    pipeline_utils.save_cache(ua_cache, UA_CACHE_FILE)
    print(f"Cache updated with {len(ua_cache)} total entries")

    # Map fields to column names and indices
    field_mapping = [
        ("Browser family", 0),
        ("Browser major", 1),
        ("Browser minor", 2),
        ("OS family", 3),
        ("OS major", 4),
        ("OS minor", 5),
        ("OS patch", 6),
        ("Device family", 7),
        ("Device brand", 8),
        ("Device type", 9),
        ("Device bot", 10),
    ]
    # Apply cached results to dataframe (much faster than parsing each row)
    for col_name_field, idx in field_mapping:
        df[col_name_field] = df[col_name].apply(
            lambda x, i=idx: pipeline_utils.get_cached_ua(ua_cache, x, i)
        )

    print("User-Agent parsing complete.")
else:
    print("Device information already extracted - skipping parsing.")

# Note: Keeping NA values for missing information is preferred over
# placeholder strings like "not specified" for proper statistical analysis

# -----------------------------------------------------------------------------
# Device Type Classification
# Classify devices as Mobile, Tablet, or PC based on User-Agent
# -----------------------------------------------------------------------------

# Classify device types and detect bots
if "Device type" not in df.columns or df["Device type"].isna().all():
    print("Classifying device types...")
    df["Device type"] = df[col_name].apply(pipeline_utils.device_type(ua_string))

# Bot detection - identifies automated crawlers and scripts
""" if "Device bot" not in df.columns or df["Device bot"].isna().all():
    print("Detecting bot traffic...")
    df["Device bot"] = df[col_name].apply(lambda x: parse(x).is_bot) """

# =============================================================================
# NETWORK SEGMENTATION ANALYSIS
# =============================================================================
# Network segments represent logical divisions of the network:
# - Segment A, B, C: Different network zones (e.g., DMZ, internal, guest)
# - Segmentation helps contain breaches and control traffic flow
#
# Binary encoding: segA = 1 if Segment A, segB = 1 if Segment B, else [0,0] for C

print("\n" + "=" * 60)
print("NETWORK SEGMENTATION ANALYSIS")
print("=" * 60)

col_name = "Network Segment"
print(df[col_name].value_counts())

if "Network Segment segA" not in df.columns:
    df = pipeline_utils.catvar_mapping(
        df, col_name, ["Segment A", "Segment B"], ["segA", "segB"]
    )
pipeline_utils.piechart_col(df, col_name)

# =============================================================================
# GEO-LOCATION DATA PARSING
# =============================================================================
# Parse "City, State" format into separate columns for geographic analysis

print("\n" + "=" * 60)
print("GEO-LOCATION DATA PARSING")
print("=" * 60)

col_name = "Geo-location Data"
print(f"Unique locations: {df[col_name].nunique()}")
print("\nTop 10 locations:")
print(df[col_name].value_counts().head(10))

if "Geo-location City" not in df.columns:
    print("Parsing geo-location data...")
    col = df.columns.get_loc(col_name)
    df.insert(col + 1, "Geo-location City", value=pd.NA)
    df.insert(col + 2, "Geo-location State", value=pd.NA)
    df[["Geo-location City", "Geo-location State"]] = df["Geo-location Data"].apply(
        lambda x: pipeline_utils.geolocation_data(x)
    )

# =============================================================================
# SECURITY SYSTEM LOGS ANALYSIS
# =============================================================================
# Firewall Logs: Records of traffic allowed/denied by firewall rules
# IDS/IPS Alerts: Intrusion Detection/Prevention System alerts
# Log Source: Where the log originated (Firewall vs Server)

print("\n" + "=" * 60)
print("SECURITY SYSTEM LOGS ANALYSIS")
print("=" * 60)

# -----------------------------------------------------------------------------
# Firewall Logs
# Binary: 1 = Log data present, 0 = No log data
# -----------------------------------------------------------------------------
print("\n--- Firewall Logs ---")
col_name = "Firewall Logs"
print(df[col_name].value_counts())

if df[col_name].dtype == "object" or "Log Data" in df[col_name].astype(str).values:
    df = pipeline_utils.catvar_mapping(df, col_name, ["Log Data"], ["/"])

# -----------------------------------------------------------------------------
# IDS/IPS Alerts
# Intrusion Detection/Prevention System alerts
# Binary: 1 = Alert data present, 0 = No alert
# -----------------------------------------------------------------------------
print("\n--- IDS/IPS Alerts ---")
col_name = "IDS/IPS Alerts"
print(df[col_name].value_counts())
# Binary: 1 = Alert data present, 0 = No alert
if df[col_name].dtype == "object" or "Alert Data" in df[col_name].astype(str).values:
    df = pipeline_utils.catvar_mapping(df, col_name, ["Alert Data"], ["/"])

# -----------------------------------------------------------------------------
# Log Source
# Origin of the log entry
# Binary: Firewall = 1, Server = 0
# -----------------------------------------------------------------------------
print("\n--- Log Source ---")
col_name = "Log Source"
if col_name in df.columns:
    print(df[col_name].value_counts())
    df = pipeline_utils.catvar_mapping(df, col_name, ["Firewall"], ["Firewall"])
elif "Log Source Firewall" in df.columns:
    print("Log Source already transformed to Log Source Firewall")

# Generate cross-tabulation between log source and firewall logs
crosstables.update(
    pipeline_utils.crosstab_col(
        df, "Log Source Firewall", "Firewall Logs", "logsource", "firewallogs"
    )
)

# -----------------------------------------------------------------------------
# Attack Signature
# Pattern matching results from signature-based detection systems
# Known Pattern A vs Known Pattern B (different attack signatures)
# Binary encoding: Pattern A = 1, Pattern B = 0
# -----------------------------------------------------------------------------
print("\n--- Attack Signature ---")
col_name = "Attack Signature"
if col_name in df.columns:
    print(df["Attack Signature"].value_counts())
    df = pipeline_utils.catvar_mapping(df, col_name, ["Known Pattern A"], ["patA"])
    pipeline_utils.piechart_col(
        df, "Attack Signature patA", names=["Known Pattern B", "Known Pattern A"]
    )
elif "Attack Signature patA" in df.columns:
    print("Attack Signature already transformed to Attack Signature patA")
    pipeline_utils.piechart_col(
        df, "Attack Signature patA", names=["Known Pattern B", "Known Pattern A"]
    )

# =============================================================================
# ATTACK TYPE SEGMENTATION
# =============================================================================
# Split dataset by attack type for comparative visualizations
# This enables side-by-side analysis of attack characteristics

print("\n" + "=" * 60)
print("ATTACK TYPE SEGMENTATION")
print("=" * 60)

df_attype = {}
attypes = ["Malware", "Intrusion", "DDoS"]
for attype in attypes:
    df_attype[attype] = df[df["Attack Type"] == attype]
    print(f"{attype}: {len(df_attype[attype]):,} records")

# =============================================================================
# GEO-VISUALIZATION: ATTACK TYPES BY ANOMALY SCORE
# =============================================================================
# 6-panel geographic visualization:
# - Rows: Malware, Intrusion, DDoS attacks
# - Columns: Source IPs (left) and Destination IPs (right)
# - Color: Anomaly score (Viridis scale - dark=low, bright=high)

print("\n" + "=" * 60)
print("GEOGRAPHIC ATTACK VISUALIZATION")
print("=" * 60)

fig = subp(
    rows=3,
    cols=2,
    specs=[
        [{"type": "scattergeo"}, {"type": "scattergeo"}],
        [{"type": "scattergeo"}, {"type": "scattergeo"}],
        [{"type": "scattergeo"}, {"type": "scattergeo"}],
    ],
    subplot_titles=(
        "Malware - Source IPs",
        "Malware - Destination IPs",
        "Intrusion - Source IPs",
        "Intrusion - Destination IPs",
        "DDoS - Source IPs",
        "DDoS - Destination IPs",
    ),
)

for i, (attype, symb) in enumerate(zip(attypes, ["diamond", "diamond", "diamond"])):
    # Source IP locations (left column)
    fig.add_trace(
        go.Scattergeo(
            lat=df_attype[attype]["Source IP latitude"],
            lon=df_attype[attype]["Source IP longitude"],
            mode="markers",
            name=f"{attype} Source",
            marker={
                "size": 4,
                "symbol": symb,
                "color": df_attype[attype]["Anomaly Scores"],
                "colorscale": "Viridis",
                "cmin": df_attype[attype]["Anomaly Scores"].min(),
                "cmax": df_attype[attype]["Anomaly Scores"].max(),
                "colorbar": {
                    "title": "Anomaly<br>Score",
                    "len": 0.25,
                    "y": 0.85 - i * 0.33,
                },
            },
            hovertemplate=f"<b>{attype} Attack Origin</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<br>Anomaly Score: %{{marker.color:.2f}}<extra></extra>",
        ),
        row=i + 1,
        col=1,
    )
    # Destination IP locations (right column)
    fig.add_trace(
        go.Scattergeo(
            lat=df_attype[attype]["Destination IP latitude"],
            lon=df_attype[attype]["Destination IP longitude"],
            mode="markers",
            name=f"{attype} Dest",
            marker={
                "size": 4,
                "symbol": symb,
                "color": df_attype[attype]["Anomaly Scores"],
                "colorscale": "Viridis",
                "cmin": df_attype[attype]["Anomaly Scores"].min(),
                "cmax": df_attype[attype]["Anomaly Scores"].max(),
                "colorbar": {
                    "title": "Anomaly<br>Score",
                    "len": 0.25,
                    "y": 0.85 - i * 0.33,
                    "x": 1.02,
                },
                "showscale": False,  # Hide duplicate colorbar
            },
            hovertemplate=f"<b>{attype} Attack Target</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<br>Anomaly Score: %{{marker.color:.2f}}<extra></extra>",
        ),
        row=i + 1,
        col=2,
    )

fig.update_geos(
    showcountries=True, showland=True, landcolor="lightgray", countrycolor="white"
)
fig.update_layout(
    height=1200,
    margin={"r": 50, "t": 80, "l": 0, "b": 0},
    title_text="Geographic Distribution of Attacks by Type and Anomaly Score",
    title_x=0.5,
    title_font_size=18,
    showlegend=False,
)
fig.show()

# =============================================================================
# PACKET LENGTH DISTRIBUTION BY ATTACK TYPE
# =============================================================================
# Strip plot showing packet length distribution for each attack type
# Helps identify if different attacks have characteristic packet sizes

print("\n" + "=" * 60)
print("PACKET LENGTH BY ATTACK TYPE")
print("=" * 60)

fig = go.Figure()

# Color palette for attack types
colors = {"Malware": "#636EFA", "Intrusion": "#EF553B", "DDoS": "#00CC96"}

for i, (attype, symb) in enumerate(zip(attypes, ["diamond", "diamond", "diamond"])):
    fig.add_trace(
        go.Scatter(
            x=df_attype[attype]["Packet Length"],
            y=[i] * df_attype[attype].shape[0],
            mode="markers",
            name=f"{attype} ({len(df_attype[attype]):,} records)",
            marker={
                "size": 4,
                "symbol": symb,
                "opacity": 0.15,
                "color": colors[attype],
            },
            hovertemplate=f"<b>{attype}</b><br>Packet Length: %{{x}} bytes<extra></extra>",
        ),
    )

fig.update_layout(
    title="Packet Length Distribution by Attack Type",
    title_font_size=16,
    xaxis_title="Packet Length (bytes)",
    yaxis=dict(
        tickmode="array", tickvals=[0, 1, 2], ticktext=attypes, title="Attack Type"
    ),
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
fig.show()

# =============================================================================
# ANOMALY SCORE DISTRIBUTION BY ATTACK TYPE
# =============================================================================
# Strip plot showing anomaly score distribution for each attack type
# Higher anomaly scores indicate more suspicious/unusual traffic patterns

print("\n" + "=" * 60)
print("ANOMALY SCORES BY ATTACK TYPE")
print("=" * 60)

fig = go.Figure()

for i, (attype, symb) in enumerate(zip(attypes, ["diamond", "diamond", "diamond"])):
    fig.add_trace(
        go.Scatter(
            x=df_attype[attype]["Anomaly Scores"],
            y=[i] * df_attype[attype].shape[0],
            mode="markers",
            name=f"{attype} ({len(df_attype[attype]):,} records)",
            marker={
                "size": 4,
                "symbol": symb,
                "opacity": 0.15,
                "color": colors[attype],
            },
            hovertemplate=f"<b>{attype}</b><br>Anomaly Score: %{{x:.2f}}<extra></extra>",
        ),
    )

fig.update_layout(
    title="Anomaly Score Distribution by Attack Type",
    title_font_size=16,
    xaxis_title="Anomaly Score (higher = more suspicious)",
    yaxis=dict(
        tickmode="array", tickvals=[0, 1, 2], ticktext=attypes, title="Attack Type"
    ),
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
fig.show()

# =============================================================================
# MULTI-VARIABLE SANKEY DIAGRAM GENERATOR
# =============================================================================
# Creates a Sankey diagram showing how categorical variable values
# flow toward different attack types. Useful for understanding
# which feature combinations characterize each attack type.


# Generate Sankey diagram with selected features
# Selected: Protocol, Traffic Type, Malware Indicators, IDS/IPS Alerts
print("\n" + "=" * 60)
print("MULTI-VARIABLE SANKEY ANALYSIS")
print("=" * 60)
print("Features included: Protocol, Traffic Type, Malware Indicators, IDS/IPS Alerts")

crosstabs = pipeline_utils.sankey_diag(
    df,
    cols_bully=[
        False,  # "Source IP country"
        False,  # "Destination IP country"
        False,  # "Source Port ephemeral"
        False,  # "Destination Port ephemeral"
        True,  # "Protocol" - INCLUDED
        False,  # "Packet Type Control"
        True,  # "Traffic Type" - INCLUDED
        True,  # "Malware Indicators" - INCLUDED
        False,  # "Alert Trigger"
        False,  # "Attack Signature patA"
        False,  # "Action Taken"
        False,  # "Severity Level"
        False,  # "Network Segment"
        False,  # "Firewall Logs"
        True,  # "IDS/IPS Alerts" - INCLUDED
        False,  # "Log Source Firewall"
    ],
)

# =============================================================================
# PARALLEL CATEGORIES DIAGRAM
# =============================================================================
#


# Generate parallel categories diagram
# Selected features: Protocol, Traffic Type, Attack Signature, Severity Level, Network Segment
# Color: Attack Signature Pattern A (to see how signature types flow through other features)
print("\n" + "=" * 60)
print("PARALLEL CATEGORIES DIAGRAM")
print("=" * 60)
print(
    "Features: Protocol, Traffic Type, Attack Signature, Severity Level, Network Segment"
)
print("Color coding: Attack Signature Pattern (A vs B)")

pipeline_utils.paracat_diag(
    df,
    [
        False,  # "Source IP country"
        False,  # "Destination IP country"
        False,  # "Source Port ephemeral"
        False,  # "Destination Port ephemeral"
        True,  # "Protocol" - INCLUDED
        False,  # "Packet Type Control"
        True,  # "Traffic Type" - INCLUDED
        False,  # "Malware Indicators"
        False,  # "Alert Trigger"
        True,  # "Attack Signature patA" - INCLUDED
        False,  # "Action Taken"
        True,  # "Severity Level" - INCLUDED
        True,  # "Network Segment" - INCLUDED
        False,  # "Firewall Logs"
        False,  # "IDS/IPS Alerts"
        False,  # "Log Source Firewall"
    ],
    colorvar="Attack Signature patA",
)


# =============================================================================
# MATTHEWS CORRELATION COEFFICIENT (MCC) ANALYSIS
# =============================================================================
# The Matthews Correlation Coefficient (MCC) measures the quality of
# binary classifications. Values range from -1 to +1:
# - +1: Perfect prediction
# -  0: No better than random
# - -1: Total disagreement
#
# For multi-class, this generalizes to the Phi coefficient.

print("\n" + "=" * 60)
print("MATTHEWS CORRELATION COEFFICIENTS")
print("=" * 60)
print("Measuring correlation between each feature and Attack Type")
print("(Values close to 0 indicate no linear correlation)\n")


catvars = np.array(
    [
        "Source IP country",
        "Destination IP country",
        "Source Port ephemeral",
        "Destination Port ephemeral",
        "Protocol",
        "Packet Type",
        "Traffic Type",
        "Malware Indicators",
        "Alert Trigger",
        "Attack Signature patA",
        "Action Taken",
        "Severity Level",
        "Network Segment",
        "Firewall Logs",
        "IDS/IPS Alerts",
        "Log Source Firewall",
    ]
)
for c in catvars:
    pipeline_utils.catvar_corr(df, c)

# =============================================================================
# CHI-SQUARE TEST OF INDEPENDENCE
# =============================================================================
# Tests whether there is a significant association between Protocol and Attack Type
# - H0: Variables are independent (no association)
# - H1: Variables are not independent (association exists)
#
# If p-value < 0.05, we reject H0 and conclude significant association

print("\n" + "=" * 60)
print("CHI-SQUARE TEST: Protocol vs Attack Type")
print("=" * 60)


# Prepare categorical variables for analysis
df_catvar = df[
    [
        "Attack Type",
        "Source Port ephemeral",
        "Destination Port ephemeral",
        "Protocol",
        "Packet Type",
        "Traffic Type",
        "Malware Indicators",
        "Alert Trigger",
        "Attack Signature patA",
        "Action Taken",
        "Severity Level",
        "Network Segment",
        "Firewall Logs",
        "IDS/IPS Alerts",
        "Log Source Firewall",
    ]
]

# Encode categorical columns as integers for contingency table
df_catvar_encoded = df_catvar.copy()
for col in df_catvar_encoded.columns:
    if df_catvar_encoded[col].dtype == "object":
        df_catvar_encoded[col] = pd.Categorical(df_catvar_encoded[col]).codes

# Perform chi-square test
res = chi2_contingency(
    pd.crosstab(df_catvar_encoded["Attack Type"], df_catvar_encoded["Protocol"])
)
print(f"Chi-square statistic: {res.statistic:.4f}")
print(f"P-value: {res.pvalue:.6f}")
print(f"Degrees of freedom: {res.dof}")
print("\nExpected frequencies:")
print(res.expected_freq)

if res.pvalue < 0.05:
    print("\n=> Significant association detected (p < 0.05)")
else:
    print("\n=> No significant association (p >= 0.05)")

# =============================================================================
# MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
# =============================================================================
# MCA is a data analysis technique for nominal categorical data,
# used to detect and represent underlying structures in a data set.
# It's the categorical equivalent of Principal Component Analysis (PCA).
#
# Key outputs:
# - Eigenvalues: Amount of variance explained by each component
# - Component coordinates: Position of category levels in reduced space
"""
print("\n" + "=" * 60)
print("MULTIPLE CORRESPONDENCE ANALYSIS (MCA)")
print("=" * 60)

# Initialize MCA with 3 components
mca = prince.MCA(
    n_components=3,
    n_iter=3,
    copy=True,
    check_input=True,
    engine="sklearn",
    random_state=42,
)

# Select categorical variables for MCA
df_catvar = df[
    [
        "Attack Type",
        "Source Port ephemeral",
        "Destination Port ephemeral",
        "Protocol",
        "Packet Type",
        "Traffic Type",
        "Malware Indicators",
        "Alert Trigger",
        "Attack Signature patA",
        "Action Taken",
        "Severity Level",
        "Network Segment",
        "Firewall Logs",
        "IDS/IPS Alerts",
        "Log Source Firewall",
    ]
]

# Fit MCA model
mca = mca.fit(df_catvar)

# Alternative MCA approaches for comparison
one_hot = pd.get_dummies(df_catvar)

# MCA with one_hot=False requires positive values (add 1 to avoid zeros)
one_hot_positive = one_hot + 1

mca_no_one_hot = prince.MCA(one_hot=False)
mca_no_one_hot = mca_no_one_hot.fit(one_hot_positive)

# Different correction methods for eigenvalue adjustment
mca_without_correction = prince.MCA(correction=None)
mca_with_benzecri_correction = prince.MCA(correction="benzecri")
mca_with_greenacre_correction = prince.MCA(correction="greenacre")

# Display MCA results
print("\nMCA Eigenvalues Summary:")
print("(Shows variance explained by each component)")
print(mca.eigenvalues_summary)"""

# =============================================================================
# TIME SERIES ANALYSIS: DAILY ATTACK PATTERNS
# =============================================================================
# Aggregate attacks by day and analyze temporal patterns
# This helps identify:
# - Seasonal patterns (weekly, monthly cycles)
# - Trend changes over time
# - Anomalous periods with unusual attack volumes

print("\n" + "=" * 60)
print("TIME SERIES ANALYSIS: DAILY ATTACK COUNTS")
print("=" * 60)

# Aggregate attacks by day
Attacks_pday = df.copy(deep=True)
Attacks_pday["date_dd"] = Attacks_pday["date"].dt.floor("d")
Attacks_pday = (
    Attacks_pday.groupby(["date_dd", "Attack Type"]).size().unstack().iloc[1:-1,]
)

print(f"Time series length: {len(Attacks_pday)} days")
print(f"Date range: {Attacks_pday.index.min()} to {Attacks_pday.index.max()}")

# -----------------------------------------------------------------------------
# Daily Attack Count Time Series Plot
# Shows raw daily counts for each attack type
# -----------------------------------------------------------------------------
fig = subp(
    rows=3,
    cols=1,
    subplot_titles=(
        "Malware Attacks per Day",
        "Intrusion Attempts per Day",
        "DDoS Attacks per Day",
    ),
    vertical_spacing=0.08,
)

# Malware time series
fig.add_trace(
    go.Scatter(
        x=Attacks_pday.index,
        y=Attacks_pday["Malware"],
        mode="lines",
        name="Malware",
        line=dict(color="#636EFA"),
        hovertemplate="<b>Malware</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Intrusion time series
fig.add_trace(
    go.Scatter(
        x=Attacks_pday.index,
        y=Attacks_pday["Intrusion"],
        mode="lines",
        name="Intrusion",
        line=dict(color="#EF553B"),
        hovertemplate="<b>Intrusion</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ),
    row=2,
    col=1,
)

# DDoS time series
fig.add_trace(
    go.Scatter(
        x=Attacks_pday.index,
        y=Attacks_pday["DDoS"],
        mode="lines",
        name="DDoS",
        line=dict(color="#00CC96"),
        hovertemplate="<b>DDoS</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=800,
    title_text="Daily Attack Counts by Type",
    title_font_size=18,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Count")
fig.show()

# -----------------------------------------------------------------------------
# ACF and PACF Analysis
# Autocorrelation helps identify time series patterns:
# - ACF: Correlation between observations at different lags
# - PACF: Direct correlation at each lag, removing intermediate effects
# These help determine ARIMA model parameters (p, d, q)
# -----------------------------------------------------------------------------
print("\n--- Autocorrelation Analysis ---")

fig = subp(
    rows=3,
    cols=2,
    subplot_titles=(
        "Malware - Autocorrelation (ACF)",
        "Malware - Partial Autocorrelation (PACF)",
        "Intrusion - Autocorrelation (ACF)",
        "Intrusion - Partial Autocorrelation (PACF)",
        "DDoS - Autocorrelation (ACF)",
        "DDoS - Partial Autocorrelation (PACF)",
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.08,
)

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

Attacktype_TSanalysis = {}
nlags = 100

# Color scheme for attack types
ts_colors = {"Malware": "#636EFA", "Intrusion": "#EF553B", "DDoS": "#00CC96"}

for i, attacktype in enumerate(["Malware", "Intrusion", "DDoS"]):
    # Calculate ACF and PACF
    Attacktype_TSanalysis[f"{attacktype}_ACF"] = acf(
        Attacks_pday[attacktype], nlags=nlags
    )
    Attacktype_TSanalysis[f"{attacktype}_PACF"] = pacf(
        Attacks_pday[attacktype], nlags=nlags
    )

    # Plot ACF
    fig.add_trace(
        go.Scatter(
            x=list(range(0, nlags + 1)),
            y=Attacktype_TSanalysis[f"{attacktype}_ACF"],
            mode="lines",
            name=f"{attacktype} ACF",
            line=dict(color=ts_colors[attacktype]),
            hovertemplate=f"<b>{attacktype} ACF</b><br>Lag: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>",
        ),
        row=i + 1,
        col=1,
    )

    # Plot PACF
    fig.add_trace(
        go.Scatter(
            x=list(range(0, nlags + 1)),
            y=Attacktype_TSanalysis[f"{attacktype}_PACF"],
            mode="lines",
            name=f"{attacktype} PACF",
            line=dict(color=ts_colors[attacktype]),
            hovertemplate=f"<b>{attacktype} PACF</b><br>Lag: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>",
        ),
        row=i + 1,
        col=2,
    )

fig.update_layout(
    height=900,
    title_text="Autocorrelation Analysis for Attack Time Series",
    title_font_size=18,
    showlegend=False,
)
fig.update_xaxes(title_text="Lag (days)")
fig.update_yaxes(title_text="Correlation")
fig.show()


# %% Save Processed Dataset
# =============================================================================
# SAVE PROCESSED DATASET
# =============================================================================
# Export the processed DataFrame with all engineered features

print("\n" + "=" * 60)
print("SAVING PROCESSED DATASET")
print("=" * 60)

df.to_parquet("data/kaloinas_eda.parquet", index=False)
print("Dataset saved to data/kaloinas_eda.parquet")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# END OF EDA PIPELINE
