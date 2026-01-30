# %% [markdown]
# # Cybersecurity Attacks Dataset - Exploratory Data Analysis
#
# This notebook performs comprehensive EDA on a cybersecurity attacks dataset,
# including IP geolocation analysis, attack pattern visualization, and
# statistical analysis of network traffic features.

# %% Library Initialization
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

import pandas as pd
import prince
import plotly.io
import os
import numpy as np

# Optional GDAL configuration for geospatial operations
# Uncomment and adjust path if needed for your environment
"""
os.environ["GDAL_LIBRARY_PATH"] = (
    "C:/Users/KalooIna/anaconda3/envs/cybersecurity_attacks/Library/bin/gdal311.dll"
) """
import geoip2.database

# Configure Plotly to render charts in the default browser
plotly.io.renderers.default = "browser"

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp
from sklearn.metrics import matthews_corrcoef
import random
import django
from user_agents import parse
from user_agents import parse as ua_parse
from django.conf import settings
from django.contrib.gis.geoip2 import GeoIP2

# -----------------------------------------------------------------------------
# Django GeoIP2 Configuration
# Configure Django settings for IP geolocation using MaxMind GeoLite2 database
# The database must be downloaded from MaxMind and placed in ./geolite2_db/
# -----------------------------------------------------------------------------
if not settings.configured:
    settings.configure(
        GEOIP_PATH="./geolite2_db", INSTALLED_APPS=["django.contrib.gis"]
    )
django.setup()

# Initialize GeoIP2 lookup service
geoIP = GeoIP2()

# Reference URLs for GeoIP2 setup and documentation
maxmind_geoip2_db_url = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
geoip2_doc_url = "https://geoip2.readthedocs.io/en/latest/"
geoip2_django_doc_url = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"

# -----------------------------------------------------------------------------
# Load Dataset
# The dataset contains network traffic records with attack classifications
# Columns include: timestamps, IP addresses, ports, protocols, attack types, etc.
# -----------------------------------------------------------------------------
df = pd.read_csv("data/df.csv", sep="|", index_col=0)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def catvar_mapping(col_name, values, name=None):
    """
    Transform categorical variables to binary (0/1) indicator columns.

    This function performs one-hot style encoding for specified categorical values,
    creating new binary columns that indicate presence/absence of each category.

    Parameters:
    -----------
    col_name : str
        Name of the categorical column to transform
    values : list
        List of categorical values to create binary indicators for
    name : list, optional
        Custom names for the new binary columns. If None, uses the values.
        Use ["/"] to replace the original column in-place.

    Returns:
    --------
    DataFrame
        Copy of the dataframe with new binary columns added

    Example:
    --------
    # Create "Protocol UDP" and "Protocol ICMP" binary columns
    df = catvar_mapping("Protocol", ["UDP", "ICMP"])
    """
    df1 = df.copy(deep=True)
    if name is None:
        name = values
    elif (len(name) == 1) and (name != ["/"]):
        col_target = f"{col_name} {name[0]}"
        df1 = df1.rename(columns={col_name: col_target})
        col_name = col_target
        name = [col_target]
    col = df1.columns.get_loc(col_name) + 1
    for val, nm in zip(values, name):
        if nm == "/":
            col_target = col_name
        elif len(name) == 1:
            col_target = nm
        else:
            col_target = f"{col_name} {nm}"
            if col_target not in df1.columns:
                df1.insert(col, col_target, value=pd.NA)
        bully = df1[col_name] == val
        df1.loc[bully, col_target] = 1
        df1.loc[~bully, col_target] = 0
        col += 1
    return df1


def piechart_col(col, names=None):
    """
    Generate an interactive pie chart for a categorical column.

    Parameters:
    -----------
    col : str
        Column name to visualize
    names : list, optional
        Custom labels for pie chart segments. If None, uses actual values.
    """
    if names is None:
        fig = px.pie(
            values=df[col].value_counts(),
            names=df[col].value_counts().index,
            title=f"Distribution of {col}",
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        fig.show()
    else:
        fig = px.pie(
            values=df[col].value_counts(),
            names=names,
            title=f"Distribution of {col}",
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        fig.show()


def ip_to_coords(ip_address):
    """
    Convert an IP address to geographic coordinates and location info.

    Uses MaxMind GeoIP2 database to lookup geographic information for IP addresses.

    Parameters:
    -----------
    ip_address : str
        IPv4 or IPv6 address to geolocate

    Returns:
    --------
    pd.Series
        Series containing [latitude, longitude, country_name, city]
        Returns NA values for any fields that cannot be resolved
    """
    print(ip_address)
    ret = pd.Series(dtype=object)
    try:
        res = geoIP.geos(ip_address).wkt
        lon, lat = res.replace("(", "").replace(")", "").split()[1:]
        ret = pd.concat([ret, pd.Series([lat, lon])], ignore_index=True)
    except:
        ret = pd.concat([ret, pd.Series([pd.NA, pd.NA])], ignore_index=True)
    try:
        res = geoIP.city(ip_address)
        ret = pd.concat(
            [ret, pd.Series([res["country_name"], res["city"]])], ignore_index=True
        )
    except:
        ret = pd.concat([ret, pd.Series([pd.NA, pd.NA])], ignore_index=True)
    return ret


# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# This section performs initial data exploration:
# - Column renaming for clarity
# - Missing value analysis
# - Target variable distribution (Attack Type)
# - Temporal distribution of attacks

# %% EDA - Data Preparation and Overview
# =============================================================================
# DATA PREPARATION
# =============================================================================
# Rename columns for better readability and consistency
# - "Timestamp" -> "date" for temporal analysis
# - Port columns renamed to indicate ephemeral port analysis
# - Alerts/Warnings -> Alert Trigger for binary encoding
df = df.rename(
    columns={
        "Timestamp": "date",
        "Source Port": "Source Port ephemeral",
        "Destination Port": "Destination Port ephemeral",
        "Alerts/Warnings": "Alert Trigger",
    }
)

# Display summary statistics for numerical columns
print("=" * 60)
print("DATASET SUMMARY STATISTICS")
print("=" * 60)
print(df.describe())

# -----------------------------------------------------------------------------
# Cross-tabulation utility function
# Used to analyze relationships between categorical variables
# -----------------------------------------------------------------------------
crosstabs = {}


def crosstab_col(col, target, name_col, name_target):
    """
    Create a normalized cross-tabulation between two categorical columns.

    Parameters:
    -----------
    col : str
        Column name for the independent variable
    target : str
        Column name for the dependent/target variable
    name_col, name_target : str
        Short names for labeling the crosstab result
    """
    name_tab = f"{name_col}_x_{name_target}"
    crosstabs[name_tab] = pd.crosstab(df[target], df[col], normalize=True) * 100


# -----------------------------------------------------------------------------
# Missing Value Analysis
# Identify columns with missing data and their percentages
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)
df_s0 = df.shape[0]
for col in df.columns:
    NA_n = sum(df[col].isna())
    if NA_n > 0:
        print(f"Missing values in {col}: {NA_n:,} / {df_s0:,} ({100*NA_n/df_s0:.2f}%)")

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
piechart_col(col_name)

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
    color_discrete_sequence=["#636EFA"]
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Attacks",
    showlegend=False
)
fig.show()


# %% [markdown]
# ## IP Address Geolocation Analysis
# This section:
# - Extracts geographic information from Source and Destination IP addresses
# - Visualizes global distribution of attack origins and targets
# - Analyzes proxy usage patterns in the network traffic

# %% IP Address Geolocation
# =============================================================================
# IP ADDRESS GEOLOCATION EXTRACTION
# =============================================================================
# Convert IP addresses to geographic coordinates using MaxMind GeoIP2
# This enables geographic visualization and analysis of attack patterns

print("=" * 60)
print("IP ADDRESS GEOLOCATION PROCESSING")
print("=" * 60)

i = 2
for destsource in ["Source", "Destination"]:
    geo_columns = [
        f"{destsource} IP latitude",
        f"{destsource} IP longitude",
        f"{destsource} IP country",
        f"{destsource} IP city",
    ]

    # Skip geolocation if already processed (columns exist)
    if all(col in df.columns for col in geo_columns):
        print(f"{destsource} IP geolocation already processed - skipping")
        continue

    print(f"Processing {destsource} IP addresses...")
    col = df.columns.get_loc(f"{destsource} IP Address") + 1

    # Insert new columns for geographic data
    for i, geo_col in enumerate(geo_columns):
        if geo_col not in df.columns:
            df.insert(col + i, geo_col, value=pd.NA)

    # Apply geolocation lookup to each IP address
    geo_data = df[f"{destsource} IP Address"].apply(ip_to_coords)
    geo_df = pd.DataFrame(geo_data.tolist(), index=df.index)

    df[geo_columns[0]] = geo_df[0]  # Latitude
    df[geo_columns[1]] = geo_df[1]  # Longitude
    df[geo_columns[2]] = geo_df[2]  # Country
    df[geo_columns[3]] = geo_df[3]  # City

# -----------------------------------------------------------------------------
# GLOBAL IP DISTRIBUTION MAP
# Orthographic projection showing attack source and destination locations
# Left globe: Source IPs (where attacks originate)
# Right globe: Destination IPs (attack targets)
# -----------------------------------------------------------------------------
fig = subp(
    rows=1,
    cols=2,
    specs=[
        [
            {"type": "scattergeo"},
            {"type": "scattergeo"},
        ]
    ],
    subplot_titles=(
        "Attack Origin Locations (Source IPs)",
        "Attack Target Locations (Destination IPs)"
    ),
)

# Source IP locations - Blue markers
fig.add_trace(
    go.Scattergeo(
        lat=df["Source IP latitude"],
        lon=df["Source IP longitude"],
        mode="markers",
        marker={"size": 5, "color": "blue", "opacity": 0.6},
        name="Source IPs",
        hovertemplate="<b>Attack Origin</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
    ),
    row=1,
    col=1,
)

# Destination IP locations - Blue markers
fig.add_trace(
    go.Scattergeo(
        lat=df["Destination IP latitude"],
        lon=df["Destination IP longitude"],
        mode="markers",
        marker={"size": 5, "color": "blue", "opacity": 0.6},
        name="Destination IPs",
        hovertemplate="<b>Attack Target</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
    ),
    row=1,
    col=2,
)

fig.update_geos(
    projection_type="orthographic",
    showcountries=True,
    showland=True,
    landcolor="lightgray",
    countrycolor="white",
    oceancolor="lightblue",
    showocean=True
)
fig.update_layout(
    height=750,
    margin={"r": 0, "t": 80, "l": 0, "b": 0},
    title_text="Global Distribution of Cyber Attack Traffic",
    title_x=0.5,
    title_font_size=18,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="center",
        x=0.5
    )
)
fig.show()

# -----------------------------------------------------------------------------
# PROXY INFORMATION COLUMNS
# Prepare columns for proxy IP geolocation data
# Proxies may be used to mask the true origin of attacks
# -----------------------------------------------------------------------------
col_name = "Proxy Information"
col = df.columns.get_loc(col_name)

for i, new_col in enumerate(
    ["Proxy latitude", "Proxy longitude", "Proxy country", "Proxy city"], start=1
):
    if new_col not in df.columns:
        df.insert(col + i, new_col, value=pd.NA)


def sankey_diag_IPs(ntop):
    """
    Create a Sankey diagram showing traffic flow between Source, Proxy, and Destination countries.

    This visualization shows how network traffic flows geographically:
    - Left nodes: Source IP countries (where traffic originates)
    - Middle nodes: Proxy countries (intermediate routing points)
    - Right nodes: Destination IP countries (final targets)

    Parameters:
    -----------
    ntop : int
        Number of top countries to display (others grouped as "other")

    Returns:
    --------
    DataFrame
        Aggregated IP flow counts grouped by source, proxy, and destination
    """
    IPs_col = {}
    labels = pd.Series(dtype="string")

    # Process each IP type: Source (SIP), Destination (DIP), Proxy (PIP)
    for IPid, dfcol in zip(
        ["SIP", "DIP", "PIP"],
        ["Source IP country", "Destination IP country", "Proxy country"],
    ):
        IPs_col[IPid] = df[dfcol].copy(deep=True)
        IPs_col[f"{IPid}labs"] = pd.Series(IPs_col[IPid].value_counts().index[:ntop])
        bully = IPs_col[IPid].isin(IPs_col[f"{IPid}labs"]) | IPs_col[IPid].isna()
        IPs_col[IPid].loc[~bully] = "other"
        IPs_col[f"{IPid}labs"] = f"{IPid} " + pd.concat(
            [IPs_col[f"{IPid}labs"], pd.Series(["other"])]
        )
        labels = pd.concat([labels, IPs_col[f"{IPid}labs"]])
    labels = list(labels.reset_index(drop=True))

    # Aggregate traffic flows between countries
    aggregIPs = pd.DataFrame(
        {
            "SIP": IPs_col["SIP"],
            "PIP": IPs_col["PIP"],
            "DIP": IPs_col["DIP"],
        }
    )
    aggregIPs = aggregIPs.groupby(by=["SIP", "PIP", "DIP"]).size().to_frame("count")

    # Build Sankey diagram links
    source = []
    target = []
    value = []
    nlvl = aggregIPs.index.nlevels
    for idx, row in aggregIPs.iterrows():
        row_labs = []
        if nlvl == 1:
            row_labs.append(f"{aggregIPs.index.name} {idx}")
        else:
            for i, val in enumerate(idx):
                row_labs.append(f"{aggregIPs.index.names[i]} {val}")
        for i in range(0, nlvl - 1):
            source.append(labels.index(row_labs[i]))
            target.append(labels.index(row_labs[i + 1]))
            value.append(row.item())

    # Create Sankey diagram with Inferno color scale
    n = len(labels)
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno, [i / (n - 1) for i in range(n)]
    )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="rgba(0, 0, 0, 0.1)", width=0.5),
                    label=labels,
                    color=colors,
                    hovertemplate="<b>%{label}</b><br>Total flows: %{value}<extra></extra>"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    hovertemplate="<b>Flow</b><br>From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Network Traffic Flow: Source → Proxy → Destination Countries",
        title_font_size=18,
        font_size=14,
        title_font_family="Avenir",
        title_font_color="black",
        annotations=[
            dict(x=0.01, y=1.05, text="<b>Source Countries</b>", showarrow=False, font_size=12),
            dict(x=0.5, y=1.05, text="<b>Proxy Countries</b>", showarrow=False, font_size=12),
            dict(x=0.99, y=1.05, text="<b>Destination Countries</b>", showarrow=False, font_size=12),
        ]
    )
    fig.show()

    return aggregIPs


# Generate IP traffic flow Sankey diagram with top 10 countries
print("\n" + "=" * 60)
print("IP TRAFFIC FLOW ANALYSIS")
print("=" * 60)
aggregIPs = sankey_diag_IPs(10)

# %% [markdown]
# ## Network Protocol and Port Analysis
# This section analyzes network layer characteristics:
# - Port classification (ephemeral vs registered)
# - Protocol distribution (TCP, UDP, ICMP)
# - Packet types and traffic categories

# %% Source and Destination Port Analysis
# =============================================================================
# PORT ANALYSIS: EPHEMERAL vs REGISTERED PORTS
# =============================================================================
# Port Classification:
# - Ephemeral ports (49152-65535): Dynamically assigned for outbound connections
# - Registered ports (1024-49151): Assigned to specific services
# - Well-known ports (0-1023): Reserved for common services (HTTP, SSH, etc.)
#
# Binary encoding: 1 = ephemeral port, 0 = registered/well-known port

print("\n" + "=" * 60)
print("PORT ANALYSIS")
print("=" * 60)

# Source Port Classification
col_name = "Source Port ephemeral"
print(f"\n--- {col_name} ---")
print("Converting to binary: 1 = ephemeral (>49151), 0 = registered/well-known")
bully = df[col_name] > 49151
df.loc[bully, col_name] = 1
df.loc[~bully, col_name] = 0
print(df[col_name].value_counts())
crosstab_col(col_name, "Attack Type", "sourceport", "attacktype")

# Destination Port Classification
col_name = "Destination Port ephemeral"
print(f"\n--- {col_name} ---")
print("Converting to binary: 1 = ephemeral (>49151), 0 = registered/well-known")
bully = df[col_name] > 49151
df.loc[bully, col_name] = 1
df.loc[~bully, col_name] = 0
print(df[col_name].value_counts())
crosstab_col(col_name, "Attack Type", "destport", "attacktype")

# =============================================================================
# PROTOCOL ANALYSIS
# =============================================================================
# Network protocols observed in the traffic:
# - TCP: Connection-oriented, reliable delivery (web, email, file transfer)
# - UDP: Connectionless, fast but unreliable (DNS, streaming, gaming)
# - ICMP: Control messages and diagnostics (ping, traceroute)
#
# Binary encoding creates indicator columns for each protocol

print("\n" + "=" * 60)
print("PROTOCOL ANALYSIS")
print("=" * 60)

col_name = "Protocol"
print(df[col_name].value_counts())
piechart_col(col_name)

# Create binary indicator columns for protocols
if "Protocol UDP" not in df.columns:
    df = catvar_mapping(col_name, ["UDP", "ICMP"])
crosstab_col(col_name, "Attack Type", col_name, "attacktype")

# =============================================================================
# PACKET LENGTH DISTRIBUTION
# =============================================================================
# Packet length can indicate:
# - Small packets: Control messages, ACKs, probes
# - Medium packets: Typical web traffic
# - Large packets: File transfers, streaming data
# - Unusual sizes: Potential attack indicators

print("\n" + "=" * 60)
print("PACKET LENGTH ANALYSIS")
print("=" * 60)

fig = px.histogram(
    df,
    x="Packet Length",
    title="Distribution of Network Packet Sizes",
    labels={"Packet Length": "Packet Length (bytes)", "count": "Frequency"},
    color_discrete_sequence=["#00CC96"]
)
fig.update_layout(
    xaxis_title="Packet Length (bytes)",
    yaxis_title="Number of Packets",
    showlegend=False
)
fig.show()

# =============================================================================
# PACKET TYPE ANALYSIS
# =============================================================================
# Packet Types:
# - Control packets: Network management (SYN, ACK, FIN, RST)
# - Data packets: Actual payload/content transmission
#
# Binary encoding: Control = 1, Data = 0

print("\n" + "=" * 60)
print("PACKET TYPE ANALYSIS")
print("=" * 60)

col_name = "Packet Type"
if col_name in df.columns:
    print(df[col_name].value_counts())
    df = catvar_mapping(col_name, ["Control"], ["Control"])
    piechart_col("Packet Type Control")
elif "Packet Type Control" in df.columns:
    print("Packet Type already transformed to Packet Type Control")
    piechart_col("Packet Type Control")

# =============================================================================
# TRAFFIC TYPE ANALYSIS
# =============================================================================
# Application layer protocols:
# - DNS: Domain Name System queries/responses (port 53)
# - HTTP: Web traffic (port 80)
# - FTP: File Transfer Protocol (ports 20, 21)
#
# Binary encoding creates indicator columns for each traffic type

print("\n" + "=" * 60)
print("TRAFFIC TYPE ANALYSIS")
print("=" * 60)

col_name = "Traffic Type"
print(df[col_name].value_counts())
if "Traffic Type DNS" not in df.columns:
    df = catvar_mapping(col_name, ["DNS", "HTTP"])
piechart_col(col_name)

# Malware Indicators
# =============================================================================
# SECURITY INDICATORS ANALYSIS
# =============================================================================
# These columns contain security-related flags and detection results

print("\n" + "=" * 60)
print("SECURITY INDICATORS ANALYSIS")
print("=" * 60)

# -----------------------------------------------------------------------------
# Malware Indicators
# Indicators of Compromise (IoC) detected by security systems
# IoC examples: malicious file hashes, suspicious domains, known attack patterns
# Binary encoding: 1 = IoC detected, 0 = No IoC detected
# -----------------------------------------------------------------------------
col_name = "Malware Indicators"
print(f"\n--- {col_name} ---")
print(df[col_name].value_counts())

if df[col_name].dtype == 'object' or "IoC Detected" in df[col_name].astype(str).values:
    df = catvar_mapping(col_name, ["IoC Detected"], ["/"])
piechart_col(col_name, names=["No IoC Detected", "IoC Detected"])

# -----------------------------------------------------------------------------
# Anomaly Scores
# Numerical score indicating how anomalous/suspicious the traffic is
# Higher scores suggest more deviation from normal behavior
# -----------------------------------------------------------------------------
print("\n--- Anomaly Scores ---")
fig = px.histogram(
    df,
    x="Anomaly Scores",
    title="Distribution of Anomaly Scores",
    labels={"Anomaly Scores": "Anomaly Score", "count": "Frequency"},
    color_discrete_sequence=["#EF553B"]
)
fig.update_layout(
    xaxis_title="Anomaly Score (higher = more suspicious)",
    yaxis_title="Number of Records",
    showlegend=False
)
fig.show()

# -----------------------------------------------------------------------------
# Alert Trigger
# Whether the traffic triggered a security alert
# Binary encoding: 1 = Alert triggered, 0 = No alert
# -----------------------------------------------------------------------------
print("\n--- Alert Trigger ---")
col_name = "Alert Trigger"
if df[col_name].dtype == 'object' or "Alert Triggered" in df[col_name].astype(str).values:
    df = catvar_mapping(col_name, ["Alert Triggered"], ["/"])
piechart_col(col_name, names=["No Alert Triggered", "Alert Triggered"])

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
    df = catvar_mapping(col_name, ["Known Pattern A"], ["patA"])
    piechart_col("Attack Signature patA", ["Known Pattern A", "Known Pattern B"])
elif "Attack Signature patA" in df.columns:
    print("Attack Signature already transformed to Attack Signature patA")
    piechart_col("Attack Signature patA", ["Known Pattern A", "Known Pattern B"])

# -----------------------------------------------------------------------------
# Action Taken
# Response action by the security system
# - Logged: Traffic recorded for analysis
# - Blocked: Traffic denied/dropped
# - Ignored: No action taken
# -----------------------------------------------------------------------------
print("\n--- Action Taken ---")
col_name = "Action Taken"
if col_name in df.columns:
    print(df[col_name].value_counts())
    df = catvar_mapping(col_name, ["Logged", "Blocked"])
    piechart_col(col_name)
elif "Action Taken Logged" in df.columns:
    print("Action Taken already transformed")

# -----------------------------------------------------------------------------
# Severity Level
# Classification of threat severity
# Ordinal encoding: Low = -1, Medium = 0, High = +1
# This preserves the natural ordering for analysis
# -----------------------------------------------------------------------------
print("\n--- Severity Level ---")
col_name = "Severity Level"
print(df[col_name].value_counts())

df.loc[df[col_name] == "Low", col_name] = -1
df.loc[df[col_name] == "Medium", col_name] = 0
df.loc[df[col_name] == "High", col_name] = +1
piechart_col(col_name, ["Low Severity", "Medium Severity", "High Severity"])

# %% [markdown]
# ## Device and User-Agent Analysis
# This section parses User-Agent strings to extract:
# - Browser information (family, version)
# - Operating system details
# - Device characteristics (mobile, tablet, PC, bot)

# %% Device Information Extraction
# =============================================================================
# USER-AGENT PARSING AND DEVICE CLASSIFICATION
# =============================================================================
# The "Device Information" column contains User-Agent strings that reveal:
# - Browser: Chrome, Firefox, Safari, Edge, IE, etc.
# - Operating System: Windows, macOS, Linux, iOS, Android
# - Device Type: Desktop PC, Mobile phone, Tablet
# - Bot Detection: Automated crawlers vs human users
#
# This information helps identify:
# - Attack vectors targeting specific platforms
# - Bot-driven attacks vs human-initiated threats
# - Vulnerable browser/OS combinations

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
        "Browser family", "Browser major", "Browser minor",
        "OS family", "OS major", "OS minor",
        "Device family", "Device brand", "Device type", "Device bot"
    ]
    for i, new_col in enumerate(device_info_cols, start=1):
        if new_col not in df.columns:
            df.insert(col + i, new_col, value=pd.NA)

    # Extract browser information
    df["Browser family"] = df[col_name].apply(
        lambda x: parse(x).browser.family if parse(x).browser.family is not None else pd.NA
    )
    df["Browser major"] = df[col_name].apply(
        lambda x: parse(x).browser.version[0]
        if parse(x).browser.version[0] is not None
        else pd.NA
    )
    df["Browser minor"] = df[col_name].apply(
        lambda x: parse(x).browser.version[1]
        if parse(x).browser.version[1] is not None
        else pd.NA
    )

    # Extract operating system information
    df["OS family"] = df[col_name].apply(
        lambda x: parse(x).os.family if parse(x).os.family is not None else pd.NA
    )
    df["OS major"] = df[col_name].apply(
        lambda x: parse(x).os.version[0]
        if len(parse(x).os.version) > 0 and parse(x).os.version[0] is not None
        else pd.NA
    )
    df["OS minor"] = df[col_name].apply(
        lambda x: parse(x).os.version[1]
        if len(parse(x).os.version) > 1 and parse(x).os.version[1] is not None
        else pd.NA
    )
    df["OS patch"] = df[col_name].apply(
        lambda x: parse(x).os.version[2]
        if len(parse(x).os.version) > 2 and parse(x).os.version[2] is not None
        else pd.NA
    )

    # Extract device information
    df["Device family"] = df[col_name].apply(
        lambda x: parse(x).device.family if parse(x).device.family is not None else pd.NA
    )
    df["Device brand"] = df[col_name].apply(
        lambda x: parse(x).device.brand if parse(x).device.brand is not None else pd.NA
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
def Device_type(ua_string):
    """
    Classify device type from User-Agent string.

    Returns:
        str: "Mobile", "Tablet", "PC", or pd.NA if unknown
    """
    try:
        if not ua_string or pd.isna(ua_string):
            return pd.NA
        ua = ua_parse(ua_string)
        if getattr(ua, "is_mobile", False):
            return "Mobile"
        if getattr(ua, "is_tablet", False):
            return "Tablet"
        if getattr(ua, "is_pc", False):
            return "PC"
        return pd.NA
    except:
        return pd.NA


# Classify device types and detect bots
if "Device type" not in df.columns or df["Device type"].isna().all():
    print("Classifying device types...")
    df["Device type"] = df[col_name].apply(Device_type)

# Bot detection - identifies automated crawlers and scripts
if "Device bot" not in df.columns or df["Device bot"].isna().all():
    print("Detecting bot traffic...")
    df["Device bot"] = df[col_name].apply(lambda x: ua_parse(x).is_bot)

# %% [markdown]
# ## Network Infrastructure Analysis
# This section analyzes network segmentation and security system logs

# %% Network Segment and Security Logs
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
    df = catvar_mapping(col_name, ["Segment A", "Segment B"], ["segA", "segB"])
piechart_col(col_name)

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


def geolocation_data(info):
    """Split 'City, State' string into separate components."""
    city, state = info.split(", ")
    return pd.Series([city, state])


if "Geo-location City" not in df.columns:
    print("Parsing geo-location data...")
    col = df.columns.get_loc(col_name)
    df.insert(col + 1, "Geo-location City", value=pd.NA)
    df.insert(col + 2, "Geo-location State", value=pd.NA)
    df[["Geo-location City", "Geo-location State"]] = df["Geo-location Data"].apply(
        lambda x: geolocation_data(x)
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

if df[col_name].dtype == 'object' or "Log Data" in df[col_name].astype(str).values:
    df = catvar_mapping(col_name, ["Log Data"], ["/"])

# -----------------------------------------------------------------------------
# IDS/IPS Alerts
# Intrusion Detection/Prevention System alerts
# Binary: 1 = Alert data present, 0 = No alert
# -----------------------------------------------------------------------------
print("\n--- IDS/IPS Alerts ---")
col_name = "IDS/IPS Alerts"
print(df[col_name].value_counts())
# Binary: 1 = Alert data present, 0 = No alert
if df[col_name].dtype == 'object' or "Alert Data" in df[col_name].astype(str).values:
    df = catvar_mapping(col_name, ["Alert Data"], ["/"])

# -----------------------------------------------------------------------------
# Log Source
# Origin of the log entry
# Binary: Firewall = 1, Server = 0
# -----------------------------------------------------------------------------
print("\n--- Log Source ---")
col_name = "Log Source"
if col_name in df.columns:
    print(df[col_name].value_counts())
    df = catvar_mapping(col_name, ["Firewall"], ["Firewall"])
elif "Log Source Firewall" in df.columns:
    print("Log Source already transformed to Log Source Firewall")

# Generate cross-tabulation between log source and firewall logs
crosstab_col("Log Source Firewall", "Firewall Logs", "logsource", "firewallogs")


# %% [markdown]
# ## Attack Type Segmentation
# Create separate DataFrames for each attack type to enable
# comparative analysis between Malware, Intrusion, and DDoS attacks

# %% Separate DataFrames by Attack Type
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


# %% [markdown]
# ## Geographic Distribution by Attack Type and Anomaly Score
# Visualize where different attack types originate and target,
# with color intensity representing the anomaly score

# %% Geographic Plot: Attack Types with Anomaly Scores
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
        "Malware - Source IPs", "Malware - Destination IPs",
        "Intrusion - Source IPs", "Intrusion - Destination IPs",
        "DDoS - Source IPs", "DDoS - Destination IPs"
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
                "colorbar": {"title": "Anomaly<br>Score", "len": 0.25, "y": 0.85 - i * 0.33},
            },
            hovertemplate=f"<b>{attype} Attack Origin</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<br>Anomaly Score: %{{marker.color:.2f}}<extra></extra>"
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
                "colorbar": {"title": "Anomaly<br>Score", "len": 0.25, "y": 0.85 - i * 0.33, "x": 1.02},
                "showscale": False,  # Hide duplicate colorbar
            },
            hovertemplate=f"<b>{attype} Attack Target</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<br>Anomaly Score: %{{marker.color:.2f}}<extra></extra>"
        ),
        row=i + 1,
        col=2,
    )

fig.update_geos(
    showcountries=True,
    showland=True,
    landcolor="lightgray",
    countrycolor="white"
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


# %% [markdown]
# ## Attack Characteristics Comparison
# Compare packet lengths and anomaly scores across different attack types

# %% Packet Length by Attack Type
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
                "color": colors[attype]
            },
            hovertemplate=f"<b>{attype}</b><br>Packet Length: %{{x}} bytes<extra></extra>"
        ),
    )

fig.update_layout(
    title="Packet Length Distribution by Attack Type",
    title_font_size=16,
    xaxis_title="Packet Length (bytes)",
    yaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=attypes,
        title="Attack Type"
    ),
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
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
                "color": colors[attype]
            },
            hovertemplate=f"<b>{attype}</b><br>Anomaly Score: %{{x:.2f}}<extra></extra>"
        ),
    )

fig.update_layout(
    title="Anomaly Score Distribution by Attack Type",
    title_font_size=16,
    xaxis_title="Anomaly Score (higher = more suspicious)",
    yaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=attypes,
        title="Attack Type"
    ),
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)
fig.show()

# %% [markdown]
# ## Multi-Variable Sankey Diagram
# Visualize the flow of traffic characteristics leading to different attack types
# This helps identify which combinations of features are associated with each attack

# %% General Cross-Table Sankey Diagram
# =============================================================================
# MULTI-VARIABLE SANKEY DIAGRAM GENERATOR
# =============================================================================
# Creates a Sankey diagram showing how categorical variable values
# flow toward different attack types. Useful for understanding
# which feature combinations characterize each attack type.


def sankey_diag(
    cols_bully=[
        True,   # "Source IP country"
        True,   # "Destination IP country"
        True,   # "Source Port ephemeral"
        True,   # "Destination Port ephemeral"
        True,   # "Protocol"
        True,   # "Packet Type Control"
        True,   # "Traffic Type"
        True,   # "Malware Indicators"
        True,   # "Alert Trigger"
        True,   # "Attack Signature patA"
        True,   # "Action Taken"
        True,   # "Severity Level"
        True,   # "Network Segment"
        True,   # "Firewall Logs"
        True,   # "IDS/IPS Alerts"
        True,   # "Log Source Firewall"
    ],
    ntop=10,
):
    """
    Generate a Sankey diagram showing feature value flows to attack types.

    Parameters:
    -----------
    cols_bully : list of bool
        Boolean flags indicating which columns to include in the diagram
    ntop : int
        Number of top countries to show (others grouped as "other")

    Returns:
    --------
    DataFrame
        Cross-tabulation percentages used to build the diagram
    """
    cols = np.array(
        [
            "Source IP country",
            "Destination IP country",
            "Source Port ephemeral",
            "Destination Port ephemeral",
            "Protocol",
            "Packet Type Control",
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
    cols = cols[np.array(cols_bully)]

    idx_ct = []
    labels = []
    if "Source IP country" in cols:
        SIP = df["Source IP country"].copy(deep=True)
        SIP.name = "SIP"
        SIPlabs = pd.Series(SIP.value_counts().index[:ntop])
        bully = SIP.isin(SIPlabs) | SIP.isna()
        SIP.loc[~bully] = "other"
        SIPlabs = "SIP " + pd.concat([SIPlabs, pd.Series(["other"])])
        labels.extend(SIPlabs.to_list())
        idx_ct = idx_ct + [SIP]
        cols = cols[cols != "Source IP country"]
    if "Destination IP country" in cols:
        DIP = df["Destination IP country"].copy(deep=True)
        DIP.name = "DIP"
        DIPlabs = pd.Series(DIP.value_counts().index[:ntop])
        bully = DIP.isin(DIPlabs) | DIP.isna()
        DIP.loc[~bully] = "other"
        DIPlabs = "DIP " + pd.concat([DIPlabs, pd.Series(["other"])])
        labels.extend(DIPlabs.to_list())
        idx_ct = idx_ct + [DIP]
        cols = cols[cols != "Destination IP country"]

    # build cross table with Attack Type in columns and multi-index of variables in index
    idx_ct = idx_ct + [df[col] for col in cols]
    print(idx_ct)
    crosstabs = pd.crosstab(index=idx_ct, columns=df["Attack Type"])

    # compute labels
    for c in np.append(cols, "Attack Type"):
        vals = df[c].unique()
        for v in vals:
            labels.append(f"{c} {v}")
    # computation of source , target , value
    source = []
    target = []
    value = []
    nlvl = crosstabs.index.nlevels
    for idx, row in crosstabs.iterrows():
        row_labs = []
        if nlvl == 1:
            row_labs.append(f"{crosstabs.index.name} {idx}")
        else:
            for i, val in enumerate(idx):
                row_labs.append(f"{crosstabs.index.names[i]} {val}")
        for attype in crosstabs.columns:
            val = row[attype]
            for i in range(0, nlvl - 1):
                source.append(labels.index(row_labs[i]))
                target.append(labels.index(row_labs[i + 1]))
                value.append(val)
            source.append(labels.index(row_labs[-1]))
            target.append(labels.index(f"Attack Type {attype}"))
            value.append(val)

    # Plot the Sankey diagram with Inferno color scale
    n = len(labels)
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno, [i / (n - 1) for i in range(n)]
    )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="rgba(0, 0, 0, 0.1)", width=0.5),
                    label=labels,
                    color=colors,
                    hovertemplate="<b>%{label}</b><br>Total: %{value}<extra></extra>"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    hovertemplate="Flow from %{source.label} to %{target.label}<br>Count: %{value}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Feature Value Flow to Attack Types",
        title_font_size=18,
        font_size=14,
        title_font_family="Avenir",
        title_font_color="black",
    )
    fig.show()

    # Return normalized cross-tabulation (percentages)
    crosstabs = crosstabs / crosstabs.sum().sum() * 100

    return crosstabs


# Generate Sankey diagram with selected features
# Selected: Protocol, Traffic Type, Malware Indicators, IDS/IPS Alerts
print("\n" + "=" * 60)
print("MULTI-VARIABLE SANKEY ANALYSIS")
print("=" * 60)
print("Features included: Protocol, Traffic Type, Malware Indicators, IDS/IPS Alerts")

crosstabs = sankey_diag(
    [
        False,  # "Source IP country"
        False,  # "Destination IP country"
        False,  # "Source Port ephemeral"
        False,  # "Destination Port ephemeral"
        True,   # "Protocol" - INCLUDED
        False,  # "Packet Type Control"
        True,   # "Traffic Type" - INCLUDED
        True,   # "Malware Indicators" - INCLUDED
        False,  # "Alert Trigger"
        False,  # "Attack Signature patA"
        False,  # "Action Taken"
        False,  # "Severity Level"
        False,  # "Network Segment"
        False,  # "Firewall Logs"
        True,   # "IDS/IPS Alerts" - INCLUDED
        False,  # "Log Source Firewall"
    ]
)


# %% [markdown]
# ## Parallel Categories Diagram
# Interactive visualization showing relationships between categorical
# variables and attack types. Lines connect category values, with
# color indicating the selected variable's values.

# %% Parallel Categories Diagram
# =============================================================================
# PARALLEL CATEGORIES DIAGRAM
# =============================================================================
# Creates an interactive parallel categories plot showing how
# different categorical feature values relate to attack types.
# The diagram allows exploration of multi-dimensional categorical relationships.


def paracat_diag(
    cols_bully=[
        True,  # "Source IP country"
        True,  # "Destination IP country"
        True,  # "Source Port ephemeral"
        True,  # "Destination Port ephemeral"
        True,  # "Protocol"
        True,  # "Packet Type Control"
        True,  # "Traffic Type"
        True,  # "Malware Indicators"
        True,  # "Alert Trigger"
        True,  # "Attack Signature patA"
        True,  # "Action Taken"
        True,  # "Severity Level"
        True,  # "Network Segment"
        True,  # "Firewall Logs"
        True,  # "IDS/IPS Alerts"
        True,  # "Log Source Firewall"
    ],
    colorvar="Attack Type",
    ntop=10,
):
    cols = np.array(
        [
            "Source IP country",
            "Destination IP country",
            "Source Port ephemeral",
            "Destination Port ephemeral",
            "Protocol",
            "Packet Type Control",
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
    cols = cols[np.array(cols_bully)]

    dims_var = {}
    if "Source IP country" in cols:
        SIP = df["Source IP country"].copy(deep=True)
        SIP.name = "SIP"
        SIPlabs = pd.Series(SIP.value_counts().index[:ntop])
        bully = SIP.isin(SIPlabs) | SIP.isna()
        SIP.loc[~bully] = "other"
        SIPlabs = "SIP " + pd.concat([SIPlabs, pd.Series(["other"])])
        cols = cols[cols != "Source IP country"]
        dims_var["SIP"] = go.parcats.Dimension(
            values=SIP,
            # categoryorder = "category ascending" ,
            label="Source IP country",
        )
        cols = np.append(cols, "SIP")
    if "Destination IP country" in cols:
        DIP = df["Destination IP country"].copy(deep=True)
        DIP.name = "DIP"
        DIPlabs = pd.Series(DIP.value_counts().index[:ntop])
        bully = DIP.isin(DIPlabs) | DIP.isna()
        DIP.loc[~bully] = "other"
        DIPlabs = "DIP " + pd.concat([DIPlabs, pd.Series(["other"])])
        cols = cols[cols != "Destination IP country"]
        dims_var["DIP"] = go.parcats.Dimension(
            values=DIP,
            # categoryorder = "category ascending" ,
            label="Destination IP country",
        )
        cols = np.append(cols, "DIP")
    for col in cols[(cols != "SIP") & (cols != "DIP")]:
        print(col)
        dims_var[col] = go.parcats.Dimension(
            values=df[col],
            # categoryorder = "category ascending" ,
            label=col,
        )

    dim_attypes = go.parcats.Dimension(
        values=df["Attack Type"],
        label="Attack Type",
        # categoryarray = [ "Malware" , "Intrusion" , "DDoS" ] ,
        ticktext=["Malware", "Intrusion", "DDoS"],
    )

    dims = [dims_var[col] for col in cols]
    dims.append(dim_attypes)

    if colorvar == "SIP":
        colorvar = SIP
    elif colorvar == "DIP":
        colorvar = DIP
    elif (colorvar in cols) or (colorvar == "Attack Type"):
        colorvar = df[colorvar]
    else:
        ValueError("colorvar must be in cols")
    catcolor = colorvar.unique()
    catcolor_n = catcolor.shape[0]
    color = colorvar.map({cat: i for i, cat in enumerate(catcolor)})
    positions = [
        i / (catcolor_n - 1) if (catcolor_n > 1) else 0 for i in range(0, catcolor_n)
    ]
    palette = px.colors.sequential.Viridis
    colors = [px.colors.sample_colorscale(palette, p)[0] for p in positions]
    colorscale = [[positions[i], colors[i]] for i in range(0, catcolor_n)]

    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=dims,
                line={
                    "color": color,
                    "colorscale": colorscale,
                    "shape": "hspline",
                },
                hoveron="color",
                hoverinfo="count+probability",
                labelfont={"size": 18, "family": "Times"},
                tickfont={"size": 16, "family": "Times"},
                arrangement="freeform",
            )
        ]
    )
    fig.show()


# Generate parallel categories diagram
# Selected features: Protocol, Traffic Type, Attack Signature, Severity Level, Network Segment
# Color: Attack Signature Pattern A (to see how signature types flow through other features)
print("\n" + "=" * 60)
print("PARALLEL CATEGORIES DIAGRAM")
print("=" * 60)
print("Features: Protocol, Traffic Type, Attack Signature, Severity Level, Network Segment")
print("Color coding: Attack Signature Pattern (A vs B)")

paracat_diag(
    [
        False,  # "Source IP country"
        False,  # "Destination IP country"
        False,  # "Source Port ephemeral"
        False,  # "Destination Port ephemeral"
        True,   # "Protocol" - INCLUDED
        False,  # "Packet Type Control"
        True,   # "Traffic Type" - INCLUDED
        False,  # "Malware Indicators"
        False,  # "Alert Trigger"
        True,   # "Attack Signature patA" - INCLUDED
        False,  # "Action Taken"
        True,   # "Severity Level" - INCLUDED
        True,   # "Network Segment" - INCLUDED
        False,  # "Firewall Logs"
        False,  # "IDS/IPS Alerts"
        False,  # "Log Source Firewall"
    ],
    colorvar="Attack Signature patA",
)


# %% [markdown]
# ## Statistical Analysis: Correlation and Chi-Square Tests
# Measure the strength of association between categorical variables
# and the target variable (Attack Type)

# %% Matthews Correlation Coefficients
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


def catvar_corr(col, target="Attack Type"):
    """Calculate Matthews correlation coefficient between a feature and target."""
    corr = matthews_corrcoef(df[target], df[col].astype(str))
    print(f"  {col:35} : {corr:+.4f}")


catvars = np.array(
    [
        "Source IP country",
        "Destination IP country",
        "Source Port ephemeral",
        "Destination Port ephemeral",
        "Protocol",
        "Packet Type Control",
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
    catvar_corr(c)


# %% [markdown]
# ## Chi-Square Test of Independence
# Test whether categorical variables are independent of Attack Type

# %% Chi-Square Independence Test
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

from scipy.stats import chi2_contingency

# Prepare categorical variables for analysis
df_catvar = df[
    [
        "Attack Type",
        "Source Port ephemeral",
        "Destination Port ephemeral",
        "Protocol",
        "Packet Type Control",
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
    if df_catvar_encoded[col].dtype == 'object':
        df_catvar_encoded[col] = pd.Categorical(df_catvar_encoded[col]).codes

# Perform chi-square test
res = chi2_contingency(pd.crosstab(df_catvar_encoded['Attack Type'], df_catvar_encoded['Protocol']))
print(f"Chi-square statistic: {res.statistic:.4f}")
print(f"P-value: {res.pvalue:.6f}")
print(f"Degrees of freedom: {res.dof}")
print(f"\nExpected frequencies:")
print(res.expected_freq)

if res.pvalue < 0.05:
    print("\n=> Significant association detected (p < 0.05)")
else:
    print("\n=> No significant association (p >= 0.05)")


# %% [markdown]
# ## Multiple Correspondence Analysis (MCA)
# Dimensionality reduction technique for categorical variables
# Similar to PCA but for categorical data

# %% Multiple Correspondence Analysis
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
        "Packet Type Control",
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
print(mca.eigenvalues_summary)


# %% [markdown]
# ## Time Series Analysis: Attack Patterns Over Time
# Analyze temporal patterns in attack occurrences using:
# - Daily attack counts by type
# - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
# - Rolling averages for trend detection

# %% Time Series Analysis - Daily Attack Counts
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
    vertical_spacing=0.08
)

# Malware time series
fig.add_trace(
    go.Scatter(
        x=Attacks_pday.index,
        y=Attacks_pday["Malware"],
        mode="lines",
        name="Malware",
        line=dict(color="#636EFA"),
        hovertemplate="<b>Malware</b><br>Date: %{x}<br>Count: %{y}<extra></extra>"
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
        hovertemplate="<b>Intrusion</b><br>Date: %{x}<br>Count: %{y}<extra></extra>"
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
        hovertemplate="<b>DDoS</b><br>Date: %{x}<br>Count: %{y}<extra></extra>"
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=800,
    title_text="Daily Attack Counts by Type",
    title_font_size=18,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
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
    horizontal_spacing=0.08
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
            hovertemplate=f"<b>{attacktype} ACF</b><br>Lag: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>"
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
            hovertemplate=f"<b>{attacktype} PACF</b><br>Lag: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>"
        ),
        row=i + 1,
        col=2,
    )

fig.update_layout(
    height=900,
    title_text="Autocorrelation Analysis for Attack Time Series",
    title_font_size=18,
    showlegend=False
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

df.to_csv("data/df.csv", sep="|")
print("Dataset saved to data/df.csv")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")


# %% [markdown]
# ## Rolling Average Trend Analysis
# Smooth the daily time series using rolling averages to identify trends

# %% Rolling Average Analysis
# =============================================================================
# ROLLING AVERAGE TREND ANALYSIS
# =============================================================================
# Apply a 30-day rolling average to smooth out daily fluctuations
# and reveal underlying trends in attack patterns

print("\n" + "=" * 60)
print("ROLLING AVERAGE TREND ANALYSIS")
print("=" * 60)

rw = 30  # 30-day rolling window
print(f"Using {rw}-day rolling average window")

tr1 = go.Scatter(
    x=Attacks_pday.index,
    y=Attacks_pday["Malware"].rolling(rw).mean(),
    name=f"Malware ({rw}-day MA)",
    line=dict(color="#636EFA", width=2),
    hovertemplate="<b>Malware</b><br>Date: %{x}<br>30-day Avg: %{y:.1f}<extra></extra>"
)
tr2 = go.Scatter(
    x=Attacks_pday.index,
    y=Attacks_pday["Intrusion"].rolling(rw).mean(),
    name=f"Intrusion ({rw}-day MA)",
    line=dict(color="#EF553B", width=2),
    hovertemplate="<b>Intrusion</b><br>Date: %{x}<br>30-day Avg: %{y:.1f}<extra></extra>"
)
tr3 = go.Scatter(
    x=Attacks_pday.index,
    y=Attacks_pday["DDoS"].rolling(rw).mean(),
    name=f"DDoS ({rw}-day MA)",
    line=dict(color="#00CC96", width=2),
    hovertemplate="<b>DDoS</b><br>Date: %{x}<br>30-day Avg: %{y:.1f}<extra></extra>"
)

# Create combined rolling average plot
fig = subp()
fig.add_trace(tr1)
fig.add_trace(tr2)
fig.add_trace(tr3)
fig.update_layout(
    title_text=f"Attack Trends: {rw}-Day Rolling Average by Attack Type",
    title_font_size=18,
    xaxis_title="Date",
    yaxis_title="Average Daily Attacks",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        title="Attack Type"
    ),
    hovermode="x unified"
)
fig.show()


# %% DDoS Short-Term Trend (15-day MA)
# =============================================================================
# DDoS SHORT-TERM TREND ANALYSIS
# =============================================================================
# 15-day rolling average for more responsive trend detection

print("\n--- DDoS 15-Day Moving Average ---")

fig = px.line(
    Attacks_pday,
    x=Attacks_pday.index,
    y=Attacks_pday["DDoS"].rolling(15).mean(),
    title="DDoS Attack Trend: 15-Day Rolling Average",
    labels={"y": "15-Day Average Attacks", "x": "Date"}
)
fig.update_traces(
    line=dict(color="#00CC96", width=2),
    hovertemplate="<b>DDoS</b><br>Date: %{x}<br>15-day Avg: %{y:.1f}<extra></extra>"
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Daily DDoS Attacks",
    showlegend=False
)
fig.show()


# %% Monthly Attack Aggregation
# =============================================================================
# MONTHLY ATTACK AGGREGATION
# =============================================================================
# Aggregate attacks by month for longer-term pattern analysis

print("\n" + "=" * 60)
print("MONTHLY ATTACK ANALYSIS")
print("=" * 60)

Attacks_pmonth = df.copy(deep=True)
Attacks_pmonth["date_mm"] = Attacks_pmonth["date"].dt.ceil("d")
Attacks_pmonth = (
    Attacks_pmonth.groupby(["date_mm", "Attack Type"]).size().unstack().iloc[1:-1,]
)

fig = px.line(
    Attacks_pmonth,
    x=Attacks_pmonth.index,
    y="DDoS",
    title="Monthly DDoS Attack Volume",
    labels={"y": "Number of Attacks", "date_mm": "Date"}
)
fig.update_traces(
    line=dict(color="#00CC96", width=2),
    hovertemplate="<b>DDoS</b><br>Date: %{x}<br>Count: %{y}<extra></extra>"
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of DDoS Attacks",
    showlegend=False
)
fig.show()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

# %%
