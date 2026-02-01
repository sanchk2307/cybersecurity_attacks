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

from sklearn.metrics import matthews_corrcoef

import plotly.io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp

import random

import src.utilities.pipeline_utility_functions as pipeline_utils
import src.utilities.django_settings as django_settings
import src.utilities.data_preparation as data_prep
# from src.utilities.data_visualization import *

# Configure Plotly to render charts in the default browser
plotly.io.renderers.default = "browser"

# Load environment variables from .env file
DOTENV_PATH = os.path.join(".env")

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

df = pipeline_utils.sankey_diag_IPs(df, ntop=10)
