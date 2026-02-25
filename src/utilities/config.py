"""Configuration, environment setup, and data loading for the cybersecurity EDA pipeline."""

import os
from pathlib import Path

# GDAL configuration for geospatial operations — must be set before any Django import
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
GDAL_LIBRARY_PATH = os.environ.get(
    "GDAL_LIBRARY_PATH",
    str(_PROJECT_ROOT / ".pixi" / "envs" / "default" / "Library" / "bin" / "gdal.dll"),
)
os.environ["GDAL_LIBRARY_PATH"] = GDAL_LIBRARY_PATH

import django  # noqa: E402
import geoip2.database  # noqa: E402, F401
import pandas as pd  # noqa: E402
import plotly.io  # noqa: E402

# Disable Arrow-backed string inference (pandas 3.x default).
# The pipeline assigns integers into string columns, which Arrow strings reject.
pd.options.future.infer_string = False
from django.conf import settings  # noqa: E402

# Plotly renderer configuration
plotly.io.renderers.default = "browser"

# Set to False to skip fig.show() calls and speed up headless runs
SHOW_FIGURES = True


def show_fig(fig):
    """Show a Plotly figure only if SHOW_FIGURES is enabled."""
    if SHOW_FIGURES:
        fig.show()

# Django GeoIP2 settings
GEOIP_PATH = "geolite2_db"

if not settings.configured:
    settings.configure(
        GEOIP_PATH=GEOIP_PATH,
        GDAL_LIBRARY_PATH=GDAL_LIBRARY_PATH,
        INSTALLED_APPS=["django.contrib.gis"],
    )
    django.setup()

from django.contrib.gis.geoip2 import GeoIP2

geoIP = GeoIP2

# Reference URLs
MAXMIND_GEOIP2_DB_URL = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
GEOIP2_DOC_URL = "https://geoip2.readthedocs.io/en/latest/"
GEOIP2_DJANGO_DOC_URL = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"

# Data paths
DATA_DIR = "data"
ORIGINAL_CSV = f"{DATA_DIR}/cybersecurity_attacks.csv"
PROCESSED_CSV = f"{DATA_DIR}/df.csv"


def load_data():
    """Load the original and processed datasets.

    If the processed CSV exists, loads it with parsed dates.
    Otherwise, loads the original CSV as both df_original and df.

    Returns
    -------
    df_original : pd.DataFrame
        Raw dataset as-is from CSV.
    df : pd.DataFrame
        Processed dataset (or raw if processed file not found).
    """
    df_original = pd.read_csv(ORIGINAL_CSV)

    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV, sep="|", index_col=0)
        df["date"] = pd.to_datetime(df["date"])
        df["date dd"] = pd.to_datetime(df["date dd"])
    else:
        print(f"Processed file {PROCESSED_CSV} not found, using original dataset.")
        df = df_original.copy()

    return df_original, df
