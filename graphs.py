## Cybersecurity Attacks Dataset - Exploratory Data Analysis
# This notebook performs comprehensive EDA on a cybersecurity attacks dataset , including IP geolocation analysis , attack pattern visualization , and statistical analysis of network traffic features .

#%% Library Initialization and Configuration
# =============================================================================
# This cell imports all required libraries and configures the environment :
# - pandas : Data manipulation and analysis
# - plotly : Interactive visualizations
# - geoip2/django : IP geolocation services
# - sklearn : Machine learning metrics ( Matthews correlation )
# - user_agents : Browser/device detection from User-Agent strings

import pandas as pd
import os
import math
import numpy as np
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
# optional GDAL configuration for geospatial operations : uncomment , adjust path and name if needed for your environment ( make sure this is the name of your gdal.dll file , rename it to appropriate version if necessary )
os.environ[ "GDAL_LIBRARY_PATH" ] = "C:/Users/KalooIna/anaconda3/envs/cybersecurity_attacks/Library/bin/gdal311.dll" 
import geoip2.database
# configure plotly to render charts in the default browser
import plotly.io
plotly.io.renderers.default = "browser"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp
from sklearn.metrics import matthews_corrcoef
import django
from user_agents import parse
from user_agents import parse as ua_parse
from django.conf import settings
# Django GeoIP2 settings : 
# - configure Django settings for IP geolocation using MaxMind GeoLite2 database
# - the database must be downloaded from MaxMind and placed in ./geolite2_db/
if not settings.configured :
    settings.configure(
        GEOIP_PATH = "data/geolite2_db" ,
        INSTALLED_APPS = [ "django.contrib.gis" ]
        )
    django.setup()
# initialize GeoIP2 lookup service
from django.contrib.gis.geoip2 import GeoIP2
geoIP = GeoIP2

# reference URLs for GeoIPE setup and documentation
maxmind_geoip2_db_url = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
geoip2_doc_url = "https://geoip2.readthedocs.io/en/latest/"
geoip2_django_doc_url = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"

# loading dataset
df_original = pd.read_csv( "data/df_original.csv" )
df = pd.read_csv( "data/df.csv" , sep = "|" , index_col = 0 )
df[ "date" ] = pd.to_datetime( df[ "date" ])
df[ "date dd" ] = pd.to_datetime( df[ "date dd" ])

# creating crosstabs dictionnary
crosstabs_x_AttackType = {}

#%% graphs

df["date"] = pd.to_datetime(df["date"])

attypes = df["Attack Type"].unique()

attype_color_map = {"DDoS": "#3B82F6", "Intrusion": "#F59E0B", "Malware": "#10B981"}

# Date histogram --------------------------------------------------------------
fig = px.histogram(
    df,
    x='date',
    color='Attack Type',
    nbins=200,
    color_discrete_map=attype_color_map,
)
fig.update_layout(
    barmode='stack', 
    xaxis_title='Date', 
    yaxis_title='number of attacks',
    title=dict(
        text=f'''
        <span style="
            font-variant: small-caps;
            font-weight: 600;
            letter-spacing: 2px;
            font-size: 24px;
            color: #333333;
        ">
            Number of cybersecurity attacks over time
        </span>
        ''',
        x=0.5,       # center the title
        xanchor='center'
    )
    )
fig.show()

# Date MW ---------------------------------------------------------------------
col_names = [ "date WD" , "date MW" , "date H" , "date M" ]
for col_name in col_names :
    fig = subp(
        rows = 1,
        cols = 4,
        specs = [[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
        subplot_titles = [
            "Global distribution of date WD",
            f"Distribution of {col_name} for { attypes[0]} attack types",
            f"Distribution of {col_name} for { attypes[1]} attack types",
            f"Distribution of {col_name} for { attypes[2]} attack types"
        ]
        )
    target_counts = df[ col_name ].value_counts( dropna = False ).sort_index()
    fig.add_trace(
        go.Pie(
            labels = target_counts.index,
            values = target_counts.values,
            name = col_name
        ),
        row = 1,
        col = 1
    )
    for a , attype in enumerate( attypes ) :
        target_counts = df[ df["Attack Type"] == attype ][ col_name ].value_counts( dropna = False ).sort_index()
        fig.add_trace(
            go.Pie(
                labels = target_counts.index,
                values = target_counts.values,
                name = col_name
            ),
            row = 1,
            col = a + 2
        )
    fig.update_layout(
        height = 600,   # adjust to taste
        title=dict(
            text=f'''
            <span style="
                font-variant: small-caps;
                font-weight: 600;
                letter-spacing: 2px;
                font-size: 24px;
                color: #333333;
            ">
                Distribution of {col_name}
            </span>
            ''',
            x=0.5,       # center the title
            xanchor='center'
        )
    )
    fig.show()

# Source Port histogram -----------------------------------------------------
col_names = ["Source Port", "Destination Port"]
for col_name in col_names :
    fig = px.histogram(
        df,
        x=col_name,
        color='Attack Type',
        nbins=200,
        color_discrete_map=attype_color_map,
        title=f'Number of cybersecurity attacks for {col_name} range',
    )
    fig.update_layout(
        barmode='stack', 
        xaxis_title='Date', 
        yaxis_title='number of attacks',
        title=dict(
            text=f'''
            <span style="
                font-variant: small-caps;
                font-weight: 600;
                letter-spacing: 2px;
                font-size: 24px;
                color: #333333;
            ">
                Distribution of {col_name}
            </span>
            ''',
            x=0.5,       # center the title
            xanchor='center'
        ))
    fig.show()

# Packet Length histogram -----------------------------------------------------
fig = px.histogram(
    df,
    x='Packet Length',
    color='Attack Type',
    nbins=200,
    color_discrete_map=attype_color_map,
)
fig.update_layout(
    barmode='stack',
    xaxis_title='Packet Length',
    yaxis_title='Number of attacks',
    title=dict(
        text=f'''
        <span style="
            font-variant: small-caps;
            font-weight: 600;
            letter-spacing: 2px;
            font-size: 24px;
            color: #333333;
        ">
            Distribution of Packet Length
        </span>
        ''',
        x=0.5,       # center the title
        xanchor='center'
    )
)
fig.show()

# Anomaly Scores histogram -----------------------------------------------------
fig = px.histogram(
    df,
    x='Anomaly Scores',
    color='Attack Type',
    nbins=200,
    color_discrete_map=attype_color_map,
)
fig.update_layout(
    barmode='stack',
    xaxis_title='Anomaly Scores',
    yaxis_title='Number of attacks',
    title=dict(
        text=f'''
        <span style="
            font-variant: small-caps;
            font-weight: 600;
            letter-spacing: 2px;
            font-size: 24px;
            color: #333333;
        ">
            Distribution of Anomaly Scores
        </span>
        ''',
        x=0.5,       # center the title
        xanchor='center'
    )
)
fig.show()

# Categorical variables ---------------------------------------------------------------------
col_names = [ "Protocol" , "Packet Type" , "Malware Indicators" , "Alerts/Warnings" , "Attack Signature" , "Action Taken" , "Severity Level" , "Browser family" , "OS family" , "Device Type" , "Network Segment" , "Proxy usage" , "Firewall Logs" , "IDS/IPS Alerts", "Log Source"]
for col_name in col_names :
    fig = subp(
        rows = 1,
        cols = 4,
        specs = [[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
        subplot_titles = [
            f"Global distribution of {col_name}",
            f"Distribution of {col_name} for { attypes[0]} attack types",
            f"Distribution of {col_name} for { attypes[1]} attack types",
            f"Distribution of {col_name} for { attypes[2]} attack types"
        ]
        )
    target_counts = df[ col_name ].value_counts( dropna = False ).sort_index()
    fig.add_trace(
        go.Pie(
            labels = target_counts.index,
            values = target_counts.values,
            name = col_name
        ),
        row = 1,
        col = 1
    )
    for a , attype in enumerate( attypes ) :
        target_counts = df[ df["Attack Type"] == attype ][ col_name ].value_counts( dropna = False ).sort_index()
        fig.add_trace(
            go.Pie(
                labels = target_counts.index,
                values = target_counts.values,
                name = col_name
            ),
            row = 1,
            col = a + 2
        )
    fig.update_layout(
        height = 600,   # adjust to taste
        # title_text = f"Distribution of {col_name}"
        
        title=dict(
            text=f'''
            <span style="
                font-variant: small-caps;
                font-weight: 600;
                letter-spacing: 2px;
                font-size: 24px;
                color: #333333;
            ">
                Distribution of {col_name}
            </span>
            ''',
            x=0.5,                     # center the title
            xanchor='center'
        )
    )
    fig.show()

# geoplot

fig = subp(
    rows = 4,
    cols = 2,
    specs = [
    [{"type": "scattergeo"}, {"type": "scattergeo"}],  # row 1: maps
    [{"type": "scattergeo"}, {"type": "scattergeo"}],          # row 2: pies
    [{"type": "scattergeo"}, {"type": "scattergeo"}],          # row 3: pies
    [{"type": "scattergeo"}, {"type": "scattergeo"}],          # row 4: pies
    ],
    subplot_titles = [
        f"Global geographical distribution of {col_names[0]}", f"Global geographical distribution of {col_names[1]}",
        f"Geographical distribution of {col_names[0]} for { attypes[0]} attack types" , f"Geographical distribution of {col_names[1]} for { attypes[0]} attack types",
        f"Geographical distribution of {col_names[0]} for { attypes[1]} attack types" , f"Geographical distribution of {col_names[1]} for { attypes[1]} attack types",
        f"Geographical distribution of {col_names[0]} for { attypes[2]} attack types" , f"Geographical distribution of {col_names[1]} for { attypes[2]} attack types"
    ],
    vertical_spacing=0.02,
    horizontal_spacing=0.03
    )
col_names = [ "Source IP" , "Destination IP"]
for c , col_name in enumerate(col_names) :
    fig.add_trace(
        go.Scattergeo(
            lon = df[f"{col_name} longitude"],
            lat = df[f"{col_name} latitude"],
            mode = "markers",
            marker = dict(
                size = 6,
                opacity = 0.6 ,
                color = "rgb(100,100,180)"
            ),
            showlegend=False
        ),
        row = 1,
        col = c + 1
    )
    for a , attype in enumerate( attypes ) :
        target_counts = df[ df["Attack Type"] == attype ][[ f"{col_name} longitude" , f"{col_name} latitude" ]]
        fig.add_trace(
            go.Scattergeo(
                lon = target_counts[f"{col_name} longitude"],
                lat = target_counts[f"{col_name} latitude"],
                mode = "markers",
                marker = dict(
                    size = 6,
                    opacity = 0.6 ,
                    color = attype_color_map[attype]
                ),
                showlegend=False
        ),
        row = a + 2 ,
        col = c + 1
    )
fig.update_layout(
    height=2400,
    title=dict(
        text='''
        <span style="
            font-variant: small-caps;
            font-weight: 600;
            letter-spacing: 2px;
            font-size: 24px;
            color: #333333;
        ">
            Geographical Distribution of Source and Destination IPs
        </span>
        ''',
        x=0.5,                     # center the title
        xanchor='center'
    )
)

# Update geo settings for all scattergeo subplots
num_geos = 8  # 4 rows x 2 cols
for i in range(1, num_geos + 1):
    fig.layout[f"geo{i}"].update(
        scope="world",
        projection_type="natural earth",
        landcolor="rgb(240,240,240)",
        oceancolor="rgb(180,220,255)",
        showland=True,
        showcountries=True,
        countrycolor="rgb(200,200,200)"
    )

fig.show()

    
