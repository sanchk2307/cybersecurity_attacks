#%% library init

import pandas as pd
import plotly.io
import os
import numpy as np
os.environ[ "GDAL_LIBRARY_PATH" ] = "C:/Users/KalooIna/anaconda3/envs/cybersecurity_attacks/Library/bin/gdal311.dll" # make sure this is the name of your gdal.dll file ( rename it to appropriate version if necessary )
import geoip2.database
plotly.io.renderers.default = "browser" # plotly settings for browser settings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp
import random
import django
from django.conf import settings
# django settings for geoIP2
settings.configure(
    GEOIP_PATH = "data/geolite2_db" ,
    INSTALLED_APPS = [ "django.contrib.gis" ]
    )
django.setup()
from django.contrib.gis.geoip2 import GeoIP2
geoIP = GeoIP2()

# useful links
maxmind_geoip2_db_url = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
geoip2_doc_url = "https://geoip2.readthedocs.io/en/latest/"
geoip2_django_doc_url = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"

# loadding dataset
df = pd.read_csv("data/cybersecurity_attacks.csv" )

# transform categorical variable to binary variables [ 0 , 1 ]
def catvar_mapping( col_name , values , name = None) : 
    df1 = df.copy( deep = True )
    if name is None :
        name = values
    elif ( len( name ) == 1 ) and ( name != [ "/" ]) :
        col_target = f"{ col_name } { name[ 0 ]}"
        df1 = df1.rename( columns = { col_name : col_target })
        col_name = col_target
        name = [ col_target ]
    col = df1.columns.get_loc( col_name ) + 1
    for val , nm in zip( values , name ) :
        if ( nm == "/" ) :
            col_target = col_name
        elif ( len( name ) == 1 ) :
            col_target = nm
        else :
            col_target = f"{ col_name } { nm }"
            df1.insert( col , col_target , value = pd.NA )
        bully = df1[ col_name ] == val
        df1.loc[ bully , col_target ] = 1
        df1.loc[ ~ bully , col_target ] = 0
        col += 1
    return df1
        
# pieichart generator for a column
def piechart_col( col , names = None ) :
    if names is None :
        fig = px.pie( 
            values = df[ col ].value_counts() ,
            names = df[ col ].value_counts().index ,
            )
        fig.show()
    else :
        fig = px.pie( 
            values = df[ col ].value_counts() ,
            names = names ,
            )
        fig.show()

# transforms IP addresses to infos : longitude , latitude , country , city
def ip_to_coords( ip_address ) :
    print( ip_address )
    ret = pd.Series( dtype = object )
    try :
        res = geoIP.geos( ip_address ).wkt
        lon , lat = res.replace( "(" , "" ).replace( ")" , "" ).split()[ 1 : ]
        ret = pd.concat([ ret , pd.Series([ lat , lon ])] , ignore_index = True )
    except :
        ret = pd.concat([ ret , pd.Series([ pd.NA , pd.NA ])] , ignore_index = True )
    try :
        res = geoIP.city( ip_address )
        ret = pd.concat([ ret , pd.Series([ res[ "country_name" ] , res[ "city" ]])] , ignore_index = True )
    except :
        ret = pd.concat([ ret , pd.Series([ pd.NA , pd.NA ])] , ignore_index = True )
    return ret
       
        
#%% EDA
     
# renaming columns
df = df.rename( columns = { 
    "Timestamp" : "date" ,
    "Source Port" : "Source Port ephemeral" ,
    "Destination Port" : "Destination Port ephemeral" ,
    "Alerts/Warnings" : "Alert Trigger" ,
    })
df = df.drop( "User Information" , axis = 1 )
print( df.describe())

# generation of crosstables for cat variables
crosstabs = {}
def crosstab_col( col ,target , name_col , name_target ) :
    name_tab = f"{ name_col }_x_{ name_target }"
    crosstabs[ name_tab ] = pd.crosstab( df[ target ] , df[ col ] , normalize = True ) * 100

# NAs
df_s0 = df.shape[ 0 ]
for col in df.columns :
    NA_n = sum( df[ col ].isna())
    if NA_n > 0 :
        print( f"number of NAs in { col } = { NA_n } / { df_s0 } = { NA_n / df_s0 } " )

# Attack Type !!!! TARGET VARIABLE !!!!
col_name = "Attack Type"
print( df[ col_name ].value_counts())
piechart_col( col_name )

# date
col_name = "date"
df[ col_name ] = pd.to_datetime( df[ col_name ])
date_end = max( df[ col_name ])
date_start = min( df[ col_name ])
print( f"dates go from { date_start } and { date_end }" )
fig = px.histogram( df , col_name )
fig.show()


#%% IP address
i = 2
for destsource in [ "Source" , "Destination" ] :
    df.insert( i , f"{ destsource } IP latitude" , value = pd.NA )
    df.insert( i + 1 , f"{ destsource } IP longitude" , value = pd.NA )
    df.insert( i + 2 , f"{ destsource } IP country" , value = pd.NA )
    df.insert( i + 3 , f"{ destsource } IP city" , value = pd.NA )
    df[[ f"{ destsource } IP latitude" , f"{ destsource } IP longitude" , f"{ destsource } IP country" , f"{ destsource } IP city" ]] = df[ f"{ destsource } IP Address" ].apply( lambda x : ip_to_coords( x ))
    i = i + 5
## IP address map graph
fig = subp(
    rows = 1 ,
    cols = 2 ,
    specs = [
        { "type" : "scattergeo" } , 
        { "type" : "scattergeo" } ,
        ] ,
    subplot_titles = (
        "Source IP locations" ,
        "Destination IP locations" 
        )
    )
fig.add_trace(
    go.Scattergeo(
        lat = df[ "Source IP latitude" ] ,
        lon = df[ "Source IP longitude" ] ,
        mode = "markers" ,
        marker = { 
            "size" : 5 ,
            "color" : "blue"
            }
        ) ,
    row = 1 , 
    col = 1
    )
fig.add_trace(
    go.Scattergeo(
        lat = df[ "Destination IP latitude" ] ,
        lon = df[ "Destination IP longitude" ] ,
        mode = "markers" ,
        marker = { 
            "size" : 5 ,
            "color" : "blue"
            }
        ) ,
    row = 1 , 
    col = 2
    )
fig.update_geos(
    projection_type = "orthographic" ,
    showcountries = True ,
    showland = True ,
    # landcolor = "LightGreen"
)
fig.update_layout(
    height = 750 ,
    margin = { 
        "r" : 0 , 
        "t" : 80 , 
        "l" : 0 , 
        "b" : 0 
        } ,
    title_text = "IP Address Locations" ,
    title_x = 0.5
)
fig.show()

# Proxy Information
## * NAs = no proxy or what ?????
col_name = "Proxy Information"
print( df[ col_name ].value_counts())
col = df.columns.get_loc( col_name )
df.insert( col + 1 , "Proxy latitude" , value = pd.NA )
df.insert( col + 2 , "Proxy longitude" , value = pd.NA )
df.insert( col + 3 , "Proxy country" , value = pd.NA )
df.insert( col + 4 , "Proxy city" , value = pd.NA )
df[[ "Proxy latitude" , "Proxy longitude" , "Proxy country" , "Proxy city" ]] = df[ "Source IP Address" ].apply( lambda x : ip_to_coords( x ))

#%% Source Port
col_name = "Source Port ephemeral"
## create boolean value for ephemeral and assigned ports
"""
    ephemeral port > 49151 = 1 
    assigned/registered port <= 49151 = 0
"""
bully = df[ col_name ] > 49151
df.loc[ bully , col_name ] = 1
df.loc[ ~ bully , col_name ] = 0
print( df[ col_name ].value_counts())
# piechart_col( col_name )
crosstab_col( col_name , "Attack Type" , "sourceport" , "attacktype" )

# Destination Port
col_name = "Destination Port ephemeral"
## create boolean value for ephemeral and assigned ports
"""
    ephemeral port > 49151 = 1 
    assigned/registered port <= 49151 = 0
"""
bully = df[ col_name ] > 49151
df.loc[ bully , col_name ] = 1
df.loc[ ~ bully , col_name ] = 0
print( df[ col_name ].value_counts())
# piechart_col( col_name )
crosstab_col( col_name , "Attack Type" , "destport" , "attacktype" )

# Protocol
col_name = "Protocol"
"""
    UDP = { 1 if Protocol = "UDP" , 0 otherwise }
    TCP = { 1 if Protocol = "TCP" , 0 otherwise }
    IMCP = [ 0 , 0 ]
"""
print( df[ col_name ].value_counts())
piechart_col( col_name )
df = catvar_mapping( col_name , [ "UDP" , "TCP" ])
### cross table Protocol x Attack Type
crosstab_col( col_name , "Attack Type" , col_name , "attacktype" )

# Packet length
fig = px.histogram( df , "Packet Length" )
fig.show()

# Packet Type
col_name = "Packet Type"
"""
    Control = 1
    Data = 0
"""
print( df[ col_name ].value_counts())
df = catvar_mapping( col_name , [ "Control" ] , [ "Control" ])
piechart_col( "Packet Type Control" )

# Traffic Type
col_name = "Traffic Type"
"""
    DNS = { 1 if Traffic Type = "DNS" , 0 otherwise }
    HTTP = { 1 if Traffic Type = "HTTP" , 0 otherwise }
    FTP = [ 0 , 0 ]
"""
print( df[ col_name ].value_counts())
df = catvar_mapping( col_name , [ "DNS" , "HTTP" ])
piechart_col( col_name )

# Malware Indicators
col_name = "Malware Indicators"
print( df[ col_name ].value_counts())
"""
    IoC Detected = 1
    pd.NA = 0
"""
df = catvar_mapping( col_name , [ "IoC Detected" ] , [ "/" ])
piechart_col( col_name )

# Anomaly Scores
fig = px.histogram( df , "Anomaly Scores" )
fig.show()

# Alert Trigger
col_name = "Alert Trigger"
# print( df[ col_name ].value_counts())
df = catvar_mapping( col_name , [ "Alert Triggered" ] , [ "/" ])
piechart_col( col_name , names = [ "Alert triggered" , "Alert not triggered" ])

# Attack Signature
col_name = "Attack Signature"
print( df[ "Attack Signature" ].value_counts())
"""
    Pattern A = 1
    Pattern B = 0
"""
df = catvar_mapping( col_name , [ "Known Pattern A" ] , [ "patA" ])
piechart_col( "Attack Signature patA" , [ "Pattern A" , "Pattern B" ])

# Action taken
col_name = "Action Taken"
print( df[ col_name ].value_counts())
df = catvar_mapping( col_name , [ "Logged" , "Blocked" ])
piechart_col( col_name )

# Severity Level
col_name = "Severity Level"
print( df[ col_name ].value_counts())
"""
    Low = - 1
    Medium = 0
    High = + 1
"""
df.loc[ df[ col_name ] == "Low" , col_name ] = - 1
df.loc[ df[ col_name ] == "Medium" , col_name ] = 0
df.loc[ df[ col_name ] == "High" , col_name ] = + 1
piechart_col( col_name , [ "Low" , "Medium" , "High" ])

# Device Information
col_name = "Device Information"
print( df[ col_name ].value_counts())
# --> to be split into several columns to be atomized [ browser/browser_version (device) ]
def atomization_DeviceInformation( info ) : # need to take process splitting for each case of device
    print( info )
    i1 , i2 = info.split(" (")
    i2 , i3 = i2.split( ")")
    i10 , i11 = i1.split( "/" )
    i2 = i2.split( "; " )
    i3 = i3.split()
    return pd.Series([ i10 , i11 , i2 , i3 ])
info = df.loc[ random.randint( 0 , df.shape[ 0 ]) , "Device Information" ]

# Network Segment
col_name = "Network Segment"
print( df[ col_name ].value_counts())
"""
    segA = { 1 if "Segment A" ; 0 otherwise }
    segB = { 1 if "Segment B" ; 0 otherwise }
    "Segment C" = [ 0 , 0 ]
"""
df = catvar_mapping( col_name , [ "Segment A" , "Segment B" ] , [ "segA" , "segB" ]) 
piechart_col( col_name )

# Geo-location Data
col_name = "Geo-location Data"
print( df[ col_name ].value_counts())
col = df.columns.get_loc( col_name )
df.insert( col + 1 , "Geo-location City" , value = pd.NA )
df.insert( col + 2 , "Geo-location State" , value = pd.NA )
def geolocation_data( info ) :
    city , state = info.split( ", " )
    return pd.Series([ city , state ])
df[[ "Geo-location City" , "Geo-location State" ]] = df[ "Geo-location Data" ].apply( lambda x : geolocation_data( x ))

# Firewall Logs
col_name = "Firewall Logs"
print( df[ col_name ].value_counts())
"""
    Log Data = 1
    pd.NA = 0
"""
df = catvar_mapping( col_name , [ "Log Data" ] , [ "/" ])
# piechart_col( col_name , [ "Log Data" , "No Log Data" ])

# IDS/IPS Alerts
col_name = "IDS/IPS Alerts"
print( df[ col_name ].value_counts())
"""
    Alert Data = 1
    pd.NA = 0
"""
bully = df[ col_name ] == "Alert Data"
df = catvar_mapping( col_name , [ "Alert Data" ] , [ "/" ])
# piechart_col( col_name , [ "Alert Data" , "No Alert Data" ])

# Log Source
col_name = "Log Source"
print( df[ col_name ].value_counts())
"""
    Firewall = 1
    Server = 0
"""
df = catvar_mapping( col_name , [ "Firewall" ] , [ "Firewall" ])
# piechart_col( col_name , [ "Firewall" , "Server" ])
crosstab_col( "Log Source Firewall" , "Firewall Logs" , "logsource" , "firewallogs" )


#%% general crosstable

crosstabs[ "general" ] = pd.crosstab( 
    index = [ 
        df[ "Source Port ephemeral" ] , 
        df[ "Destination Port ephemeral"] ,
        df[ "Protocol" ] ,
        df[ "Packet Type Control" ] ,
        df[ "Traffic Type" ] ,
        df[ "Malware Indicators" ] ,
        df[ "Alert Trigger" ] ,
        df[ "Attack Signature patA" ] ,
        df[ "Action Taken" ] ,
        df[ "Severity Level" ] ,
        df[ "Network Segment" ] ,
        df[ "Firewall Logs" ] ,
        df[ "IDS/IPS Alerts" ] ,
        df[ "Log Source Firewall" ]
    ] , 
    columns = df[ "Attack Type" ])


#%% SARIMA analysis on Attack type

Attacks_pday = df.copy( deep = True )
Attacks_pday[ "date_dd" ] = Attacks_pday[ "date" ].dt.floor( "d" )
Attacks_pday = Attacks_pday.groupby([ "date_dd" , "Attack Type" ]).size().unstack().iloc[ 1 : - 1 , ]

# plot n attacks per day
fig = subp(
    rows = 3 ,
    cols = 1 ,
    subplot_titles = (
        "Malware" ,
        "Intrusion" ,
        "DDos" ,
        )
    )
fig.add_trace(
    go.Scatter(
        x = Attacks_pday.index ,
        y = Attacks_pday[ "Malware" ] ,
    ) ,
    row = 1 ,
    col = 1 ,
    )
fig.add_trace(
    go.Scatter(
        x = Attacks_pday.index ,
        y = Attacks_pday[ "Intrusion" ] ,
    ) ,
    row = 2 ,
    col = 1 ,
    )
fig.add_trace(
    go.Scatter(
        x = Attacks_pday.index ,
        y = Attacks_pday[ "DDoS" ] ,
    ) ,
    row = 3 ,
    col = 1 ,
    )

# plot ACF & PACF

fig = subp(
    rows = 3 ,
    cols = 2 ,
    subplot_titles = (
        "Malware ACF" ,
        "Malware PACF" ,
        "Intrusion ACF" ,
        "Intrusion PACF" ,
        "DDoS ACF" ,
        "DDoS PACF" ,
        )
    )
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go

Attacktype_TSanalysis = {}
nlags = 100
for i , attacktype in enumerate([ "Malware" , "Intrustion" , "DDoS" ]) :
    Attacktype_TSanalysis[ f"{ attacktype }_ACF" ] = acf( Attacks_pday[ "Malware" ] ,  nlags = nlags )
    Attacktype_TSanalysis[ f"{ attacktype }_PACF" ] = pacf( Attacks_pday[ "Malware" ] ,  nlags = nlags )
    fig.add_trace(
        go.Scatter(
            x = list( range( 0 , nlags + 1 )) , 
            y = Attacktype_TSanalysis[ f"{ attacktype }_ACF" ] ,
            mode = "lines" ,
            name = "ACF" ,
        ) ,
        row = i + 1 ,
        col = 1 ,
        )
    fig.add_trace(
        go.Scatter(
            x = list( range( 0 , nlags + 1 )) , 
            y = Attacktype_TSanalysis[ f"{ attacktype }_PACF" ] ,
            mode = "lines" ,
            name = "PACF" ,
        ) ,
        row = i + 1,
        col = 2 ,
        )
fig.show()
