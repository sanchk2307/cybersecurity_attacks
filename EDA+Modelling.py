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

#%% Utility functions
# =============================================================================

def catvar_corr( col , target = "Attack Type" ) :
    """
        Calculate Matthews correlation coefficient between a feature and target.
    """
    
    corr = matthews_corrcoef( df[ target ], df[ col ].astype( str ))
    print( f"phi corr between { target } and { col } = { corr : +.4f }" )

def catvar_mapping( col_name , values , name = None ) : 
    """
    Transform categorical variables to binary { 0 , 1 } indicator columns

    This function performs one-hot style encoding for specified categorical values ,
    creating new binary columns that indicate presence/absence of each category .

    Parameters :
    ------------
    col_name : str
        Name of the categorical column to transform
    values : list
        List of categorical values to create binary indicators for
    name : list, optional
        Custom names for the new binary columns . If None , uses the values .
        Use [ "/" ] to replace the original column in-place .

    Returns :
    ---------
    DataFrame
        Copy of the dataframe with new binary columns added

    Example :
    ---------
    # Create "Protocol UDP" and "Protocol ICMP" binary columns
    df = catvar_mapping( "Protocol", [ "UDP" , "ICMP" ])
    """
    df1 = df.copy( deep = True )
    if name is None :
        name = values
    elif ( len( name ) == 1 ) and ( name != [ "/" ]) :
        col_target = f"{ col_name } { name[ 0 ]}"
        df1 = df1.rename( columns = { col_name : col_target })
        col_name = col_target
        name = [ col_target ]
    col = df1.columns.get_loc( col_name ) + 1
    print( "\nNew columns created :" )
    for val , nm in zip( values , name ) :
        if ( nm == "/" ) :
            col_target = col_name
        elif ( len( name ) == 1 ) :
            col_target = nm
        else :
            col_target = f"{ col_name } { nm }"
            if col_target not in df1.columns :
                df1.insert( col , col_target , value = pd.NA )
        bully = df1[ col_name ] == val
        df1.loc[ bully , col_target ] = 1
        df1.loc[ ~ bully , col_target ] = 0
        col += 1
        print( col_target )
    return df1
        
def piechart_col( df , col , names = None ) :
    """
    Generate an interactive pie chart for a categorical column .

    Parameters :
    ------------
    col : str
        Column name to visualize
    names : list , optional
        Custom labels for pie chart segments . If None , uses actual values .
    """
    if names is None :
        fig = px.pie( 
            values = df[ col ].value_counts( dropna = False ) ,
            names = df[ col ].value_counts( dropna = False  ).index ,
            title = f"Distribution of { col }" ,
            )
        fig.update_traces(
            textposition = "inside" ,
            textinfo = "percent + label" ,
            hovertemplate = "<b>%{label}</b><br>Count : %{value}<br>Percentage : %{percent}<extra></extra>"
            )
        fig.show()
    else :
        fig = px.pie( 
            values = df[ col ].value_counts() ,
            names = names ,
            title = f"Distribution of {col}" ,
            )
        fig.update_traces(
            textposition = "inside" ,
            textinfo = "percent + label" ,
            hovertemplate = "<b>%{label}</b><br>Count : %{value}<br>Percentage : %{percent}<extra></extra>"
            )
        fig.show()

#%% Utility functions : data preparation ______________________________________

def ip_to_coords( df , ip_address ) :
    """
    Convert an IP address to geographic coordinates and location info .

    Uses MaxMind GeoIP2 database to lookup geographic information for IP addresses .

    Parameters :
    ------------
    ip_address : str
        IPv4 or IPv6 address to geolocate

    Returns :
    ---------
    pd.Series
        Series containing [latitude, longitude, country_name, city]
        Returns NA values for any fields that cannot be resolved
    """
    print( f"\n{ip_address}" )
    ret = pd.Series( dtype = object )
    try :
        print("trying lat lon")
        res = geoIP.geos( ip_address ).wkt
        lon , lat = res.replace( "(" , "" ).replace( ")" , "" ).split()[ 1 : ]
        ret = pd.concat([ 
            ret , 
            pd.Series([ 
                lat , 
                lon 
                ])
            ] , 
            ignore_index = True 
            )
    except :
        ret = pd.concat([ 
            ret , 
            pd.Series([ 
                pd.NA , 
                pd.NA 
                ])
            ] , 
            ignore_index = True 
            )
    try :
        print("trying cit count")
        res = geoIP.city( ip_address )
        ret = pd.concat([ 
            ret , 
            pd.Series([ 
                res[ "country_name" ] ,
                res[ "city" ]]
                )] , 
            ignore_index = True 
            )
    except :
        ret = pd.concat([ 
            ret , 
            pd.Series([ 
                pd.NA ,
                pd.NA ])
            ] ,
            ignore_index = True 
            )
    print(f"ret = {ret}")
    return ret
    
def geolocation_data( info ) :
    """Split 'City, State' string into separate components."""
    city , state = info.split( ", " )
    return pd.Series([ city , state ])
    
def extract_ipv4_prefix( SourDest , n_octets ) :
    """
    Extract the first n octets from an IPv4 column.

    Parameters
    ----------
    SourDest : str
        Name of the IPv4 column
    n_octets : int
        Number of octets to keep (1, 2, or 3)

    Returns
    -------
    pandas.Series
        Series containing the IPv4 prefix
    """

    if n_octets not in ( 1 , 2 , 3 ) :
        raise ValueError("n_octets must be 1, 2, or 3")

    return df[ f"{SourDest} IP Address" ].str.split( "." ).str[ : n_octets ].str.join( "." )

def Device_type( ua_string ) :
    """
    Classify device type from User-Agent string .
    Returns :
        str : "Mobile" , "Tablet" , "PC" , or pd.NA if pd.NA
    """
    try :
        if not ua_string or pd.isna( ua_string ) :
            return pd.NA
        ua = ua_parse( ua_string )
        if getattr( ua , "is_mobile" , False ) :
            return "Mobile"
        if getattr( ua , "is_tablet" , False ) :
            return "Tablet"
        if getattr( ua , "is_pc", False ) :
            return "PC"
        return pd.NA # replaced "Unknown" with pd.NA
    except :
        return pd.NA # replaced "Unknown" with pd.NA
    
#%% Utility functions : diagrams ______________________________________________

def sankey_diag_IPs( df , ntop ) :
    """
    Create a Sankey diagram showing traffic flow between Source , Proxy , and Destination countries .

    This visualization shows how network traffic flows geographically :
    - Left nodes : Source IP countries ( where traffic originates )
    - Middle nodes : Proxy countries ( intermediate routing points )
    - Right nodes : Destination IP countries ( final targets )

    Parameters :
    ------------
    ntop : int
        Number of top countries to display ( others grouped as "other" )

    Returns :
    ---------
    DataFrame
        Aggregated IP flow counts grouped by source, proxy, and destination
    """
    
    IPs_col = {}
    labels = pd.Series( dtype = "string" )
    
    # process each IP type: Source (SIP), Destination (DIP), Proxy (PIP)
    for IPid , dfcol in zip([ "SIP" , "DIP" , "PIP" ] , [ "Source IP country" , "Destination IP country" , "Proxy country" ]) :
        IPs_col[ IPid ] = df[ dfcol ].copy( deep = True )
        IPs_col[ f"{ IPid }labs" ] = pd.Series( IPs_col[ IPid ].value_counts().index[ : ntop ])
        bully = ( IPs_col[ IPid ].isin( IPs_col[ f"{IPid}labs" ]) | IPs_col[ IPid ].isna())
        IPs_col[ IPid ].loc[ ~ bully ] = "other"
        IPs_col[ f"{ IPid }labs" ] = f"{ IPid } " + pd.concat([ IPs_col[ f"{ IPid }labs" ] , pd.Series([ "other" ])])
        labels = pd.concat([ labels , IPs_col[ f"{ IPid }labs" ]])
    labels = list( labels.reset_index( drop = True ))
    
    # aggregate traffic flow between countries
    aggregIPs = pd.DataFrame({
        "SIP" : IPs_col[ "SIP" ] ,
        "PIP" : IPs_col[ "PIP" ] ,
        "DIP" : IPs_col[ "DIP" ] ,
        })
    aggregIPs = aggregIPs.groupby( by = [ 
        "SIP" , 
        "PIP" , 
        "DIP"
        ]).size().to_frame( "count" )

    # build Sankey diag links
    source = []
    target = []
    value = []
    nlvl = aggregIPs.index.nlevels
    for idx , row in aggregIPs.iterrows() :
        row_labs = []
        if ( nlvl == 1 ) :
            row_labs.append( f"{ aggregIPs.index.name } { idx }" )
        else :
            for i , val in enumerate( idx ) :
                row_labs.append( f"{ aggregIPs.index.names[ i ] } { val }" )
        for i in range( 0 , nlvl - 1 ) :
            source.append( labels.index( row_labs[ i ]))
            target.append( labels.index( row_labs[ i + 1 ]))
            value.append( row.item())
    
    # create Sankey diagram with Inferno color scale
    n = len( labels )
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno ,
        [ i / ( n - 1 ) for i in range( n )]
        ) 
    fig = go.Figure( data = [ go.Sankey(
        node = { 
          "pad" : 15 ,
          "thickness" : 20 ,
          "line" : { 
              "color" : "rgba( 0 , 0 , 0 , 0.1 )" ,
              "width" : 0.5 
              } ,
          "label" : labels ,
          "color" : colors ,
          } ,
        link = {
          "source" : source ,
          "target" : target ,
          "value" : value 
          }
        )])
    fig.update_layout(
        title_text = "Network Traffic Flow : Source → Proxy → Destination Countries" ,
        # font_family = "Courier New" ,
        # font_color = "blue" , 
        title_font_size = 18 ,
        font_size = 14 ,
        title_font_family = "Avenir" ,
        title_font_color = "black" ,
        annotations = [
            {
                "x" : 0.01 ,
                "y" : 1.05, 
                "text" : "<b>Source countries</b>" , 
                "showarrow" : False , 
                "font_size" : 12
                } ,
            {
                "x" : 0.5 ,
                "y" : 1.05, 
                "text" : "<b>Source countries</b>" , 
                "showarrow" : False , 
                "font_size" : 12
                } ,
            {
                "x" : 0.99 ,
                "y" : 1.05, 
                "text" : "<b>Destination countries</b>" , 
                "showarrow" : False , 
                "font_size" : 12
                }
            ]
        )
    fig.show()
    
    return aggregIPs

def sankey_diag( cols_bully = [
        True , # "Source IP country"
        True , # "Destination IP country"
        True , # "Source Port ephemeral"
        True , # "Destination Port ephemeral"
        True , # "Protocol"
        True , # "Packet Type Control"
        True , # "Traffic Type"
        True , # "Malware Indicators"
        True , # "Alert Trigger"
        True , # "Attack Signature patA"
        True , # "Action Taken"
        True , # "Severity Level"
        True , # "Browser family" ,
        True , # "Browser major" ,
        True , # "Browser minor" ,
        True , # "OS family" ,
        True , # "OS major" ,
        True , # "OS minor" ,
        True , # "OS patch" ,
        True , # "Device family" ,
        True , # "Device brand" ,
        True , # "Device type" ,
        True , # "Device bot",
        True , # "Network Segment"
        True , # "Firewall Logs"
        True , # "IDS/IPS Alerts"
        True , # "Log Source Firewall"
        ] , ntop = 10 ) :
    
    """
    Generate a Sankey diagram showing feature value flows to attack types .

    Parameters :
    ------------
    cols_bully : list of bool
        Boolean flags indicating which columns to include in the diagram
    ntop : int
        Number of top countries to show ( others grouped as "other" )

    Returns :
    ---------
    DataFrame
        Cross-tabulation percentages used to build the diagram
    """
    
    cols = np.array([
        "Source IP country" ,
        "Destination IP country" ,
        "Source Port ephemeral" ,
        "Destination Port ephemeral" ,
        "Protocol" , 
        "Packet Type Control" ,
        "Traffic Type" ,
        "Malware Indicators" ,
        "Alert Trigger" ,
        "Attack Signature patA" ,
        "Action Taken" ,
        "Severity Level" ,
        "Browser family" ,
        "Browser major" ,
        "Browser minor" ,
        "OS family" ,
        "OS major" ,
        "OS minor" ,
        "OS patch" ,
        "Device family" ,
        "Device brand" ,
        "Device type" ,
        "Device bot",
        "Network Segment" ,
        "Firewall Logs" ,
        "IDS/IPS Alerts" ,
        "Log Source Firewall"
        ])
    cols = cols[ np.array( cols_bully )]
    
    idx_ct =  []
    labels = []
    if "Source IP country" in cols :
        SIP = df[ "Source IP country" ].copy( deep = True )
        SIP.name = "SIP"
        SIPlabs = pd.Series( SIP.value_counts().index[ : ntop ])
        bully = ( SIP.isin( SIPlabs ) | SIP.isna())
        SIP.loc[ ~ bully ] = "other"
        SIPlabs = "SIP " + pd.concat([ SIPlabs , pd.Series([ "other" ])])
        labels.extend( SIPlabs.to_list())
        idx_ct = idx_ct + [ SIP ]
        cols = cols[ cols != "Source IP country" ]
    if "Destination IP country" in cols :
        DIP = df[ "Destination IP country" ].copy( deep = True )
        DIP.name = "DIP"
        DIPlabs = pd.Series( DIP.value_counts().index[ : ntop ])
        bully = ( DIP.isin( DIPlabs ) | DIP.isna())
        DIP.loc[ ~ bully ] = "other"
        DIPlabs = "DIP " + pd.concat([ DIPlabs , pd.Series([ "other" ])])
        labels.extend( DIPlabs.to_list())
        idx_ct = idx_ct + [ DIP ]
        cols = cols[ cols != "Destination IP country" ]
        
    # build cross table with Attack Type in columns and multi-index of variables in index
    idx_ct = idx_ct + [ df[ col ] for col in cols ]
    print( idx_ct )
    crosstabs_x_AttackType = pd.crosstab( 
        index = idx_ct ,
        columns = df[ "Attack Type" ]
        )
    
    # compute labels
    for c in np.append( cols , "Attack Type" ) :
        vals = df[ c ].unique()
        for v in vals :
            labels.append( f"{ c } { v }" )
    # computation of source , target , value
    source = []
    target = []
    value = []
    nlvl = crosstabs_x_AttackType.index.nlevels
    for idx , row in crosstabs_x_AttackType.iterrows()  :
        row_labs = []
        if ( nlvl == 1 ) :
            row_labs.append( f"{ crosstabs_x_AttackType.index.name } { idx }" )
        else :
            for i , val in enumerate( idx ) :
                row_labs.append( f"{ crosstabs_x_AttackType.index.names[ i ] } { val }" )
        for attype in crosstabs_x_AttackType.columns :
            val = row[ attype ]
            for i in range( 0 , nlvl - 1 ) :
                source.append( labels.index( row_labs[ i ]))
                target.append( labels.index( row_labs[ i + 1 ]))
                value.append( val )
            source.append( labels.index( row_labs[ - 1 ]))
            target.append( labels.index( f"Attack Type { attype }" ))
            value.append( val )
    
    # plot the sankey diagram
    n = len( labels )
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno ,
        [ i / ( n - 1 ) for i in range( n )]
        ) 
    fig = go.Figure( data = [ go.Sankey(
        node = dict(
          pad = 15 ,
          thickness = 20 ,
          line = dict( 
              color = "rgba( 0 , 0 , 0 , 0.1 )" ,
              width = 0.5 
              ) ,
          label = labels ,
          color = colors ,
          ) ,
        link = dict(
          source = source ,
          target = target ,
          value = value 
          ))])
    fig.update_layout(
        title_text = "Sankey Diagram" ,
        # font_family = "Courier New" ,
        # font_color = "blue" , 
        font_size = 20 ,
        title_font_family = "Avenir" ,
        title_font_color = "black",
        )
    fig.show()
    
    crosstabs_x_AttackType = crosstabs_x_AttackType / crosstabs_x_AttackType.sum().sum() * 100
    
    return crosstabs_x_AttackType

def paracat_diag( cols_bully = [
        True , # "Source IP country"
        True , # "Destination IP country"
        True , # "Source Port ephemeral"
        True , # "Destination Port ephemeral"
        True , # "Protocol"
        True , # "Packet Type Control"
        True , # "Traffic Type"
        True , # "Malware Indicators"
        True , # "Alert Trigger"
        True , # "Attack Signature patA"
        True , # "Action Taken"
        True , # "Severity Level"
        True , # "Network Segment"
        True , # "Firewall Logs"
        True , # "IDS/IPS Alerts"
        True , # "Log Source Firewall"
        True , # "Attack Type max n d"
        True , # "Protocol max n d"
        True , # "Traffic Type max n d"
        True , # "Alert Trigger max n d"
        True , # "Firewall Logs max n d"
        True , # "Attack Signature patA max n d"
        True , # "Action Taken max n d"
        True , # "Browser family max n d"
        ] , 
        colorvar = "Attack Type" ,
        ntop = 10 ,
        ) :
    
    cols = np.array([
        "Source IP country" ,
        "Destination IP country" ,
        "Source Port ephemeral" ,
        "Destination Port ephemeral" ,
        "Protocol" , 
        "Packet Type Control" ,
        "Traffic Type" ,
        "Malware Indicators" ,
        "Alert Trigger" ,
        "Attack Signature patA" ,
        "Action Taken" ,
        "Severity Level" ,
        "Network Segment" ,
        "Firewall Logs" ,
        "IDS/IPS Alerts" ,
        "Log Source Firewall" ,
        "Attack Type max n d" ,
        "Protocol max n d" ,
        "Traffic Type max n d" ,
        "Alert Trigger max n d" ,
        "Firewall Logs max n d" ,
        "Attack Signature patA max n d" ,
        "Action Taken max n d" ,
        "Browser family max n d"
        ])
    cols = cols[ np.array( cols_bully )]
    
    dims_var = {}
    if "Source IP country" in cols :
        SIP = df[ "Source IP country" ].copy( deep = True )
        SIP.name = "SIP"
        SIPlabs = pd.Series( SIP.value_counts().index[ : ntop ])
        bully = ( SIP.isin( SIPlabs ) | SIP.isna())
        SIP.loc[ ~ bully ] = "other"
        SIPlabs = "SIP " + pd.concat([ SIPlabs , pd.Series([ "other" ])])
        cols = cols[ cols != "Source IP country" ]
        dims_var[ "SIP" ] = go.parcats.Dimension(
            values = SIP ,
            # categoryorder = "category ascending" ,
            label = "Source IP country"
            )
        cols = np.append( cols , "SIP" )
    if "Destination IP country" in cols :
        DIP = df[ "Destination IP country" ].copy( deep = True )
        DIP.name = "DIP"
        DIPlabs = pd.Series( DIP.value_counts().index[ : ntop ])
        bully = ( DIP.isin( DIPlabs ) | DIP.isna())
        DIP.loc[ ~ bully ] = "other"
        DIPlabs = "DIP " + pd.concat([ DIPlabs , pd.Series([ "other" ])])
        cols = cols[ cols != "Destination IP country" ]
        dims_var[ "DIP" ] = go.parcats.Dimension(
            values = DIP ,
            # categoryorder = "category ascending" ,
            label = "Destination IP country"
            )
        cols = np.append( cols , "DIP" )
    for col in cols[( cols != "SIP" ) & ( cols != "DIP" )] :
        print( col )
        dims_var[ col ] = go.parcats.Dimension(
            values = df[ col ] ,
            # categoryorder = "category ascending" ,
            label = col
            )
    
    dim_attypes = go.parcats.Dimension(
        values = df[ "Attack Type" ], 
        label = "Attack Type", 
        # categoryarray = [ "Malware" , "Intrusion" , "DDoS" ] ,
        ticktext = [ "Malware" , "Intrusion" , "DDoS" ]
        )
    
    dims = [ dims_var[ col ] for col in cols ]
    dims.append( dim_attypes )
    
    if colorvar == "SIP" :
        colorvar = SIP
    elif colorvar == "DIP" :
        colorvar = DIP
    elif ( colorvar in cols ) or ( colorvar == "Attack Type" ) :
        colorvar = df[ colorvar ]
    else :
        ValueError( "colorvar must be in cols" )
    catcolor = colorvar.unique()
    catcolor_n = catcolor.shape[ 0 ]
    color = colorvar.map({ cat : i for i , cat in enumerate( catcolor )})
    positions = [ i / ( catcolor_n - 1 ) if ( catcolor_n > 1 ) else 0 for i in range( 0 , catcolor_n )]
    palette = px.colors.sequential.Viridis
    colors = [ px.colors.sample_colorscale( palette , p )[ 0 ] for p in positions ]
    colorscale = [[ positions[ i ] , colors[ i ]] for i in range( 0 , catcolor_n )]
    
    fig = go.Figure( data = [ go.Parcats( 
        dimensions = dims ,
        line = { 
            "color" : color ,
            "colorscale" : colorscale ,
            "shape" : "hspline" ,
            } ,
        hoveron = "color" ,
        hoverinfo = "count+probability" ,
        labelfont = { "size" : 18 , "family" : "Times" } ,
        tickfont = { "size" : 16 , "family" : "Times" } ,
        arrangement = "freeform" ,
        )])
    fig.show()

#%% Utility functions : feature engineering ___________________________________

def dynamic_threshold( row , attype , dynamic_threshold_pars ) :
    """
    Determine whether a row satisfies a dynamically computed threshold.

    Parameters
    ----------
    row : pandas.Series
        A dataframe row containing at least:
        - "n obs" : number of observations
        - attype : value to compare against threshold
    attype : str
        Column name representing the metric to evaluate.
    dynamic_threshold_pars : tuple
        ( n1 , d1 , n2 , d2 )
        Two reference points defining a linear threshold:
            at n_obs = n1 → threshold = d1
            at n_obs = n2 → threshold = d2

    Returns
    -------
    bool
        True if row[ attype ] ≥ computed threshold , else False.
    """
    n1 , d1 , n2 , d2 = dynamic_threshold_pars
    # compute parameters a and b of the linear threshold function
    a = ( d2 - d1 ) / ( n2 - n1 )
    b = d1 - a * n1
    # compute threshold value for current row using
    threshold = a * row[ "n obs" ] + b
    # compare actual value against computed threshold
    bully = row[ attype ] >= threshold
    return bully

def contvar_classes_builder_byval( df , col , n_classes ) :
    """
    Create value-based bins for a continuous variable.

    The column is split into equal-width intervals between
    floor( min ) and ceil( max ).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column to bin.
    col : str
        Name of continuous variable to discretize.
    n_classes : int
        Number of equal-width bins.

    Returns
    -------
    str
        Name of the newly created class column.
    """
    
    colclass_name = f"{ col } classes"
    colmax = math.ceil( df_train[ col ].max())
    colmin = math.floor( df_train[ col ].min())
    stp = int(( colmax - colmin ) / n_classes )
    for class_bound_low in range( colmin , colmax , stp ) :
        class_bound_up = class_bound_low + stp
        if ( class_bound_low == colmin ) :
            row_target = df[ df[ col ] < class_bound_up ].index
            df.loc[ row_target , colclass_name ] = f"class : min - { class_bound_up }"
        elif ( class_bound_up >= colmax ) :
            row_target = df[ df[ col ] >= class_bound_low ].index
            df.loc[ row_target , colclass_name ] = f"class : { class_bound_low } - max"
        else :
            row_target = df[( df[ col ] >= class_bound_low ) & ( df[ col ] < class_bound_up )].index
            df.loc[ row_target , colclass_name ] = f"class : { class_bound_low } - { class_bound_up }"
    return colclass_name
def contvar_classes_builder_bycount_bynobs( df , col , n_obs ) :
    """
    Create bins for a continuous variable such that each bin
    contains at least n_obs observations ( except possibly last ).

    Bins are formed on sorted values and only split when:
        - value changes
        - current bin size ≥ n_obs

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing column to discretize.
    col : str
        Continuous variable to bin.
    n_obs : int
        Minimum number of observations per bin.

    Returns
    -------
    str
        Name of the newly created class column.
    """
    
    # name of the new class column
    colclass_name = f"{ col } classes"

    # sort column values in ascending order
    target_col = df[[ col ]].sort_values( by = col )[ col ]
    
    current_indices = []    # row indices in current bin
    current_values = []     # corresponding values
    val_prev = None         # previous value ( for detecting change )

    # iterate through sorted values
    for idx , val in target_col.items() :
        if (
            val_prev is not None
            and val != val_prev
            and len( current_indices ) >= n_obs
        ) : # if ( value changed ) and ( we already reached minimum n_obs ) then ( close current bin and start a new one )
            # bounds of completed bin
            class_min = min( current_values )
            class_max = max( current_values )
            # create class label
            class_label = f"class : { class_min } - { class_max }"
            # assign label to corresponding rows
            df.loc[ current_indices , colclass_name ] = class_label
            # reset bin containers
            current_indices = []
            current_values = []
        
        # add current observation to active bin
        current_indices.append( idx )
        current_values.append( val )
        val_prev = val
        
    # after loop , assign remaining observations to final bin
    if current_indices :
        class_min = min( current_values )
        class_max = max( current_values )
        class_label = f"class : { class_min } - { class_max }"
        df.loc[ current_indices , colclass_name ] = class_label
        
    return colclass_name
def contvar_classes_builder_bycount_bynclasses( df , col , n_classes ) :
    colclass_name = ""
    return colclass_name

def crosstab_col( df , cols ) :
    """
    Create a normalized cross-tabulation between a categorical columns and "Attack Type"

    Parameters :
    ------------
    col : list
        List of column names for the categorical variables
    """
    
    if ( len( cols ) == 1 ) :
        crosstab_name = cols[ 0 ]
    else :
        crosstab_name = "{ " + " , ".join( cols ) + " }"
        
    attypes = df[ "Attack Type" ].unique()
    crosstabs_x_AttackType[ crosstab_name ] = pd.crosstab( 
        index = [ df[ col ] for col in cols ] , 
        columns = df[ "Attack Type" ] , 
        margins = True ,
        margins_name = "n obs" ,
        normalize = False 
        )
    crosstabs_x_AttackType[ crosstab_name ] = crosstabs_x_AttackType[ crosstab_name ].iloc[ : - 1 , ]
    for attype in attypes :
        crosstabs_x_AttackType[ crosstab_name ][ attype ] = crosstabs_x_AttackType[ crosstab_name ][ attype ] / crosstabs_x_AttackType[ crosstab_name ][ "n obs" ] * 100
    crosstabs_x_AttackType[ crosstab_name ][ "% obs" ] = crosstabs_x_AttackType[ crosstab_name ][ "n obs" ] / df.shape[ 1 ]
    
    return crosstab_name

def df_bias_classes( df , crosstab_name , threshold , colclass_names , dynamic_threshold_pars ) :
    """
    Create bias classification columns based on filtered crosstab results.

    For each unique "Attack Type" value in df :
        1. Select the corresponding crosstab of the dependent variable of interest
        2. Filter rows using a dynamic threshold condition
        3. Create a bias column assigning ordinal class levels depending on predefined percentage thresholds

    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe containing:
            - "Attack Type"
            - class columns listed in colclass_names
    crosstab_name : str
        Key used to select the appropriate crosstab from
        global dictionary `crosstabs_x_AttackType`.
    threshold : numeric
        Base threshold included in the predefined threshold set.
    colclass_names : list of str
        Column name(s) used to match class labels.
        Can represent a single column or multiple columns ( MultiIndex case ).
    dynamic_threshold_pars : tuple
        Parameters passed to dynamic_threshold().

    Returns
    -------
    list
        Names of all generated bias columns.
    """
    
    # extract all distinct attack types present in dataframe
    attypes = df[ "Attack Type" ].unique()
    X_col = []   # store names of generated bias columns
    
    # iterate over each attack type
    for attype in attypes :
        target_crosstab = crosstabs_x_AttackType[ crosstab_name ]   # retrieve corresponding crosstab
        
        target_crosstab = target_crosstab[ target_crosstab.apply( lambda row : dynamic_threshold( row , attype , dynamic_threshold_pars ) , axis = 1 )] # filter rows using dynamic threshold logic ( only keep rows satisfying dynamic_threshold condition )
        bias_col_name = f"{ crosstab_name } , bias { attype }"  # create name for bias column specific to this attack type
        X_col.append( bias_col_name )
        df[ bias_col_name ] = 0 # initialize bias column with default class 0
        
        # iterate through predefined threshold classes
        for p_class , threshold_class in enumerate({ threshold , 40 , 50 , 60 , 75 , 90 }) :
            p_class = p_class + 1   # convert enumerate index to 1-based class index
            bias = target_crosstab[ attype ]    # extract bias values for current attack type
            bias = bias[ bias >= threshold_class ].index    # keep only classes above current threshold
            
            if not isinstance( bias , pd.MultiIndex ) : # if index is single-level , then assign class number to matching rows in df
                df.loc[ df[ df[ colclass_names ].isin( bias )].index , bias_col_name ] = p_class
            else :  # if index is MultiIndex , then :
                midx = list( zip( * ( df[ col ] for col in colclass_names ))) # build tuple representation of class columns
                midx = pd.Series( midx ).isin( bias )   # identify rows matching MultiIndex entries
                df.loc[ midx , bias_col_name ] = p_class    # assign class number
    return X_col

#%% Utility functions : modelling _____________________________________________

def logit( attypes , x_train , x_test , y_train , y_test ) :
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    attypes : array-like
        Attack type labels ( not used inside this function ).
    x_train : array-like
        Training feature matrix.
    x_test : array-like
        Test feature matrix ( not used inside this function ).
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels ( not used inside this function ).

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Fitted Logistic Regression model.
    """
    
    model = skl.linear_model.LogisticRegression(
        solver = "lbfgs" ,
        max_iter = 1000
    )
    model.fit( x_train , y_train )
    
    return model

def randomforrest( attypes , x_train , x_test , y_train , y_test ) :
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    attypes : array-like
        Attack type labels ( not used inside this function ).
    x_train : array-like
        Training feature matrix.
    x_test : array-like
        Test feature matrix ( not used inside this function ).
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels ( not used inside this function ).

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Fitted Random Forest model.
    """
    
    model = skl.ensemble.RandomForestClassifier(
        n_estimators = 100 ,
        max_depth = None ,
        random_state = 1 ,
        n_jobs = - 1
        )
    model.fit( x_train , y_train )
    return model

def model_predictions( model , x_test ) :
    """
    Generate predictions and class probabilities from a trained model.

    Parameters
    ----------
    model : sklearn-like estimator
        Fitted classification model implementing:
            - predict()
            - predict_proba()
    x_test : array-like
        Test feature matrix used for inference.

    Returns
    -------
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted class probabilities.
        Shape typically:
            ( n_samples , n_classes )
    """
    
    y_pred = model.predict( x_test )
    y_prob = model.predict_proba( x_test )
    
    return y_pred , y_prob

def model_metrics( attypes , model , x_train , x_test , y_train , y_test , y_pred , y_prob ) :
    """
    Compute and visualize classification metrics.

    This function performs :
        1. Confusion matrix visualization
        2. Accuracy and classification report
        3. Multiclass ROC curves with AUC
        4. Mutual Information feature importance

    Parameters
    ----------
    attypes : list-like
        Ordered class labels used for display and evaluation.
    model : fitted estimator
        Trained classification model ( not directly used here ).
    x_train , x_test : array-like
        Feature matrices.
    y_train , y_test : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted class probabilities.
    """
    
    # confusion matrix ________________________________________________________
    confmat = pd.DataFrame(
        skl.metrics.confusion_matrix( y_test , y_pred ) ,
        index = attypes ,
        columns = attypes
    )
    fig = px.imshow(
        confmat ,
        text_auto = True ,
        color_continuous_scale = "Magma" ,
        title = "Confusion Matrix"
    )
    fig.update_layout(
        xaxis_title = "Predicted Class" ,
        yaxis_title = "True Class"
    )
    fig.show()
    
    # metrics evaluation ______________________________________________________
    accscore = skl.metrics.accuracy_score( y_test , y_pred )
    print( f"Accuracy = { accscore }" )
    print( "\nClassification Report :\n" )
    print( skl.metrics.classification_report( y_test , y_pred , target_names = attypes ))
    
    # ROC curve computation ___________________________________________________
    y_test_bin = label_binarize( y_test , classes = np.arange( len( attypes )))
    n_classes = y_test_bin.shape[ 1 ]
    fig_roc = go.Figure()
    for yclass in range( n_classes ) :
        fpr , tpr , _ = skl.metrics.roc_curve( y_test_bin[ : , yclass ] , y_prob[ : , yclass ])
        roc_auc = skl.metrics.auc( fpr , tpr )
    
        fig_roc.add_trace( go.Scatter(
            x = fpr ,
            y = tpr ,
            mode = "lines" ,
            name = f"{ attypes[ yclass ]} ( AUC = { roc_auc })"
        ))
    ## random baseline
    fig_roc.add_trace(go.Scatter(
        x = [ 0 , 1 ] ,
        y = [ 0 , 1 ] ,
        mode = "lines" ,
        line = dict( dash = "dash" ) ,
        name = "Random"
    ))
    fig_roc.update_layout(
        title = "Multiclass ROC Curve" ,
        xaxis_title = "False Positive Rate" ,
        yaxis_title = "True Positive Rate" ,
        legend_title = "Classes" ,
        template = "plotly_dark"
    )
    fig_roc.show()
    
    # marginal information ____________________________________________________
    marginfo_scores = skl.feature_selection.mutual_info_classif(
        x_train ,
        y_train ,
        discrete_features = "auto" ,
        random_state = 100
    )
    marginfo = pd.Series(
        marginfo_scores ,
        index = x_train.columns
    ).sort_values( ascending = False )
    fig = px.bar(
        marginfo ,
        x = list( marginfo.values ) ,
        y = marginfo.index ,
        orientation = "h" ,
        text = marginfo.values ,
        title = "Marginal Information Scores" ,
        labels = { "x" : "Marginal Information score" , "y" : "Feature" }
        )
    fig.update_traces( 
        texttemplate = "%{x:.4f}" , 
        textposition = "outside" )
    fig.update_layout( 
        yaxis = dict( autorange = "reversed" ) ,
        barcornerradius = 15 ,
        bargap = 0.2 ,
        uniformtext_minsize = 8 ,
        uniformtext_mode = "hide"
        )
    fig.show()

def feature_engineering( 
        features_mask , 
        threshold_floor , 
        contvar_nobs_b_class ,
        dynamic_threshold_pars 
        ) :
    """
    Build engineered bias-based feature columns according to a feature mask.

    The function :
        1. Selects feature groups based on features_mask .
        2. Builds class bins ( for continuous variables when needed ) .
        3. Computes crosstabs for selected column combinations .
        4. Applies df_bias_classes() to generate ordinal bias features .
        5. Returns list of generated feature column names .

    Parameters
    ----------
    features_mask : iterable of bool
        Mask selecting which feature groups to activate .
    threshold_floor : numeric
        Base threshold used inside df_bias_classes() .
    contvar_nobs_b_class : int
        Minimum number of observations per bin when discretizing
        continuous variables.
    dynamic_threshold_pars : tuple
        Parameters passed to dynamic_threshold() .

    Returns
    -------
    list
        Names of all generated engineered feature columns.
    """
    
    # define all available feature groups
    features_existing = np.array([
        "Source Port & Destination Port" ,
        "Source IP latitude/longitude combination & Destination IP latitude/longitude combination" ,
        "Sounce IP Country , Destination IP country combination" ,
        "date dd" ,
        "Network & Traffic" ,
        "Security Response" ,
        "Time decomposition" ,
        "Anomaly Scores" ,
        "Packet Length"
        ])
    features_selected = features_existing[ np.array( features_mask )] # select only activated feature groups
    X_cols = [] # store names of generated columns
    
    # Source Port & Destination Port
    # -------------------------------------------------------------------------
    if "Source Port & Destination Port" in features_selected :
        
        # process both source and destination ports independently
        for SourDest in [ "Source" , "Destination" ] :
            target_col = f"{ SourDest } Port"
            # discretize continuous port values into bins
            colclass_name = contvar_classes_builder_bycount_bynobs( 
                df , 
                target_col , 
                contvar_nobs_b_class 
                )
            # build crosstab on discretized class
            crosstab_name = crosstab_col( df , [ colclass_name ])
            # generate bias features
            X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , crosstab_name , dynamic_threshold_pars ))
            
    # Source IP latitude/longitude combination & Destination IP latitude/longitude combination
    # -------------------------------------------------------------------------
    if "Source IP latitude/longitude combination & Destination IP latitude/longitude combination" in features_selected : 
        for SourDest in [ 
                "Source" , 
                "Destination" 
                ] :
            colclass_names = []
            # discretize longitude and latitude separately
            for lonlat in [ "longitude" , "latitude" ] :
                target_col = f"{ SourDest } IP { lonlat }"
                # colclass_name = classes_builder( target_col , 40 )
                colclass_name = contvar_classes_builder_bycount_bynobs( df , target_col , contvar_nobs_b_class )
                colclass_names.append( colclass_name )
            # extend with categorical security-related variables
            colclass_names.extend([
                # "Protocol" , # <------- these for 
                # "Packet Type" ,
                # "Traffic Type" ,
                # "Attack Signature patA" ,
                # "Network Segment" ,
                # "Proxy usage" ,
                
                "Malware Indicators" ,
                "Alerts/Warnings" ,
                "Action Taken" ,
                "Severity Level" ,
                "Log Source" ,
                "Firewall Logs" ,
                "IDS/IPS Alerts" ,
                ])
            # build crosstab
            crosstab_name = crosstab_col( df , [ col for col in colclass_names ])
            # generate bias features
            X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , colclass_names , dynamic_threshold_pars ))
            
    # Sounce IP Country , Destination IP country combination
    # -------------------------------------------------------------------------
    if "Sounce IP Country , Destination IP country combination" in features_selected :
        colclass_names = [ 
            "Source IP country" , 
            "Destination IP country" ,
            
            # "Protocol" , 
            # "Packet Type" ,
            # "Traffic Type" ,
            # "Attack Signature" ,
            # "Network Segment" ,
            # "Proxy usage" ,
            
            # "Malware Indicators" ,
            # "Alerts/Warnings" ,
            # "Action Taken" ,
            # "Severity Level" ,
            # "Log Source" ,
            # "Firewall Logs" ,
            # "IDS/IPS Alerts" ,
            
            ]
        crosstab_name = crosstab_col( df , [ col for col in colclass_names ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , colclass_names , dynamic_threshold_pars ))
        
    # date dd
    # -------------------------------------------------------------------------
    if "date dd" in features_selected :
        crosstab_name = crosstab_col( df , [ "date dd" ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , crosstab_name , dynamic_threshold_pars ))
        
    # Network & Traffic
    # -------------------------------------------------------------------------
    if "Network & Traffic" in features_selected :
        columns = [
            "Protocol" , 
            "Packet Type" ,
            "Traffic Type" ,
            "Attack Signature" ,
            "Network Segment" ,
            "Proxy usage" ,
            
            "Browser family" ,
            "OS family" ,
            "Device type" ,
            ]
        crosstab_name = crosstab_col( df , [ col for col in columns ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , columns , dynamic_threshold_pars ))
        
    # Security Response
    # -------------------------------------------------------------------------
    if "Security Response" in features_selected : 
        columns = [
            "Malware Indicators" ,
            "Alerts/Warnings" ,
            "Action Taken" ,
            "Severity Level" ,
            "Log Source" ,
            "Firewall Logs" ,
            "IDS/IPS Alerts" ,
            "Severity Level" ,
            
            "Browser family" ,
            "OS family" ,
            "Device type" ,
            ]
        crosstab_name = crosstab_col( df , [ col for col in columns ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , columns , dynamic_threshold_pars ))
        
    # time
    # -------------------------------------------------------------------------
    if "time decomposition" in features_selected :
        columns = { 
            "date MW" , 
            "date WD" , 
            "date H" ,
            "date M" , # <------------ buddy here dropped my accurage by 2 whole percent, turned my marginal info to dust !!!!!!!!!!!
            # "Browser family" ,
            # "OS family" ,
            # "Device type" ,
            # "Malware Indicators" ,
            }
        crosstab_name = crosstab_col( df , [ col for col in columns ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , columns , dynamic_threshold_pars ))
        
    # Anomaly Scores
    # -------------------------------------------------------------------------
    if "Anomaly Scores" in features_selected : 
        colclass_name = contvar_classes_builder_bycount_bynobs( df , "Anomaly Scores" , contvar_nobs_b_class )
        crosstab_name = crosstab_col( df , [ colclass_name ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , crosstab_name , dynamic_threshold_pars ))
        
    # Packet Length
    # -------------------------------------------------------------------------
    if "Packet Length" in features_selected : 
        colclass_name = contvar_classes_builder_bycount_bynobs( df , "Packet Length" , contvar_nobs_b_class )
        # colclass_names = [ colclass_name , "Protocol" ]
        crosstab_name = crosstab_col( df , [ colclass_name ])
        X_cols.extend( df_bias_classes( df , crosstab_name , threshold_floor , crosstab_name , dynamic_threshold_pars ))
        
    return X_cols

def Modelling_PIPELINE( testp , features_mask , threshold_floor , contvar_nobs_b_class , dynamic_threshold_pars , model_type = "logit" , split_before_training = False ) :
    """
    End-to-end modelling pipeline.

    The function :
        1. Optionally performs train / test split before or after feature engineering .
        2. Builds engineered bias-based features .
        3. Encodes target labels .
        4. Trains selected model type .
        5. Computes predictions and evaluation metrics .
        6. Returns test predictions .

    Parameters
    ----------
    testp : float
        Proportion of dataset used for testing ( 0 < testp < 1 ) .
    features_mask : iterable of bool
        Controls which feature groups are activated .
    threshold_floor : numeric
        Base bias threshold passed to feature engineering .
    contvar_nobs_b_class : int
        Minimum number of observations per continuous bin .
    dynamic_threshold_pars : tuple
        Parameters controlling dynamic threshold function .
    model_type : str
        "logit" or "randomforrest" .
    split_before_training : bool
        If False :
            Feature engineering is performed on full dataset
            before train / test split .
        If True :
            Dataset is split first , then features are used
            on train and test subsets .

    Returns
    -------
    y_pred : array-like
        Predicted class labels for test set .
    """
    
    global df
    attypes = df[ "Attack Type" ].unique()
    
    # =========================================================================
    # CASE 1 : SPLIT AFTER FEATURE ENGINEERING
    # =========================================================================
    if split_before_training == False :
        # build engineered features on entire dataset
        X_cols = feature_engineering( 
            features_mask , 
            threshold_floor , 
            contvar_nobs_b_class , 
            dynamic_threshold_pars 
            )
        df = df.copy() # copy dataframe to avoid fragmentation
        
        # separate predictors and target
        x_vars = df[ X_cols ]
        y_var = df[ "Attack Type" ]
        
        # encode categorical target into integers
        y = LabelEncoder()
        y_var = y.fit_transform( y_var )
        
        # stratif tryain / test split
        x_train , x_test , y_train , y_test = skl.model_selection.train_test_split(
            x_vars ,
            y_var ,
            test_size = testp ,
            random_state = 50 ,
            stratify = y_var
        )
    # =========================================================================
    # CASE 2 : SPLIT BEFORE FEATURE ENGINEERING ----- not operational
    # =========================================================================
    else :
        # split raw dataset first
        df_train , df_test = skl.model_selection.train_test_split(
            df ,
            test_size = testp ,
            train_size = ( 1 - testp ) ,
            stratify = df[ "Attack Type" ] ,
            random_state = 42
            )
        
        # build engineered features on entire train and test dataset
        X_cols = feature_engineering( 
            features_mask , 
            threshold_floor , 
            contvar_nobs_b_class , 
            dynamic_threshold_pars 
            ) # df to be modified !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # copy dataframes to avoid fragmentation
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        # extract engineered features from train and test splits
        x_train = df_train[ X_cols ]
        x_test = df_test[ X_cols ]
        
        y = LabelEncoder()
        # encode training labels
        y_train = df_train[ "Attack Type" ]
        y_train = y.fit_transform( y_train )
        # encode test labels
        y_test = df_test[ "Attack Type" ]
        y_test = y.fit_transform( y_test )
    
    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    if model_type == "logit" :
        model = logit( attypes , x_train , x_test , y_train , y_test )
    elif model_type == "randomforrest" :
        model = randomforrest( attypes , x_train , x_test , y_train , y_test )
    else :
        print( "model type is nor recognized" )
    
    # =========================================================================
    # PREDICTION AND EVALUATION
    # =========================================================================
    y_pred , y_prob = model_predictions( model , x_test )
    model_metrics( 
        attypes , 
        model , 
        x_train , 
        x_test , 
        y_train ,
        y_test , 
        y_pred , 
        y_prob 
        )
    
    return y_pred

#%% Exploratory Data Analysis (EDA) & Data Preparation
# =============================================================================

# This section performs initial data exploration :
# - Column renaming for clarity
# - Missing value analysis
# - Target variable distribution ( Attack Type )
# - Temporal distribution of attacks

# Rename columns for better readability and consistency
# - "Timestamp" -> "date" for temporal analysis
# - Port columns renamed to indicate ephemeral port analysis
# - Alerts/Warnings -> Alert Trigger for binary encoding

def EDA_DataPrep_PIPELINE( df ) :
    # renaming columns
    # -------------------------------------------------------------------------
    df = df.rename( columns = { 
        "Timestamp" : "date" ,
        # "Source Port" : "Source Port ephemeral" ,
        # "Destination Port" : "Destination Port ephemeral" ,
        # "Alerts/Warnings" : "Alert Trigger" ,
        })
    df = df.drop([ "User Information" , "Payload Data" ] , axis = 1 )
    
    # display summary statistics for numerical columns
    # -------------------------------------------------------------------------
    print( "Dataset summary statistics" , "-" * 60 , df.describe())
    
    # missing value analysis : identify columns with missing data and their %
    # -------------------------------------------------------------------------
    print( "Missing value analysis" , "-" * 60 )
    df_s0 = df.shape[ 0 ]
    for col in df.columns :
        NA_n = sum( df[ col ].isna())
        if NA_n > 0 :
            print( f"number of NAs in { col } = { NA_n } / { df_s0 } = { NA_n / df_s0 }" )
            
# Attack Type : target variable analysis
# -----------------------------------------------------------------------------
    ## This is the primary classification target with 3 classes:
    ## - Malware : Malicious software attacks
    ## - Intrusion : Unauthorized access attempts
    ## - DDoS : Distributed Denial of Service attacks
    col_name = "Attack Type"
    print( "Target variiable : Attack Type distribution" , "-" * 60 , df[ col_name ].value_counts())
    piechart_col( df , col_name )

# date
# -----------------------------------------------------------------------------

    col_name = "date"
    df[ col_name ] = pd.to_datetime( df[ col_name ])
    date_end = max( df[ col_name ])
    date_start = min( df[ col_name ])
    
    print( "Temporal distribution" , "-" * 60 ,  f"dataset time range : from { date_start } to { date_end }" )
    fig = px.histogram(
        df , 
        x = col_name ,
        title = "Distribution of cyberattack types over time" ,
        labels = { 
            "date" : "date" ,
            "count" : "number of attacks"
            } ,
        color_discrete_sequence = [ "#636EFA" ]
        )
    fig.update_layout(
        xaxis_title = "Date" ,
        yaxis_title = "Number of Attacks" ,
        showlegend = False
    )
    fig.show()
    
    col = df.columns.get_loc( col_name )
    col_insert = [ 
            "date dd" ,
            "date MW" ,
            # "date MW1" , # day = [ 1 ; 7 ]
            # "date MW2" , # day = [ 8 ; 14 ]
            # "date MW3" , # day = [ 15 ; 22 ]
            "date WD" ,
            # "date WD1" , # day = monday
            # "date WD2" , # day = tuesday
            # "date WD3" , # day = wednesday
            # "date WD4" , # day = thursday
            # "date WD5" , # day = friday
            # "date WD6" , # day = saturday
            "date H" ,
            # "date H1" , # hour = [ 0300 ; 0900 [
            # "date H2" , # hour = [ 0900 ; 1500 [
            # "date H3" , # hour = [ 1500 ; 2100 [
            "date M" ,
            # "date M1" , # month = january
            # "date M2" , # month = february
            # "date M3" , # month = march
            # "date M4" , # month = april
            # "date M5" , # month = may
            # "date M6" , # month = june
            # "date M7" , # month = july
            # "date M8" , # month = august
            # "date M9" , # month = september
            # "date M10" , # month = october
            # "date M11" , # month = november
            ]
    for col , colnew_name in zip( range( col + 1 , col + len( col_insert ) + 1 ) , col_insert ) :
        if colnew_name in [ 
                "date dd" ,
                "date MW" , 
                "date WD" , 
                "date H" , 
                "date M" 
                ] :
            df.insert( col , colnew_name , value = pd.NA )
        else :
            df.insert( col , colnew_name , value = 0 )
    
    df[ "date dd" ] = df[ "date" ].dt.floor( "d" )
    sched = [ 0 , 7 , 14 , 22 ]
    for day_filter in range( 0 , len( sched ) - 1 ) :
        day_bully = ( df[ "date" ].dt.day > sched[ day_filter ]) & ( df[ "date" ].dt.day <= sched[ day_filter + 1 ])
        # df.loc[ day_bully ,  f"date MW{ day_filter + 1 }" ] = 1
        df.loc[ day_bully ,  "date MW" ] = f"MW{ day_filter + 1 }"
    df[ "date MW" ] = df[ "date MW" ].fillna( "MW4" )
    for day_filter in range( 0 , 6 ) :
        day_bully = df[ "date" ].dt.weekday == day_filter
        # df.loc[ day_bully , f"date WD{ day_filter + 1 }" ] = 1
        df.loc[ day_bully , "date WD" ] = f"WD{ day_filter + 1 }"
    df[ "date WD" ] = df[ "date WD" ].fillna( "WD7" )
    sched = [ 3 , 9 , 15 , 21 ]
    for day_filter in range( 0 , len( sched ) - 1 ) :
        day_bully = ( df[ "date" ].dt.hour >= sched[ day_filter ]) & ( df[ "date" ].dt.hour < sched[ day_filter + 1 ])
        # df.loc[ day_bully , f"date H{ day_filter + 1}" ] = 1
        df.loc[ day_bully , "date H" ] = f"H{ day_filter + 1}"
    df[ "date H" ] = df[ "date H" ].fillna( "H4" )
    for day_filter in range( 1 , 12 ) :
        day_bully = df[ "date" ].dt.month == day_filter
        # df.loc[ day_bully , f"date M{ day_filter }" ] = 1
        df.loc[ day_bully , "date M" ] = f"M{ day_filter }"
    df[ "date M" ] = df[ "date M" ].fillna( "M12" )
    
# IP address
# -----------------------------------------------------------------------------
    # This section :
    # - Extracts geographic information from Source and Destination IP addresses
    # - Visualizes global distribution of attack origins and targets
    # - Analyzes proxy usage patterns in the network traffic
    
    # IP address geolocation extraction _______________________________________
    # Convert IP addresses to geographic coordinates using MaxMind GeoIP2
    # This enables geographic visualization and analysis of attack patterns
    
    ## insert new columns for geographic data
    for destsource in [ "Source" , "Destination" ] :
        col = df.columns.get_loc( f"{ destsource } IP Address" )
        col_insert = [
            f"{ destsource } IP 3octet" ,
            f"{ destsource } IP 2octet" ,
            f"{ destsource } IP 1octet" ,
            f"{ destsource } IP latitude" ,
            f"{ destsource } IP longitude" ,
            f"{ destsource } IP country" ,
            f"{ destsource } IP city"
            ]
        for col , colnew_name in zip( range( col + 1 , col + len( col_insert ) + 1 ) , col_insert ) :
            df.insert( col , colnew_name , value = pd.NA )
        df[[ 
            f"{ destsource } IP latitude" , 
            f"{ destsource } IP longitude" , 
            f"{ destsource } IP country" , 
            f"{ destsource } IP city" 
            ]] = df[ f"{ destsource } IP Address" ].apply( lambda x : ip_to_coords( df , x ))
        for i in range( 1 , 4 ) :
            df[ f"{ destsource } IP {i}octet" ] = extract_ipv4_prefix( destsource , i )
    
    ## global IP address distribution map _____________________________________
    ### orthographic projection showing attack source and destination locations
    ### left globe : Source IPs ( where attacks originate )
    ### right globe : Destination IPs ( attack targets )
    fig = subp(
        rows = 1 ,
        cols = 2 ,
        specs = [
            [
                { "type" : "scattergeo" } ,
                { "type" : "scattergeo" } ,
            ]
            ] ,
        subplot_titles = (
            "Source IP locations" ,
            "Destination IP locations" 
            )
        )
    ## Source IP locations ____________________________________________________
    fig.add_trace(
        go.Scattergeo(
            lat = df[ "Source IP latitude" ] ,
            lon = df[ "Source IP longitude" ] ,
            mode = "markers" ,
            marker = { 
                "size" : 5 ,
                "color" : "blue" ,
                "opacity" : 0.6
                } ,
            name = "Source IPs" ,
            hovertemplate = "<b>Attack Origin</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
            ) ,
        row = 1 , 
        col = 1
        )
    ## Destination IP locations _______________________________________________
    fig.add_trace(
        go.Scattergeo(
            lat = df[ "Destination IP latitude" ] ,
            lon = df[ "Destination IP longitude" ] ,
            mode = "markers" ,
            marker = { 
                "size" : 5 ,
                "color" : "blue" ,
                "opacity" : 0.6
                } ,
            name = "Desination IPs" ,
            hovertemplate = "<b>Attack Target</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
            ) ,
        row = 1 , 
        col = 2
        )
    fig.update_geos(
        # projection_type = "orthographic" ,
        showcountries = True ,
        showland = True
        # landcolor = "lightgreen" ,
        # countrycolor = "white ,
        # oceancolor = "lightblue" ,
        # showocean = True
    )
    fig.update_layout(
        height = 750 ,
        margin = { 
            "r" : 0 , 
            "t" : 80 , 
            "l" : 0 , 
            "b" : 0 
            } ,
        title_text = "Global Distribution of Cyber Attack Traffic" ,
        title_x = 0.5 ,
        title_font_size = 18 ,
        showlegend = True ,
        legend = {
            "orientation" : "h" ,
            "yanchor" : "bottom" ,
            "y" : - 0.1 ,
            "xanchor" : "center" ,
            "x" : 0.5
            }
        )
    fig.show()
    
# Proxy Information
# -----------------------------------------------------------------------------
    # Prepare columns for proxy IP geolocation data
    # Proxies may be used to mask the true origin of attacks
    col_name = "Proxy Information"
    
    ## insert new columns for geographic data
    col = df.columns.get_loc( col_name )
    col_insert = [ 
            "Proxy usage" ,
            "Proxy latitude" ,
            "Proxy longitude" ,
            "Proxy country" ,
            "Proxy city"
            ]
    for col , colnew_name in zip( range( col + 1 , col + len( col_insert ) + 1 ) , col_insert ) :
        if ( col_insert == "Proxy usage" ) :
            df.insert( col , colnew_name , value = 0 )
            df.loc[ ~ df[ "Proxy Information" ].isna(), "Proxy usage" ] = 1
        else :
            df.insert( col , colnew_name , value = pd.NA )
    df[[ "Proxy latitude" , "Proxy longitude" , "Proxy country" , "Proxy city" ]] = df[ "Proxy Information" ].apply( lambda x : ip_to_coords( df , x ))
    
    print( "Traffic flow analysis" , "-" * 60 )
    # aggregIPs = sankey_diag_IPs( df , 10 )
    
# Source Port
# -----------------------------------------------------------------------------
    col_name = "Source Port"
    print( df[ col_name ].value_counts())
    ## histogram
    fig = px.histogram(
        df , 
        x = col_name ,
        title = f"Distribution of { col_name } values" ,
        labels = { 
            "date" : "date" ,
            "count" : "number of attacks"
            } ,
        color_discrete_sequence = [ "#636EFA" ]
        )
    fig.update_layout(
        xaxis_title = col_name ,
        yaxis_title = "Number of attacks" ,
        showlegend = False
    )
    fig.show()
    
# Destination Port
# -----------------------------------------------------------------------------
    col_name = "Destination Port"
    print( df[ col_name ].value_counts())
    ## histogram
    fig = px.histogram(
        df , 
        x = col_name ,
        title = f"Distribution of { col_name } values" ,
        labels = { 
            "date" : "date" ,
            "count" : "number of attacks"
            } ,
        color_discrete_sequence = [ "#636EFA" ]
        )
    fig.update_layout(
        xaxis_title = col_name ,
        yaxis_title = "Number of attacks" ,
        showlegend = False
    )
    fig.show()
    
# Protocol
# -----------------------------------------------------------------------------
    col_name = "Protocol"
    """
        UDP = { 1 if Protocol = "UDP" , 0 otherwise } ; connectionless , fast but unreliable ( DNS , streaming , gaming )
        TCP = { 1 if Protocol = "TCP" , 0 otherwise } ; connection-oriented , reliable delivery ( web , email , file transfer )
        IMCP = [ 0 , 0 ] ; control messages and diagnostics ( ping , traceroute )
    """
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Packet length
# -----------------------------------------------------------------------------
    # Packet length can indicate:
        # - Small packets : Control messages , ACKs , probes
        # - Medium packets : Typical web traffic
        # - Large packets : File transfers , streaming data
        # - Unusual sizes : Potential attack indicators
    fig = px.histogram( 
        df , 
        "Packet Length" ,
        title = "Distribution of network pack sizes" ,
        labels = {
            "Packet Length" : "Packet Length ( bytes )" ,
            "count" : "Frequency" 
            } ,
            color_discrete_sequence = [ "#00CC96" ]
        )
    fig.update_layout(
        xaxis_title = "Packet length ( bytes )" ,
        yaxis_title = "Number of packets" ,
        showlegend = False
    )
    fig.show()
    
# Packet Type
# -----------------------------------------------------------------------------
    # Packet Types :
        # - Control packets : Network management ( SYN , ACK , FIN , RST )
        # - Data packets : Actual payload / content transmission
    col_name = "Packet Type"
    print( df[ col_name ].value_counts())
    piechart_col( df , "Packet Type" )
    
# Traffic Type
# -----------------------------------------------------------------------------
    # Application layer protocols :
        # - DNS : Domain Name System queries / responses ( port 53 )
        # - HTTP : Web traffic ( port 80 )
        # - FTP : File Transfer Protocol ( ports 20, 21 )
    col_name = "Traffic Type"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Malware Indicators
# -----------------------------------------------------------------------------
    # Indicators of Compromise ( IoC ) detected by security systems
    # IoC examples : malicious file hashes , suspicious domains , known attack patterns
    col_name = "Malware Indicators"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Anomaly Scores
# -----------------------------------------------------------------------------
    # numerical score indicating how anomalous / suspicious the traffic is
    # higher scores suggest more deviation from normal behavior
    fig = px.histogram(
        df ,
        x = "Anomaly Scores" ,
        title = "Distribution of Anomaly Scores" ,
        labels = {
            "Anomaly Scores" : "Anomaly Score" ,
            "count" : "Frequency" } ,
        color_discrete_sequence = [ "#EF553B" ]
    )
    fig.update_layout(
        xaxis_title = "Anomaly Score ( higher = more suspicious )" ,
        yaxis_title = "Number of Records" ,
        showlegend = False
    )
    fig.show()
    
# Alerts/Warnigns
# -----------------------------------------------------------------------------
    # Whether the traffic triggered a security alert
    col_name = "Alerts/Warnings"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Attack Signature
# -----------------------------------------------------------------------------
    # Pattern matching results from signature-based detection systems
    # Known Pattern A vs Known Pattern B ( different attack signatures )
    col_name = "Attack Signature"
    print( df[ "Attack Signature" ].value_counts())
    piechart_col( df , col_name )
    
# Action taken
# -----------------------------------------------------------------------------
    # Response action by the security system
        # - Logged : Traffic recorded for analysis
        # - Blocked : Traffic denied/dropped
        # - Ignored : No action taken
    col_name = "Action Taken"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Severity Level
# -----------------------------------------------------------------------------
    # Classification of threat severity
    col_name = "Severity Level"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name , [ "Low" , "Medium" , "High" ])
    
# Device Information
# -----------------------------------------------------------------------------
    # This section parses User-Agent strings to extract :
        # - Browser information ( family , version )
        # - Operating system details
        # - Device characteristics ( mobile , tablet , PC , bot )
        
    # User-agent parsing and device classification ____________________________
    # The "Device Information" column contains User-Agent strings that reveal :
        # - Browser : Chrome , Firefox , Safari , Edge , IE , etc.
        # - Operating System : Windows , macOS , Linux , iOS , Android
        # - Device Type : Desktop PC , Mobile phone , Tablet
        # - Bot Detection : Automated crawlers vs human users
    # This information helps identify :
        # - Attack vectors targeting specific platforms
        # - Bot-driven attacks vs human-initiated threats
        # - Vulnerable browser / OS combinations
    col_name = "Device Information"
    print( df[ col_name ].value_counts())
    col = df.columns.get_loc( col_name )
    col_insert = [ 
            "Browser family" ,
            "Browser major" ,
            "Browser minor" ,
            "OS family" ,
            "OS major" ,
            "OS minor" ,
            "OS patch" ,
            "Device family" ,
            "Device brand" ,
            "Device type" ,
            "Device bot"
            ]
    for col , colnew_name in zip( range( col + 1 , col + len( col_insert ) + 1 ) , col_insert ) :
        df.insert( col , colnew_name , value = pd.NA )
    df[ "Browser family" ] = df[ col_name ].apply( lambda x : parse( x ).browser.family if parse( x ).browser.family is not None else pd.NA )
    df[ "Browser major" ] = df[ col_name ].apply( lambda x : parse( x ).browser.version[ 0 ] if parse( x ).browser.version[ 0 ] is not None else pd.NA )
    df[ "Browser minor" ] = df[ col_name ].apply( lambda x: parse( x ).browser.version[ 1 ] if parse( x ).browser.version[ 1 ] is not None else pd.NA )
    df[ "OS family" ] = df[ col_name ].apply( lambda x : parse( x ).os.family if parse( x ).os.family is not None else pd.NA )
    df[ "OS major" ] = df[ col_name ].apply( lambda x : parse( x ).os.version[ 0 ] if len( parse( x ).os.version ) > 0 and parse( x ).os.version[ 0 ] is not None else pd.NA )
    df[ "OS minor" ] = df[ col_name ].apply( lambda x : parse( x ).os.version[ 1 ] if len( parse( x ).os.version ) > 1 and parse( x ).os.version[ 1 ] is not None else pd.NA )
    df[ "OS patch" ] = df[ col_name ].apply( lambda x : parse( x ).os.version[ 2 ] if len( parse( x ).os.version ) > 2 and parse( x ).os.version[ 2 ] is not None else pd.NA )
    df[ "Device family" ] = df[ col_name ].apply (lambda x : parse( x ).device.family if parse( x ).device.family is not None else pd.NA )
    df[ "Device brand" ] = df[ col_name].apply( lambda x : parse( x ).device.brand if parse( x ).device.brand is not None else pd.NA )
    df[ "Device type" ] = df[ col_name ].apply( lambda x : Device_type( x ))
    df[ "Device bot" ] = df[ col_name ].apply( lambda x: ua_parse( x ).is_bot )
    
    col_name = "Browser family"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Network Segment
# -----------------------------------------------------------------------------
    # this section analyzes network segmentation and security system logs
    # network segments represent logical divisions of the network :
        # - Segment A , B , C : Different network zones ( e.g. , DMZ , internal , guest )
        # - Segmentation helps contain breaches and control traffic flow
    col_name = "Network Segment"
    print( df[ col_name ].value_counts()) 
    piechart_col( df , col_name )
    
# Geo-location Data
# -----------------------------------------------------------------------------
    # parse "City, State" format into separate columns for geographic analysis
    col_name = "Geo-location Data"
    print( df[ col_name ].value_counts())
    col = df.columns.get_loc( col_name )
    df.insert( col + 1 , "Geo-location City" , value = pd.NA )
    df.insert( col + 2 , "Geo-location State" , value = pd.NA )
    df[[ "Geo-location City" , "Geo-location State" ]] = df[ "Geo-location Data" ].apply( lambda x : geolocation_data( x ))
    
# Firewall Logs
# -----------------------------------------------------------------------------
    # records of traffic allowed / denied by firewall rules
    col_name = "Firewall Logs"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# IDS/IPS Alerts
# -----------------------------------------------------------------------------
    # Intrusion Detection / Prevention System alerts
    col_name = "IDS/IPS Alerts"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
# Log Source
# -----------------------------------------------------------------------------
    # origin of the log entry
    col_name = "Log Source"
    print( df[ col_name ].value_counts())
    piechart_col( df , col_name )
    
    return df

df = EDA_DataPrep_PIPELINE( df_original )


#%% Modelling PIPELINE
# -----------------------------------------------------------------------------

crosstabs_x_AttackType = {}

features_mask = [
        True , # "Source Port & Destination Port"
        False , # "Source IP latitude/longitude combination & Destination IP latitude/longitude combination" -----> sometimes works, other times doesnt, is people selective
        False , # "Sounce IP Country , Destination IP country combination" -----> sometimes works, other times doesnt, is people selective ( related to Network & Traffic + Security Response + time/device info/malware aggregate )
        True , # "date dd"
        True , # "Network & Traffic + device info"
        True , # "Security Response + device info"
        True , # "time"
        True , # "Anomaly Scores"
        True , # "Packet Length"
        ]
threshold_floor = 37.5
nobs_floor = 10
y_pred = Modelling_PIPELINE( 0.1 , features_mask , threshold_floor , 15 , [ 5 , 100 , nobs_floor , threshold_floor ] , "logit" , False )

