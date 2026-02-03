import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utilities.django_settings import geoIP
from user_agents import parse
from sklearn.metrics import matthews_corrcoef

# =============================================================================
# IP GEOLOCATION CACHE
# Added by Claude: Cache IP lookups to disk to avoid repeated GeoIP2 queries
# =============================================================================
IP_CACHE_FILE = "data/ip_geo_cache.json"


def load_cache(filename):
    """Load cached data from disk."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache, filename):
    """Save cache to disk."""
    with open(filename, "w") as f:
        json.dump(cache, f)


# Global cache loaded once at import
_ip_cache = load_cache(IP_CACHE_FILE)

# =============================================================================
# UTILITY FUNCTIONS For the pipeline
# =============================================================================


def catvar_mapping(df, col_name, values, name=None):
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


def piechart_col(df, col, names=None):
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
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )
        fig.show()
    else:
        fig = px.pie(
            values=df[col].value_counts(),
            names=names,
            title=f"Distribution of {col}",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
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


def ip_geolocation_processing(df: pd.DataFrame):
    """
    Process IP address columns to extract geographic coordinates and location info.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing IP address columns to be geolocated
    Returns:
    --------
    pd.DataFrame
        DataFrame with new columns for latitude, longitude, country, and city
    """

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

        ip_col = f"{destsource} IP Address"
        print(f"Processing {destsource} IP addresses...")
        col = list(df.columns).index(ip_col) + 1

        # Insert new columns for geographic data
        for i, geo_col in enumerate(geo_columns):
            if geo_col not in df.columns:
                df.insert(col + i, geo_col, value=pd.NA)

        # Added by Claude: Parallel processing with caching for performance
        # - Only lookup unique IPs (not every row)
        # - Use ThreadPoolExecutor for parallel lookups
        # - Cache results to ip_geo_cache.json for subsequent runs

        global _ip_cache
        unique_ips = list(df[ip_col].dropna().unique())

        # Check cache for already-resolved IPs
        cached_ips = [ip for ip in unique_ips if ip in _ip_cache]
        uncached_ips = [ip for ip in unique_ips if ip not in _ip_cache]
        print(
            f"  {len(unique_ips)} unique IPs: {len(cached_ips)} cached, {len(uncached_ips)} to lookup..."
        )

        # Start with cached results
        ip_geo_map = {ip: _ip_cache[ip] for ip in cached_ips}

        # Lookup uncached IPs in parallel
        if uncached_ips:
            completed = 0
            with ThreadPoolExecutor(max_workers=14) as executor:
                futures = {executor.submit(ip_to_coords, ip): ip for ip in uncached_ips}
                for future in as_completed(futures):
                    ip = futures[future]
                    result = future.result()
                    # Convert Series to list, replacing pd.NA with None for JSON serialization
                    result_list = [None if pd.isna(v) else v for v in result.tolist()]
                    ip_geo_map[ip] = result_list
                    _ip_cache[ip] = result_list
                    completed += 1
                    if completed % 500 == 0:
                        print(f"  Progress: {completed}/{len(uncached_ips)}")
            print(f"  Completed {len(uncached_ips)} new lookups")
            save_cache(_ip_cache, IP_CACHE_FILE)

        # Map results back to dataframe (cache stores lists with None for missing values)
        for i, geo_col in enumerate(geo_columns):
            df[geo_col] = df[ip_col].map(
                lambda ip, i=i: ip_geo_map.get(ip, [None] * 4)[i]
                if pd.notna(ip)
                else None
            )

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
            "Attack Target Locations (Destination IPs)",
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
            hovertemplate="<b>Attack Origin</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>",
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
            hovertemplate="<b>Attack Target</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>",
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
        showocean=True,
    )
    fig.update_layout(
        height=750,
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        title_text="Global Distribution of Cyber Attack Traffic",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    fig.show()

    return df


# -----------------------------------------------------------------------------
# PROXY INFORMATION COLUMNS
# Prepare columns for proxy IP geolocation data
# Proxies may be used to mask the true origin of attacks
# -----------------------------------------------------------------------------
# Modified by Claude: Added actual geolocation extraction (was only creating empty columns)
def proxy_info_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    -----------------------------------------------------------------------------
    Extract geolocation data from proxy IP addresses.
    Proxies may be used to mask the true origin of attacks.
    -----------------------------------------------------------------------------

    Args:
        df: the input dataframe with "Proxy Information" column containing IP addresses
    Returns:
        df: the dataframe with proxy geolocation columns populated
    """
    col_name = "Proxy Information"
    if col_name not in df.columns:
        raise KeyError(f"{col_name!r} not found in DataFrame columns")

    geo_columns = ["Proxy latitude", "Proxy longitude", "Proxy country", "Proxy city"]

    # Skip if already processed
    if all(col in df.columns for col in geo_columns):
        if df[geo_columns].notna().any().any():
            print("Proxy geolocation already processed - skipping")
            return df

    print("Processing Proxy IP addresses...")
    col = list(df.columns).index(col_name) + 1

    # Insert columns if they don't exist
    for i, new_col in enumerate(geo_columns):
        if new_col not in df.columns:
            df.insert(col + i, new_col, value=None)

    # Lookup only unique IPs for performance (parallel, with caching)
    global _ip_cache
    unique_ips = list(df[col_name].dropna().unique())

    # Check cache for already-resolved IPs
    cached_ips = [ip for ip in unique_ips if ip in _ip_cache]
    uncached_ips = [ip for ip in unique_ips if ip not in _ip_cache]
    print(
        f"  {len(unique_ips)} unique proxy IPs: {len(cached_ips)} cached, {len(uncached_ips)} to lookup..."
    )

    # Start with cached results
    ip_geo_map = {ip: _ip_cache[ip] for ip in cached_ips}

    # Lookup uncached IPs in parallel
    if uncached_ips:
        completed = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(ip_to_coords, ip): ip for ip in uncached_ips}
            for future in as_completed(futures):
                ip = futures[future]
                result = future.result()
                # Convert Series to list, replacing pd.NA with None for JSON serialization
                result_list = [None if pd.isna(v) else v for v in result.tolist()]
                ip_geo_map[ip] = result_list
                _ip_cache[ip] = result_list
                completed += 1
                if completed % 500 == 0:
                    print(f"  Progress: {completed}/{len(uncached_ips)}")
        print(f"  Completed {len(uncached_ips)} new lookups")
        save_cache(_ip_cache, IP_CACHE_FILE)

    # Map results back to dataframe (cache stores lists)
    for i, geo_col in enumerate(geo_columns):
        df[geo_col] = df[col_name].map(
            lambda ip, i=i: ip_geo_map.get(ip, [None] * 4)[i] if pd.notna(ip) else None
        )

    return df


def sankey_diag_IPs(df: pd.DataFrame, ntop: int) -> pd.DataFrame:
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
        # Added by Claude: Replace NaN with "Unknown" to prevent groupby dropping rows
        IPs_col[IPid] = IPs_col[IPid].fillna("Unknown")
        # Added by Claude: Include "Unknown" in labels for Sankey diagram
        IPs_col[f"{IPid}labs"] = f"{IPid} " + pd.concat(
            [IPs_col[f"{IPid}labs"], pd.Series(["other", "Unknown"])]
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
                    hovertemplate="<b>%{label}</b><br>Total flows: %{value}<extra></extra>",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    hovertemplate="<b>Flow</b><br>From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>",
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
            dict(
                x=0.01,
                y=1.05,
                text="<b>Source Countries</b>",
                showarrow=False,
                font_size=12,
            ),
            dict(
                x=0.5,
                y=1.05,
                text="<b>Proxy Countries</b>",
                showarrow=False,
                font_size=12,
            ),
            dict(
                x=0.99,
                y=1.05,
                text="<b>Destination Countries</b>",
                showarrow=False,
                font_size=12,
            ),
        ],
    )
    fig.show()
    
    return aggregIPs
    
def parse_ua_string(ua_string):
        """Parse User-Agent string once and extract all fields."""
        if not ua_string or pd.isna(ua_string):
            return [None] * 11

        try:
            ua = parse(ua_string)

            # Extract all fields from single parse
            browser_family = ua.browser.family if ua.browser.family else None
            browser_major = ua.browser.version[0] if len(ua.browser.version) > 0 else None
            browser_minor = ua.browser.version[1] if len(ua.browser.version) > 1 else None

            os_family = ua.os.family if ua.os.family else None
            os_major = ua.os.version[0] if len(ua.os.version) > 0 else None
            os_minor = ua.os.version[1] if len(ua.os.version) > 1 else None
            os_patch = ua.os.version[2] if len(ua.os.version) > 2 else None

            device_family = ua.device.family if ua.device.family else None
            device_brand = ua.device.brand if ua.device.brand else None

            # Device type classification
            if ua.is_mobile:
                device_type = "Mobile"
            elif ua.is_tablet:
                device_type = "Tablet"
            elif ua.is_pc:
                device_type = "PC"
            else:
                device_type = None

            device_bot = ua.is_bot

            return [browser_family, browser_major, browser_minor,
                    os_family, os_major, os_minor, os_patch,
                    device_family, device_brand, device_type, device_bot]
        except Exception:
            return [None] * 11
        
def get_cached_ua(ua_cache, ua_string, field_index):
    """
    Retrieve a specific field from cached User-Agent parsing results.

    Parameters
    ----------
    ua_cache : dict
        Dictionary mapping User-Agent strings to lists of parsed field values.
        Expected format: {ua_string: [browser_family, browser_major, browser_minor,
                            os_family, os_major, os_minor, os_patch,
                            device_family, device_brand, device_type, device_bot]}
    ua_string : str
        The User-Agent string to look up in the cache.
    field_index : int
        Index of the field to retrieve (0-10 corresponding to the parsed fields).

    Returns
    -------
    Any
        The cached field value, or pd.NA if the string is missing, not cached,
        or the field value is None.
    """
    if not ua_string or pd.isna(ua_string):
        return pd.NA
    result = ua_cache.get(ua_string)
    if result is None:
        return pd.NA
    val = result[field_index]
    return pd.NA if val is None else val

def device_type(ua_string):
    """
    Classify device type from User-Agent string.

    Returns:
        str: "Mobile", "Tablet", "PC", or pd.NA if unknown
    """
    try:
        if not ua_string or pd.isna(ua_string):
            return pd.NA
        ua = parse(ua_string)
        if getattr(ua, "is_mobile", False):
            return "Mobile"
        if getattr(ua, "is_tablet", False):
            return "Tablet"
        if getattr(ua, "is_pc", False):
            return "PC"
        return pd.NA
    except:
        return pd.NA
    
def geolocation_data(info):
    """Split 'City, State' string into separate components."""
    city, state = info.split(", ")
    return pd.Series([city, state])

def crosstab_col(df, col, target, col_name, target_name) -> dict[str, pd.DataFrame]:
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
    Returns:
    pd.crosstab
        Normalized cross-tabulation DataFrame
    """
    table_name = f"{col_name}_x_{target_name}"
    return {table_name : pd.crosstab(df[target], df[col], normalize=True) * 100}

def sankey_diag(
    df: pd.DataFrame,
    cols_bully=[
        None,   # "Source IP country"
        None,   # "Destination IP country"
        None,   # "Source Port ephemeral"
        None,   # "Destination Port ephemeral"
        None,   # "Protocol"
        None,   # "Packet Type Control"
        None,   # "Traffic Type"
        None,   # "Malware Indicators"
        None,   # "Alert Trigger"
        None,   # "Attack Signature patA"
        None,   # "Action Taken"
        None,   # "Severity Level"
        None,   # "Network Segment"
        None,   # "Firewall Logs"
        None,   # "IDS/IPS Alerts"
        None,   # "Log Source Firewall"
    ],
    ntop=10,
    ) -> pd.DataFrame:
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

# Creates an interactive parallel categories plot showing how
# different categorical feature values relate to attack types.
# The diagram allows exploration of multi-dimensional categorical relationships.


def paracat_diag(
    df  : pd.DataFrame,
    cols_bully=[
        None,  # "Source IP country"
        None,  # "Destination IP country"
        None,  # "Source Port ephemeral"
        None,  # "Destination Port ephemeral"
        None,  # "Protocol"
        None,  # "Packet Type Control"
        None,  # "Traffic Type"
        None,  # "Malware Indicators"
        None,  # "Alert Trigger"
        None,  # "Attack Signature patA"
        None,  # "Action Taken"
        None,  # "Severity Level"
        None,  # "Network Segment"
        None,  # "Firewall Logs"
        None,  # "IDS/IPS Alerts"
        None,  # "Log Source Firewall"
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
        SIP = df["Source IP country"].copy(deep=None)
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


def catvar_corr(df, col, target="Attack Type"):
    """Calculate Matthews correlation coefficient between a feature and target."""
    corr = matthews_corrcoef(df[target], df[col].astype(str))
    print(f"  {col:35} : {corr:+.4f}")