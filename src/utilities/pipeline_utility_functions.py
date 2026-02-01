import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from src.utilities.django_settings import geoIP

# =============================================================================
# IP GEOLOCATION CACHE
# Added by Claude: Cache IP lookups to disk to avoid repeated GeoIP2 queries
# =============================================================================
IP_CACHE_FILE = "data/ip_geo_cache.json"


def load_ip_cache():
    """Load cached IP geolocation data from disk."""
    if os.path.exists(IP_CACHE_FILE):
        try:
            with open(IP_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_ip_cache(cache):
    """Save IP geolocation cache to disk."""
    with open(IP_CACHE_FILE, "w") as f:
        json.dump(cache, f)


# Global cache loaded once at import
_ip_cache = load_ip_cache()

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
            save_ip_cache(_ip_cache)

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
        save_ip_cache(_ip_cache)

    # Map results back to dataframe (cache stores lists)
    for i, geo_col in enumerate(geo_columns):
        df[geo_col] = df[col_name].map(
            lambda ip, i=i: ip_geo_map.get(ip, [None] * 4)[i] if pd.notna(ip) else None
        )

    return df


def sankey_diag_IPs(df, ntop):
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
