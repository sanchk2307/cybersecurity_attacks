"""Data preparation utilities: IP geolocation, device parsing, IPv4 prefix extraction.

Heavy I/O-bound functions (IP geolocation) are parallelized using
ThreadPoolExecutor. CPU-bound functions (UA string parsing) use
ProcessPoolExecutor for true parallelism past the GIL.
Results are cached to disk so subsequent runs skip already-resolved lookups.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from user_agents import parse as ua_parse

import geoip2.database

from src.utilities.config import GEOIP_PATH

# Number of workers: max logical cores - 2, minimum 1
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# When True, ip_to_coords_parallel and parse_device_info_parallel run
# lookups on the main thread instead of spawning executor pools.
SEQUENTIAL_MODE = False

# Cache directory
CACHE_DIR = Path("data/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IP_CACHE_FILE = CACHE_DIR / "ip_geo_cache.json"
UA_CACHE_FILE = CACHE_DIR / "ua_parse_cache.json"

# Sentinel used to represent pd.NA in JSON (JSON has no NA concept)
_NA = "__NA__"

# GeoIP2 City database reader (used for lat/lon + country/city lookups)
_CITY_DB = Path(GEOIP_PATH) / "GeoLite2-City.mmdb"
_COUNTRY_DB = Path(GEOIP_PATH) / "GeoLite2-Country.mmdb"

_city_reader = None
_country_reader = None

if _CITY_DB.exists():
    _city_reader = geoip2.database.Reader(str(_CITY_DB))
elif _COUNTRY_DB.exists():
    _country_reader = geoip2.database.Reader(str(_COUNTRY_DB))
    print(
        f"WARNING: {_CITY_DB} not found — lat/lon lookups will return NA. "
        f"Download GeoLite2-City.mmdb from https://dev.maxmind.com/geoip/geolite2-free-geolocation-data "
        f"and place it in '{GEOIP_PATH}/'."
    )
else:
    print(
        f"WARNING: No GeoIP2 databases found in '{GEOIP_PATH}/'. "
        f"IP geolocation will be unavailable."
    )


def _load_json_cache(path):
    """Load a JSON cache file, returning an empty dict if missing or corrupt."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_json_cache(path, cache):
    """Atomically write a JSON cache file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(path)


def _from_cache_val(v):
    """Convert a cache value back, replacing sentinel with pd.NA."""
    return pd.NA if v == _NA else v


def _to_cache_val(v):
    """Convert a value for JSON storage, replacing pd.NA with sentinel."""
    if v is pd.NA or (isinstance(v, float) and pd.isna(v)):
        return _NA
    return v


# ─── IP geolocation ──────────────────────────────────────────────────────────


def ip_to_coords(ip_address):
    """Convert an IP address to geographic coordinates and location info.

    Uses geoip2.database.Reader directly (no Django/GDAL dependency).
    Prefers the City database (has lat/lon + country + city).
    Falls back to Country database (country only, no coordinates).

    Parameters
    ----------
    ip_address : str
        IPv4 or IPv6 address to geolocate.

    Returns
    -------
    tuple
        (latitude, longitude, country_name, city).
        Returns pd.NA for any fields that cannot be resolved.
    """
    lat, lon, country, city = pd.NA, pd.NA, pd.NA, pd.NA
    if _city_reader is not None:
        try:
            res = _city_reader.city(ip_address)
            if res.location.latitude is not None:
                lat = res.location.latitude
            if res.location.longitude is not None:
                lon = res.location.longitude
            if res.country.name is not None:
                country = res.country.name
            if res.city.name is not None:
                city = res.city.name
        except Exception:
            pass
    elif _country_reader is not None:
        try:
            res = _country_reader.country(ip_address)
            if res.country.name is not None:
                country = res.country.name
        except Exception:
            pass
    return (lat, lon, country, city)


def ip_to_coords_parallel(series):
    """Apply ip_to_coords to a Series in parallel with disk caching.

    Cached results are loaded first; only uncached IPs are looked up.
    New results are appended to the cache file after completion.

    Parameters
    ----------
    series : pd.Series
        Series of IP address strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [latitude, longitude, country, city].
    """
    cache = _load_json_cache(IP_CACHE_FILE)

    results = [None] * len(series)
    index = series.index
    to_lookup = {}  # ip_str -> list of positions (deduplicated)

    # Resolve from cache first, dedup uncached IPs
    cache_hits = 0
    for pos, ip in enumerate(series.values):
        ip_str = str(ip)
        if ip_str in cache:
            cached = cache[ip_str]
            results[pos] = tuple(_from_cache_val(v) for v in cached)
            cache_hits += 1
        else:
            if ip_str not in to_lookup:
                to_lookup[ip_str] = []
            to_lookup[ip_str].append(pos)

    total = len(series)
    unique_to_lookup = len(to_lookup)
    # print(f"  IP geolocation: {cache_hits}/{total} from cache, {unique_to_lookup} unique IPs to resolve")

    if to_lookup:
        new_cache_entries = {}

        if SEQUENTIAL_MODE:
            for ip in to_lookup:
                result = ip_to_coords(ip)
                for pos in to_lookup[ip]:
                    results[pos] = result
                new_cache_entries[ip] = tuple(_to_cache_val(v) for v in result)
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_ip = {
                    executor.submit(ip_to_coords, ip): ip for ip in to_lookup
                }
                for future in as_completed(future_to_ip):
                    ip = future_to_ip[future]
                    result = future.result()
                    for pos in to_lookup[ip]:
                        results[pos] = result
                    new_cache_entries[ip] = tuple(_to_cache_val(v) for v in result)

        # Update and save cache
        cache.update(new_cache_entries)
        _save_json_cache(IP_CACHE_FILE, cache)

    return pd.DataFrame(
        results,
        index=index,
        columns=["latitude", "longitude", "country", "city"],
    )


# ─── Geolocation data splitting ──────────────────────────────────────────────


def geolocation_data(info):
    """Split 'City, State' string into separate components.

    Parameters
    ----------
    info : str
        String in "City, State" format.

    Returns
    -------
    tuple
        (city, state).
    """
    city, state = info.split(", ")
    return (city, state)


def geolocation_data_parallel(series):
    """Split "City, State" strings into separate columns using vectorized ops.

    Parameters
    ----------
    series : pd.Series
        Series of "City, State" strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [city, state].
    """
    split = series.str.split(", ", n=1, expand=True)
    split.columns = ["city", "state"]
    return split


# ─── IPv4 prefix extraction ──────────────────────────────────────────────────


def extract_ipv4_prefix(df, SourDest, n_octets):
    """Extract the first n octets from an IPv4 column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the IP address column.
    SourDest : str
        Name prefix of the IPv4 column (e.g. "Source" or "Destination").
    n_octets : int
        Number of octets to keep (1, 2, or 3).

    Returns
    -------
    pd.Series
        Series containing the IPv4 prefix.
    """
    if n_octets not in (1, 2, 3):
        raise ValueError("n_octets must be 1, 2, or 3")
    return df[f"{SourDest} IP Address"].str.split(".").str[:n_octets].str.join(".")


# ─── User-Agent parsing ──────────────────────────────────────────────────────


def _parse_ua_string(ua_string):
    """Parse a single User-Agent string and extract all device fields at once.

    This function is a top-level function so it can be pickled for use with
    ProcessPoolExecutor.  It returns None instead of pd.NA because pd.NA
    cannot be pickled across process boundaries.  Callers must convert None
    back to pd.NA after collecting results.

    Parameters
    ----------
    ua_string : str
        Raw User-Agent string.

    Returns
    -------
    tuple
        (browser_family, browser_major, browser_minor,
         os_family, os_major, os_minor, os_patch,
         device_family, device_brand, device_type, device_bot)
        Missing values are represented as None (not pd.NA).
    """
    try:
        if not ua_string or ua_string != ua_string:  # NaN != NaN
            return (None,) * 11
        ua = ua_parse(ua_string)

        browser_family = ua.browser.family if ua.browser.family is not None else None
        bv = ua.browser.version
        browser_major = bv[0] if len(bv) > 0 and bv[0] is not None else None
        browser_minor = bv[1] if len(bv) > 1 and bv[1] is not None else None

        os_family = ua.os.family if ua.os.family is not None else None
        ov = ua.os.version
        os_major = ov[0] if len(ov) > 0 and ov[0] is not None else None
        os_minor = ov[1] if len(ov) > 1 and ov[1] is not None else None
        os_patch = ov[2] if len(ov) > 2 and ov[2] is not None else None

        device_family = ua.device.family if ua.device.family is not None else None
        device_brand = ua.device.brand if ua.device.brand is not None else None

        if getattr(ua, "is_mobile", False):
            device_type = "Mobile"
        elif getattr(ua, "is_tablet", False):
            device_type = "Tablet"
        elif getattr(ua, "is_pc", False):
            device_type = "PC"
        else:
            device_type = None

        device_bot = ua.is_bot

        return (
            browser_family,
            browser_major,
            browser_minor,
            os_family,
            os_major,
            os_minor,
            os_patch,
            device_family,
            device_brand,
            device_type,
            device_bot,
        )
    except Exception:
        return (None,) * 11


def _none_to_na(t):
    """Convert None values in a tuple back to pd.NA."""
    return tuple(pd.NA if v is None else v for v in t)


_UA_COLUMNS = [
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


def parse_device_info_parallel(series):
    """Parse User-Agent strings in parallel with disk caching.

    Cached results are loaded first; only uncached strings are parsed.
    Duplicate UA strings within the batch are resolved once and reused.

    Parameters
    ----------
    series : pd.Series
        Series of User-Agent strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for all parsed device fields.
    """
    cache = _load_json_cache(UA_CACHE_FILE)

    results = [None] * len(series)
    to_parse = {}  # ua_string -> list of positions

    # Resolve from cache + dedup
    cache_hits = 0
    for pos, ua_str in enumerate(series.values):
        key = str(ua_str)
        if key in cache:
            cached = cache[key]
            results[pos] = tuple(_from_cache_val(v) for v in cached)
            cache_hits += 1
        else:
            if key not in to_parse:
                to_parse[key] = []
            to_parse[key].append(pos)

    total = len(series)
    unique_to_parse = len(to_parse)
    # print(f"  UA parsing: {cache_hits}/{total} from cache, {unique_to_parse} unique strings to parse")

    if to_parse:
        new_cache_entries = {}
        ua_strings = list(to_parse.keys())

        if SEQUENTIAL_MODE:
            for ua_str in ua_strings:
                raw_result = _parse_ua_string(ua_str)
                result = _none_to_na(raw_result)
                for pos in to_parse[ua_str]:
                    results[pos] = result
                new_cache_entries[ua_str] = tuple(_to_cache_val(v) for v in result)
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_key = {
                    executor.submit(_parse_ua_string, ua_str): ua_str
                    for ua_str in ua_strings
                }
                for future in as_completed(future_to_key):
                    ua_str = future_to_key[future]
                    raw_result = future.result()  # contains None, not pd.NA
                    result = _none_to_na(raw_result)
                    for pos in to_parse[ua_str]:
                        results[pos] = result
                    new_cache_entries[ua_str] = tuple(_to_cache_val(v) for v in result)

        # Update and save cache
        cache.update(new_cache_entries)
        _save_json_cache(UA_CACHE_FILE, cache)

    return pd.DataFrame(results, index=series.index, columns=_UA_COLUMNS)


# ─── Backward compatibility ──────────────────────────────────────────────────


def Device_type(ua_string):
    """Classify device type from User-Agent string.

    Parameters
    ----------
    ua_string : str
        Raw User-Agent string.

    Returns
    -------
    str or pd.NA
        "Mobile", "Tablet", "PC", or pd.NA.
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
    except Exception:
        return pd.NA
