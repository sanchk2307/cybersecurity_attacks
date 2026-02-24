"""Data preparation utilities: IP geolocation, device parsing, IPv4 prefix extraction.

Heavy parsing functions are parallelized using ThreadPoolExecutor with
(max logical cores - 2) workers. Results are cached to disk so subsequent
runs skip already-resolved lookups.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from user_agents import parse as ua_parse

from src.config import geoIP

# Number of workers: max logical cores - 2, minimum 1
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# Cache directory
CACHE_DIR = Path("data/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IP_CACHE_FILE = CACHE_DIR / "ip_geo_cache.json"
UA_CACHE_FILE = CACHE_DIR / "ua_parse_cache.json"

# Sentinel used to represent pd.NA in JSON (JSON has no NA concept)
_NA = "__NA__"


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
    try:
        res = geoIP.geos(ip_address).wkt
        lon_s, lat_s = res.replace("(", "").replace(")", "").split()[1:]
        lat, lon = lat_s, lon_s
    except Exception:
        pass
    try:
        res = geoIP.city(ip_address)
        country = res["country_name"]
        city = res["city"]
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
    to_lookup = []  # (position, ip) pairs that need a live lookup

    # Resolve from cache first
    cache_hits = 0
    for pos, ip in enumerate(series.values):
        ip_str = str(ip)
        if ip_str in cache:
            cached = cache[ip_str]
            results[pos] = tuple(_from_cache_val(v) for v in cached)
            cache_hits += 1
        else:
            to_lookup.append((pos, ip_str))

    total = len(series)
    # print(f"  IP geolocation: {cache_hits}/{total} from cache, {len(to_lookup)} to resolve")

    if to_lookup:
        new_cache_entries = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pos = {
                executor.submit(ip_to_coords, ip): (pos, ip) for pos, ip in to_lookup
            }
            done_count = 0
            for future in as_completed(future_to_pos):
                pos, ip = future_to_pos[future]
                result = future.result()
                results[pos] = result
                new_cache_entries[ip] = tuple(_to_cache_val(v) for v in result)
                done_count += 1
                # if done_count % 5000 == 0:
                # print(f"  IP geolocation: {done_count}/{len(to_lookup)} resolved")

        # print(f"  IP geolocation: {len(to_lookup)}/{len(to_lookup)} resolved")

        # Update and save cache
        cache.update(new_cache_entries)
        _save_json_cache(IP_CACHE_FILE, cache)
        # print(f"  IP cache saved ({len(cache)} total entries)")

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
    """Apply geolocation_data to a Series in parallel using threads.

    Parameters
    ----------
    series : pd.Series
        Series of "City, State" strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [city, state].
    """
    results = [None] * len(series)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pos = {
            executor.submit(geolocation_data, val): pos
            for pos, val in enumerate(series.values)
        }
        for future in as_completed(future_to_pos):
            pos = future_to_pos[future]
            results[pos] = future.result()

    return pd.DataFrame(
        results,
        index=series.index,
        columns=["city", "state"],
    )


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
    """
    try:
        if not ua_string or pd.isna(ua_string):
            return (pd.NA,) * 11
        ua = ua_parse(ua_string)

        browser_family = ua.browser.family if ua.browser.family is not None else pd.NA
        bv = ua.browser.version
        browser_major = bv[0] if len(bv) > 0 and bv[0] is not None else pd.NA
        browser_minor = bv[1] if len(bv) > 1 and bv[1] is not None else pd.NA

        os_family = ua.os.family if ua.os.family is not None else pd.NA
        ov = ua.os.version
        os_major = ov[0] if len(ov) > 0 and ov[0] is not None else pd.NA
        os_minor = ov[1] if len(ov) > 1 and ov[1] is not None else pd.NA
        os_patch = ov[2] if len(ov) > 2 and ov[2] is not None else pd.NA

        device_family = ua.device.family if ua.device.family is not None else pd.NA
        device_brand = ua.device.brand if ua.device.brand is not None else pd.NA

        if getattr(ua, "is_mobile", False):
            device_type = "Mobile"
        elif getattr(ua, "is_tablet", False):
            device_type = "Tablet"
        elif getattr(ua, "is_pc", False):
            device_type = "PC"
        else:
            device_type = pd.NA

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
        return (pd.NA,) * 11


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
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_key = {
                executor.submit(_parse_ua_string, ua_str): ua_str for ua_str in to_parse
            }
            done_count = 0
            for future in as_completed(future_to_key):
                ua_str = future_to_key[future]
                result = future.result()
                # Assign result to all positions with this UA string
                for pos in to_parse[ua_str]:
                    results[pos] = result
                new_cache_entries[ua_str] = tuple(_to_cache_val(v) for v in result)
                done_count += 1
                # if done_count % 5000 == 0:
                # print(f"  UA parsing: {done_count}/{unique_to_parse} resolved")

        # print(f"  UA parsing: {unique_to_parse}/{unique_to_parse} resolved")

        # Update and save cache
        cache.update(new_cache_entries)
        _save_json_cache(UA_CACHE_FILE, cache)
        # print(f"  UA cache saved ({len(cache)} total entries)")

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
