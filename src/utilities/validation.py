# ============================================================
# VALIDATION FUNCTIONS MODULE
# ============================================================
"""Reusable validation functions for Tab 1 & Tab 3"""

from datetime import datetime, date
import ipaddress
import pandas as pd


def validate_timestamp(ts_str):
    """Validate timestamp format and date constraints.
    Accepts multiple formats for flexibility with flexible spacing.
    Returns: (is_valid, error_message)
    """
    if not ts_str or str(ts_str).strip() == "":
        return False, "Timestamp is required"
    
    ts_str = str(ts_str).strip()
    
    # Normalize multiple spaces to single space for matching
    ts_normalized = " ".join(ts_str.split())  # Replace multiple spaces with single space
    
    # Try multiple timestamp formats
    formats = [
        "%Y-%m-%d %H:%M:%S",              # 2023-05-30 06:33:58
        "%m/%d/%Y %I:%M:%S %p",           # 5/30/2023 06:33:58 AM
        "%m/%d/%Y %H:%M:%S",              # 5/30/2023 06:33:58
        "%m/%d/%Y %H:%M",              # 5/30/2023 06:33
        "%Y/%m/%d %H:%M:%S",              # 2023/05/30 06:33:58
        "%Y-%m-%d %I:%M:%S %p",          # 2023-05-02 03:33:58 PM
        "%Y-%m-%d %I:%M:%S",             # 2023-05-02 03:33:58
        "%Y-%m-%d %I:%M",             # 2023-05-02 03:33
    ]
    
    for fmt in formats:
        try:
            ts = datetime.strptime(ts_normalized, fmt)
            print('ts', ts)
            if ts.date() > date.today():
                return False, "Date cannot be in the future"
            return True, ""
        except ValueError:
            continue
    
    return False, "Invalid timestamp. Use formats like: YYYY-MM-DD HH:MM:SS or M/D/YYYY H:MM:SS AM/PM"


def validate_ip_address(ip_str):
    """Validate IPv4 address format.
    Returns: (is_valid, error_message)
    """
    if not ip_str or ip_str.strip() == "":
        return False, "IP address is required"
    
    try:
        ipaddress.IPv4Address(ip_str)
        return True, ""
    except (ipaddress.AddressValueError, ValueError):
        return False, "Invalid IPv4 address (e.g., 192.168.1.1)"


def validate_ip_optional(ip_str):
    """Validate IP address if provided (optional field).
    Returns: (is_valid, error_message)
    """
    if not ip_str or ip_str.strip() == "":
        return True, ""  # Optional field
    return validate_ip_address(ip_str)


def validate_port(port_val):
    """Validate port number (0-65535).
    Returns: (is_valid, error_message)
    """
    try:
        port_val = int(port_val)
        if port_val < 0 or port_val > 65535:
            return False, "Port must be 0-65535"
        return True, ""
    except (ValueError, TypeError):
        return False, "Port must be numeric"


def validate_packet_length(length_val):
    """Validate packet length (0-2000).
    Returns: (is_valid, error_message)
    """
    try:
        length_val = int(length_val)
        if length_val < 0 or length_val > 2000:
            return False, "Packet length must be 0-2000"
        return True, ""
    except (ValueError, TypeError):
        return False, "Packet length must be numeric"


def validate_anomaly_score(score_val):
    """Validate anomaly score (0.0-100.0).
    Returns: (is_valid, error_message)
    """
    try:
        score_val = float(score_val)
        if score_val < 0.0 or score_val > 100.0:
            return False, "Score must be 0.0-100.0"
        return True, ""
    except (ValueError, TypeError):
        return False, "Score must be numeric"


def validate_user_agent(ua_str, is_optional=True):
    """Validate user agent string.
    Returns: (is_valid, error_message)
    """
    if not ua_str or ua_str.strip() == "":
        if is_optional:
            return True, ""  # Optional field
        else:
            return False, "Device Information is required"
    if len(ua_str) < 10:
        return False, "Device Information too short (min 10 chars)"
    return True, ""


def validate_time_hms(time_str):
    """Validate time string in HH:MM:SS format.
    Returns: (is_valid, error_message)
    """
    if not time_str or time_str.strip() == "":
        return False, "Time is required"
    
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
        if time_obj.hour > 23:
            return False, "Hours must be 0-23"
        if time_obj.minute > 59:
            return False, "Minutes must be 0-59"
        if time_obj.second > 59:
            return False, "Seconds must be 0-59"
        return True, ""
    except ValueError:
        return False, "Invalid input or format. Use format: HH:MM:SS (e.g., 06:33:58)"


def validate_dropdown(value, allowed_values, field_name):
    """Validate dropdown selection against allowed values.
    Returns: (is_valid, error_message)
    """
    if value not in allowed_values:
        allowed_str = " or ".join(allowed_values)
        return False, f"{field_name} must be {allowed_str}"
    return True, ""


def validate_csv_row(row, row_index):
    """
    Validate a single row from CSV for mandatory fields.
    Returns (is_valid, error_message).
    """
    errors = []
    
    # Define valid values for dropdown fields (REQUIRED - must have a value)
    required_dropdowns = {
        "Protocol": ["TCP", "UDP", "ICMP"],
        "Packet Type": ["Data", "Control"],
        "Traffic Type": ["HTTP", "DNS", "FTP"],
        "Severity Level": ["Low", "Medium", "High"],
        "Action Taken": ["Blocked", "Logged", "Ignored"],
        "Attack Signature": ["Known Pattern A", "Known Pattern B"],
        "Network Segment": ["Segment A", "Segment B", "Segment C"],
        "Log Source": ["Firewall", "Server"],
    }
    
    # Define valid values for dropdown fields (OPTIONAL - empty cell means "None")
    optional_with_none_dropdowns = {
        "Malware Indicators": ["None", "IoC Detected"],
        "Alerts/Warnings": ["None", "Alert Triggered"],
        "Firewall Logs": ["None", "Log Data"],
        "IDS/IPS Alerts": ["None", "Alert Data"],
    }
    
    # Timestamp validation
    timestamp_val = row.get("Timestamp", "")
    if pd.isna(timestamp_val) or str(timestamp_val).strip() == "":
        errors.append("Timestamp is required")
    else:
        is_valid, err_msg = validate_timestamp(str(timestamp_val))
        if not is_valid:
            errors.append(f"Timestamp: {err_msg}")
    
    # Source IP validation
    source_ip = row.get("Source IP Address", "")
    if pd.isna(source_ip) or str(source_ip).strip() == "":
        errors.append("Source IP Address is required")
    else:
        is_valid, err_msg = validate_ip_address(str(source_ip))
        if not is_valid:
            errors.append(f"Source IP Address: {err_msg}")
    
    # Destination IP validation
    dest_ip = row.get("Destination IP Address", "")
    if pd.isna(dest_ip) or str(dest_ip).strip() == "":
        errors.append("Destination IP Address is required")
    else:
        is_valid, err_msg = validate_ip_address(str(dest_ip))
        if not is_valid:
            errors.append(f"Destination IP Address: {err_msg}")
    
    # Port validations
    try:
        source_port = int(row.get("Source Port", 0))
        is_valid, err_msg = validate_port(source_port)
        if not is_valid:
            errors.append(f"Source Port: {err_msg}")
    except (ValueError, TypeError):
        errors.append("Source Port must be numeric")
    
    try:
        dest_port = int(row.get("Destination Port", 0))
        is_valid, err_msg = validate_port(dest_port)
        if not is_valid:
            errors.append(f"Destination Port: {err_msg}")
    except (ValueError, TypeError):
        errors.append("Destination Port must be numeric")
    
    # Packet length validation
    try:
        pkt_len = int(row.get("Packet Length", 0))
        is_valid, err_msg = validate_packet_length(pkt_len)
        if not is_valid:
            errors.append(f"Packet Length: {err_msg}")
    except (ValueError, TypeError):
        errors.append("Packet Length must be numeric")
    
    # Anomaly score validation
    try:
        anom_score = float(row.get("Anomaly Scores", 0))
        is_valid, err_msg = validate_anomaly_score(anom_score)
        if not is_valid:
            errors.append(f"Anomaly Scores: {err_msg}")
    except (ValueError, TypeError):
        errors.append("Anomaly Scores must be numeric")
    
    # REQUIRED dropdown field validations (must have a value)
    for field_name, allowed_values in required_dropdowns.items():
        field_value = row.get(field_name, "")
        allowed_str = " or ".join(allowed_values)
        if pd.isna(field_value) or str(field_value).strip() == "":
            errors.append(f"{field_name} is required and must be {allowed_str}")
        else:
            if str(field_value).strip() not in allowed_values:
                errors.append(f"{field_name} must be {allowed_str}")
    
    # OPTIONAL dropdown field validations (empty cell = "None" is valid)
    for field_name, allowed_values in optional_with_none_dropdowns.items():
        field_value = row.get(field_name, "")
        # If empty/NaN, treat as valid (equivalent to "None")
        if not (pd.isna(field_value) or str(field_value).strip() == ""):
            # Only validate if value provided
            if str(field_value).strip() not in allowed_values:
                allowed_str = " or ".join(allowed_values)
                errors.append(f"{field_name} must be {allowed_str}")
    
    # Device info validation (mandatory)
    device_info = row.get("Device Information", "")
    if pd.isna(device_info) or str(device_info).strip() == "":
        errors.append("Device Information is required")
    else:
        is_valid, err_msg = validate_user_agent(str(device_info), is_optional=False)
        if not is_valid:
            errors.append(f"Device Information: {err_msg}")
    
    # Proxy info validation (optional but if provided, validate)
    proxy_info = row.get("Proxy Information", "")
    if not pd.isna(proxy_info) and str(proxy_info).strip() != "":
        is_valid, err_msg = validate_ip_optional(str(proxy_info))
        if not is_valid:
            errors.append(f"Proxy Information: {err_msg}")
    
    if errors:
        # Format errors with bullet points on separate lines
        error_str = f"Row {row_index + 1}:\n  • " + "\n  • ".join(errors)
        return False, error_str
    return True, ""
