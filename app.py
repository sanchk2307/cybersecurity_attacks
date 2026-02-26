from datetime import datetime, date
import re
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import folium
from streamlit_folium import st_folium
from user_agents import parse as ua_parse
from src.utilities.helpers import engineer_features_for_new_row, ports_engineer_features_for_new_row
import hashlib
from pipeline import main as run_pipeline
import sys

# Import validation functions
from src.utilities.validation import (
    validate_timestamp,
    validate_ip_address,
    validate_ip_optional,
    validate_port,
    validate_packet_length,
    validate_anomaly_score,
    validate_user_agent,
    validate_time_hms,
    validate_dropdown,
    validate_csv_row,
)


def show_validation_error(message):
    """Display validation error in red, small text."""
    st.markdown(f"<small style='color: red;'>{message}</small>", unsafe_allow_html=True)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cyber Attack Detector", page_icon="🛡️", layout="wide")

# ============================================================
# LOAD MODEL & DATA
# ============================================================
# @st.cache_resource
# def load_model(model_type):
#     try:
#         with open(f"models/{model_type}_model.pkl", "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         sys.argv = "--no-figures --sequential"
#         st.error(f"Model file for {model_type} not found.")
#         run_pipeline()  # Run the pipeline to create the model if not found
#         with open(f"models/{model_type}_model.pkl", "rb") as f:
#             return pickle.load(f)

@st.cache_resource
def load_model(model_type: str):
    model_path = f"models/{model_type}_model.pkl"
 
    # First attempt: load existing model if present
    try:
        with open(f"models/{model_type}_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Display error only after running pipeline to create the model
        pass
 
    # If model is missing – run the pipeline to create it.
    # Use sequential, no‑figure mode to be lighter on resources.
    sys.argv = ["--no-figures", "--sequential"]
    with st.spinner(f"Training '{model_type}' model. This can take a few minutes..."):
        run_pipeline()
 
    # Second attempt: load the freshly trained model.
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Model file for `{model_type}` not found even after running the pipeline. "
            "Please check that training completed successfully."
        )
        raise

@st.cache_data
def load_dataset():
    return pd.read_csv("data/cybersecurity_attacks.csv")

# ============================================================
# FEATURE ENGINEERING PIPELINE AND HELPER FUNCTIONS ON USER INPUT
# ============================================================

def _predict_batch(df_input, feature_fn, model, target_names):
    """
    Runs feature engineering + prediction on a batch of rows.
    Returns df_output (with Prediction & Confidence), pred_labels, probabilities.
    """
    X_input = feature_fn(df_input)
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)

    pred_labels = [target_names[p] for p in predictions]

    df_output = df_input.copy()
    df_output["Prediction"] = pred_labels
    df_output["Confidence"] = [
        f"{probabilities[i][predictions[i]] * 100:.1f}%"
        for i in range(len(predictions))
    ]
    return df_output, pred_labels, probabilities

def _render_batch_results(df_output, df_upload, pred_labels, key_suffix=""):
    """
    Shared UI for tab 1 results: metrics, accuracy, pagination, table, download.
    """
    st.markdown("---")
    st.subheader("Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df_output))
    col2.metric("🔴 DDoS", pred_labels.count("DDoS"))
    col3.metric("🟡 Intrusion", pred_labels.count("Intrusion"))
    col4.metric("🟢 Malware", pred_labels.count("Malware"))

    # We can check whether the uploaded CSV has "Attack Type" and test the accuracy
    display_cols = ["Prediction", "Confidence"]
    if "Attack Type" in df_upload.columns:
        df_output["Correct"] = df_output["Prediction"] == df_output["Attack Type"]
        accuracy = df_output["Correct"].mean() * 100
        st.write(f"**Accuracy on uploaded data: {accuracy:.1f}%**")
        display_cols = ["Attack Type"] + display_cols + ["Correct"]

    display_cols = ["Source Port", "Destination Port", "Protocol"] + display_cols
    display_cols = [c for c in display_cols if c in df_output.columns]

    records_per_page_output = min(10, len(df_output))
    number_of_pages_output = max(
        1,
        (len(df_output) + records_per_page_output - 1) // records_per_page_output,
    )

    if st.session_state.current_page > number_of_pages_output:
        st.session_state.current_page = number_of_pages_output
    page_output = int(st.session_state.current_page)

    start_idx_output = (page_output - 1) * records_per_page_output
    end_idx_output = min(start_idx_output + records_per_page_output, len(df_output))

    st.caption(
        f"Showing records: **{start_idx_output + 1}-{end_idx_output}** "
        f"(page: **{page_output}** of **{number_of_pages_output}**)"
    )

    df_temp_output = df_output.iloc[start_idx_output:end_idx_output][display_cols]
    st.dataframe(df_temp_output, use_container_width=True)

    csv_result = df_output.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv_result,
        "predictions.csv",
        "text/csv",
        key=f"download_{key_suffix}",
    )

# ============================================================
# RUNNING PREDICTIONS
# ============================================================
def run_predictions(model_type, tab):
    model = model_data["model"]
    target_names = model_data["target_names"]
    label_encoder = model_data["label_encoder"]

    # -------- TAB 1: CSV Upload (batch) --------
    if tab == 1:
        df_upload = st.session_state.uploaded_df

        if model_type in ["logit", "randomforrest"]:
            feature_fn = lambda df: engineer_features_for_new_row(df, model_data)
        elif model_type == "extra_trees":
            feature_fn = lambda df: ports_engineer_features_for_new_row(df, model_data)
        else:
            st.error(f"Unknown model_type: {model_type}")
            return

        with st.spinner("Running feature engineering & prediction..."):
            df_output, pred_labels, _ = _predict_batch(
                df_upload, feature_fn, model, target_names
            )

        _render_batch_results(
            df_output=df_output,
            df_upload=df_upload,
            pred_labels=pred_labels,
            key_suffix=f"{model_type}_tab1",
        )
        return

    # -------- TAB 2: Pick a Row from dataset --------
    if tab == 2:
        with st.spinner("Running prediction..."):
            if model_type in ["logit", "randomforrest"]:
                df_transformed = model_data["df_transformed"]
                X_cols = model_data["X_cols"]
                X_input = df_transformed.loc[[row_number], X_cols].values.astype(float)
                X_input = np.nan_to_num(X_input, nan=0.0)
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                pred_label = label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100

            elif model_type == "extra_trees":
                X_input = ports_engineer_features_for_new_row(selected_row, model_data)
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                pred_label = target_names[prediction]
                confidence = probabilities[prediction] * 100
            else:
                st.error(f"Unknown model_type: {model_type}")
                return

        show_prediction(pred_label, confidence, probabilities, target_names)

        if pred_label == actual_type:
            st.success(
                f"✅ Correct! Model predicted **{pred_label}**, actual is **{actual_type}**."
            )
        else:
            st.error(
                f"❌ Wrong. Model predicted **{pred_label}**, actual is **{actual_type}**."
            )
        return

    # -------- TAB 3: Manual input (single row) --------
    if tab == 3:
        with st.spinner("Running feature engineering & prediction..."):
            if model_type in ["logit", "randomforrest"]:
                X_input = engineer_features_for_new_row(row, model_data)
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                pred_label = label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100
            elif model_type == "extra_trees":
                X_input = ports_engineer_features_for_new_row(row, model_data)
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                pred_label = target_names[prediction]
                confidence = probabilities[prediction] * 100
            else:
                st.error(f"Unknown model_type: {model_type}")
                return

        show_prediction(pred_label, confidence, probabilities, target_names)

# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================
def show_prediction(pred_label, confidence, probabilities, target_names):
    colors = {"DDoS": "#3B82F6", "Intrusion": "#F59E0B", "Malware": "#10B981"}
    icons = {"DDoS": "🌊", "Intrusion": "🕵️", "Malware": "🦠"}

    color = colors.get(pred_label, "#888888")
    icon = icons.get(pred_label, "⚪")

    st.markdown("---")
    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div style="
            border: 3px solid {color};
            border-radius: 12px;
            padding: 18px 24px;
            background-color: {color}18;
            display: inline-block;
            min-width: 260px;
            margin-bottom: 12px;
        ">
            <h2 style="color:{color}; margin:0;">{icon} {pred_label}</h2>
            <p style="color:#888; margin:4px 0 0 0;">Confidence: <strong style="color:{color};">{confidence:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bars
    st.write("**Probability per class:**")
    for i, name in enumerate(target_names):
        pct = float(probabilities[i]) * 100
        st.progress(float(probabilities[i]), text=f"{name}: {pct:.1f}%")

# ============================================================
# APP HEADER
# ============================================================
st.title("🛡️ Cyber Attack Detector")
st.caption(
    "Machine Learning pipeline for predicting cyber attack types from network traffic data."
)

# Load dataset first (needed for training)
try:
    df_dataset = load_dataset()
except FileNotFoundError:
    st.error(
        "❌ Dataset not found. Place `cybersecurity_attacks.csv` in the `data/` folder."
    )
    st.stop()

# Load model using modelling()
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
    This application predicts the type of cyber attack 
    (DDoS, Intrusion, or Malware) from raw network traffic data.
    """
    )
    model_options = st.selectbox(
        "Model",
        options=[
            ("extra_trees", "Extra Trees"),
            ("logit", "Logit"),
            ("randomforrest", "Random Forest"),
        ],
        format_func=lambda x: x[1],
        index=0,
        key="model_select",
        disabled=False  # Locksto dropdown values only
    )
    selected_model_type = model_options[0]
    try:
        model_data = load_model(selected_model_type)
    except FileNotFoundError:
        st.error(
            f"Model not found. Add the '{selected_model_type}_model.pkl' in the 'models/' folder."
        )

    st.markdown("---")
    st.subheader("Model Performance")
    accscore = model_data.get("accscore", None)
    if accscore is not None:
        st.metric("Accuracy (test set)", f"{accscore * 100:.1f}%")
    else:
        st.metric("Accuracy", "—")

    with st.expander("📊 Confusion Matrix", expanded=False):
        fig_cm = model_data.get("fig_cm")
        if fig_cm is not None:
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Not available.")

    with st.expander("📈 ROC Curve", expanded=False):
        fig_roc = model_data.get("fig_roc")
        if fig_roc is not None:
            # Ensure legend shows AUC with only 2 digits after the decimal, even if the
            # figure comes from an older pickled model.
            auc_re = re.compile(r"^(?P<label>.*)\(\s*AUC\s*=\s*(?P<auc>[0-9]*\.?[0-9]+)\s*\)\s*$")
            for tr in fig_roc.data:
                if getattr(tr, "name", "") == "Random":
                    continue
                m = auc_re.match(getattr(tr, "name", "") or "")
                if m:
                    label = m.group("label").rstrip()
                    auc_val = float(m.group("auc"))
                    tr.name = f"{label}(AUC = {auc_val:.2f})"
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Not available.")



    st.markdown("---")
    st.subheader("Pipeline")
    st.write(
        """
    1. Raw data input (25 columns)
    2. Feature engineering (cleaning, encoding, parsing)
    3. Target encoding
    4. Prediction based on selected model
    5. Result display
    """
    )

    st.markdown("---")
    st.caption("DSTI MSc Project — Cyber Attack Detection")

st.markdown("---")

# ============================================================
# THREE TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📂 CSV Upload", "🔢 Pick a Row", "✏️ Manual Input"])

# ============================================================
# TAB 1: CSV UPLOAD
# ============================================================
with tab1:
    st.subheader("Upload a CSV file")
    st.caption(
        "Upload rows from the original dataset. The pipeline handles feature engineering automatically."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # We store session state to make sure that the state persists between reruns while we use pagination
        st.session_state.setdefault("uploaded_df", None)
        st.session_state.setdefault("uploaded_file_hash", None)
        st.session_state.setdefault("current_page", 1)
        st.session_state.setdefault("csv_validation_errors", [])

        # We reload the file if the name, size or hash changes
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if st.session_state.uploaded_file_hash != file_hash:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.current_page = 1
            st.session_state.csv_validation_errors = []  # Reset errors on new file

        df_upload = st.session_state.uploaded_df
        st.write(f"✅ **{len(df_upload)} rows loaded**")

        records_per_page = min(10, len(df_upload))
        number_of_pages = max(
            1, (len(df_upload) + records_per_page - 1) // records_per_page
        )
        if st.session_state.current_page > number_of_pages:
            # In case, we remove rows from the editable table
            st.session_state.current_page = number_of_pages
        page = int(st.session_state.current_page)
        start_idx = (page - 1) * records_per_page
        end_idx = min(start_idx + records_per_page, len(df_upload))
        st.caption(
            f"Showing records: **{start_idx + 1}-{end_idx}** (page: **{page}** of **{number_of_pages}**)"
        )

        df_temp = df_upload.iloc[start_idx:end_idx].copy()
        
        # Reset index to start from 1 for display
        df_temp_display = df_temp.reset_index(drop=True)
        df_temp_display.index = df_temp_display.index + 1
        
        edited_df_temp = st.data_editor(
            df_temp_display,
            use_container_width=True,
            num_rows="fixed",
            key=f"df_edit_page_{page}",
        )

        # Map edits back to the original df in session_state using original indices
        edited_df_temp.index = edited_df_temp.index - 1  # Convert back to 0-indexed
        st.session_state.uploaded_df.loc[edited_df_temp.index, :] = edited_df_temp
        df_upload = st.session_state.uploaded_df
        
        # Real-time validation of all rows
        validation_errors = []
        for idx, row in df_upload.iterrows():
            is_valid, error_msg = validate_csv_row(row, idx)
            if not is_valid:
                validation_errors.append(error_msg)
        st.session_state.csv_validation_errors = validation_errors

        _, btn_prev, curr_page, btn_next = st.columns(
            [24, 1, 1, 1], vertical_alignment="center"
        )
        with btn_prev:
            if (
                st.button(
                    "",
                    key="prev_page",
                    icon=":material/arrow_left:",
                    disabled=st.session_state.current_page <= 1,
                )
                and st.session_state.current_page > 1
            ):
                st.session_state.current_page -= 1
        with curr_page:
            st.caption(f"**{st.session_state.current_page}**", text_alignment="center")
        with btn_next:
            if (
                st.button(
                    "",
                    key="next_page",
                    icon=":material/arrow_right:",
                    disabled=st.session_state.current_page >= number_of_pages,
                )
                and st.session_state.current_page < number_of_pages
            ):
                st.session_state.current_page += 1

        # Display validation errors below button (real-time) with proper formatting
        if st.session_state.csv_validation_errors:
            # Count total field errors from all rows
            total_field_errors = sum(err.count("•") for err in st.session_state.csv_validation_errors)
            st.warning(f"⚠️ **{total_field_errors} validation issue(s) found in {len(st.session_state.csv_validation_errors)} row(s):**")
            for err in st.session_state.csv_validation_errors[:15]:  # Show first 15 rows with errors
                st.markdown(f"{err}")  # Already formatted with bullet points
            if len(st.session_state.csv_validation_errors) > 15:
                st.markdown(f"... and {len(st.session_state.csv_validation_errors) - 15} more row(s) with errors")
        
        if st.button("🔍 Run Prediction", key="btn_csv", disabled=bool(st.session_state.csv_validation_errors)):
            st.success("✅ All rows validated successfully!")
            run_predictions(selected_model_type, 1)
# ============================================================
# TAB 2: PICK A ROW FROM DATASET
# ============================================================
with tab2:
    st.subheader("Pick a row from the test set")
    st.caption(
        "Select a row from the held-out test split. The app will load it and predict the attack type."
    )

    test_indices = model_data["test_indices"]
    test_position = st.number_input(
        f"Row position in test set (0 – {len(test_indices) - 1})",
        min_value=0,
        max_value=len(test_indices) - 1,
        value=0,
        step=1,
    )
    row_number = test_indices[test_position]  # original dataset index
    selected_row = df_dataset.iloc[[row_number]]

    st.caption(
        f"Dataset index: {row_number} | Test set size: {len(test_indices):,} rows"
    )
    st.write("**Selected row:**")
    st.dataframe(selected_row, use_container_width=True)

    actual_type = selected_row["Attack Type"].values[0]
    st.write(f"**Actual Attack Type:** {actual_type}")

    if st.button("🔍 Run Prediction", key="btn_row"):
        run_predictions(selected_model_type, 2)
# ============================================================
# FUNCTIONS TO DISPLAY GEO-LOCATION OF IP ADDRESS
# ============================================================

@st.cache_data
def geocode_ip(ip_address):
    """
    Geocode an IP address to latitude and longitude using ip-api.com
    Returns: lat, lon, city, country
    """
    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                return (
                    data["lat"],
                    data["lon"],
                    data.get("city", "Unknown"),
                    data.get("country", "Unknown"),
                )
    except Exception as e:
        st.warning(f"Could not geocode {ip_address}: {str(e)}")
    return None, None, None, None

# ============================================================
# TAB 3: MANUAL INPUT
# ============================================================
with tab3:
    st.subheader("Manual input")
    st.caption("Fill in all fields manually. The pipeline handles the rest.")
    
    # Initialize session state for Tab 3 validation
    if "tab3_validation" not in st.session_state:
        st.session_state.tab3_validation = {}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Network Info**")
        
        # Timestamp: Date picker + Time input (HH:MM:SS in single field)
        st.write("Timestamp *")
        col_date, col_time = st.columns(2)
        with col_date:
            timestamp_date = st.date_input(
                "Date",
                value=date(2023, 5, 30),
                max_value=date.today(),
                label_visibility="collapsed",
                key="tab3_timestamp_date"
            )
        with col_time:
            timestamp_time_str = st.text_input(
                "Time (HH:MM:SS)",
                value="06:33:58",
                max_chars=8,
                label_visibility="collapsed",
                key="tab3_timestamp_time"
            )
        
        # Validate time format (HH:MM:SS)
        is_valid_time, err_time = validate_time_hms(timestamp_time_str)
        if not is_valid_time:
            show_validation_error(err_time)
            st.session_state.tab3_validation["timestamp"] = False
        else:
            timestamp = f"{timestamp_date} {timestamp_time_str}"
            is_valid_ts, err_ts = validate_timestamp(timestamp)
            st.session_state.tab3_validation["timestamp"] = is_valid_ts
            if not is_valid_ts:
                show_validation_error(err_ts)
        
        # Source IP Address
        st.write("Source IP Address *")
        source_ip = st.text_input(
            "Source IP",
            value="103.216.15.12",
            label_visibility="collapsed",
            key="tab3_source_ip"
        )
        is_valid_src_ip, err_src_ip = validate_ip_address(source_ip)
        st.session_state.tab3_validation["source_ip"] = is_valid_src_ip
        if not is_valid_src_ip:
            show_validation_error(err_src_ip)
        
        # Destination IP Address
        st.write("Destination IP Address *")
        dest_ip = st.text_input(
            "Destination IP",
            value="84.9.164.252",
            label_visibility="collapsed",
            key="tab3_dest_ip"
        )
        is_valid_dst_ip, err_dst_ip = validate_ip_address(dest_ip)
        st.session_state.tab3_validation["dest_ip"] = is_valid_dst_ip
        if not is_valid_dst_ip:
            show_validation_error(err_dst_ip)
        
        # Geocode IP addresses
        src_lat, src_lon, src_city, src_country = geocode_ip(source_ip) if is_valid_src_ip else (None, None, None, None)
        dst_lat, dst_lon, dst_city, dst_country = geocode_ip(dest_ip) if is_valid_dst_ip else (None, None, None, None)

        # Build folium map
        if src_lat and dst_lat:
            center_lat = (src_lat + dst_lat) / 2
            center_lon = (src_lon + dst_lon) / 2
        elif src_lat:
            center_lat, center_lon = src_lat, src_lon
        else:
            center_lat, center_lon = 20.0, 0.0

        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap"
        )

        if src_lat and src_lon:
            folium.CircleMarker(
                location=[src_lat, src_lon],
                radius=10,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.8,
                tooltip=f"🔴 Source: {src_city}, {src_country}\nIP: {source_ip}",
            ).add_to(m)

        if dst_lat and dst_lon:
            folium.CircleMarker(
                location=[dst_lat, dst_lon],
                radius=10,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.8,
                tooltip=f"🔵 Destination: {dst_city}, {dst_country}\nIP: {dest_ip}",
            ).add_to(m)

        if src_lat and dst_lat:
            folium.PolyLine(
                locations=[[src_lat, src_lon], [dst_lat, dst_lon]],
                color="orange",
                weight=2,
                dash_array="5",
            ).add_to(m)

        st_folium(m, width=430, height=300)

        # Source Port
        st.write("Source Port *")
        source_port = st.number_input(
            "Source Port",
            # min_value=0,
            # max_value=65535,
            value=31225,
            label_visibility="collapsed",
            key="tab3_source_port"
        )
        is_valid_src_port, err_src_port = validate_port(source_port)
        st.session_state.tab3_validation["source_port"] = is_valid_src_port
        if not is_valid_src_port:
            show_validation_error(err_src_port)
        
        # Destination Port
        st.write("Destination Port *")
        dest_port = st.number_input(
            "Destination Port",
            # min_value=0,
            # max_value=65535,
            value=17616,
            label_visibility="collapsed",
            key="tab3_dest_port"
        )
        is_valid_dst_port, err_dst_port = validate_port(dest_port)
        st.session_state.tab3_validation["dest_port"] = is_valid_dst_port
        if not is_valid_dst_port:
            show_validation_error(err_dst_port)
        
        # Protocol
        st.write("Protocol *")
        protocol_options = ["TCP", "UDP", "ICMP"]
        protocol = st.selectbox(
            "Protocol",
            protocol_options,
            label_visibility="collapsed",
            key="tab3_protocol",
            index=0
        )
        is_valid_protocol, err_protocol = validate_dropdown(protocol, protocol_options, "Protocol")
        st.session_state.tab3_validation["protocol"] = is_valid_protocol
        if not is_valid_protocol:
            show_validation_error(err_protocol)
        
        # Packet Length
        st.write("Packet Length *")
        packet_length = st.slider(
            "Packet Length",
            min_value=0,
            max_value=2000,
            value=503,
            step=1,
            label_visibility="collapsed",
            key="tab3_packet_length"
        )
        is_valid_pkt_len, err_pkt_len = validate_packet_length(packet_length)
        st.session_state.tab3_validation["packet_length"] = is_valid_pkt_len
        if not is_valid_pkt_len:
            show_validation_error(err_pkt_len)
        
        # Packet Type
        st.write("Packet Type *")
        packet_type_options = ["Data", "Control"]
        packet_type = st.selectbox(
            "Packet Type",
            packet_type_options,
            label_visibility="collapsed",
            key="tab3_packet_type",
            index=0
        )
        is_valid_ptype, err_ptype = validate_dropdown(packet_type, packet_type_options, "Packet Type")
        st.session_state.tab3_validation["packet_type"] = is_valid_ptype
        if not is_valid_ptype:
            show_validation_error(err_ptype)
        
        # Traffic Type
        st.write("Traffic Type *")
        traffic_type_options = ["HTTP", "DNS", "FTP"]
        traffic_type = st.selectbox(
            "Traffic Type",
            traffic_type_options,
            label_visibility="collapsed",
            key="tab3_traffic_type",
            index=0
        )
        is_valid_ttype, err_ttype = validate_dropdown(traffic_type, traffic_type_options, "Traffic Type")
        st.session_state.tab3_validation["traffic_type"] = is_valid_ttype
        if not is_valid_ttype:
            show_validation_error(err_ttype)

    with col2:
        st.write("**Security Info**")
        
        # Anomaly Score
        st.write("Anomaly Scores *")
        anomaly_score = st.slider(
            "Anomaly Scores",
            min_value=0.0,
            max_value=100.0,
            value=28.67,
            step=0.01,
            label_visibility="collapsed",
            key="tab3_anomaly_score"
        )
        is_valid_anom, err_anom = validate_anomaly_score(anomaly_score)
        st.session_state.tab3_validation["anomaly_score"] = is_valid_anom
        if not is_valid_anom:
            show_validation_error(err_anom)
        
        # Severity
        st.write("Severity Level *")
        severity_options = ["Low", "Medium", "High"]
        severity = st.selectbox(
            "Severity Level",
            severity_options,
            label_visibility="collapsed",
            key="tab3_severity",
            index=0
        )
        is_valid_sev, err_sev = validate_dropdown(severity, severity_options, "Severity Level")
        st.session_state.tab3_validation["severity"] = is_valid_sev
        if not is_valid_sev:
            show_validation_error(err_sev)
        
        # Action
        st.write("Action Taken *")
        action_options = ["Blocked", "Logged", "Ignored"]
        action = st.selectbox(
            "Action Taken",
            action_options,
            label_visibility="collapsed",
            key="tab3_action",
            index=0
        )
        is_valid_act, err_act = validate_dropdown(action, action_options, "Action Taken")
        st.session_state.tab3_validation["action"] = is_valid_act
        if not is_valid_act:
            show_validation_error(err_act)
        
        # Attack Signature
        st.write("Attack Signature *")
        attack_sig_options = ["Known Pattern A", "Known Pattern B"]
        attack_sig = st.selectbox(
            "Attack Signature",
            attack_sig_options,
            label_visibility="collapsed",
            key="tab3_attack_sig",
            index=0
        )
        is_valid_asig, err_asig = validate_dropdown(attack_sig, attack_sig_options, "Attack Signature")
        st.session_state.tab3_validation["attack_sig"] = is_valid_asig
        if not is_valid_asig:
            show_validation_error(err_asig)
        
        # Network Segment
        st.write("Network Segment *")
        network_seg_options = ["Segment A", "Segment B", "Segment C"]
        network_seg = st.selectbox(
            "Network Segment",
            network_seg_options,
            label_visibility="collapsed",
            key="tab3_network_seg",
            index=0
        )
        is_valid_nseg, err_nseg = validate_dropdown(network_seg, network_seg_options, "Network Segment")
        st.session_state.tab3_validation["network_seg"] = is_valid_nseg
        if not is_valid_nseg:
            show_validation_error(err_nseg)
        
        # Log Source
        st.write("Log Source *")
        log_source_options = ["Firewall", "Server"]
        log_source = st.selectbox(
            "Log Source",
            log_source_options,
            label_visibility="collapsed",
            key="tab3_log_source",
            index=0
        )
        is_valid_lsrc, err_lsrc = validate_dropdown(log_source, log_source_options, "Log Source")
        st.session_state.tab3_validation["log_source"] = is_valid_lsrc
        if not is_valid_lsrc:
            show_validation_error(err_lsrc)
        
        # Malware Indicators
        st.write("Malware Indicators *")
        malware_ind_options = ["None", "IoC Detected"]
        malware_ind = st.selectbox(
            "Malware Indicators",
            malware_ind_options,
            label_visibility="collapsed",
            key="tab3_malware_ind",
            index=0
        )
        is_valid_mw, err_mw = validate_dropdown(malware_ind, malware_ind_options, "Malware Indicators")
        st.session_state.tab3_validation["malware_ind"] = is_valid_mw
        if not is_valid_mw:
            show_validation_error(err_mw)
        
        # Alerts/Warnings
        st.write("Alerts/Warnings *")
        alerts_warn_options = ["None", "Alert Triggered"]
        alerts_warn = st.selectbox(
            "Alerts/Warnings",
            alerts_warn_options,
            label_visibility="collapsed",
            key="tab3_alerts_warn",
            index=0
        )
        is_valid_aw, err_aw = validate_dropdown(alerts_warn, alerts_warn_options, "Alerts/Warnings")
        st.session_state.tab3_validation["alerts_warn"] = is_valid_aw
        if not is_valid_aw:
            show_validation_error(err_aw)

    with col3:
        st.write("**Other Info**")
        
        # Firewall Logs
        st.write("Firewall Logs *")
        firewall_logs_options = ["None", "Log Data"]
        firewall_logs = st.selectbox(
            "Firewall Logs",
            firewall_logs_options,
            label_visibility="collapsed",
            key="tab3_firewall_logs",
            index=0
        )
        is_valid_fw, err_fw = validate_dropdown(firewall_logs, firewall_logs_options, "Firewall Logs")
        st.session_state.tab3_validation["firewall_logs"] = is_valid_fw
        if not is_valid_fw:
            show_validation_error(err_fw)
        
        # IDS/IPS Alerts
        st.write("IDS/IPS Alerts *")
        ids_alerts_options = ["None", "Alert Data"]
        ids_alerts = st.selectbox(
            "IDS/IPS Alerts",
            ids_alerts_options,
            label_visibility="collapsed",
            key="tab3_ids_alerts",
            index=0
        )
        is_valid_ids, err_ids = validate_dropdown(ids_alerts, ids_alerts_options, "IDS/IPS Alerts")
        st.session_state.tab3_validation["ids_alerts"] = is_valid_ids
        if not is_valid_ids:
            show_validation_error(err_ids)
        
        # User Information (Optional)
        st.write("User Information")
        user_info = st.text_input(
            "User Information",
            value="Reyansh Dugal",
            label_visibility="collapsed",
            key="tab3_user_info"
        )
        st.session_state.tab3_validation["user_info"] = True  # Optional
        
        # Device Information (Mandatory)
        st.write("Device Information *")
        device_info = st.text_input(
            "Device Information",
            value="Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.2; Trident/5.0)",
            label_visibility="collapsed",
            key="tab3_device_info"
        )
        is_valid_ua, err_ua = validate_user_agent(device_info, is_optional=False)
        st.session_state.tab3_validation["device_info"] = is_valid_ua
        if not is_valid_ua:
            show_validation_error(err_ua)
        
        # Geo-location Data (Optional)
        st.write("Geo-location Data")
        geo_location = st.text_input(
            "Geo-location Data",
            value="Mumbai, Maharashtra",
            label_visibility="collapsed",
            key="tab3_geo_location"
        )
        st.session_state.tab3_validation["geo_location"] = True  # Optional
        
        # Proxy Information (Optional)
        st.write("Proxy Information")
        proxy_info = st.text_input(
            "Proxy Information (leave empty if none)",
            value="",
            label_visibility="collapsed",
            key="tab3_proxy_info"
        )
        is_valid_proxy, err_proxy = validate_ip_optional(proxy_info)
        st.session_state.tab3_validation["proxy_info"] = is_valid_proxy
        if not is_valid_proxy:
            show_validation_error(err_proxy)
        
        # Payload Data (Optional)
        st.write("Payload Data")
        payload_data = st.text_area(
            "Payload Data",
            value="Sample payload data",
            height=68,
            label_visibility="collapsed",
            key="tab3_payload_data"
        )
        st.session_state.tab3_validation["payload_data"] = True  # Optional
    
    # Check if all mandatory validations pass
    mandatory_fields = [
        "timestamp", "source_ip", "dest_ip", "source_port", "dest_port", 
        "protocol", "packet_length", "packet_type", "traffic_type",
        "anomaly_score", "severity", "action", "attack_sig", "network_seg",
        "log_source", "malware_ind", "alerts_warn", "firewall_logs", "ids_alerts", "device_info"
    ]
    
    all_valid = all(st.session_state.tab3_validation.get(field, False) for field in mandatory_fields)
    
    # Show info message if button is disabled
    if not all_valid:
        st.info("ℹ️ Missing mandatory fields (marked with *) or invalid input")
        st.button("🔍 Run Prediction", key="btn_manual", disabled=True)
    else:
        if st.button("🔍 Run Prediction", key="btn_manual", disabled=False):
            row = pd.DataFrame(
                [
                    {
                        "Timestamp": timestamp,
                        "Source IP Address": source_ip,
                        "Destination IP Address": dest_ip,
                        "Source Port": source_port,
                        "Destination Port": dest_port,
                        "Protocol": protocol,
                        "Packet Length": packet_length,
                        "Packet Type": packet_type,
                        "Traffic Type": traffic_type,
                        "Payload Data": payload_data,
                        "Malware Indicators": (
                            "IoC Detected" if malware_ind == "IoC Detected" else np.nan
                        ),
                        "Anomaly Scores": anomaly_score,
                        "Alerts/Warnings": (
                            "Alert Triggered"
                            if alerts_warn == "Alert Triggered"
                            else np.nan
                        ),
                        "Attack Signature": attack_sig,
                        "Action Taken": action,
                        "Severity Level": severity,
                        "User Information": user_info if user_info.strip() else np.nan,
                        "Device Information": device_info if device_info.strip() else np.nan,
                        "Network Segment": network_seg,
                        "Geo-location Data": geo_location if geo_location.strip() else np.nan,
                        "Proxy Information": proxy_info if proxy_info.strip() else np.nan,
                        "Firewall Logs": (
                            "Log Data" if firewall_logs == "Log Data" else np.nan
                        ),
                        "IDS/IPS Alerts": (
                            "Alert Data" if ids_alerts == "Alert Data" else np.nan
                        ),
                        "Log Source": log_source,
                    }
                ]
            )
            run_predictions(selected_model_type, 3)