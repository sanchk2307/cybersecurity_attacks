import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import folium
from streamlit_folium import st_folium
from user_agents import parse as ua_parse
from helpers import feature_engineering, Device_type, geolocation_data, modelling, preprocess_for_modelling, engineer_features_for_new_row

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cyber Attack Detector",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model(df_dataset):
    """
    Trains the model using modelling() from helpers.py
    and caches it so it only runs once.
    """
    crosstabs_x_AttackType = {}
    # Preprocess raw CSV to create all columns expected by feature_engineering()
    df_work = preprocess_for_modelling(df_dataset)
    y_pred, model, label_encoder, X_cols, fig_cm, fig_roc, accscore, test_indices = modelling(
        df=df_work,
        crosstabs_x_AttackType=crosstabs_x_AttackType,
        testp=0.1,
        features_mask=[True, False, False, True, True, True, True, True, True],
        threshold_floor=37.5,
        contvar_nobs_b_class=15,
        dynamic_threshold_pars=[5, 100, 10, 37.5],
        model_type="randomforrest",
        split_before_training=False
    )
    target_names = label_encoder.classes_
    return {
        "model": model,
        "label_encoder": label_encoder,
        "target_names": target_names,
        "crosstabs_x_AttackType": crosstabs_x_AttackType,
        "X_cols": X_cols,
        "df_transformed": df_work,   # df with bias columns added — used for row-level prediction
        "test_indices": test_indices,    # original row indices from the test split
        "threshold_floor": 37.5,
        "dynamic_threshold_pars": [5, 100, 10, 37.5],
        "fig_cm": fig_cm,
        "fig_roc": fig_roc,
        "accscore": accscore,
    }

@st.cache_data
def load_dataset():
    return pd.read_csv('data/cybersecurity_attacks.csv')

# ============================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df_input, model_data):
    """
    Takes a raw row (all 25 columns) from the dataset
    and transforms it into the feature vector the model expects.
    This runs in the background - the user never sees this.
    """
    df = df_input.copy()
    
    # --- Binary columns: NaN = 0, value = 1 ---
    for col in ['Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 'IDS/IPS Alerts']:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)
        else:
            df[col] = 0
    
    if 'Proxy Information' in df.columns:
        df['has_proxy'] = df['Proxy Information'].notna().astype(int)
    else:
        df['has_proxy'] = 0
    
    # --- Timestamp features ---
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['hour'] = df['Timestamp'].dt.hour.fillna(12).astype(int)
        df['dayofweek'] = df['Timestamp'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    else:
        df['hour'], df['dayofweek'], df['is_weekend'] = 12, 0, 0
    
    # --- Port features ---
    def port_cat(port):
        try:
            port = int(port)
            if port <= 1023: return 'wellknown'
            elif port <= 49151: return 'registered'
            else: return 'dynamic'
        except:
            return 'registered'
    
    df['src_port_cat'] = df['Source Port'].apply(port_cat)
    df['dst_port_cat'] = df['Destination Port'].apply(port_cat)
    df['port_diff'] = abs(df['Source Port'].astype(float) - df['Destination Port'].astype(float))
    
    # --- Device Information parsing ---
    if 'Device Information' in df.columns:
        df['os_family'] = df['Device Information'].apply(lambda x: ua_parse(str(x)).os.family)
        df['browser_family'] = df['Device Information'].apply(lambda x: ua_parse(str(x)).browser.family)
        df['device_type'] = df['Device Information'].apply(
            lambda x: 'Mobile' if ua_parse(str(x)).is_mobile 
            else ('Tablet' if ua_parse(str(x)).is_tablet 
            else ('PC' if ua_parse(str(x)).is_pc else 'Other'))
        )
    else:
        df['os_family'], df['browser_family'], df['device_type'] = 'Windows', 'Chrome', 'PC'
    
    # --- Total alerts ---
    df['total_alerts'] = (df['Malware Indicators'] + df['Alerts/Warnings'] + 
                          df['Firewall Logs'] + df['IDS/IPS Alerts'])
    
    # --- Build feature vector using bias columns from training ---
    # For manual/CSV input: assign bias=0 for unknown bins (conservative default)
    X_cols = model_data['X_cols']
    for col in X_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[X_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    return X

# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================
def show_prediction(pred_label, confidence, probabilities, target_names):
    colors = {'DDoS': '#FF4B4B', 'Intrusion': '#FFA500', 'Malware': '#00CC66'}
    icons  = {'DDoS': '🔴',      'Intrusion': '🟡',      'Malware': '🟢'}

    color = colors.get(pred_label, '#888888')
    icon  = icons.get(pred_label, '⚪')

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
        unsafe_allow_html=True
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
st.caption("Machine Learning pipeline for predicting cyber attack types from network traffic data.")

# Load dataset first (needed for training)
try:
    df_dataset = load_dataset()
except FileNotFoundError:
    st.error("❌ Dataset not found. Place `cybersecurity_attacks.csv` in the `data/` folder.")
    st.stop()

# Train / load model using modelling()
try:
    with st.spinner("⏳ Training model, please wait..."):
        model_data = load_model(df_dataset)
    model        = model_data['model']
    target_names = model_data['target_names']
    label_encoder = model_data['label_encoder']
    crosstabs_x_AttackType = model_data['crosstabs_x_AttackType']
except Exception as e:
    st.error(f"❌ Model training failed: {e}")
    st.stop()

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
    st.caption("Upload rows from the original dataset. The pipeline handles feature engineering automatically.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"✅ **{len(df_upload)} rows loaded**")
        st.dataframe(df_upload.head(5), use_container_width=True, height=200)
        
        if st.button("🔍 Run Prediction", key='btn_csv'):
            with st.spinner("Running feature engineering & prediction..."):
                X_input = engineer_features_for_new_row(df_upload, model_data)
                predictions = model.predict(X_input)
                probabilities = model.predict_proba(X_input)
                
                pred_labels = [target_names[p] for p in predictions]
                df_upload['Prediction'] = pred_labels
                df_upload['Confidence'] = [f"{probabilities[i][predictions[i]]*100:.1f}%" 
                                           for i in range(len(predictions))]
            
            # Summary metrics
            st.markdown("---")
            st.subheader("Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", len(df_upload))
            col2.metric("🔴 DDoS", pred_labels.count('DDoS'))
            col3.metric("🟡 Intrusion", pred_labels.count('Intrusion'))
            col4.metric("🟢 Malware", pred_labels.count('Malware'))
            
            # Show if Attack Type column exists (for comparison)
            display_cols = ['Prediction', 'Confidence']
            if 'Attack Type' in df_upload.columns:
                df_upload['Correct'] = df_upload['Prediction'] == df_upload['Attack Type']
                accuracy = df_upload['Correct'].mean() * 100
                st.write(f"**Accuracy on uploaded data: {accuracy:.1f}%**")
                display_cols = ['Attack Type'] + display_cols + ['Correct']
            
            display_cols = ['Source Port', 'Destination Port', 'Protocol'] + display_cols
            display_cols = [c for c in display_cols if c in df_upload.columns]
            st.dataframe(df_upload[display_cols], use_container_width=True)
            
            # Download button
            csv_result = df_upload.to_csv(index=False)
            st.download_button("📥 Download Results", csv_result, "predictions.csv", "text/csv")

# ============================================================
# TAB 2: PICK A ROW FROM DATASET
# ============================================================
with tab2:
    st.subheader("Pick a row from the test set")
    st.caption("Select a row from the held-out test split. The app will load it and predict the attack type.")

    test_indices = model_data['test_indices']
    test_position = st.number_input(
        f"Row position in test set (0 – {len(test_indices)-1})",
        min_value=0, max_value=len(test_indices)-1, value=0, step=1
    )
    row_number = test_indices[test_position]   # original dataset index
    selected_row = df_dataset.iloc[[row_number]]

    st.caption(f"Dataset index: {row_number} | Test set size: {len(test_indices):,} rows")
    st.write("**Selected row:**")
    st.dataframe(selected_row, use_container_width=True)

    actual_type = selected_row['Attack Type'].values[0]
    st.write(f"**Actual Attack Type:** {actual_type}")

    if st.button("🔍 Run Prediction", key='btn_row'):
        with st.spinner("Running prediction..."):
            # Use the pre-transformed df from training (bias columns already computed)
            df_transformed = model_data['df_transformed']
            X_cols = model_data['X_cols']

            X_input = df_transformed.loc[[row_number], X_cols].values.astype(float)
            X_input = np.nan_to_num(X_input, nan=0.0)

            prediction    = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]
            pred_label    = label_encoder.inverse_transform([prediction])[0]
            confidence    = probabilities[prediction] * 100

        show_prediction(pred_label, confidence, probabilities, target_names)

        if pred_label == actual_type:
            st.success(f"✅ Correct! Model predicted **{pred_label}**, actual is **{actual_type}**.")
        else:
            st.error(f"❌ Wrong. Model predicted **{pred_label}**, actual is **{actual_type}**.")

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
        response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data['lat'], data['lon'], data.get('city', 'Unknown'), data.get('country', 'Unknown')
    except Exception as e:
        st.warning(f"Could not geocode {ip_address}: {str(e)}")
    return None, None, None, None
    
# ============================================================
# TAB 3: MANUAL INPUT
# ============================================================
with tab3:
    st.subheader("Manual input")
    st.caption("Fill in all fields manually. The pipeline handles the rest.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Network Info**")
        timestamp = st.text_input("Timestamp", value="2023-05-30 06:33:58")
        source_ip = st.text_input("Source IP Address", value="103.216.15.12")
        dest_ip = st.text_input("Destination IP Address", value="84.9.164.252")
        
        # Geocode IP addresses
        src_lat, src_lon, src_city, src_country = geocode_ip(source_ip)
        dst_lat, dst_lon, dst_city, dst_country = geocode_ip(dest_ip)

        # Build folium map
        if src_lat and dst_lat:
            # Center map between the two points
            center_lat = (src_lat + dst_lat) / 2
            center_lon = (src_lon + dst_lon) / 2
        elif src_lat:
            center_lat, center_lon = src_lat, src_lon
        else:
            center_lat, center_lon = 20.0, 0.0  # default world view

        m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

        # Source marker (red)
        if src_lat and src_lon:
            folium.CircleMarker(
                location=[src_lat, src_lon],
                radius=10,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                tooltip=f"🔴 Source: {src_city}, {src_country}\nIP: {source_ip}"
            ).add_to(m)

        # Destination marker (blue)
        if dst_lat and dst_lon:
            folium.CircleMarker(
                location=[dst_lat, dst_lon],
                radius=10,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.8,
                tooltip=f"🔵 Destination: {dst_city}, {dst_country}\nIP: {dest_ip}"
            ).add_to(m)

        # Draw a line between source and destination
        if src_lat and dst_lat:
            folium.PolyLine(
                locations=[[src_lat, src_lon], [dst_lat, dst_lon]],
                color='orange',
                weight=2,
                dash_array='5'
            ).add_to(m)

        st_folium(m, width=430, height=300)

        source_port = st.number_input("Source Port", min_value=0, max_value=65535, value=31225)
        dest_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=17616)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        packet_length = st.slider("Packet Length", min_value=0, max_value=2000, value=503, step=1)
        packet_type = st.selectbox("Packet Type", ["Data", "Control"])
        traffic_type = st.selectbox("Traffic Type", ["HTTP", "DNS", "FTP"])
    
    with col2:
        st.write("**Security Info**")
        anomaly_score = st.slider("Anomaly Scores", min_value=0.0, max_value=100.0, value=28.67, step=0.01)
        severity = st.selectbox("Severity Level", ["Low", "Medium", "High"])
        action = st.selectbox("Action Taken", ["Blocked", "Logged", "Ignored"])
        attack_sig = st.selectbox("Attack Signature", ["Known Pattern A", "Known Pattern B"])
        network_seg = st.selectbox("Network Segment", ["Segment A", "Segment B", "Segment C"])
        log_source = st.selectbox("Log Source", ["Firewall", "Server"])
        malware_ind = st.selectbox("Malware Indicators", ["None", "IoC Detected"])
        alerts_warn = st.selectbox("Alerts/Warnings", ["None", "Alert Triggered"])
    
    with col3:
        st.write("**Other Info**")
        firewall_logs = st.selectbox("Firewall Logs", ["None", "Log Data"])
        ids_alerts = st.selectbox("IDS/IPS Alerts", ["None", "Alert Data"])
        user_info = st.text_input("User Information", value="Reyansh Dugal")
        device_info = st.text_input("Device Information", 
                                     value="Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.2; Trident/5.0)")
        geo_location = st.text_input("Geo-location Data", value="Mumbai, Maharashtra")
        proxy_info = st.text_input("Proxy Information (leave empty if none)", value="")
        payload_data = st.text_area("Payload Data", value="Sample payload data", height=68)
    
    if st.button("🔍 Run Prediction", key='btn_manual'):
        row = pd.DataFrame([{
            'Timestamp': timestamp,
            'Source IP Address': source_ip,
            'Destination IP Address': dest_ip,
            'Source Port': source_port,
            'Destination Port': dest_port,
            'Protocol': protocol,
            'Packet Length': packet_length,
            'Packet Type': packet_type,
            'Traffic Type': traffic_type,
            'Payload Data': payload_data,
            'Malware Indicators': 'IoC Detected' if malware_ind == 'IoC Detected' else np.nan,
            'Anomaly Scores': anomaly_score,
            'Alerts/Warnings': 'Alert Triggered' if alerts_warn == 'Alert Triggered' else np.nan,
            'Attack Signature': attack_sig,
            'Action Taken': action,
            'Severity Level': severity,
            'User Information': user_info,
            'Device Information': device_info,
            'Network Segment': network_seg,
            'Geo-location Data': geo_location,
            'Proxy Information': proxy_info if proxy_info != "" else np.nan,
            'Firewall Logs': 'Log Data' if firewall_logs == 'Log Data' else np.nan,
            'IDS/IPS Alerts': 'Alert Data' if ids_alerts == 'Alert Data' else np.nan,
            'Log Source': log_source,
        }])
        
        with st.spinner("Running feature engineering & prediction..."):
            X_input = engineer_features_for_new_row(row, model_data)
            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]
            pred_label = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100
        
        show_prediction(pred_label, confidence, probabilities, target_names)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application predicts the type of cyber attack 
    (DDoS, Intrusion, or Malware) from raw network traffic data.
    """)

    st.markdown("---")
    st.subheader("Model Performance")
    accscore = model_data.get('accscore', None)
    if accscore is not None:
        st.metric("Accuracy (test set)", f"{accscore * 100:.1f}%")
    else:
        st.metric("Accuracy", "—")

    with st.expander("📊 Confusion Matrix", expanded=False):
        fig_cm = model_data.get('fig_cm')
        if fig_cm is not None:
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Not available.")

    with st.expander("📈 ROC Curve", expanded=False):
        fig_roc = model_data.get('fig_roc')
        if fig_roc is not None:
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Not available.")

    st.markdown("---")
    st.subheader("Pipeline")
    st.write("""
    1. Raw data input (25 columns)
    2. Feature engineering (cleaning, encoding, parsing)
    3. Bias-based target encoding
    4. Random Forest prediction
    5. Result display
    """)

    st.markdown("---")
    st.caption("DSTI MSc Project — Cyber Attack Detection")
