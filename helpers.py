import pandas as pd
import numpy as np
import math
import sklearn as skl
from sklearn.preprocessing import LabelEncoder, label_binarize
from user_agents import parse as ua_parse
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PREPROCESS DATASET (replaces what script.py does, without Django/GeoIP)
# Creates all columns that feature_engineering() expects
# ============================================================
def preprocess_for_modelling(df_raw):
    """
    Takes the raw cybersecurity_attacks.csv and produces a df
    with all columns expected by feature_engineering().
    Mirrors what script.py does, without Django/GeoIP2.
    """
    df = df_raw.copy()

    # --- Rename columns to match script.py conventions ---
    df = df.rename(columns={
        "Timestamp": "date",
        "Alerts/Warnings": "Alert Trigger",
    })

    # --- Date decomposition (mirrors script.py date section) ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date dd"] = df["date"].dt.floor("d")

    # Month-week schedule: MW1=hours 0-7, MW2=7-14, MW3=14-22, MW4=rest
    def get_mw(dt):
        if pd.isna(dt): return "MW4"
        h = dt.hour
        if h < 7: return "MW1"
        elif h < 14: return "MW2"
        elif h < 22: return "MW3"
        return "MW4"
    df["date MW"] = df["date"].apply(get_mw)

    # Weekday: WD1=Mon ... WD7=Sun
    def get_wd(dt):
        if pd.isna(dt): return "WD7"
        return f"WD{dt.weekday() + 1}"
    df["date WD"] = df["date"].apply(get_wd)

    # Hour schedule: H1=3-9, H2=9-15, H3=15-21, H4=rest
    def get_h(dt):
        if pd.isna(dt): return "H4"
        h = dt.hour
        if 3 <= h < 9: return "H1"
        elif 9 <= h < 15: return "H2"
        elif 15 <= h < 21: return "H3"
        return "H4"
    df["date H"] = df["date"].apply(get_h)

    # Minute: M1-M12 (5-min buckets)
    def get_m(dt):
        if pd.isna(dt): return "M12"
        bucket = min(dt.minute // 5, 11) + 1
        return f"M{bucket}"
    df["date M"] = df["date"].apply(get_m)

    # --- Device Information: Browser family, OS family, Device type ---
    if "Device Information" in df.columns:
        df["Browser family"] = df["Device Information"].apply(
            lambda x: ua_parse(str(x)).browser.family if pd.notna(x) else "Other"
        )
        df["OS family"] = df["Device Information"].apply(
            lambda x: ua_parse(str(x)).os.family if pd.notna(x) else "Other"
        )
        df["Device type"] = df["Device Information"].apply(Device_type).fillna("Other")
    else:
        df["Browser family"] = "Other"
        df["OS family"] = "Other"
        df["Device type"] = "Other"

    # --- Proxy usage: 1 if proxy present, 0 otherwise ---
    if "Proxy Information" in df.columns:
        df["Proxy usage"] = df["Proxy Information"].notna().astype(int).astype(str)
    else:
        df["Proxy usage"] = "0"

    # --- Attack Signature patA: first part of Attack Signature ---
    if "Attack Signature" in df.columns:
        df["Attack Signature patA"] = df["Attack Signature"].apply(
            lambda x: str(x).split(" ")[0] if pd.notna(x) else "Unknown"
        )
    else:
        df["Attack Signature patA"] = "Unknown"

    # --- Log Source Firewall: 1 if Log Source == Firewall ---
    if "Log Source" in df.columns:
        df["Log Source Firewall"] = (df["Log Source"] == "Firewall").astype(int).astype(str)
    else:
        df["Log Source Firewall"] = "0"

    # --- Alert Trigger binary ---
    if "Alert Trigger" in df.columns:
        df["Alert Trigger"] = df["Alert Trigger"].notna().astype(int).astype(str)

    # --- Malware Indicators binary ---
    if "Malware Indicators" in df.columns:
        df["Malware Indicators"] = df["Malware Indicators"].notna().astype(int).astype(str)

    return df

# ============================================================
# DEVICE TYPE
# ============================================================
def Device_type(ua_string):
    try:
        if not ua_string or pd.isna(ua_string):
            return pd.NA
        ua = ua_parse(ua_string)
        if getattr(ua, "is_mobile", False): return "Mobile"
        if getattr(ua, "is_tablet", False): return "Tablet"
        if getattr(ua, "is_pc", False): return "PC"
        return pd.NA
    except:
        return pd.NA

# ============================================================
# GEOLOCATION DATA PARSER
# ============================================================
def geolocation_data(info):
    """Split 'City, State' string into separate components."""
    try:
        city, state = info.split(", ")
        return pd.Series([city, state])
    except:
        return pd.Series([pd.NA, pd.NA])

# ============================================================
# ENGINEER FEATURES FOR NEW ROW
# Properly assigns bias scores by looking up trained crosstabs
# and bin boundaries — used for Manual Input and CSV Upload
# ============================================================
def engineer_features_for_new_row(df_raw, model_data):
    """
    Takes a raw DataFrame (manual input or CSV upload),
    runs preprocess_for_modelling(), then assigns bias scores
    by looking up the stored crosstabs and continuous-var bin boundaries.
    Returns a numpy array ready for model.predict().
    """
    df = preprocess_for_modelling(df_raw)

    crosstabs            = model_data['crosstabs_x_AttackType']
    df_train             = model_data['df_transformed']
    X_cols               = model_data['X_cols']
    threshold_floor      = model_data.get('threshold_floor', 37.5)
    dyn_pars             = model_data.get('dynamic_threshold_pars', [5, 100, 10, 37.5])
    attypes              = list(model_data['target_names'])

    # ---- 1. Assign class-bin labels for continuous variables ----
    # (e.g. Source Port 31225 → "class : 29184 - 32767")
    cont_class_cols = [c for c in df_train.columns if c.endswith(' classes')]
    for col_class in cont_class_cols:
        base_col = col_class[: -len(' classes')]
        if base_col not in df.columns:
            df[col_class] = None
            continue

        # Parse all bin ranges from training data
        bin_ranges = []
        for lbl in df_train[col_class].dropna().unique():
            try:
                lo, hi = map(float, lbl.replace('class : ', '').split(' - '))
                bin_ranges.append((lo, hi, lbl))
            except:
                pass
        bin_ranges.sort(key=lambda x: x[0])

        def _assign_bin(val, br=bin_ranges):
            try:
                v = float(val)
                for lo, hi, lbl in br:
                    if lo <= v <= hi:
                        return lbl
                # Outside range: clamp to nearest boundary bin
                if br:
                    return br[0][2] if v < br[0][0] else br[-1][2]
            except:
                pass
            return None

        df[col_class] = df[base_col].apply(_assign_bin)

    # ---- 2. Assign bias scores for each feature column ----
    for x_col in X_cols:
        if x_col in df.columns:
            continue

        parts = x_col.rsplit(' , bias ', 1)
        if len(parts) != 2:
            df[x_col] = 0
            continue

        crosstab_name, attype = parts
        if crosstab_name not in crosstabs or attype not in attypes:
            df[x_col] = 0
            continue

        ct = crosstabs[crosstab_name]
        df[x_col] = 0

        is_multi = crosstab_name.startswith('{ ') and crosstab_name.endswith(' }')
        key_cols = (
            [c.strip() for c in crosstab_name[2:-2].split(' , ')]
            if is_multi else [crosstab_name]
        )

        if any(c not in df.columns for c in key_cols):
            continue

        for idx in df.index:
            try:
                if is_multi:
                    key = tuple(str(df.at[idx, c]) for c in key_cols)
                else:
                    key = df.at[idx, key_cols[0]]
                    if pd.isna(key):
                        continue

                if key not in ct.index:
                    continue

                row_ct = ct.loc[key]
                n_obs  = float(row_ct['n obs'])
                pct    = float(row_ct[attype])

                score = 0
                for p, thresh in enumerate([threshold_floor, 40, 50, 60, 75, 90]):
                    if dynamic_threshold({'n obs': n_obs, attype: pct}, attype, dyn_pars) and pct >= thresh:
                        score = p + 1
                df.at[idx, x_col] = score
            except Exception:
                pass

    X = df[X_cols].fillna(0).values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    return X

def dynamic_threshold(row, attype, dynamic_threshold_pars):
    n1, d1, n2, d2 = dynamic_threshold_pars
    a = (d2 - d1) / (n2 - n1)
    b = d1 - a * n1
    threshold = a * row["n obs"] + b
    return row[attype] >= threshold

# ============================================================
# CONTVAR CLASSES BUILDER (by n_obs)
# ============================================================
def contvar_classes_builder_bycount_bynobs(df, col, n_obs):
    colclass_name = f"{col} classes"
    target_col = df[[col]].sort_values(by=col)[col]
    current_indices = []
    current_values = []
    val_prev = None

    for idx, val in target_col.items():
        if (
            val_prev is not None
            and val != val_prev
            and len(current_indices) >= n_obs
        ):
            class_min = min(current_values)
            class_max = max(current_values)
            class_label = f"class : {class_min} - {class_max}"
            df.loc[current_indices, colclass_name] = class_label
            current_indices = []
            current_values = []

        current_indices.append(idx)
        current_values.append(val)
        val_prev = val

    if current_indices:
        class_min = min(current_values)
        class_max = max(current_values)
        class_label = f"class : {class_min} - {class_max}"
        df.loc[current_indices, colclass_name] = class_label

    return colclass_name

# ============================================================
# CROSSTAB COL
# ============================================================
def crosstab_col(df, cols, crosstabs_x_AttackType):
    """
    crosstabs_x_AttackType must be passed explicitly (was global in script.py)
    """
    if len(cols) == 1:
        crosstab_name = cols[0]
    else:
        crosstab_name = "{ " + " , ".join(cols) + " }"

    attypes = df["Attack Type"].unique()
    crosstabs_x_AttackType[crosstab_name] = pd.crosstab(
        index=[df[col] for col in cols],
        columns=df["Attack Type"],
        margins=True,
        margins_name="n obs",
        normalize=False
    )
    crosstabs_x_AttackType[crosstab_name] = crosstabs_x_AttackType[crosstab_name].iloc[:-1, ]
    for attype in attypes:
        crosstabs_x_AttackType[crosstab_name][attype] = (
            crosstabs_x_AttackType[crosstab_name][attype] /
            crosstabs_x_AttackType[crosstab_name]["n obs"] * 100
        )
    return crosstab_name

# ============================================================
# DF BIAS CLASSES
# ============================================================
def df_bias_classes(df, crosstab_name, threshold, colclass_names, dynamic_threshold_pars, crosstabs_x_AttackType):
    attypes = df["Attack Type"].unique()
    X_col = []

    for attype in attypes:
        target_crosstab = crosstabs_x_AttackType[crosstab_name]
        target_crosstab = target_crosstab[
            target_crosstab.apply(
                lambda row: dynamic_threshold(row, attype, dynamic_threshold_pars), axis=1
            )
        ]
        bias_col_name = f"{crosstab_name} , bias {attype}"
        X_col.append(bias_col_name)
        df[bias_col_name] = 0

        for p_class, threshold_class in enumerate({threshold, 40, 50, 60, 75, 90}):
            p_class = p_class + 1
            bias = target_crosstab[attype]
            bias = bias[bias >= threshold_class].index

            if not isinstance(bias, pd.MultiIndex):
                df.loc[df[df[colclass_names].isin(bias)].index, bias_col_name] = p_class
            else:
                midx = list(zip(*(df[col] for col in colclass_names)))
                midx = pd.Series(midx).isin(bias)
                df.loc[midx, bias_col_name] = p_class
    return X_col

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df, crosstabs_x_AttackType, features_mask, threshold_floor, contvar_nobs_b_class, dynamic_threshold_pars):
    features_existing = np.array([
        "Source Port & Destination Port",
        "Source IP latitude/longitude combination & Destination IP latitude/longitude combination",
        "Sounce IP Country , Destination IP country combination",
        "date dd",
        "Network & Traffic",
        "Security Response",
        "time decomposition",
        "Anomaly Scores",
        "Packet Length"
    ])
    features_selected = features_existing[np.array(features_mask)]
    X_cols = []

    if "Source Port & Destination Port" in features_selected:
        for SourDest in ["Source", "Destination"]:
            target_col = f"{SourDest} Port"
            colclass_name = contvar_classes_builder_bycount_bynobs(df, target_col, contvar_nobs_b_class)
            crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
            X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, crosstab_name, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "Anomaly Scores" in features_selected:
        colclass_name = contvar_classes_builder_bycount_bynobs(df, "Anomaly Scores", contvar_nobs_b_class)
        crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, crosstab_name, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "Packet Length" in features_selected:
        colclass_name = contvar_classes_builder_bycount_bynobs(df, "Packet Length", contvar_nobs_b_class)
        crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, crosstab_name, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "date dd" in features_selected:
        crosstab_name = crosstab_col(df, ["date dd"], crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, crosstab_name, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "Network & Traffic" in features_selected:
        columns = ["Protocol", "Packet Type", "Traffic Type", "Attack Signature patA",
                   "Network Segment", "Proxy usage", "Browser family", "OS family", "Device type"]
        crosstab_name = crosstab_col(df, columns, crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, columns, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "Security Response" in features_selected:
        columns = ["Malware Indicators", "Alert Trigger", "Action Taken", "Severity Level",
                   "Log Source Firewall", "Firewall Logs", "IDS/IPS Alerts",
                   "Browser family", "OS family", "Device type"]
        crosstab_name = crosstab_col(df, columns, crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, columns, dynamic_threshold_pars, crosstabs_x_AttackType))

    if "time decomposition" in features_selected:
        columns = ["date MW", "date WD", "date H", "date M"]
        crosstab_name = crosstab_col(df, columns, crosstabs_x_AttackType)
        X_cols.extend(df_bias_classes(df, crosstab_name, threshold_floor, columns, dynamic_threshold_pars, crosstabs_x_AttackType))

    return X_cols

# ============================================================
# LOGIT
# ============================================================
def logit(attypes, x_train, x_test, y_train, y_test):
    model = skl.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train)
    return model

# ============================================================
# RANDOM FOREST
# ============================================================
def randomforrest(attypes, x_train, x_test, y_train, y_test):
    model = skl.ensemble.RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=1, n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model

# ============================================================
# MODEL PREDICTIONS
# ============================================================
def model_predictions(model, x_test):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    return y_pred, y_prob

# ============================================================
# MODEL METRICS
# ============================================================
def model_metrics(attypes, model, x_train, x_test, y_train, y_test, y_pred, y_prob):
    """
    Returns (fig_confmat, fig_roc, accscore) instead of calling fig.show()
    so it works both in scripts and Streamlit.
    """
    confmat = pd.DataFrame(
        skl.metrics.confusion_matrix(y_test, y_pred),
        index=attypes, columns=attypes
    )
    fig = px.imshow(confmat, text_auto=True, color_continuous_scale="Magma", title="Confusion Matrix")
    fig.update_layout(xaxis_title="Predicted Class", yaxis_title="True Class")

    accscore = skl.metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accscore}")
    print("\nClassification Report:\n")
    print(skl.metrics.classification_report(y_test, y_pred, target_names=attypes))

    y_test_bin = label_binarize(y_test, classes=np.arange(len(attypes)))
    n_classes = y_test_bin.shape[1]
    fig_roc = go.Figure()
    for yclass in range(n_classes):
        fpr, tpr, _ = skl.metrics.roc_curve(y_test_bin[:, yclass], y_prob[:, yclass])
        roc_auc = skl.metrics.auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{attypes[yclass]} (AUC = {roc_auc:.4f})"
        ))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="Multiclass ROC Curve",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
    return fig, fig_roc, accscore

# ============================================================
# MODELLING (from script.py — refactored: no global df)
# ============================================================
def modelling(df, crosstabs_x_AttackType, testp, features_mask, threshold_floor, contvar_nobs_b_class, dynamic_threshold_pars, model_type="logit", split_before_training=False):
    attypes = df["Attack Type"].unique()

    if split_before_training == False:
        X_cols = feature_engineering(df, crosstabs_x_AttackType, features_mask, threshold_floor, contvar_nobs_b_class, dynamic_threshold_pars)
        df = df.copy()

        x_vars = df[X_cols]
        y_var = df["Attack Type"]

        y = LabelEncoder()
        y_var = y.fit_transform(y_var)

        x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(
            x_vars, y_var, test_size=testp, random_state=50, stratify=y_var
        )
    else:
        df_train, df_test = skl.model_selection.train_test_split(
            df, test_size=testp, train_size=(1 - testp),
            stratify=df["Attack Type"], random_state=42
        )
        X_cols = feature_engineering(df, crosstabs_x_AttackType, features_mask, threshold_floor, contvar_nobs_b_class, dynamic_threshold_pars)
        df_train = df_train.copy()
        df_test = df_test.copy()

        x_train = df_train[X_cols]
        x_test = df_test[X_cols]

        y = LabelEncoder()
        y_train = y.fit_transform(df_train["Attack Type"])
        y_test = y.fit_transform(df_test["Attack Type"])

    if model_type == "logit":
        model = logit(attypes, x_train, x_test, y_train, y_test)
    elif model_type == "randomforrest":
        model = randomforrest(attypes, x_train, x_test, y_train, y_test)
    else:
        print("model type is not recognized")
        return None

    y_pred, y_prob = model_predictions(model, x_test)
    fig_cm, fig_roc, accscore = model_metrics(
        attypes, model, x_train, x_test, y_train, y_test, y_pred, y_prob
    )

    test_indices = x_test.index if split_before_training == False else df_test.index

    return y_pred, model, y, X_cols, fig_cm, fig_roc, accscore, test_indices