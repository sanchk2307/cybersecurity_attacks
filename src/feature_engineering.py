"""Feature engineering utilities: binning, crosstabs, bias classes, and aggregate features."""

import math

import numpy as np
import pandas as pd


def dynamic_threshold(row, attype, dynamic_threshold_pars):
    """Determine whether a row satisfies a dynamically computed threshold.

    Parameters
    ----------
    row : pd.Series
        A dataframe row containing at least "n obs" and attype columns.
    attype : str
        Column name representing the metric to evaluate.
    dynamic_threshold_pars : tuple
        (n1, d1, n2, d2) -- two reference points defining a linear threshold.

    Returns
    -------
    bool
        True if row[attype] >= computed threshold.
    """
    n1, d1, n2, d2 = dynamic_threshold_pars
    a = (d2 - d1) / (n2 - n1)
    b = d1 - a * n1
    threshold = a * row["n obs"] + b
    return row[attype] >= threshold


def contvar_classes_builder_byval(df, df_train, col, n_classes):
    """Create value-based bins for a continuous variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the class column to.
    df_train : pd.DataFrame
        Training DataFrame used to compute bin boundaries.
    col : str
        Name of continuous variable to discretize.
    n_classes : int
        Number of equal-width bins.

    Returns
    -------
    str
        Name of the newly created class column.
    """
    colclass_name = f"{col} classes"
    colmax = math.ceil(df_train[col].max())
    colmin = math.floor(df_train[col].min())
    stp = int((colmax - colmin) / n_classes)
    for class_bound_low in range(colmin, colmax, stp):
        class_bound_up = class_bound_low + stp
        if class_bound_low == colmin:
            row_target = df[df[col] < class_bound_up].index
            df.loc[row_target, colclass_name] = f"class : min - {class_bound_up}"
        elif class_bound_up >= colmax:
            row_target = df[df[col] >= class_bound_low].index
            df.loc[row_target, colclass_name] = f"class : {class_bound_low} - max"
        else:
            row_target = df[
                (df[col] >= class_bound_low) & (df[col] < class_bound_up)
            ].index
            df.loc[row_target, colclass_name] = (
                f"class : {class_bound_low} - {class_bound_up}"
            )
    return colclass_name


def contvar_classes_builder_bycount_bynobs(df, col, n_obs):
    """Create bins for a continuous variable with at least n_obs observations each.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing column to discretize.
    col : str
        Continuous variable to bin.
    n_obs : int
        Minimum number of observations per bin.

    Returns
    -------
    str
        Name of the newly created class column.
    """
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


def contvar_classes_builder_bycount_bynclasses(df, col, n_classes):
    """Create equal-count bins for a continuous variable (stub).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing column to discretize.
    col : str
        Continuous variable to bin.
    n_classes : int
        Number of bins.

    Returns
    -------
    str
        Name of the newly created class column (empty string -- not implemented).
    """
    colclass_name = ""
    return colclass_name


def crosstab_col(df, cols, crosstabs_x_AttackType):
    """Create a normalized cross-tabulation between categorical columns and "Attack Type".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns and "Attack Type".
    cols : list
        List of column names for the categorical variables.
    crosstabs_x_AttackType : dict
        Dictionary to store the crosstab results (mutated in place).

    Returns
    -------
    str
        Key name of the crosstab in the dictionary.
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
        normalize=False,
    )
    crosstabs_x_AttackType[crosstab_name] = crosstabs_x_AttackType[crosstab_name].iloc[
        :-1,
    ]
    for attype in attypes:
        crosstabs_x_AttackType[crosstab_name][attype] = (
            crosstabs_x_AttackType[crosstab_name][attype]
            / crosstabs_x_AttackType[crosstab_name]["n obs"]
            * 100
        )
    crosstabs_x_AttackType[crosstab_name]["% obs"] = (
        crosstabs_x_AttackType[crosstab_name]["n obs"] / df.shape[1]
    )

    return crosstab_name


def df_bias_classes(df, crosstab_name, threshold, colclass_names, dynamic_threshold_pars, crosstabs_x_AttackType):
    """Create bias classification columns based on filtered crosstab results.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe (mutated in place to add bias columns).
    crosstab_name : str
        Key in crosstabs_x_AttackType dictionary.
    threshold : numeric
        Base threshold included in the predefined threshold set.
    colclass_names : list of str or str
        Column name(s) used to match class labels.
    dynamic_threshold_pars : tuple
        Parameters passed to dynamic_threshold().
    crosstabs_x_AttackType : dict
        Dictionary containing the crosstab DataFrames.

    Returns
    -------
    list
        Names of all generated bias columns.
    """
    attypes = df["Attack Type"].unique()
    X_col = []

    for attype in attypes:
        target_crosstab = crosstabs_x_AttackType[crosstab_name]
        target_crosstab = target_crosstab[
            target_crosstab.apply(
                lambda row: dynamic_threshold(row, attype, dynamic_threshold_pars),
                axis=1,
            )
        ]
        bias_col_name = f"{crosstab_name} , bias {attype}"
        X_col.append(bias_col_name)
        df[bias_col_name] = 0

        for p_class, threshold_class in enumerate(
            {threshold, 40, 50, 60, 75, 90}
        ):
            p_class = p_class + 1
            bias = target_crosstab[attype]
            bias = bias[bias >= threshold_class].index

            if not isinstance(bias, pd.MultiIndex):
                df.loc[
                    df[df[colclass_names].isin(bias)].index, bias_col_name
                ] = p_class
            else:
                midx = list(zip(*(df[col] for col in colclass_names)))
                midx = pd.Series(midx).isin(bias)
                df.loc[midx, bias_col_name] = p_class
    return X_col


def coln_max(row, timedim):
    """Return the column name(s) with the maximum value in a row.

    Parameters
    ----------
    row : pd.Series
        Row from a pivot table of counts.
    timedim : str
        Time dimension label (unused in current implementation).

    Returns
    -------
    str
        Formatted string of column names with the max value.
    """
    cols_target = row.index[row == row.max()].astype(str)
    return "{ " + " + ".join(cols_target) + " }"


def ln(row):
    """Compute natural logarithm, returning 0 for zero input.

    Parameters
    ----------
    row : numeric
        Input value.

    Returns
    -------
    float
        log(row) if row != 0, else 0.
    """
    if row != 0:
        return math.log(row)
    else:
        return 0


def build_daily_aggregates(df, target_cols):
    """Build daily aggregate features for the given target columns.

    For each target column, computes daily counts, max-category indicators,
    exponential and log transforms, and binary mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame (mutated in place).
    target_cols : list of str
        Column names to aggregate by day.

    Returns
    -------
    dict
        Dictionary of daily pivot DataFrames keyed by "col, d".
    """
    from src.utils import catvar_mapping

    df_n = {}
    for target_col in target_cols:
        # print(f"\nWorking on {target_col}")
        for timedim in ["d"]:
            df_n_name = f"{target_col} , {timedim}"
            df_n[df_n_name] = df.loc[:, ["date", "date dd", target_col]]
            df_n[df_n_name] = (
                df_n[df_n_name]
                .groupby(["date dd", target_col])
                .size()
                .unstack()
                .fillna(0)
            )

            target_col_values = df[target_col].value_counts().index

            max_n_d_col = f"{target_col} max n {timedim}"
            df_n[df_n_name][max_n_d_col] = df_n[df_n_name][
                [value for value in target_col_values]
            ].apply(lambda x: coln_max(x, timedim), axis=1)

            for lags in [0, 1]:
                if lags == 0:
                    max_n_dlags_col = max_n_d_col
                    df[max_n_dlags_col] = df["date dd"].map(
                        df_n[df_n_name][max_n_d_col]
                    )
                    for tarcol in df[target_col].value_counts().index:
                        df[f"{tarcol} n d exp"] = df["date dd"].map(
                            df_n[df_n_name][tarcol].apply(math.exp)
                        )
                        df[f"{tarcol} n d ln"] = df["date dd"].map(
                            df_n[df_n_name][tarcol].apply(ln)
                        )
                else:
                    max_n_dlags_col = f"{max_n_d_col}-{lags}"
                    df[max_n_dlags_col] = (
                        df["date dd"] - pd.Timedelta(days=lags)
                    ).map(df_n[df_n_name][max_n_d_col])

                df = catvar_mapping(
                    df,
                    max_n_dlags_col,
                    df[max_n_dlags_col].value_counts().index[:-1],
                )

    return df_n


def feature_engineering(
    df,
    crosstabs_x_AttackType,
    features_mask,
    threshold_floor,
    contvar_nobs_b_class,
    dynamic_threshold_pars,
):
    """Build engineered bias-based feature columns according to a feature mask.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame (mutated in place).
    crosstabs_x_AttackType : dict
        Dictionary to store crosstab results.
    features_mask : iterable of bool
        Mask selecting which feature groups to activate.
    threshold_floor : numeric
        Base threshold used inside df_bias_classes().
    contvar_nobs_b_class : int
        Minimum number of observations per bin.
    dynamic_threshold_pars : tuple
        Parameters passed to dynamic_threshold().

    Returns
    -------
    list
        Names of all generated engineered feature columns.
    """
    features_existing = np.array(
        [
            "Source Port & Destination Port",
            "Source IP latitude/longitude combination & Destination IP latitude/longitude combination",
            "Sounce IP Country , Destination IP country combination",
            "date dd",
            "Network & Traffic",
            "Security Response",
            "time decomposition",
            "Anomaly Scores",
            "Packet Length",
        ]
    )
    features_selected = features_existing[np.array(features_mask)]
    X_cols = []

    # Source Port & Destination Port
    if "Source Port & Destination Port" in features_selected:
        for SourDest in ["Source", "Destination"]:
            target_col = f"{SourDest} Port"
            colclass_name = contvar_classes_builder_bycount_bynobs(
                df, target_col, contvar_nobs_b_class
            )
            crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
            X_cols.extend(
                df_bias_classes(
                    df, crosstab_name, threshold_floor, crosstab_name,
                    dynamic_threshold_pars, crosstabs_x_AttackType,
                )
            )

    # Source/Destination IP latitude/longitude combination
    if (
        "Source IP latitude/longitude combination & Destination IP latitude/longitude combination"
        in features_selected
    ):
        for SourDest in ["Source", "Destination"]:
            colclass_names = []
            for lonlat in ["longitude", "latitude"]:
                target_col = f"{SourDest} IP {lonlat}"
                colclass_name = contvar_classes_builder_bycount_bynobs(
                    df, target_col, contvar_nobs_b_class
                )
                colclass_names.append(colclass_name)
            colclass_names.extend(
                [
                    "Malware Indicators",
                    "Alert Trigger",
                    "Action Taken",
                    "Severity Level",
                    "Log Source Firewall",
                    "Firewall Logs",
                    "IDS/IPS Alerts",
                ]
            )
            crosstab_name = crosstab_col(
                df, [col for col in colclass_names], crosstabs_x_AttackType
            )
            X_cols.extend(
                df_bias_classes(
                    df, crosstab_name, threshold_floor, colclass_names,
                    dynamic_threshold_pars, crosstabs_x_AttackType,
                )
            )

    # Source IP Country, Destination IP country combination
    if "Sounce IP Country , Destination IP country combination" in features_selected:
        colclass_names = ["Source IP country", "Destination IP country"]
        crosstab_name = crosstab_col(
            df, [col for col in colclass_names], crosstabs_x_AttackType
        )
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, colclass_names,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # date dd
    if "date dd" in features_selected:
        crosstab_name = crosstab_col(df, ["date dd"], crosstabs_x_AttackType)
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, crosstab_name,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # Network & Traffic
    if "Network & Traffic" in features_selected:
        columns = [
            "Protocol",
            "Packet Type Control",
            "Traffic Type",
            "Attack Signature patA",
            "Network Segment",
            "Proxy usage",
            "Browser family",
            "OS family",
            "Device type",
        ]
        crosstab_name = crosstab_col(
            df, [col for col in columns], crosstabs_x_AttackType
        )
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, columns,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # Security Response
    if "Security Response" in features_selected:
        columns = [
            "Malware Indicators",
            "Alert Trigger",
            "Action Taken",
            "Severity Level",
            "Log Source Firewall",
            "Firewall Logs",
            "IDS/IPS Alerts",
            "Severity Level",
            "Browser family",
            "OS family",
            "Device type",
        ]
        crosstab_name = crosstab_col(
            df, [col for col in columns], crosstabs_x_AttackType
        )
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, columns,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # time decomposition
    if "time decomposition" in features_selected:
        columns = {"date MW", "date WD", "date H", "date M"}
        crosstab_name = crosstab_col(
            df, [col for col in columns], crosstabs_x_AttackType
        )
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, columns,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # Anomaly Scores
    if "Anomaly Scores" in features_selected:
        colclass_name = contvar_classes_builder_bycount_bynobs(
            df, "Anomaly Scores", contvar_nobs_b_class
        )
        crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, crosstab_name,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    # Packet Length
    if "Packet Length" in features_selected:
        colclass_name = contvar_classes_builder_bycount_bynobs(
            df, "Packet Length", contvar_nobs_b_class
        )
        crosstab_name = crosstab_col(df, [colclass_name], crosstabs_x_AttackType)
        X_cols.extend(
            df_bias_classes(
                df, crosstab_name, threshold_floor, crosstab_name,
                dynamic_threshold_pars, crosstabs_x_AttackType,
            )
        )

    return X_cols
