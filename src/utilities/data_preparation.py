# =============================================================================
# DATA PREPARATION
# =============================================================================
# Rename columns for better readability and consistency
# - "Timestamp" -> "date" for temporal analysis
# - Port columns renamed to indicate ephemeral port analysis
# - Alerts/Warnings -> Alert Trigger for binary encoding

def rename_columns(df):
    """
    Rename columns for clarity and consistency.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with original column names

    Returns:
    --------
    pd.DataFrame
        DataFrame with renamed columns
    """
    return df.rename(
        columns={
            "Timestamp": "date",
            "Source Port": "Source Port ephemeral",
            "Destination Port": "Destination Port ephemeral",
            "Alerts/Warnings": "Alert Trigger",
        }
    )

# -----------------------------------------------------------------------------
# Cross-tabulation utility function
# Used to analyze relationships between categorical variables
# -----------------------------------------------------------------------------


def crosstab_col(col, target, name_col, name_target):
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
    """
    crosstabs = {}

    name_tab = f"{name_col}_x_{name_target}"
    crosstabs[name_tab] = pd.crosstab(df[target], df[col], normalize=True) * 100

def missing_values_analysis(df):
    
    # -----------------------------------------------------------------------------
    # Missing Value Analysis
    # Identify columns with missing data and their percentages
    # -----------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    df_s0 = df.shape[0]
    for col in df.columns:
        NA_n = sum(df[col].isna())
        if NA_n > 0:
            print(f"Missing values in {col}: {NA_n:,} / {df_s0:,} ({100*NA_n/df_s0:.2f}%)")
    print("=" * 60)