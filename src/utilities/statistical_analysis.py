"""Statistical analysis: Matthews Correlation Coefficient and Chi-Square tests."""

import numpy as np
from scipy.stats import chi2_contingency

from  src.utilities.utils import catvar_corr


def run_mcc_analysis(df):
    """Compute Matthews Correlation Coefficient for categorical variables vs Attack Type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical columns and "Attack Type".
    """
    catvars = np.array(
        [
            "Source IP country",
            "Destination IP country",
            "Source Port",
            "Destination Port",
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
    for c in catvars:
        catvar_corr(df, c)


def run_chi_square_test(df):
    """Run Chi-Square independence test on categorical features vs Attack Type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical columns and "Attack Type".
    """
    df_catvar = df[
        [
            "Attack Type",
            "Source Port",
            "Destination Port",
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
    ]
    # Encode all columns as numeric category codes for chi-square computation
    df_encoded = df_catvar.apply(lambda col: col.astype("category").cat.codes)
    res = chi2_contingency(df_encoded.values.T[1:])
    print(
        f"Chi-square statistic : {res.statistic:.4f}",
        f"\nP-value : {res.pvalue:.6f}",
        f"\nDegrees of freedom : {res.dof}",
        f"\nExpected frequencies : {res.expected_freq}",
    )
    if res.pvalue < 0.05:
        print("\n=> Significant association detected ( p < 0.05 )")
    else:
        print("\n=> No significant association ( p >= 0.05 )")
