"""Time series analysis: daily attack patterns, ACF/PACF, and cross-variable analysis."""

from itertools import product

import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp

from src.config import show_fig


def plot_daily_attack_counts(Attacks_pday):
    """Plot daily attack counts by type.

    Parameters
    ----------
    Attacks_pday : pd.DataFrame
        DataFrame with date index and columns for each attack type.
    """
    fig = subp(
        rows=3,
        cols=1,
        subplot_titles=(
            "Malware Attacks per Day",
            "Intrusion Attempts per Day",
            "DDoS Attacks per Day",
        ),
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=Attacks_pday.index,
            y=Attacks_pday["Malware"],
            mode="lines",
            name="Malware",
            line={"color": "#636EFA"},
            hovertemplate="<b>Malware</b><br>Date : %{x}<br>Count : %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Attacks_pday.index,
            y=Attacks_pday["Intrusion"],
            mode="lines",
            name="Intrusion",
            line={"color": "#EF553B"},
            hovertemplate="<b>Intrusion</b><br>Date : %{x}<br>Count : %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Attacks_pday.index,
            y=Attacks_pday["DDoS"],
            mode="lines",
            name="DDoS",
            line={"color": "#00CC96"},
            hovertemplate="<b>DDoS</b><br>Date : %{x}<br>Count : %{y}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    show_fig(fig)


def plot_acf_pacf(Attacks_pday, nlags=100):
    """Plot ACF and PACF for each attack type.

    Parameters
    ----------
    Attacks_pday : pd.DataFrame
        DataFrame with date index and columns for each attack type.
    nlags : int
        Number of lags for ACF/PACF computation.
    """
    from statsmodels.tsa.stattools import acf, pacf

    fig = subp(
        rows=3,
        cols=2,
        subplot_titles=(
            "Malware ACF", "Malware PACF",
            "Intrusion ACF", "Intrusion PACF",
            "DDoS ACF", "DDoS PACF",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    Attacktype_TSanalysis = {}
    colors = {
        "Malware": "#636EFA",
        "Intrusion": "#EF553B",
        "DDoS": "#00CC96",
    }
    for i, attacktype in enumerate(["Malware", "Intrusion", "DDoS"]):
        Attacktype_TSanalysis[f"{attacktype}_ACF"] = acf(
            Attacks_pday[attacktype], nlags=nlags
        )
        Attacktype_TSanalysis[f"{attacktype}_PACF"] = pacf(
            Attacks_pday[attacktype], nlags=nlags
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(0, nlags + 1)),
                y=Attacktype_TSanalysis[f"{attacktype}_ACF"],
                mode="lines",
                name=f"{attacktype} ACF",
                line={"color": colors[attacktype]},
                hovertemplate=f"<b>{attacktype} ACF</b><br>Lag : %{{x}}<br>Correlation : %{{y:.3f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(0, nlags + 1)),
                y=Attacktype_TSanalysis[f"{attacktype}_PACF"],
                mode="lines",
                name=f"{attacktype} PACF",
                line={"color": colors[attacktype]},
                hovertemplate=f"<b>{attacktype} PACF</b><br>Lag : %{{x}}<br>Correlation : %{{y:.3f}}<extra></extra>",
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(
        height=900,
        title_text="Autocorrelation Analysis for Attack Time Series",
        title_font_size=18,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Lag (days)")
    fig.update_yaxes(title_text="Correlation")
    show_fig(fig)


def plot_ts_by_variable(df, target_col="Network Segment", nlags=100):
    """Time series analysis by a categorical variable cross Attack Type.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame containing date, target_col, and "Attack Type".
    target_col : str
        Categorical column to cross with Attack Type.
    nlags : int
        Number of lags for ACF/PACF computation.
    """
    import pandas as pd
    from statsmodels.tsa.stattools import acf, pacf

    df_TSA_pday = {}
    df["date"] = pd.to_datetime(df["date"])
    df_TSA_pday[target_col] = df.copy(deep=True)
    df_TSA_pday[target_col]["date_dd"] = df_TSA_pday[target_col]["date"].dt.floor("D")
    df_TSA_pday[target_col] = (
        df_TSA_pday[target_col]
        .groupby(["date_dd", target_col, "Attack Type"])
        .size()
        .unstack()
        .fillna(0)
    )

    attypes = df["Attack Type"].value_counts().index
    target_values = df[target_col].value_counts().index

    # Daily counts plot
    fig = subp(
        rows=len(attypes),
        cols=len(target_values),
        subplot_titles=[
            f"Attack Type = {attype} , {target_col} = {target_value} per Day"
            for target_value in target_values
            for attype in attypes
        ],
        vertical_spacing=0.08,
    )
    for v, value in enumerate(target_values):
        df_TSA_pday[value] = df_TSA_pday[target_col].xs(value, level=target_col)
        for a, attype in enumerate(attypes):
            fig.add_trace(
                go.Scatter(
                    x=df_TSA_pday[value].index,
                    y=df_TSA_pday[value][attype],
                    mode="lines",
                    name=f"{attype} x {value}",
                    hovertemplate=f"<b>{attype}</b><br>Date : %{{x}}<br>Count : %{{y}}<extra></extra>",
                ),
                row=a + 1,
                col=v + 1,
            )
    show_fig(fig)

    # ACF and PACF analysis
    TSA_PACF = {}
    fig = subp(
        rows=len(attypes) * 2,
        cols=len(target_values),
        subplot_titles=[
            f"{metric} , Attack Type = {attype} , {target_col} = {target_value} per Day"
            for target_value in target_values
            for metric in ["ACF", "PACF"]
            for attype in attypes
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )
    for (a, attype), (v, value) in product(
        enumerate(attypes), enumerate(target_values)
    ):
        ACF_name = f"{attype} x {value} ACF"
        PACF_name = f"{attype} x {value} PACF"
        TSA_PACF[ACF_name] = acf(df_TSA_pday[value][attype], nlags=nlags)
        TSA_PACF[PACF_name] = pacf(df_TSA_pday[value][attype], nlags=nlags)
        fig.add_trace(
            go.Bar(
                x=list(range(0, nlags + 1)),
                y=TSA_PACF[ACF_name],
                name=ACF_name,
                hovertemplate=f"<b>{ACF_name}</b><br>Lag : %{{x}}<br>Correlation : %{{y:.3f}}<extra></extra>",
            ),
            row=1 + (v * 2),
            col=a + 1,
        )
        fig.add_trace(
            go.Bar(
                x=list(range(0, nlags + 1)),
                y=TSA_PACF[PACF_name],
                name=PACF_name,
                hovertemplate=f"<b>{PACF_name}</b><br>Lag : %{{x}}<br>Correlation : %{{y:.3f}}<extra></extra>",
            ),
            row=2 + (v * 2),
            col=a + 1,
        )
    fig.update_layout(
        height=3000,
        title_text="Autocorrelation Analysis",
        title_font_size=18,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Lag (days)")
    fig.update_yaxes(title_text="Correlation")
    show_fig(fig)
