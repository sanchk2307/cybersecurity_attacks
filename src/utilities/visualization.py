"""Visualization functions: geographic plots, strip plots, and interpretation histograms."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots as subp

from src.utilities.config import show_fig


def plot_geo_attack_types(df, df_attype, attypes):
    """6-panel geographic visualization of attack types by anomaly score.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame.
    df_attype : dict
        Dictionary of DataFrames keyed by attack type.
    attypes : list
        List of attack type names.
    """
    fig = subp(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
        ],
        subplot_titles=("Source IP locations", "Destination IP locations"),
    )
    for i, (attype, symb) in enumerate(
        zip(attypes, ["diamond", "diamond", "diamond"])
    ):
        fig.add_trace(
            go.Scattergeo(
                lat=df_attype[attype]["Source IP latitude"],
                lon=df_attype[attype]["Source IP longitude"],
                mode="markers",
                marker={
                    "size": 4,
                    "symbol": symb,
                    "color": df_attype[attype]["Anomaly Scores"],
                    "colorscale": "Viridis",
                    "cmin": df_attype[attype]["Anomaly Scores"].min(),
                    "cmax": df_attype[attype]["Anomaly Scores"].max(),
                    "colorbar": {"title": "Anomaly Score"},
                },
                hovertemplate=f"<b>{attype} Attack Origin</b><br>Lat : %{{lat:.2f}}<br>Lon : %{{lon:.2f}}<br>Anomaly Score : %{{marker.color:.2f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scattergeo(
                lat=df_attype[attype]["Destination IP latitude"],
                lon=df_attype[attype]["Destination IP longitude"],
                mode="markers",
                marker={
                    "size": 4,
                    "symbol": symb,
                    "color": df_attype[attype]["Anomaly Scores"],
                    "colorscale": "Viridis",
                    "cmin": df_attype[attype]["Anomaly Scores"].min(),
                    "cmax": df_attype[attype]["Anomaly Scores"].max(),
                    "colorbar": {"title": "Anomaly Score"},
                },
                hovertemplate=f"<b>{attype} Attack Target</b><br>Lat : %{{lat:.2f}}<br>Lon : %{{lon:.2f}}<br>Anomaly Score : %{{marker.color:.2f}}<extra></extra>",
            ),
            row=i + 1,
            col=2,
        )
    fig.update_geos(showcountries=True, showland=True)
    fig.update_layout(
        height=4000,
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        title_text="Geographic Distribution of Attacks by Type and Anomaly Score",
        title_x=0.5,
        title_font_size=18,
        showlegend=False,
    )
    show_fig(fig)


def plot_strip_packet_length(df_attype, attypes):
    """Strip plot showing packet length distribution for each attack type.

    Parameters
    ----------
    df_attype : dict
        Dictionary of DataFrames keyed by attack type.
    attypes : list
        List of attack type names.
    """
    colors = {
        "Malware": "#636EFA",
        "Intrusion": "#EF553B",
        "DDoS": "#00CC96",
    }
    fig = go.Figure()
    for i, (attype, symb) in enumerate(
        zip(attypes, ["diamond", "diamond", "diamond"])
    ):
        fig.add_trace(
            go.Scatter(
                x=df_attype[attype]["Packet Length"],
                y=[i] * df_attype[attype].shape[0],
                mode="markers",
                name=f"{attype} ({len(df_attype[attype]):,} records)",
                marker={
                    "size": 4,
                    "symbol": symb,
                    "opacity": 0.1,
                    "color": colors[attype],
                },
                hovertemplate=f"<b>{attype}</b><br>Packet Length : %{{x}} bytes<extra></extra>",
            ),
        )
    fig.update_layout(
        title="Packet Length Distribution by Attack Type",
        title_font_size=16,
        xaxis_title="Packet Length ( bytes )",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=attypes,
            title="Attack Type",
        ),
        height=400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    show_fig(fig)


def plot_strip_anomaly_scores(df_attype, attypes):
    """Strip plot showing anomaly score distribution for each attack type.

    Parameters
    ----------
    df_attype : dict
        Dictionary of DataFrames keyed by attack type.
    attypes : list
        List of attack type names.
    """
    colors = {
        "Malware": "#636EFA",
        "Intrusion": "#EF553B",
        "DDoS": "#00CC96",
    }
    fig = go.Figure()
    for i, (attype, symb) in enumerate(
        zip(attypes, ["diamond", "diamond", "diamond"])
    ):
        fig.add_trace(
            go.Scatter(
                x=df_attype[attype]["Anomaly Scores"],
                y=[i] * df_attype[attype].shape[0],
                mode="markers",
                name=f"{attype} ({len(df_attype[attype]):,} records)",
                marker={
                    "size": 4,
                    "symbol": symb,
                    "opacity": 0.1,
                    "color": colors[attype],
                },
                hovertemplate=f"<b>{attype}</b><br>Anomaly Score : %{{x}}<extra></extra>",
            ),
        )
    fig.update_layout(
        title="Anomaly Score Distribution by Attack Type",
        title_font_size=16,
        xaxis_title="Anomaly Score (higher = more suspicious)",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=attypes,
            title="Attack Type",
        ),
        height=400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    show_fig(fig)


def plot_interpretation_histograms(df, attypes):
    """Plot bias class interpretation histograms for anomaly scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing bias columns and "Anomaly Scores".
    attypes : list
        List of attack type names.
    """
    color_map = {
        "1": "#1f77b4",
        "2": "#ff7f0e",
        "3": "#2ca02c",
        "4": "#00ff1a",
        "5": "#aaaa00",
        "6": "#b600a1",
    }
    for target in ["Anomaly Scores", "Source Port", "Destination Port"]:
        fig = subp(
            rows=len(attypes),
            cols=1,
            shared_xaxes=True,
            subplot_titles=attypes,
        )
        for row, attype in enumerate(attypes):
            for lvl in df[f"Anomaly Scores classes , bias {attype}"].unique()[1:]:
                vals = df[
                    df[f"Anomaly Scores classes , bias {attype}"] == lvl
                ]["Anomaly Scores"]
                fig.add_trace(
                    go.Histogram(
                        x=vals,
                        name=f"{attype} class {lvl}",
                        nbinsx=20,
                        marker=dict(color=color_map[str(lvl)]),
                        opacity=0.6,
                    ),
                    row=row + 1,
                    col=1,
                )
        fig.update_layout(title=target, barmode="stack")
        show_fig(fig)
