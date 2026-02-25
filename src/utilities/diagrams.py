"""Sankey and parallel categories diagram generators."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utilities.config import show_fig


def sankey_diag_IPs(df, ntop):
    """Create a Sankey diagram showing traffic flow between Source, Proxy, and Destination countries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing IP country columns.
    ntop : int
        Number of top countries to display (others grouped as "other").

    Returns
    -------
    pd.DataFrame
        Aggregated IP flow counts grouped by source, proxy, and destination.
    """
    IPs_col = {}
    labels = pd.Series(dtype="string")

    for IPid, dfcol in zip(
        ["SIP", "DIP", "PIP"],
        ["Source IP country", "Destination IP country", "Proxy country"],
    ):
        IPs_col[IPid] = df[dfcol].copy(deep=True)
        IPs_col[f"{IPid}labs"] = pd.Series(IPs_col[IPid].value_counts().index[:ntop])
        bully = IPs_col[IPid].isin(IPs_col[f"{IPid}labs"]) | IPs_col[IPid].isna()
        IPs_col[IPid].loc[~bully] = "other"
        IPs_col[f"{IPid}labs"] = f"{IPid} " + pd.concat(
            [IPs_col[f"{IPid}labs"], pd.Series(["other"])]
        )
        labels = pd.concat([labels, IPs_col[f"{IPid}labs"]])
    labels = list(labels.reset_index(drop=True))

    aggregIPs = pd.DataFrame(
        {
            "SIP": IPs_col["SIP"],
            "PIP": IPs_col["PIP"],
            "DIP": IPs_col["DIP"],
        }
    )
    aggregIPs = aggregIPs.groupby(by=["SIP", "PIP", "DIP"]).size().to_frame("count")

    source = []
    target = []
    value = []
    nlvl = aggregIPs.index.nlevels
    for idx, row in aggregIPs.iterrows():
        row_labs = []
        if nlvl == 1:
            row_labs.append(f"{aggregIPs.index.name} {idx}")
        else:
            for i, val in enumerate(idx):
                row_labs.append(f"{aggregIPs.index.names[i]} {val}")
        for i in range(0, nlvl - 1):
            source.append(labels.index(row_labs[i]))
            target.append(labels.index(row_labs[i + 1]))
            value.append(row.item())

    n = len(labels)
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno, [i / (n - 1) for i in range(n)]
    )
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "rgba( 0 , 0 , 0 , 0.1 )", "width": 0.5},
                    "label": labels,
                    "color": colors,
                },
                link={"source": source, "target": target, "value": value},
            )
        ]
    )
    fig.update_layout(
        title_text="Network Traffic Flow : Source → Proxy → Destination Countries",
        title_font_size=18,
        font_size=14,
        title_font_family="Avenir",
        title_font_color="black",
        annotations=[
            {
                "x": 0.01,
                "y": 1.05,
                "text": "<b>Source countries</b>",
                "showarrow": False,
                "font_size": 12,
            },
            {
                "x": 0.5,
                "y": 1.05,
                "text": "<b>Proxy countries</b>",
                "showarrow": False,
                "font_size": 12,
            },
            {
                "x": 0.99,
                "y": 1.05,
                "text": "<b>Destination countries</b>",
                "showarrow": False,
                "font_size": 12,
            },
        ],
    )
    show_fig(fig)

    return aggregIPs


def sankey_diag(
    df,
    crosstabs_x_AttackType,
    cols_bully=[True] * 26,
    ntop=10,
):
    """Generate a Sankey diagram showing feature value flows to attack types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and "Attack Type".
    crosstabs_x_AttackType : dict
        Dictionary to store cross-tabulation results (mutated in place).
    cols_bully : list of bool
        Boolean flags indicating which columns to include.
    ntop : int
        Number of top countries to show (others grouped as "other").

    Returns
    -------
    pd.DataFrame
        Cross-tabulation percentages used to build the diagram.
    """
    cols = np.array(
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
            "Network Segment",
            "Firewall Logs",
            "IDS/IPS Alerts",
            "Log Source Firewall",
        ]
    )
    cols = cols[np.array(cols_bully)]

    idx_ct = []
    labels = []
    if "Source IP country" in cols:
        SIP = df["Source IP country"].copy(deep=True)
        SIP.name = "SIP"
        SIPlabs = pd.Series(SIP.value_counts().index[:ntop])
        bully = SIP.isin(SIPlabs) | SIP.isna()
        SIP.loc[~bully] = "other"
        SIPlabs = "SIP " + pd.concat([SIPlabs, pd.Series(["other"])])
        labels.extend(SIPlabs.to_list())
        idx_ct = idx_ct + [SIP]
        cols = cols[cols != "Source IP country"]
    if "Destination IP country" in cols:
        DIP = df["Destination IP country"].copy(deep=True)
        DIP.name = "DIP"
        DIPlabs = pd.Series(DIP.value_counts().index[:ntop])
        bully = DIP.isin(DIPlabs) | DIP.isna()
        DIP.loc[~bully] = "other"
        DIPlabs = "DIP " + pd.concat([DIPlabs, pd.Series(["other"])])
        labels.extend(DIPlabs.to_list())
        idx_ct = idx_ct + [DIP]
        cols = cols[cols != "Destination IP country"]

    idx_ct = idx_ct + [df[col] for col in cols]
    # print(idx_ct)
    ct = pd.crosstab(index=idx_ct, columns=df["Attack Type"])

    for c in np.append(cols, "Attack Type"):
        vals = df[c].unique()
        for v in vals:
            labels.append(f"{c} {v}")

    source = []
    target = []
    value = []
    nlvl = ct.index.nlevels
    for idx, row in ct.iterrows():
        row_labs = []
        if nlvl == 1:
            row_labs.append(f"{ct.index.name} {idx}")
        else:
            for i, val in enumerate(idx):
                row_labs.append(f"{ct.index.names[i]} {val}")
        for attype in ct.columns:
            val = row[attype]
            for i in range(0, nlvl - 1):
                source.append(labels.index(row_labs[i]))
                target.append(labels.index(row_labs[i + 1]))
                value.append(val)
            source.append(labels.index(row_labs[-1]))
            target.append(labels.index(f"Attack Type {attype}"))
            value.append(val)

    n = len(labels)
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Inferno, [i / (n - 1) for i in range(n)]
    )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="rgba( 0 , 0 , 0 , 0.1 )", width=0.5),
                    label=labels,
                    color=colors,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(
        title_text="Sankey Diagram",
        font_size=20,
        title_font_family="Avenir",
        title_font_color="black",
    )
    show_fig(fig)

    ct = ct / ct.sum().sum() * 100

    return ct


def paracat_diag(
    df,
    cols_bully=[True] * 23,
    colorvar="Attack Type",
    ntop=10,
):
    """Generate a parallel categories diagram for categorical features vs Attack Type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and "Attack Type".
    cols_bully : list of bool
        Boolean flags indicating which columns to include.
    colorvar : str
        Column name to use for coloring the diagram lines.
    ntop : int
        Number of top countries to show (others grouped as "other").
    """
    cols = np.array(
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
            "Attack Type max n d",
            "Protocol max n d",
            "Traffic Type max n d",
            "Alert Trigger max n d",
            "Firewall Logs max n d",
            "Attack Signature patA max n d",
            "Action Taken max n d",
            "Browser family max n d",
        ]
    )
    cols = cols[np.array(cols_bully)]

    dims_var = {}
    SIP = None
    DIP = None
    if "Source IP country" in cols:
        SIP = df["Source IP country"].copy(deep=True)
        SIP.name = "SIP"
        SIPlabs = pd.Series(SIP.value_counts().index[:ntop])
        bully = SIP.isin(SIPlabs) | SIP.isna()
        SIP.loc[~bully] = "other"
        cols = cols[cols != "Source IP country"]
        dims_var["SIP"] = go.parcats.Dimension(
            values=SIP, label="Source IP country"
        )
        cols = np.append(cols, "SIP")
    if "Destination IP country" in cols:
        DIP = df["Destination IP country"].copy(deep=True)
        DIP.name = "DIP"
        DIPlabs = pd.Series(DIP.value_counts().index[:ntop])
        bully = DIP.isin(DIPlabs) | DIP.isna()
        DIP.loc[~bully] = "other"
        cols = cols[cols != "Destination IP country"]
        dims_var["DIP"] = go.parcats.Dimension(
            values=DIP, label="Destination IP country"
        )
        cols = np.append(cols, "DIP")
    for col in cols[(cols != "SIP") & (cols != "DIP")]:
        # print(col)
        dims_var[col] = go.parcats.Dimension(values=df[col], label=col)

    dim_attypes = go.parcats.Dimension(
        values=df["Attack Type"],
        label="Attack Type",
        ticktext=["Malware", "Intrusion", "DDoS"],
    )

    dims = [dims_var[col] for col in cols]
    dims.append(dim_attypes)

    if colorvar == "SIP" and SIP is not None:
        colorvar = SIP
    elif colorvar == "DIP" and DIP is not None:
        colorvar = DIP
    elif (colorvar in cols) or (colorvar == "Attack Type"):
        colorvar = df[colorvar]
    else:
        raise ValueError("colorvar must be in cols")
    catcolor = colorvar.unique()
    catcolor_n = catcolor.shape[0]
    color = colorvar.map({cat: i for i, cat in enumerate(catcolor)})
    positions = [
        i / (catcolor_n - 1) if (catcolor_n > 1) else 0
        for i in range(0, catcolor_n)
    ]
    palette = px.colors.sequential.Viridis
    colors = [px.colors.sample_colorscale(palette, p)[0] for p in positions]
    colorscale = [[positions[i], colors[i]] for i in range(0, catcolor_n)]

    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=dims,
                line={
                    "color": color,
                    "colorscale": colorscale,
                    "shape": "hspline",
                },
                hoveron="color",
                hoverinfo="count+probability",
                labelfont={"size": 18, "family": "Times"},
                tickfont={"size": 16, "family": "Times"},
                arrangement="freeform",
            )
        ]
    )
    show_fig(fig)
