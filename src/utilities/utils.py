"""General utility functions for categorical encoding and simple charts."""

import warnings

import pandas as pd
import plotly.express as px
from sklearn.metrics import matthews_corrcoef

from src.utilities.config import show_fig

# Suppress fragmentation warning — handled via df.copy() at pipeline checkpoints
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")


def catvar_corr(df, col, target="Attack Type"):
    """Calculate Matthews correlation coefficient between a feature and target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both columns.
    col : str
        Feature column name.
    target : str
        Target column name.
    """
    corr = matthews_corrcoef(df[target], df[col].astype(str))
    # print(f"phi corr between {target} and {col} = {corr:+.4f}")


def catvar_mapping(df, col_name, values, name=None):
    """Transform categorical variables to binary {0, 1} indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_name : str
        Name of the categorical column to transform.
    values : list
        List of categorical values to create binary indicators for.
    name : list, optional
        Custom names for the new binary columns. If None, uses the values.
        Use ["/"] to replace the original column in-place.

    Returns
    -------
    pd.DataFrame
        The same dataframe, modified in-place with new binary columns added.
    """
    if name is None:
        name = values
    elif (len(name) == 1) and (name != ["/"]):
        col_target = f"{col_name} {name[0]}"
        df.rename(columns={col_name: col_target}, inplace=True)
        col_name = col_target
        name = [col_target]
    col_pos = df.columns.get_loc(col_name) + 1
    insert_offset = 0
    for val, nm in zip(values, name):
        if nm == "/":
            col_target = col_name
        elif len(name) == 1:
            col_target = nm
        else:
            col_target = f"{col_name} {nm}"
        col_data = (df[col_name] == val).astype("int64")
        if nm == "/" or len(name) == 1:
            # Overwrite existing column
            df[col_target] = col_data
        else:
            if col_target not in df.columns:
                df.insert(col_pos + insert_offset, col_target, col_data)
                insert_offset += 1
            else:
                df[col_target] = col_data
    return df


def piechart_col(df, col, names=None):
    """Generate an interactive pie chart for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    col : str
        Column name to visualize.
    names : dict or list, optional
        Custom labels for pie chart segments.
        - dict: maps actual column values to display labels, e.g. {0: "No", 1: "Yes"}.
        - list: positional mapping over index-sorted value_counts (legacy).
        - None: uses actual column values as labels.
    """
    vc = df[col].value_counts()
    if names is None:
        display_names = vc.index
    elif isinstance(names, dict):
        display_names = [names.get(v, v) for v in vc.index]
    else:
        vc = vc.sort_index()
        if len(vc) != len(names):
            display_names = vc.index
        else:
            display_names = names
    fig = px.pie(
        values=vc.values,
        names=display_names,
        title=f"Distribution of {col}",
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent + label",
        hovertemplate="<b>%{label}</b><br>Count : %{value}<br>Percentage : %{percent}<extra></extra>",
    )
    show_fig(fig)
