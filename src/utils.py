"""General utility functions for categorical encoding and simple charts."""

import warnings

import pandas as pd
import plotly.express as px
from sklearn.metrics import matthews_corrcoef

from src.config import show_fig

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
        Copy of the dataframe with new binary columns added.
    """
    df1 = df.copy(deep=True)
    if name is None:
        name = values
    elif (len(name) == 1) and (name != ["/"]):
        col_target = f"{col_name} {name[0]}"
        df1 = df1.rename(columns={col_name: col_target})
        col_name = col_target
        name = [col_target]
    col_pos = df1.columns.get_loc(col_name) + 1
    # Pre-compute all new column Series, then concat once
    new_col_series = {}
    targets_in_order = []
    for val, nm in zip(values, name):
        if nm == "/":
            col_target = col_name
        elif len(name) == 1:
            col_target = nm
        else:
            col_target = f"{col_name} {nm}"
        bully = df1[col_name] == val
        col_data = pd.Series(0, index=df1.index, dtype="int64")
        col_data.loc[bully] = 1
        if nm == "/" or len(name) == 1:
            # Overwrite existing column
            df1[col_target] = col_data
        else:
            if col_target not in df1.columns:
                new_col_series[col_target] = col_data
                targets_in_order.append(col_target)
            else:
                df1[col_target] = col_data
    if new_col_series:
        new_cols = pd.DataFrame(
            {c: new_col_series[c] for c in targets_in_order},
            index=df1.index,
        )
        df1 = pd.concat(
            [df1.iloc[:, :col_pos], new_cols, df1.iloc[:, col_pos:]],
            axis=1,
        )
    return df1


def piechart_col(df, col, names=None):
    """Generate an interactive pie chart for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    col : str
        Column name to visualize.
    names : list, optional
        Custom labels for pie chart segments. If None, uses actual values.
    """
    if names is None:
        fig = px.pie(
            values=df[col].value_counts(),
            names=df[col].value_counts().index,
            title=f"Distribution of {col}",
        )
    else:
        fig = px.pie(
            values=df[col].value_counts(),
            names=names,
            title=f"Distribution of {col}",
        )
    fig.update_traces(
        textposition="inside",
        textinfo="percent + label",
        hovertemplate="<b>%{label}</b><br>Count : %{value}<br>Percentage : %{percent}<extra></extra>",
    )
    show_fig(fig)
