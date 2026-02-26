"""Machine learning modelling: training, prediction, and evaluation."""

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn as skl
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.utilities.config import show_fig

MODELS_DIR = Path("models")
from  src.utilities.feature_engineering import feature_engineering, ports_feature_engineering


def logit(attypes, x_train, x_test, y_train, y_test):
    """Train a Logistic Regression classifier.

    Parameters
    ----------
    attypes : array-like
        Attack type labels (not used inside this function).
    x_train : array-like
        Training feature matrix.
    x_test : array-like
        Test feature matrix (not used inside this function).
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels (not used inside this function).

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Fitted Logistic Regression model.
    """
    model = skl.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train)
    return model


def randomforrest(attypes, x_train, x_test, y_train, y_test, n_jobs=-1):
    """Train a Random Forest classifier with hyperparameter tuning.

    Uses RandomizedSearchCV with 3-fold stratified cross-validation to find
    good hyperparameters, then refits on the full training set.

    Parameters
    ----------
    attypes : array-like
        Attack type labels (not used inside this function).
    x_train : array-like
        Training feature matrix.
    x_test : array-like
        Test feature matrix (not used inside this function).
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels (not used inside this function).
    n_jobs : int
        Number of parallel jobs for CV and forest fitting (-1 = all CPUs).

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Fitted Random Forest model with tuned hyperparameters.
    """
    from sklearn.model_selection import RandomizedSearchCV

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.1, 0.2],
    }

    base_model = skl.ensemble.RandomForestClassifier(
        random_state=1, n_jobs=n_jobs,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=30,
        cv=3,
        scoring="f1_macro",
        random_state=1,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(x_train, y_train)

    print(f"Best RF params: {search.best_params_}")
    print(f"Best CV F1 (macro): {search.best_score_:.4f}")

    return search.best_estimator_


def model_predictions(model, x_test):
    """Generate predictions and class probabilities from a trained model.

    Parameters
    ----------
    model : sklearn-like estimator
        Fitted classification model.
    x_test : array-like
        Test feature matrix.

    Returns
    -------
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted class probabilities.
    """
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    return y_pred, y_prob


def model_metrics(attypes, model, x_train, x_test, y_train, y_test, y_pred, y_prob):
    """Compute and visualize classification metrics.

    Generates confusion matrix, accuracy, classification report,
    multiclass ROC curves, and mutual information feature importance.

    Parameters
    ----------
    attypes : list-like
        Ordered class labels.
    model : fitted estimator
        Trained classification model.
    x_train, x_test : array-like
        Feature matrices.
    y_train, y_test : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted class probabilities.
    """
    # Confusion matrix
    confmat = pd.DataFrame(
        skl.metrics.confusion_matrix(y_test, y_pred),
        index=attypes,
        columns=attypes,
    )
    fig = px.imshow(
        confmat,
        text_auto=True,
        color_continuous_scale="Magma",
        title="Confusion Matrix",
    )
    fig.update_layout(xaxis_title="Predicted Class", yaxis_title="True Class")
    show_fig(fig)

    # Metrics evaluation
    accscore = skl.metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accscore}")
    print("\nClassification Report :\n")
    print(skl.metrics.classification_report(y_test, y_pred, target_names=attypes))

    # ROC curve
    y_test_bin = label_binarize(y_test, classes=np.arange(len(attypes)))
    n_classes = y_test_bin.shape[1]
    fig_roc = go.Figure()
    for yclass in range(n_classes):
        fpr, tpr, _ = skl.metrics.roc_curve(
            y_test_bin[:, yclass], y_prob[:, yclass]
        )
        roc_auc = skl.metrics.auc(fpr, tpr)
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{attypes[yclass]} ( AUC = {roc_auc})",
            )
        )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random",
        )
    )
    fig_roc.update_layout(
        title="Multiclass ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Classes",
        template="plotly_dark",
    )
    show_fig(fig_roc)

    # Precision-Recall curves
    fig_pr = go.Figure()
    for yclass in range(n_classes):
        precision, recall, _ = skl.metrics.precision_recall_curve(
            y_test_bin[:, yclass], y_prob[:, yclass]
        )
        ap = skl.metrics.average_precision_score(
            y_test_bin[:, yclass], y_prob[:, yclass]
        )
        fig_pr.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{attypes[yclass]} ( AP = {ap:.3f})",
            )
        )
    fig_pr.update_layout(
        title="Multiclass Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title="Classes",
        template="plotly_dark",
    )
    show_fig(fig_pr)

    # Mutual information
    marginfo_scores = skl.feature_selection.mutual_info_classif(
        x_train, y_train, discrete_features="auto", random_state=100
    )
    marginfo = pd.Series(marginfo_scores, index=x_train.columns).sort_values(
        ascending=False
    )
    fig = px.bar(
        marginfo,
        x=list(marginfo.values),
        y=marginfo.index,
        orientation="h",
        text=marginfo.values,
        title="Mutual Information Scores",
        labels={"x": "Marginal Information score", "y": "Feature"},
    )
    fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        barcornerradius=15,
        bargap=0.2,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    show_fig(fig)
    return accscore, fig_roc, fig


def modelling(
    df,
    crosstabs_x_AttackType,
    testp,
    features_mask,
    threshold_floor,
    contvar_nobs_b_class,
    dynamic_threshold_pars,
    model_type="logit",
    split_before_training=False,
    n_jobs=-1,
):
    """End-to-end modelling pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame.
    crosstabs_x_AttackType : dict
        Crosstab dictionary (will be reset inside).
    testp : float
        Proportion of dataset used for testing.
    features_mask : iterable of bool
        Controls which feature groups are activated.
    threshold_floor : numeric
        Base bias threshold.
    contvar_nobs_b_class : int
        Minimum number of observations per continuous bin.
    dynamic_threshold_pars : tuple
        Parameters controlling dynamic threshold function.
    model_type : str
        "logit" or "randomforrest".
    split_before_training : bool
        If False, feature engineering on full dataset before split.
        If True, split first then build features (not fully operational).

    Returns
    -------
    y_pred : array-like
        Predicted class labels for test set.
    df : pd.DataFrame
        DataFrame after feature engineering (potentially modified).
    """
    attypes = df["Attack Type"].unique()

    # Reset crosstabs for modelling
    crosstabs_x_AttackType.clear()
    crosstabs_for_save = {}

    if not split_before_training:
        X_cols = feature_engineering(
            df,
            crosstabs_x_AttackType,
            features_mask,
            threshold_floor,
            contvar_nobs_b_class,
            dynamic_threshold_pars,
        )
        crosstabs_for_save = dict(crosstabs_x_AttackType)
        # Extract only the columns needed for modelling, then free the crosstabs
        x_vars = df[X_cols].copy()
        y_var = df["Attack Type"].copy()
        crosstabs_x_AttackType.clear()
        gc.collect()

        y = LabelEncoder()
        y_var = y.fit_transform(y_var)

        x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(
            x_vars, y_var, test_size=testp, random_state=50, stratify=y_var
        )
        del x_vars, y_var
        gc.collect()
    else:
        df_train, df_test = skl.model_selection.train_test_split(
            df,
            test_size=testp,
            train_size=(1 - testp),
            stratify=df["Attack Type"],
            random_state=42,
        )

        X_cols = feature_engineering(
            df,
            crosstabs_x_AttackType,
            features_mask,
            threshold_floor,
            contvar_nobs_b_class,
            dynamic_threshold_pars,
        )
        crosstabs_for_save = dict(crosstabs_x_AttackType)
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
        model = randomforrest(attypes, x_train, x_test, y_train, y_test, n_jobs=n_jobs)
    else:
        print("model type is not recognized")
        return None, df

    y_pred, y_prob = model_predictions(model, x_test)
    accscore, fig_roc, fig_cm = model_metrics(
        attypes, model, x_train, x_test, y_train, y_test, y_pred, y_prob
    )
    test_indices = x_test.index if split_before_training == False else df_test.index

    save_data = {
        "model": model,
        "label_encoder": y,
        "target_names": y.classes_.tolist(),
        "crosstabs_x_AttackType": crosstabs_for_save,
        "X_cols": X_cols,
        "df_transformed": df,
        "test_indices": test_indices,    # original row indices from the test split
        "threshold_floor": 37.5,
        "dynamic_threshold_pars": [5, 100, 10, 37.5],
        "fig_cm": fig_cm,
        "fig_roc": fig_roc,
        "accscore": accscore,
    }

    # Save trained model
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Model saved to {model_path}")

    return y_pred, df

def ports_modelling(df, split_before_training = False):
    attypes = df["Attack Type"].unique()
    # Encoding the target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['Attack Type'])
    target_names = le.classes_.tolist()

    df, src_port_means, dst_port_means = ports_feature_engineering(df)

    # Selecting the final features
    feature_cols = (
        # Numeric
        ['Packet Length', 'Anomaly Scores', 'Source Port', 'Destination Port',
        'port_diff', 'hour', 'dayofweek'] +
        # Binary
        ['Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 
        'IDS/IPS Alerts', 'has_proxy', 'is_weekend', 'total_alerts'] +
        # Target encoded
        [col for col in df.columns if col.endswith('_te')]
    )

    X = df[feature_cols]
    y = df['target']
    # Managing potential NaNs
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = skl.ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=20, 
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred, y_prob = model_predictions(model, X_test)
    accscore, fig_roc, fig_cm = model_metrics(
        attypes, model, X_train, X_test, y_train, y_test, y_pred, y_prob
    )

    save_data = {
        'model': model,
        'label_encoder': le,
        'target_names': target_names,
        # "crosstabs_x_AttackType": crosstabs_x_AttackType,
        'X_cols': feature_cols,
        # "df_transformed": df,
        "test_indices": X_test.index,    # original row indices from the test split
        # "threshold_floor": 37.5,
        # "dynamic_threshold_pars": [5, 100, 10, 37.5],
        'src_port_means': src_port_means,
        'dst_port_means': dst_port_means,
        "fig_cm": fig_cm,
        "fig_roc": fig_roc,
        "accscore": accscore,
    }

    # Save trained model
    model_type = "extra_trees"
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Model saved to {model_path}")

    return y_pred, df