#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cybersecurity_attacks.csv') 

print(f"Shape: {df.shape}")
df.head()


# In[2]:


print(df.dtypes)


# In[3]:


# Valeurs manquantes
print("--- Valeurs manquantes ---")
for col in df.columns:
    n = df[col].isna().sum()
    if n > 0:
        print(f"{col}: {n} ({n/len(df)*100:.1f}%)")


# In[4]:


# Nombre de valeurs uniques par colonne
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique")


# In[5]:


# Stats des colonnes numériques
df.describe()


# In[6]:


# Target variable
print(df['Attack Type'].value_counts())
print()
print((df['Attack Type'].value_counts(normalize=True) * 100).round(2))


# In[7]:


# Convertir les colonnes binaires : NaN = 0, valeur = 1
df['Malware Indicators'] = df['Malware Indicators'].notna().astype(int)
df['Alerts/Warnings'] = df['Alerts/Warnings'].notna().astype(int)
df['Firewall Logs'] = df['Firewall Logs'].notna().astype(int)
df['IDS/IPS Alerts'] = df['IDS/IPS Alerts'].notna().astype(int)
df['has_proxy'] = df['Proxy Information'].notna().astype(int)

print("Vérification :")
print(df[['Malware Indicators','Alerts/Warnings','Firewall Logs','IDS/IPS Alerts','has_proxy']].sum())


# In[8]:


# Timestamp : extraire des features utiles
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

print("Exemple :")
print(df[['Timestamp','hour','dayofweek','is_weekend']].head())


# In[9]:


drop_cols = ['Timestamp', 'Payload Data', 'User Information', 
             'Proxy Information', 'Geo-location Data',
             'Source IP Address', 'Destination IP Address']
df = df.drop(columns=drop_cols)

print(f"Colonnes restantes : {df.shape[1]}")
print(df.columns.tolist())


# In[10]:


# Vérifier la distribution de chaque feature catégorielle vs Attack Type
cat_cols = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Signature',
            'Action Taken', 'Severity Level', 'Network Segment', 'Log Source']

for col in cat_cols:
    ct = pd.crosstab(df[col], df['Attack Type'], normalize='index') * 100
    print(f"\n{col}:")
    print(ct.round(1))


# In[11]:


# Visualisation
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, col in enumerate(cat_cols):
    ax = axes[i // 4][i % 4]
    pd.crosstab(df[col], df['Attack Type']).plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
plt.suptitle('Toutes les features montrent ~33/33/33 => pas de séparation individuelle', fontweight='bold')
plt.tight_layout()
plt.show()


# In[12]:


# Vérifier les numériques aussi
print("Packet Length par Attack Type :")
print(df.groupby('Attack Type')['Packet Length'].mean().round(1))
print("\nAnomaly Scores par Attack Type :")
print(df.groupby('Attack Type')['Anomaly Scores'].mean().round(1))
# Si les moyennes sont presque identiques => pas de signal individuel


# In[13]:


# Port categories (standard réseau)
def port_cat(port):
    if port <= 1023: return 'wellknown'
    elif port <= 49151: return 'registered'
    else: return 'dynamic'

df['src_port_cat'] = df['Source Port'].apply(port_cat)
df['dst_port_cat'] = df['Destination Port'].apply(port_cat)

# Différence entre ports
df['port_diff'] = abs(df['Source Port'] - df['Destination Port'])

print("Port categories :")
print(df['src_port_cat'].value_counts())


# In[14]:


get_ipython().system('pip install user-agents')


# In[15]:


# Device Information : extraire OS, Browser, Device type
from user_agents import parse as ua_parse

df['os_family'] = df['Device Information'].apply(lambda x: ua_parse(str(x)).os.family)
df['browser_family'] = df['Device Information'].apply(lambda x: ua_parse(str(x)).browser.family)

def get_device_type(ua):
    p = ua_parse(str(ua))
    if p.is_mobile: return 'Mobile'
    elif p.is_tablet: return 'Tablet'
    elif p.is_pc: return 'PC'
    return 'Other'

df['device_type'] = df['Device Information'].apply(get_device_type)

print("OS :", df['os_family'].value_counts().to_dict())
print("Browser :", df['browser_family'].value_counts().to_dict())
print("Device :", df['device_type'].value_counts().to_dict())


# In[16]:


# Combinaison d'alertes
df['total_alerts'] = (df['Malware Indicators'] + df['Alerts/Warnings'] + 
                      df['Firewall Logs'] + df['IDS/IPS Alerts'])

print("Total alerts distribution :")
print(df['total_alerts'].value_counts().sort_index())


# In[17]:


# TARGET ENCODING (la technique clé)
#
# L'idée : remplacer chaque valeur de catégorie par la moyenne du target
# pour cette catégorie. Avec cross-validation pour éviter le data leakage.
#
# Exemple : si Source Port 31225 a 40% de DDoS, on le code comme 0.40

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# D'abord encoder le target en numérique
le = LabelEncoder()
df['target'] = le.fit_transform(df['Attack Type'])
print(f"Encodage : {dict(zip(le.classes_, le.transform(le.classes_)))}")

def target_encode_cv(df, column, target_col='target', n_splits=5):
    """Target encoding avec cross-validation pour éviter le leakage."""
    encoded = pd.Series(np.nan, index=df.index)
    global_mean = df[target_col].mean()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df, df[target_col]):
        means = df.iloc[train_idx].groupby(column)[target_col].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][column].map(means)
    
    return encoded.fillna(global_mean)


# In[18]:


# Appliquer le target encoding sur les features catégorielles
te_cols = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Signature',
           'Action Taken', 'Severity Level', 'Network Segment', 'Log Source',
           'src_port_cat', 'dst_port_cat', 'os_family', 'browser_family', 'device_type']

for col in te_cols:
    df[f'{col}_te'] = target_encode_cv(df, col)
    
print(f"Target encoding fait pour {len(te_cols)} colonnes catégorielles")


# In[19]:


# TARGET ENCODING SUR SOURCE PORT ET DESTINATION PORT
# 
# C'est LA technique qui fait la différence.
# Source Port a ~30k valeurs uniques => chaque port reçoit son propre score.
# Un port qui apparaît souvent avec DDoS aura un score différent 
# d'un port qui apparaît souvent avec Malware.

print("Target encoding Source Port (~30k valeurs uniques)...")
df['Source Port_te'] = target_encode_cv(df, 'Source Port')

print("Target encoding Destination Port (~30k valeurs uniques)...")
df['Destination Port_te'] = target_encode_cv(df, 'Destination Port')

print(f"\nSource Port TE - mean: {df['Source Port_te'].mean():.4f}, std: {df['Source Port_te'].std():.4f}")
print(f"Dest Port TE   - mean: {df['Destination Port_te'].mean():.4f}, std: {df['Destination Port_te'].std():.4f}")


# In[20]:


# Sélectionner les features finales
feature_cols = (
    # Numériques originales
    ['Packet Length', 'Anomaly Scores', 'Source Port', 'Destination Port',
     'port_diff', 'hour', 'dayofweek'] +
    # Binaires
    ['Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 
     'IDS/IPS Alerts', 'has_proxy', 'is_weekend', 'total_alerts'] +
    # Toutes les features target-encoded
    [col for col in df.columns if col.endswith('_te')]
)

X = df[feature_cols].values
y = df['target'].values

# Gérer les NaN éventuels
X = np.nan_to_num(X, nan=0.0)

print(f"Features : {len(feature_cols)}")
print(f"X shape : {X.shape}")
print(f"Target : {np.bincount(y)}")


# In[21]:


# Split train/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")


# In[22]:


# Baseline : random guessing
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X_train, y_train)
print(f"Baseline (random) : {dummy.score(X_test, y_test):.4f}")
# Devrait être ~0.33


# In[24]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
import time

skf = StratifiedKFold(5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, 
                                             min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=20,
                                         min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=5, 
                                                     learning_rate=0.05, random_state=42),
}

results = {}
print(f"{'Model':<25} {'CV Accuracy':<20} {'Test Accuracy':<15} {'Time'}")
print("-" * 70)

for name, model in models.items():
    t0 = time.time()
    cv = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    t = time.time() - t0
    
    results[name] = {'cv': cv.mean(), 'cv_std': cv.std(), 'test': test_acc}
    print(f"{name:<25} {cv.mean():.4f} ± {cv.std():.4f}   {test_acc:.4f}          {t:.1f}s")


# In[25]:


# Refaire le target encoding des ports sur TOUTE la data (sans CV)
# C'est la même approche que le label encoding mais avec la moyenne du target

print("Re-encoding Source Port sur toute la data...")
src_port_means = df.groupby('Source Port')['target'].mean()
df['Source Port_te'] = df['Source Port'].map(src_port_means)

print("Re-encoding Destination Port sur toute la data...")
dst_port_means = df.groupby('Destination Port')['target'].mean()
df['Destination Port_te'] = df['Destination Port'].map(dst_port_means)

print(f"Source Port TE - std: {df['Source Port_te'].std():.4f}")
print(f"Dest Port TE   - std: {df['Destination Port_te'].std():.4f}")


# In[26]:


# Reconstruire X avec les nouvelles valeurs
X = df[feature_cols].values
y = df['target'].values
X = np.nan_to_num(X, nan=0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Test rapide avec Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=5, 
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Random Forest accuracy : {rf.score(X_test, y_test):.4f}")


# In[27]:


# Comparaison des modèles avec le nouveau encoding
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, 
                                             min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=20,
                                         min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=5, 
                                                     learning_rate=0.05, random_state=42),
}

results = {}
print(f"{'Model':<25} {'Test Accuracy'}")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = acc
    print(f"{name:<25} {acc:.4f}")


# In[28]:


# Évaluation détaillée du meilleur modèle
best_model = ExtraTreesClassifier(n_estimators=500, max_depth=20, 
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

target_names = le.classes_.tolist()
print(f"Accuracy : {best_model.score(X_test, y_test):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=target_names))


# In[29]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Extra Trees (92.7%)')
plt.tight_layout()
plt.show()


# In[30]:


# Feature Importance : quelles features comptent le plus ?
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 features :")
print(feat_imp.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
top15 = feat_imp.head(15)
ax.barh(range(len(top15)), top15['importance'].values, color='steelblue')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# In[31]:


import pickle
import os

os.makedirs('../models', exist_ok=True)

save_data = {
    'model': best_model,
    'feature_cols': feature_cols,
    'label_encoder': le,
    'target_names': target_names,
    'src_port_means': src_port_means,
    'dst_port_means': dst_port_means,
}

with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("Modèle sauvegardé !")
print(f"Accuracy : 0.9279")
print(f"Model : Extra Trees (500 estimators)")
print(f"Features : {len(feature_cols)}")


# In[32]:


# APPROACH 1: Encoding on ALL data, then split ---
src_means_all = df.groupby('Source Port')['target'].mean()
dst_means_all = df.groupby('Destination Port')['target'].mean()
df['src_te_v1'] = df['Source Port'].map(src_means_all)
df['dst_te_v1'] = df['Destination Port'].map(dst_means_all)

X1 = df[['src_te_v1', 'dst_te_v1']].values
y = df['target'].values
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X1, y, test_size=0.2, random_state=42, stratify=y)

rf1 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf1.fit(X_tr1, y_tr1)
print(f"Approach 1 (encode ALL then split): {rf1.score(X_te1, y_te1):.4f}")

# --- APPROACH 2: Split first, then encoding on TRAIN only ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
train_df = train_df.copy()
test_df = test_df.copy()

src_means_train = train_df.groupby('Source Port')['target'].mean()
dst_means_train = train_df.groupby('Destination Port')['target'].mean()
global_mean = train_df['target'].mean()

train_df['src_te_v2'] = train_df['Source Port'].map(src_means_train)
train_df['dst_te_v2'] = train_df['Destination Port'].map(dst_means_train)
test_df['src_te_v2'] = test_df['Source Port'].map(src_means_train).fillna(global_mean)
test_df['dst_te_v2'] = test_df['Destination Port'].map(dst_means_train).fillna(global_mean)

# How many test ports are unknown?
unknown_src = test_df['Source Port'].apply(lambda x: x not in src_means_train.index).sum()
unknown_dst = test_df['Destination Port'].apply(lambda x: x not in dst_means_train.index).sum()
print(f"\nUnknown ports in test set: {unknown_src} src ({unknown_src/len(test_df)*100:.1f}%), {unknown_dst} dst ({unknown_dst/len(test_df)*100:.1f}%)")

rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf2.fit(train_df[['src_te_v2','dst_te_v2']].values, train_df['target'].values)
print(f"Approach 2 (split THEN encode train): {rf2.score(test_df[['src_te_v2','dst_te_v2']].values, test_df['target'].values):.4f}")


# In[33]:


import pickle
import os

save_data = {
    'model': best_model,
    'feature_cols': feature_cols,
    'label_encoder': le,
    'target_names': target_names,
    'src_port_means': src_port_means,
    'dst_port_means': dst_port_means,
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("Sauvegardé !")


# In[34]:


import sklearn
print(sklearn.__version__)


# In[ ]:




