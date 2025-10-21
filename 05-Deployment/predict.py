# %% [markdown]
# ### Churn prediction code from 04-Evaluation Metrics

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# %%
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# %%
# !wget $data

# %%
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# %%
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# %%
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

# %%
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
C = 1.0
n_splits = 5

# %%
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# %%
scores

# %%
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc

# %% [markdown]
# ### 5.2 Saving and loading the model
# 
# * Saving the model using Pickle
# * Load the model from Pickle
# * Save Jupyter Notebook to Python script

# %%
import pickle

# %%
output_file = f'model_C={C}.bin'
output_file

# %% [markdown]
# Saving the model and dictionary vectorizer

# %%
f_out = open(output_file, 'wb')  # wb - write bytes
pickle.dump((dv, model), f_out)
f_out.close()

# %%
with open(output_file, 'wb') as f_out:  # same as above but compact
    pickle.dump((dv, model), f_out)


# %% [markdown]
# Load the model. Be sure to restart the kernel if you want to ensure none of the above is in memory.

# %%
import pickle

# %%
input_file = 'model_C=1.0.bin'

# %% [markdown]
# Note sci-kit learn must be installed on the computer opening the model or pickle will fail.

# %%
with open(input_file, 'rb') as f_in: 
    (dv, model) = pickle.load(f_in)

# %%
dv, model

# %% [markdown]
# Customer we want to score.

# %%
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# %% [markdown]
# Turn customer into a feature matrix.

# %%
X = dv.transform([customer]) # vectorizer expects a list so wrap in []
X

# %%
model.predict_proba(X)[0, 1]  # probability that the customer will churn

# %%



