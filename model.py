from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats

import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


data = pd.read_csv('data/base_modelo.csv')
# print(data.info())
# print(data.head())

sns.countplot(x='ind_malo', data=data)
plt.title('Target Variable (ind_malo)')
plt.savefig('targetvariable.png')

X = data.drop('ind_malo', axis=1)
y = data['ind_malo']

print('------------------------')
print(y.value_counts())
print('------------------------')

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


models = {
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = {}


def gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1


def ks(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return np.max(tpr - fpr)


for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    # print(y_pred)

    auc = roc_auc_score(y_test, y_pred)
    gini_score = gini(y_test, y_pred)
    ks_statistic = ks(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)

    results[name] = {'AUC': auc, 'Gini': gini_score,
                     'K-S': ks_statistic, 'Log Loss': logloss}
    print(f'{name}:')
    print(f'  AUC: {auc:.4f}')
    print(f'  Gini: {gini_score:.4f}')
    print(f'  K-S: {ks_statistic:.4f}')
    print(f'  Log Loss: {logloss:.4f}')

results_data = pd.DataFrame.from_dict(results, orient='index')
print('\nResults:')
print(results_data)

plt.figure(figsize=(12, 6))
results_data.plot(kind='bar')
plt.title('Comparación de modelos')
plt.ylabel('Métrica')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print('-------------------------')

# Gradient Boosting Best model

best_model = models['Gradient Boosting']
pipeline = Pipeline(
    steps=[('preprocessor', preprocessor), ('classifier', best_model)])
pipeline.fit(X, y)

non_payment_prob = pipeline.predict_proba(X)[:, 1]
data['non_payment_probability'] = non_payment_prob


media = non_payment_prob.mean()
mediana = np.median(non_payment_prob)
desv_est = non_payment_prob.std()

umbral = media + desv_est
high_risk_client = data[data['non_payment_probability'] > umbral]
high_risk_percentage = len(high_risk_client) / len(data) * 100

with open('estadisticas_impago.txt', 'w') as f:
    f.write(f"Probabilidad media de impago: {non_payment_prob.mean():.4f}\n")
    f.write(
        f"Mediana de la probabilidad de impago: {np.median(non_payment_prob):.4f}\n")
    f.write(
        f"Desviación estándar de la probabilidad de impago: {non_payment_prob.std():.4f}\n")
    f.write(
        f'Porcentaje de créditos con alto riesgo de impago: {high_risk_percentage:.2f}%'
    )


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.hist(non_payment_prob, bins=50, edgecolor='black')
ax1.set_title('Probabilidades de Impago')
ax1.set_xlabel('Probabilidad de Impago')
ax1.set_ylabel('Conteo')

n, bins, patches = ax2.hist(non_payment_prob, bins=50,
                            density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(0, 1, 100)
p = stats.norm.pdf(x, media, desv_est)
ax2.plot(x, p, 'r', linewidth=2)
ax2.axvline(x=media, color='green', linestyle='--', linewidth=2)
ax2.set_title(
    'Distribución de Probabilidades de Impago')
ax2.set_xlabel('Probabilidad de Impago')
ax2.set_ylabel('Densidad')
ax2.set_xlim(0, 1)
ax2.legend(['Distribución Normal', f'Media ({media:.2f})', 'Data'])

ax2.text(0.95, 0.95, f'Media: {media:.2f}\nMediana: {mediana:.2f}\nDesv. Est.: {desv_est:.2f}',
         verticalalignment='top', horizontalalignment='right',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('distribución_impago_prob.png')
plt.close()
