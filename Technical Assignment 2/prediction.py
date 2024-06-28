import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ambil data
data = pd.read_csv('ai4i2020.csv')

# tampilkan data secara kesimpulan 
print(data.head())
print(data.info())
print(data.describe())

# cek data kosong
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
# kalo ada hapus
data = data.dropna()

# visualisasikan data berdasarkan distribusi Tool wear [min]
plt.figure(figsize=(12, 6))
sns.histplot(data['Tool wear [min]'], kde=True)
plt.title('Tool Wear Distribution')
plt.show()

# visualisasi korelasi
plt.figure(figsize=(12, 6))
numerical_data = data.select_dtypes(include=['float', 'int'])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# penetapan x dan y
X = data.drop(columns=['Machine failure', 'UDI', 'Product ID'])
y = data['Machine failure']

# Cek kolom yang bukan float dan int
non_numeric_cols = X.select_dtypes(exclude=['float', 'int']).columns
print("Non-numeric columns:", non_numeric_cols)
X = X.drop(columns=non_numeric_cols)

# membagi data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# memilih model logistic regression 
model_lr = LogisticRegression(random_state=42)
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_search_lr = GridSearchCV(estimator=model_lr, param_grid=param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression: ", grid_search_lr.best_params_)

# eval
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, roc_auc

metrics_lr = evaluate_model(best_model_lr, X_test, y_test)

print(f'Logistic Regression - Accuracy: {metrics_lr[0]:.4f}, Precision: {metrics_lr[1]:.4f}, Recall: {metrics_lr[2]:.4f}, F1-score: {metrics_lr[3]:.4f}, ROC AUC: {metrics_lr[4]:.4f}')

# Confusion matrix untuk Logistic Regression
conf_matrix_lr = confusion_matrix(y_test, best_model_lr.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# grafik probabilitas kegagalan berdasarkan "Process temperature [K]"
for failure_type in y_test.unique():
    df_failure = data[data['Machine failure'] == failure_type]
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df_failure, x='Process temperature [K]')
    plt.title(f'Probabilitas kegagalan berdasarkan Process temperature [K] ({failure_type})')
    plt.ylabel('Probability Density')
    plt.show()
