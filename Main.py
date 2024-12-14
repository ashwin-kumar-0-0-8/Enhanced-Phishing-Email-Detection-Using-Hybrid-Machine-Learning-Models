import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = '/content/Phishing_Legitimate_full.csv'  # Update the path to your dataset
phishing_full_data = pd.read_csv(file_path)

# Separate features and target variable
X = phishing_full_data.drop(['id', 'CLASS_LABEL'], axis=1)
y = phishing_full_data['CLASS_LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training data for validation (used for stacking)
X_train_base, X_val_meta, y_train_base, y_val_meta = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_base_scaled = scaler.fit_transform(X_train_base)
X_val_meta_scaled = scaler.transform(X_val_meta)
X_test_scaled = scaler.transform(X_test)

# Hybrid Model 1: Decision Tree + Logistic Regression
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_base_scaled, y_train_base)
X_train_dt = dt_model.predict_proba(X_train_base_scaled)[:, 1].reshape(-1, 1)
X_val_dt = dt_model.predict_proba(X_val_meta_scaled)[:, 1].reshape(-1, 1)
X_test_dt = dt_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

lr_model_dt = LogisticRegression(random_state=42)
lr_model_dt.fit(X_train_dt, y_train_base)

y_pred_dt_lr_test = lr_model_dt.predict(X_test_dt)
accuracy_dt_lr_test = accuracy_score(y_test, y_pred_dt_lr_test)

# Hybrid Model 2: Random Forest + Gradient Boosting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_base_scaled, y_train_base)
X_train_rf = rf_model.predict_proba(X_train_base_scaled)[:, 1].reshape(-1, 1)
X_val_rf = rf_model.predict_proba(X_val_meta_scaled)[:, 1].reshape(-1, 1)
X_test_rf = rf_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_rf, y_train_base)

y_pred_rf_gb_test = gb_model.predict(X_test_rf)
accuracy_rf_gb_test = accuracy_score(y_test, y_pred_rf_gb_test)

# Hybrid Model 3: Naive Bayes + K-Nearest Neighbors (KNN)
nb_model = GaussianNB()
nb_model.fit(X_train_base_scaled, y_train_base)
X_train_nb = nb_model.predict_proba(X_train_base_scaled)[:, 1].reshape(-1, 1)
X_val_nb = nb_model.predict_proba(X_val_meta_scaled)[:, 1].reshape(-1, 1)
X_test_nb = nb_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_nb, y_train_base)

y_pred_nb_knn_test = knn_model.predict(X_test_nb)
accuracy_nb_knn_test = accuracy_score(y_test, y_pred_nb_knn_test)

# Hybrid Model 4: SVM + AdaBoost
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_base_scaled, y_train_base)
X_train_svm = svm_model.predict_proba(X_train_base_scaled)[:, 1].reshape(-1, 1)
X_val_svm = svm_model.predict_proba(X_val_meta_scaled)[:, 1].reshape(-1, 1)
X_test_svm = svm_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train_svm, y_train_base)

y_pred_svm_ada_test = ada_model.predict(X_test_svm)
accuracy_svm_ada_test = accuracy_score(y_test, y_pred_svm_ada_test)

# Stacking: Meta-learner (Logistic Regression) to combine the results of the four hybrid models
X_meta_train = np.column_stack((X_val_dt, X_val_rf, X_val_nb, X_val_svm))
X_meta_test = np.column_stack((X_test_dt, X_test_rf, X_test_nb, X_test_svm))

# Meta-Learner: Logistic Regression
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(X_meta_train, y_val_meta)

# Final prediction using the meta-learner
y_pred_meta = meta_learner.predict(X_meta_test)
accuracy_meta = accuracy_score(y_test, y_pred_meta)

# Create DataFrame for plotting
accuracy_data = {
    'Model': ['Hybrid Model 1 (DT + LR)', 'Hybrid Model 2 (RF + GB)',
              'Hybrid Model 3 (NB + KNN)', 'Hybrid Model 4 (SVM + AdaBoost)', 'Stacked Model (Meta-Learner)'],
    'Accuracy': [accuracy_dt_lr_test, accuracy_rf_gb_test,
                 accuracy_nb_knn_test, accuracy_svm_ada_test, accuracy_meta]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Plot the bar graph
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='mako')
plt.title('Hybrid Model Accuracy Comparison with Meta-Learner')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()

# Print accuracy scores
print(accuracy_data)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Function to calculate and display metrics
def print_metrics(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Extracts True Negatives, False Positives, False Negatives, and True Positives
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate FPR and FNR
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate

    print(f"Metrics for {model_name}:")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*40)

# Use the function for each model
print_metrics(y_test, y_pred_dt_lr_test, "Hybrid Model 1 (DT + LR)")
print_metrics(y_test, y_pred_rf_gb_test, "Hybrid Model 2 (RF + GB)")
print_metrics(y_test, y_pred_nb_knn_test, "Hybrid Model 3 (NB + KNN)")
print_metrics(y_test, y_pred_svm_ada_test, "Hybrid Model 4 (SVM + AdaBoost)")
print_metrics(y_test, y_pred_meta, "Meta-Learner (Stacked Model)")
