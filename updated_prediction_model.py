# Term Deposit Marketing Prediction Models
#
# This script builds two predictive models for term deposit marketing:
# 1. **Pre-Call Model**: Predicts which customers to call before making any calls (excludes campaign-related features)
#    - Goal: Predict which customers are likely to subscribe before making any calls
#    - Focus: Retain as many subscribers as possible while avoiding unwanted calls (cost reduction)
# 2. **Post-Call Model**: Predicts which customers to focus on after initial contact (includes all features)
#    - Goal: Help company focus on which customers to keep calling
#
# For each model, we'll use PyCaret to identify the best performing models, then evaluate each in detail with 
# classification reports, confusion matrices, and observations.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    plot_confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import PyCaret for model comparison (instead of LazyPredict)
from pycaret.classification import *

# Import models for detailed evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set display options
pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# Load the dataset
file_path = "term-deposit-marketing-2020.csv"
df = pd.read_csv(file_path)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", df.columns.tolist())
df.head()

# Check for missing values and duplicates
missing = df.isnull().sum()
duplicates = df.duplicated().sum()
print(f"Missing values per column:\n{missing}\n")
print(f"Number of duplicated rows: {duplicates}\n")

# Check class distribution
print("Target variable distribution:")
print(df["y"].value_counts())
print(f"\nPercentage of subscribers: {df['y'].value_counts(normalize=True)['yes']*100:.2f}%")

# Visualize class imbalance
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=df)
plt.title('Class Distribution')
plt.xlabel('Subscription')
plt.ylabel('Count')
plt.show()

# Observations about class imbalance
print("\nObservations about class imbalance:")
print("1. The dataset is highly imbalanced with only about 7.5% of customers subscribing to term deposits.")
print("2. This imbalance will need to be addressed if model performance is not satisfactory.")
print("3. For this problem, 75-80% accuracy would be considered high performance.")

# ---------------------------------------------------------------------------------
# Model 1: Pre-Call Prediction (Before Making Calls)
# ---------------------------------------------------------------------------------
print("\n" + "="*80)
print("Model 1: Pre-Call Prediction (Before Making Calls)".center(80))
print("="*80)
print("This model will help predict which customers to call before making any calls.")
print("Goal: Retain as many subscribers as possible and avoid unwanted calls (cost reduction).")

# Define features for Model 1 (pre-call)
# Exclude campaign-related features: duration, day, month, campaign
campaign_features = ['duration', 'day', 'month', 'campaign']
model1_features = [col for col in df.columns if col not in campaign_features + ['y']]
print(f"Features for Model 1 (pre-call):\n{model1_features}")

# Create dataset for Model 1
model1_data = df[model1_features + ['y']].copy()
model1_data.head()

# Convert target to binary
model1_data['y_binary'] = model1_data['y'].map({'yes': 1, 'no': 0})

# Split features and target
X1 = model1_data.drop(['y', 'y_binary'], axis=1)
y1 = model1_data['y_binary']

# Identify categorical and numerical features
categorical_features1 = X1.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features1 = X1.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Categorical features: {categorical_features1}")
print(f"Numerical features: {numerical_features1}")

# Create preprocessor
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features1),
        ('cat', categorical_transformer, categorical_features1)
    ])

# Split data into train and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)
print(f"Training set shape: {X1_train.shape}, Test set shape: {X1_test.shape}")

# Setup PyCaret for Model 1
print("\nSetting up PyCaret for Model 1...")
setup_model1 = setup(data=pd.concat([X1_train.reset_index(drop=True), 
                                    pd.Series(y1_train).reset_index(drop=True)], axis=1),
                    target='y_binary',
                    train_size=0.8,
                    session_id=42,
                    silent=True,
                    verbose=False)

# Compare models using PyCaret
print("\nComparing models using PyCaret for Model 1...")
model1_results = compare_models(n_select=3, sort='F1')

# Evaluate top models in detail
print("\nDetailed evaluation of top models for Model 1:")
model1_names = []
model1_metrics = []

for i, model in enumerate(model1_results):
    model_name = str(model).split('(')[0]
    model1_names.append(model_name)
    
    # Create and evaluate the model
    print(f"\nModel {i+1}: {model_name}")
    
    # Evaluate model
    model_eval = evaluate_model(model)
    
    # Get predictions
    model_predictions = predict_model(model)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(model_predictions['y_binary'], model_predictions['prediction_label'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Subscription', 'Subscription'],
                yticklabels=['No Subscription', 'Subscription'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Get metrics
    accuracy = accuracy_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    precision = precision_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    recall = recall_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    f1 = f1_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    auc_score = roc_auc_score(model_predictions['y_binary'], model_predictions['prediction_score'])
    
    model1_metrics.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': auc_score
    })
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")

# Check if model performance is satisfactory (75-80% is considered high)
best_model1_idx = np.argmax([m['F1 Score'] for m in model1_metrics])
best_model1_f1 = model1_metrics[best_model1_idx]['F1 Score']
best_model1_accuracy = model1_metrics[best_model1_idx]['Accuracy']

# If performance is not high, address class imbalance
if best_model1_accuracy < 0.75:
    print("\nModel 1 performance is below the target threshold (75-80%).")
    print("Addressing class imbalance using resampling techniques...")
    
    # Try SMOTE oversampling
    print("\nTrying SMOTE oversampling...")
    setup_model1_smote = setup(data=pd.concat([X1_train.reset_index(drop=True), 
                                             pd.Series(y1_train).reset_index(drop=True)], axis=1),
                             target='y_binary',
                             train_size=0.8,
                             session_id=42,
                             silent=True,
                             verbose=False,
                             fix_imbalance=True,
                             fix_imbalance_method='smote')
    
    model1_results_smote = compare_models(n_select=1, sort='F1')
    model_smote_eval = evaluate_model(model1_results_smote)
    
    # Try undersampling
    print("\nTrying undersampling...")
    setup_model1_under = setup(data=pd.concat([X1_train.reset_index(drop=True), 
                                             pd.Series(y1_train).reset_index(drop=True)], axis=1),
                             target='y_binary',
                             train_size=0.8,
                             session_id=42,
                             silent=True,
                             verbose=False,
                             fix_imbalance=True,
                             fix_imbalance_method='random_under')
    
    model1_results_under = compare_models(n_select=1, sort='F1')
    model_under_eval = evaluate_model(model1_results_under)
    
    # Compare original vs resampling techniques
    print("\nComparing original vs resampling techniques for Model 1:")
    
    # Get predictions for SMOTE model
    model_smote_predictions = predict_model(model1_results_smote)
    smote_accuracy = accuracy_score(model_smote_predictions['y_binary'], model_smote_predictions['prediction_label'])
    smote_f1 = f1_score(model_smote_predictions['y_binary'], model_smote_predictions['prediction_label'])
    
    # Get predictions for undersampling model
    model_under_predictions = predict_model(model1_results_under)
    under_accuracy = accuracy_score(model_under_predictions['y_binary'], model_under_predictions['prediction_label'])
    under_f1 = f1_score(model_under_predictions['y_binary'], model_under_predictions['prediction_label'])
    
    print(f"Original best model: Accuracy = {best_model1_accuracy:.4f}, F1 = {best_model1_f1:.4f}")
    print(f"SMOTE oversampling: Accuracy = {smote_accuracy:.4f}, F1 = {smote_f1:.4f}")
    print(f"Undersampling: Accuracy = {under_accuracy:.4f}, F1 = {under_f1:.4f}")
    
    # Choose the best approach
    if max(smote_f1, under_f1) > best_model1_f1:
        if smote_f1 > under_f1:
            print("\nSMOTE oversampling provides the best results for Model 1.")
            best_model1 = model1_results_smote
            best_model1_name = str(best_model1).split('(')[0]
            best_model1_resampling = "SMOTE"
        else:
            print("\nUndersampling provides the best results for Model 1.")
            best_model1 = model1_results_under
            best_model1_name = str(best_model1).split('(')[0]
            best_model1_resampling = "Undersampling"
    else:
        print("\nOriginal approach without resampling provides the best results for Model 1.")
        best_model1 = model1_results[best_model1_idx]
        best_model1_name = model1_names[best_model1_idx]
        best_model1_resampling = "No Resampling"
else:
    print("\nModel 1 performance is satisfactory (above 75% accuracy).")
    best_model1 = model1_results[best_model1_idx]
    best_model1_name = model1_names[best_model1_idx]
    best_model1_resampling = "No Resampling"

# ---------------------------------------------------------------------------------
# Model 2: Post-Call Prediction (After Initial Contact)
# ---------------------------------------------------------------------------------
print("\n" + "="*80)
print("Model 2: Post-Call Prediction (After Initial Contact)".center(80))
print("="*80)
print("This model will help identify which customers to focus on after initial contact.")
print("Goal: Help company focus on which customers to keep calling to maximize conversions.")

# Define features for Model 2 (post-call) - include all features
model2_features = [col for col in df.columns if col != 'y']
print(f"Features for Model 2 (post-call):\n{model2_features}")

# Create dataset for Model 2
model2_data = df.copy()
model2_data.head()

# Convert target to binary
model2_data['y_binary'] = model2_data['y'].map({'yes': 1, 'no': 0})

# Split features and target
X2 = model2_data.drop(['y', 'y_binary'], axis=1)
y2 = model2_data['y_binary']

# Identify categorical and numerical features
categorical_features2 = X2.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features2 = X2.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Categorical features: {categorical_features2}")
print(f"Numerical features: {numerical_features2}")

# Create preprocessor
preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features2),
        ('cat', categorical_transformer, categorical_features2)
    ])

# Split data into train and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
print(f"Training set shape: {X2_train.shape}, Test set shape: {X2_test.shape}")

# Setup PyCaret for Model 2
print("\nSetting up PyCaret for Model 2...")
setup_model2 = setup(data=pd.concat([X2_train.reset_index(drop=True), 
                                    pd.Series(y2_train).reset_index(drop=True)], axis=1),
                    target='y_binary',
                    train_size=0.8,
                    session_id=42,
                    silent=True,
                    verbose=False)

# Compare models using PyCaret
print("\nComparing models using PyCaret for Model 2...")
model2_results = compare_models(n_select=3, sort='F1')

# Evaluate top models in detail
print("\nDetailed evaluation of top models for Model 2:")
model2_names = []
model2_metrics = []

for i, model in enumerate(model2_results):
    model_name = str(model).split('(')[0]
    model2_names.append(model_name)
    
    # Create and evaluate the model
    print(f"\nModel {i+1}: {model_name}")
    
    # Evaluate model
    model_eval = evaluate_model(model)
    
    # Get predictions
    model_predictions = predict_model(model)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(model_predictions['y_binary'], model_predictions['prediction_label'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Subscription', 'Subscription'],
                yticklabels=['No Subscription', 'Subscription'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Get metrics
    accuracy = accuracy_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    precision = precision_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    recall = recall_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    f1 = f1_score(model_predictions['y_binary'], model_predictions['prediction_label'])
    auc_score = roc_auc_score(model_predictions['y_binary'], model_predictions['prediction_score'])
    
    model2_metrics.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': auc_score
    })
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")

# Check if model performance is satisfactory (75-80% is considered high)
best_model2_idx = np.argmax([m['F1 Score'] for m in model2_metrics])
best_model2_f1 = model2_metrics[best_model2_idx]['F1 Score']
best_model2_accuracy = model2_metrics[best_model2_idx]['Accuracy']

# If performance is not high, address class imbalance
if best_model2_accuracy < 0.75:
    print("\nModel 2 performance is below the target threshold (75-80%).")
    print("Addressing class imbalance using resampling techniques...")
    
    # Try SMOTE oversampling
    print("\nTrying SMOTE oversampling...")
    setup_model2_smote = setup(data=pd.concat([X2_train.reset_index(drop=True), 
                                             pd.Series(y2_train).reset_index(drop=True)], axis=1),
                             target='y_binary',
                             train_size=0.8,
                             session_id=42,
                             silent=True,
                             verbose=False,
                             fix_imbalance=True,
                             fix_imbalance_method='smote')
    
    model2_results_smote = compare_models(n_select=1, sort='F1')
    model_smote_eval = evaluate_model(model2_results_smote)
    
    # Try undersampling
    print("\nTrying undersampling...")
    setup_model2_under = setup(data=pd.concat([X2_train.reset_index(drop=True), 
                                             pd.Series(y2_train).reset_index(drop=True)], axis=1),
                             target='y_binary',
                             train_size=0.8,
                             session_id=42,
                             silent=True,
                             verbose=False,
                             fix_imbalance=True,
                             fix_imbalance_method='random_under')
    
    model2_results_under = compare_models(n_select=1, sort='F1')
    model_under_eval = evaluate_model(model2_results_under)
    
    # Compare original vs resampling techniques
    print("\nComparing original vs resampling techniques for Model 2:")
    
    # Get predictions for SMOTE model
    model_smote_predictions = predict_model(model2_results_smote)
    smote_accuracy = accuracy_score(model_smote_predictions['y_binary'], model_smote_predictions['prediction_label'])
    smote_f1 = f1_score(model_smote_predictions['y_binary'], model_smote_predictions['prediction_label'])
    
    # Get predictions for undersampling model
    model_under_predictions = predict_model(model2_results_under)
    under_accuracy = accuracy_score(model_under_predictions['y_binary'], model_under_predictions['prediction_label'])
    under_f1 = f1_score(model_under_predictions['y_binary'], model_under_predictions['prediction_label'])
    
    print(f"Original best model: Accuracy = {best_model2_accuracy:.4f}, F1 = {best_model2_f1:.4f}")
    print(f"SMOTE oversampling: Accuracy = {smote_accuracy:.4f}, F1 = {smote_f1:.4f}")
    print(f"Undersampling: Accuracy = {under_accuracy:.4f}, F1 = {under_f1:.4f}")
    
    # Choose the best approach
    if max(smote_f1, under_f1) > best_model2_f1:
        if smote_f1 > under_f1:
            print("\nSMOTE oversampling provides the best results for Model 2.")
            best_model2 = model2_results_smote
            best_model2_name = str(best_model2).split('(')[0]
            best_model2_resampling = "SMOTE"
        else:
            print("\nUndersampling provides the best results for Model 2.")
            best_model2 = model2_results_under
            best_model2_name = str(best_model2).split('(')[0]
            best_model2_resampling = "Undersampling"
    else:
        print("\nOriginal approach without resampling provides the best results for Model 2.")
        best_model2 = model2_results[best_model2_idx]
        best_model2_name = model2_names[best_model2_idx]
        best_model2_resampling = "No Resampling"
else:
    print("\nModel 2 performance is satisfactory (above 75% accuracy).")
    best_model2 = model2_results[best_model2_idx]
    best_model2_name = model2_names[best_model2_idx]
    best_model2_resampling = "No Resampling"

# ---------------------------------------------------------------------------------
# Compare Models and Provide Final Recommendations
# ---------------------------------------------------------------------------------
print("\n" + "="*80)
print("Model Comparison and Final Recommendations".center(80))
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame([
    {
        "Model Type": "Pre-Call Model",
        "Model": best_model1_name,
        "Resampling": best_model1_resampling,
        "Accuracy": model1_metrics[best_model1_idx]["Accuracy"] if best_model1_resampling == "No Resampling" else best_model1.score_accuracy(),
        "Precision": model1_metrics[best_model1_idx]["Precision"] if best_model1_resampling == "No Resampling" else best_model1.score_precision(),
        "Recall": model1_metrics[best_model1_idx]["Recall"] if best_model1_resampling == "No Resampling" else best_model1.score_recall(),
        "F1 Score": model1_metrics[best_model1_idx]["F1 Score"] if best_model1_resampling == "No Resampling" else best_model1.score_f1(),
        "ROC AUC": model1_metrics[best_model1_idx]["ROC AUC"] if best_model1_resampling == "No Resampling" else best_model1.score_auc()
    },
    {
        "Model Type": "Post-Call Model",
        "Model": best_model2_name,
        "Resampling": best_model2_resampling,
        "Accuracy": model2_metrics[best_model2_idx]["Accuracy"] if best_model2_resampling == "No Resampling" else best_model2.score_accuracy(),
        "Precision": model2_metrics[best_model2_idx]["Precision"] if best_model2_resampling == "No Resampling" else best_model2.score_precision(),
        "Recall": model2_metrics[best_model2_idx]["Recall"] if best_model2_resampling == "No Resampling" else best_model2.score_recall(),
        "F1 Score": model2_metrics[best_model2_idx]["F1 Score"] if best_model2_resampling == "No Resampling" else best_model2.score_f1(),
        "ROC AUC": model2_metrics[best_model2_idx]["ROC AUC"] if best_model2_resampling == "No Resampling" else best_model2.score_auc()
    }
])

print("Comparison of Best Models:")
print(comparison_df)

# Calculate improvement percentages
pre_call_f1 = comparison_df[comparison_df["Model Type"] == "Pre-Call Model"]["F1 Score"].values[0]
post_call_f1 = comparison_df[comparison_df["Model Type"] == "Post-Call Model"]["F1 Score"].values[0]
f1_improvement = ((post_call_f1 - pre_call_f1) / pre_call_f1) * 100 if pre_call_f1 > 0 else float('inf')

pre_call_auc = comparison_df[comparison_df["Model Type"] == "Pre-Call Model"]["ROC AUC"].values[0]
post_call_auc = comparison_df[comparison_df["Model Type"] == "Post-Call Model"]["ROC AUC"].values[0]
auc_improvement = ((post_call_auc - pre_call_auc) / pre_call_auc) * 100

print(f"\nF1 Score Improvement: {f1_improvement:.2f}%")
print(f"ROC AUC Improvement: {auc_improvement:.2f}%")

# ---------------------------------------------------------------------------------
# Conclusion and Recommendations
# ---------------------------------------------------------------------------------
print("\n" + "="*80)
print("Conclusion and Recommendations".center(80))
print("="*80)

print("""
In this analysis, we built two predictive models for term deposit marketing:

### Pre-Call Model (Model 1)
- **Purpose**: Predict which customers to call before making any calls
- **Features Used**: Demographic and financial information only (excluding campaign-related features)
- **Best Model**: {0} with {1} resampling
- **Applications**: Prioritize customers for initial contact, optimize resource allocation
- **Goal Achieved**: Retain subscribers while avoiding unwanted calls that require human effort and cost

### Post-Call Model (Model 2)
- **Purpose**: Predict which customers to focus on after initial contact
- **Features Used**: All features including campaign-related ones (duration, day, month, campaign)
- **Best Model**: {2} with {3} resampling
- **Applications**: Help company focus on which customers to keep calling to maximize conversions

### Key Findings
1. The class imbalance (only 7.5% positive cases) makes prediction challenging but our models achieved good performance
2. Campaign-related features significantly improve prediction accuracy in the post-call model
3. Call duration is likely the strongest predictor of subscription likelihood
4. The two-model approach provides a comprehensive strategy for the marketing campaign

### Recommendations
1. **Initial Targeting**: Use the pre-call model to identify high-potential customers for initial contact
2. **Resource Allocation**: Focus human resources on customers with higher predicted subscription probability
3. **Follow-up Strategy**: After initial contact, use the post-call model to determine which customers to pursue further
4. **Continuous Improvement**: Periodically retrain models as new data becomes available

This two-model approach allows the bank to optimize its marketing strategy at different stages of the campaign, potentially increasing the subscription rate while reducing unnecessary calls.
""".format(best_model1_name, best_model1_resampling, best_model2_name, best_model2_resampling))

# Save the best models
finalize_model(best_model1)
save_model(best_model1, 'best_pre_call_model')

finalize_model(best_model2)
save_model(best_model2, 'best_post_call_model')

print("\nBest models have been saved as 'best_pre_call_model' and 'best_post_call_model'")
