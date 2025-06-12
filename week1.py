# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Set up visualization style
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Load the dataset
def load_data(filepath):
    """Load the customer churn dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Data preprocessing
def preprocess_data(df):
    """Clean and preprocess the data"""
    
    # Drop customer ID as it's not useful for prediction
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, handle empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values (only in TotalCharges for this dataset)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert binary categorical variables to numerical (0/1)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    
    # Convert other categorical variables using label encoding
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod']
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis and visualization"""
    
    # Plot churn distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Customer Churn Distribution')
    plt.show()
    
    # Plot numerical features distribution
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Numerical Features Distribution')
    plt.show()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

# Feature engineering
def feature_engineering(df):
    """Create new features that might help prediction"""
    
    # Create a tenure group feature
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                              labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
    
    # Create a feature for ratio of total charges to tenure
    df['ChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 0.001)  # Avoid division by zero
    
    # One-hot encode the tenure group
    df = pd.get_dummies(df, columns=['TenureGroup'], drop_first=True)
    
    return df

# Prepare data for modeling
def prepare_data(df):
    """Split data into features and target, then train-test split"""
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargePerMonth']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, X.columns

# Model training and evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and evaluate their performance"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': report,
            'confusion_matrix': cm
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)
    
    return results

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Create a DataFrame with feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    elif hasattr(model, 'coef_'):
        # For logistic regression
        coefficients = model.coef_[0]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Coefficients")
        plt.bar(range(len(coefficients)), coefficients)
        plt.xticks(range(len(coefficients)), feature_names, rotation=90)
        plt.tight_layout()
        plt.show()
        
        return None
    
    else:
        print("Model doesn't support feature importance analysis.")
        return None

# Main workflow
def main():
    # Load data
    df = load_data('telecom_churn_data.csv')  # Replace with your file path
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform EDA
    perform_eda(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance of the best model (XGBoost in this case)
    best_model = results['XGBoost']['model']
    feature_importance = analyze_feature_importance(best_model, feature_names)
    
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Optional: Hyperparameter tuning (example for XGBoost)
    print("\nPerforming hyperparameter tuning for XGBoost...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best ROC AUC score: ", grid_search.best_score_)

if _name_ == "_main_":
    main()