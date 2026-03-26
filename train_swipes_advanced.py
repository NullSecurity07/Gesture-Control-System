import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb

# Constants
DATA_FILE = "swipe_data_cvzone.csv"
MODEL_FILE = "swipe_model.pkl"

def augment_data(X, y):
    augmented_X = list(X)
    augmented_y = list(y)
    
    for i in range(len(X)):
        # Add noise
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
    return np.array(augmented_X), np.array(augmented_y)

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run collect_swipes.py first.")
        return

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    X = df.drop(["label", "label_encoded"], axis=1).values
    y = df['label_encoded'].values

    # Data Augmentation
    print("Augmenting data...")
    X_aug, y_aug = augment_data(X, y)
    
    print(f"Dataset shape after augmentation: {X_aug.shape}")

    # K-Fold Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    print("Training XGBoost model with K-Fold Cross-Validation...")
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_aug, y_aug)):
        print(f"\n--- FOLD {fold+1}/5 ---")
        X_train, X_val = X_aug[train_index], X_aug[val_index]
        y_train, y_val = y_aug[train_index], y_aug[val_index]

        # Build XGBoost model
        model = xgb.XGBClassifier(objective='multi:softmax', 
                                  num_class=len(le.classes_),
                                  eval_metric='mlogloss',
                                  use_label_encoder=False)
        
        # Train the model
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc * 100:.2f}%")

    print(f"\nAverage Cross-Validation Accuracy: {np.mean(accuracies) * 100:.2f}%")

    # Train final model on all data
    print("\nTraining final model on all augmented data...")
    final_model = xgb.XGBClassifier(objective='multi:softmax', 
                                     num_class=len(le.classes_),
                                     eval_metric='mlogloss',
                                     use_label_encoder=False)
    final_model.fit(X_aug, y_aug, verbose=False)
    
    joblib.dump(final_model, MODEL_FILE)
    print(f"XGBoost model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()