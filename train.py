import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = "data"
LABELS = ["background", "thumbs_up"] # 0: background, 1: thumbs_up
IMG_SIZE = (64, 64)

def get_hog_features(image):
    win_size = IMG_SIZE
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image).flatten()

def load_data():
    X = []
    y = []
    
    print("Loading data...")
    for label in LABELS:
        path = os.path.join(DATA_DIR, label)
        class_num = LABELS.index(label)
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read {img_name}. Skipping.")
                    continue
                img = cv2.resize(img, IMG_SIZE)
                
                features = get_hog_features(img)
                X.append(features)
                y.append(class_num)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                
    return np.array(X), np.array(y)

def train():
    X, y = load_data()
    if len(X) == 0:
        print("No data loaded. Exiting training.")
        return
        
    print(f"Loaded {len(X)} samples.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training SVM with GridSearchCV...")
    
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    # Create a GridSearchCV object
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=3)
    
    # Fit the model
    grid.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters found: ", grid.best_params_)
    
    # Use the best estimator to make predictions
    model = grid.best_estimator_
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(model, "gesture_model.pkl")
    print("Model saved as 'gesture_model.pkl'")

if __name__ == "__main__":
    train()
