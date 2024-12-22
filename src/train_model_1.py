import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def handle_file_for_train_1(file_path):
    # 讀取資料
    df = pd.read_csv(file_path)
    X = df.drop(columns=['User Behavior Class'])
    y = df['User Behavior Class']

    # 資料分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # 模型訓練與比較
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "SVC": SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    }
    results = {}
    for model_name, model in models.items():
        # 使用交叉驗證評估模型
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_score = cv_scores.mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[model_name] = {"accuracy": acc, "cv_score": mean_cv_score, "model": model}
        print(f"Model: {model_name}, Accuracy: {acc}, Cross-Validation Score: {mean_cv_score}")

    # 儲存最佳模型
    best_model_name = max(results, key=lambda x: results[x]['cv_score'])
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, "user_behavior_model.pkl")

    # 載入最佳模型進行預測
    model = joblib.load("user_behavior_model.pkl")
    predictions = model.predict(X)
    return predictions, y
