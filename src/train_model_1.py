import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 讀取資料
df = pd.read_csv("User Behavior Class.csv")
X = df.drop(columns=['User Behavior Class'])
y = df['User Behavior Class']

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 模型訓練與比較
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = {"accuracy": acc, "model": model}

# 儲存最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]["model"]
joblib.dump(best_model, "best_model.pkl")

# 輸出最佳模型資訊
print(f"最佳模型: {best_model_name}, 準確率: {results[best_model_name]['accuracy']}")
