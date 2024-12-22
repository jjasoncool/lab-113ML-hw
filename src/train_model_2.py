import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import openpyxl  # 新增這行
import xgboost as xgb  # 新增這行
from sklearn.ensemble import RandomForestRegressor  # 保留這行

def handle_file_for_train_2(file_path, use_xgboost=False):
    # 讀取資料
    data = pd.read_excel(file_path)
    # Define features and targets
    X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
            'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
            'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28',
            'X29', 'X30']]
    Y = data[['Y1', 'Y2', 'Y3']]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    if use_xgboost:
        # Initialize the base estimator and the multi-output regressor with XGBoost
        base_estimator = xgb.XGBRegressor(
            n_estimators=200,  # 增加樹的數量
            max_depth=20,  # 設置樹的最大深度
            learning_rate=0.1,  # 設置學習率
            tree_method='hist',  # 使用 CPU 的 hist 方法
            device='cuda',  # 使用 GPU
            random_state=42
        )
    else:
        # Initialize the base estimator and the multi-output regressor with RandomForest
        base_estimator = RandomForestRegressor(
            n_estimators=200,  # 增加樹的數量
            max_depth=20,  # 設置樹的最大深度
            min_samples_split=5,  # 設置分裂所需的最小樣本數
            min_samples_leaf=2,  # 設置葉節點所需的最小樣本數
            random_state=42
        )

    model = MultiOutputRegressor(base_estimator)

    # Fit the model
    model.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = model.predict(X_test)

    # Calculate R2 score for each target
    r2_scores = {}
    for i, target in enumerate(['Y1', 'Y2', 'Y3']):
        r2_scores[target] = r2_score(Y_test[target], Y_pred[:, i])
        print(f"R2 Score for {target}: {r2_scores[target]}")

    # If you want the average R2 score across all targets
    average_r2 = sum(r2_scores.values()) / len(r2_scores)
    print(f"Average R2 Score: {average_r2}")

    return Y_pred, Y_test
