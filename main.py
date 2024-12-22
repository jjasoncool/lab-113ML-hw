import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.train_model_1 import handle_file_for_model_1

# === 功能 1: 使用模型 1 預測 ===
def predict_behavior_1():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        predictions, real_data = handle_file_for_model_1(file_path)
        if len(predictions) != len(real_data):
            raise ValueError("預測結果與實際資料的長度不一致")
        else:
            # 找出不同的部分
            differences = np.where(real_data != predictions)[0]

            # 創建子圖
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # 第一張圖：顯示不同的部分
            ax1.scatter(differences, real_data[differences], label='Real Data', marker='o', color='blue')
            ax1.scatter(differences, predictions[differences], label='Predictions', marker='x', color='red')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Value')
            ax1.set_title('Differences between Predictions and Real Data')
            ax1.legend()
            ax1.grid(True)

            # 第二張圖：顯示預測與實際資料
            ax2.plot(real_data, label='Real Data', marker='o')
            ax2.plot(predictions, label='Predictions', marker='x')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Value')
            ax2.set_title('Predictions vs Real Data')
            ax2.legend()
            ax2.grid(True)

            # 顯示圖表
            plt.tight_layout()
            plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"匯入錯誤: {e}")

# === 功能 2: 使用模型 2 預測 ===
def predict_behavior_2():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        data = pd.read_excel(file_path).values.tolist()
        # 載入模型
        model = joblib.load("model_2.pkl")
        predictions = model.predict(data)
        messagebox.showinfo("Prediction Result", f"使用者行為模式為: 類別 {predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"匯入錯誤: {e}")

# === 功能切換 ===
def switch_function():
    selected_function = selected_option.get()
    if selected_function == "使用行為分類":
        predict_behavior_1()
    elif selected_function == "工作機台參數迴歸預測":
        predict_behavior_2()
    else:
        messagebox.showinfo("功能提示", "尚未實作其他功能。")

# 建立 Tkinter 視窗
window = tk.Tk()
window.title("多功能使用者行為分類系統")

# 功能選擇
selected_option = tk.StringVar(value="使用行為分類")
tk.Label(window, text="選擇功能:").grid(row=0, column=0)
tk.OptionMenu(window, selected_option, "使用行為分類", "工作機台參數迴歸預測").grid(row=0, column=1, columnspan=3)

# 預測按鈕
predict_button = tk.Button(window, text="執行功能", command=switch_function)
predict_button.grid(row=2, column=4, sticky="e", padx=10, pady=5)

# 運行視窗
window.mainloop()
