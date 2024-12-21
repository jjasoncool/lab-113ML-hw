import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import pandas as pd

# === 功能 1: 使用模型 1 預測 ===
def predict_behavior_1():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        data = pd.read_csv(file_path).values.tolist()
        # 載入模型
        model = joblib.load("model_1.pkl")
        predictions = model.predict(data)
        messagebox.showinfo("Prediction Result", f"使用者行為模式為: 類別 {predictions}")
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
