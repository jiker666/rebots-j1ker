import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 加载数据
data = pd.read_csv(r"E:\1\美赛\2025_Problem_C_Data\processed_athletes_data_with_noc_and_host.csv")

# 提取预测奖牌总数的特征和目标变量
X_total_medals = data[['Year', 'NOC', 'Is_Host', 'Total_events_4_Years', 'Total_events_8_Years', 'Total_events_12_Years',
                       'Total_disciplines_4_Years', 'Total_disciplines_8_Years', 'Total_disciplines_12_Years',
                       'Total_sports_4_Years', 'Total_sports_8_Years', 'Total_sports_12_Years', 'Participants_4_Years',
                       'Participants_8_Years', 'Participants_12_Years', 'Gold_4_Years', 'Gold_8_Years', 'Gold_12_Years',
                       'Total_medals_4_Years', 'Total_medals_8_Years', 'Total_medals_12_Years']]
y_total_medals = data['Total_medals']

# 对 NOC 进行独热编码
X_total_medals = pd.get_dummies(X_total_medals, columns=['NOC'])

# 划分训练集和测试集（预测奖牌总数）
X_train_total_medals, X_test_total_medals, y_train_total_medals, y_test_total_medals = train_test_split(
    X_total_medals, y_total_medals, test_size=0.2, random_state=42)

# 对特征进行标准化处理（预测奖牌总数）
scaler_total_medals = StandardScaler()
numerical_columns_total_medals = [col for col in X_train_total_medals.columns if col != 'Year' and not col.startswith('NOC_')]
X_train_total_medals[numerical_columns_total_medals] = scaler_total_medals.fit_transform(X_train_total_medals[numerical_columns_total_medals])
X_test_total_medals[numerical_columns_total_medals] = scaler_total_medals.transform(X_test_total_medals[numerical_columns_total_medals])

# 创建并训练预测奖牌总数的模型
model_total_medals = LinearRegression()
model_total_medals.fit(X_train_total_medals, y_train_total_medals)

# 在测试集上进行预测（奖牌总数）
y_pred_total_medals = model_total_medals.predict(X_test_total_medals)

# 评估预测奖牌总数的模型
r2_total_medals = r2_score(y_test_total_medals, y_pred_total_medals)
mse_total_medals = mean_squared_error(y_test_total_medals, y_pred_total_medals)

# 可视化：绘制预测与实际值的比较
plt.figure(figsize=(8, 6))
plt.scatter(y_test_total_medals, y_pred_total_medals, color='blue', alpha=0.5)
plt.plot([y_test_total_medals.min(), y_test_total_medals.max()], [y_test_total_medals.min(), y_test_total_medals.max()], color='red', linestyle='--')
plt.title('Total Medal Prediction vs Actual (R² = {:.2f})'.format(r2_total_medals))
plt.xlabel('Actual Total Medals')
plt.ylabel('Predicted Total Medals')
plt.grid(True)

# 创建 tkinter 窗口
root = tk.Tk()
root.title("Olympic Medal Prediction")

# 将 matplotlib 图形嵌入 tkinter 窗口
canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# 启动 tkinter 窗口
root.mainloop()
