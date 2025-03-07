import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

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

print(f'预测奖牌总数模型的 R2 分数: {r2_total_medals}')
print(f'预测奖牌总数模型的均方误差: {mse_total_medals}')

# 提取预测金牌数的特征和目标变量
X_gold = data[['Year', 'NOC', 'Is_Host', 'Total_events_4_Years', 'Total_events_8_Years', 'Total_events_12_Years',
               'Total_disciplines_4_Years', 'Total_disciplines_8_Years', 'Total_disciplines_12_Years',
               'Total_sports_4_Years', 'Total_sports_8_Years', 'Total_sports_12_Years', 'Participants_4_Years',
               'Participants_8_Years', 'Participants_12_Years', 'Gold_4_Years', 'Gold_8_Years', 'Gold_12_Years',
               'Total_medals_4_Years', 'Total_medals_8_Years', 'Total_medals_12_Years']]
y_gold = data['Gold']

# 对 NOC 进行独热编码
X_gold = pd.get_dummies(X_gold, columns=['NOC'])

# 划分训练集和测试集（预测金牌数）
X_train_gold, X_test_gold, y_train_gold, y_test_gold = train_test_split(
    X_gold, y_gold, test_size=0.2, random_state=42)

# 对特征进行标准化处理（预测金牌数）
scaler_gold = StandardScaler()
numerical_columns_gold = [col for col in X_train_gold.columns if col != 'Year' and not col.startswith('NOC_')]
X_train_gold[numerical_columns_gold] = scaler_gold.fit_transform(X_train_gold[numerical_columns_gold])
X_test_gold[numerical_columns_gold] = scaler_gold.transform(X_test_gold[numerical_columns_gold])

# 创建并训练预测金牌数的模型
model_gold = LinearRegression()
model_gold.fit(X_train_gold, y_train_gold)

# 在测试集上进行预测（金牌数）
y_pred_gold = model_gold.predict(X_test_gold)

# 评估预测金牌数的模型
r2_gold = r2_score(y_test_gold, y_pred_gold)
mse_gold = mean_squared_error(y_test_gold, y_pred_gold)

print(f'预测金牌数模型的 R2 分数: {r2_gold}')
print(f'预测金牌数模型的均方误差: {mse_gold}')

# 获取最后两届奥运会的年份
last_two_years = sorted(data['Year'].unique())[-2:]

# 筛选出最后两届奥运会的数据
last_two_years_data = data[data['Year'].isin(last_two_years)]

# 按 NOC 分组计算除年份和 NOC 外其他特征的均值
grouped_mean = last_two_years_data.groupby('NOC')[['Is_Host', 'Total_events_4_Years', 'Total_events_8_Years', 'Total_events_12_Years',
                                                   'Total_disciplines_4_Years', 'Total_disciplines_8_Years', 'Total_disciplines_12_Years',
                                                   'Total_sports_4_Years', 'Total_sports_8_Years', 'Total_sports_12_Years',
                                                   'Participants_4_Years', 'Participants_8_Years', 'Participants_12_Years',
                                                   'Gold_4_Years', 'Gold_8_Years', 'Gold_12_Years',
                                                   'Total_medals_4_Years', 'Total_medals_8_Years', 'Total_medals_12_Years']].mean()

# 准备 2028 年的预测数据，只保留前两届有数据的国家
noc_list = [noc for noc in data['NOC'].unique() if noc in grouped_mean.index]
X_2028 = []
for noc in noc_list:
    feature_row = [2028, noc] + grouped_mean.loc[noc].tolist()
    X_2028.append(feature_row)

# 正确设置列名
columns = ['Year', 'NOC'] + grouped_mean.columns.tolist()
X_2028 = pd.DataFrame(X_2028, columns=columns)

X_2028_total_medals = pd.get_dummies(X_2028, columns=['NOC'])
X_2028_gold = pd.get_dummies(X_2028, columns=['NOC'])

# 确保 2028 年数据的列与训练数据的列一致（奖牌总数）
missing_cols_total_medals = set(X_train_total_medals.columns) - set(X_2028_total_medals.columns)
for col in missing_cols_total_medals:
    X_2028_total_medals[col] = 0
X_2028_total_medals = X_2028_total_medals[X_train_total_medals.columns]

# 对 2028 年预测数据进行标准化处理（奖牌总数）
X_2028_total_medals[numerical_columns_total_medals] = scaler_total_medals.transform(X_2028_total_medals[numerical_columns_total_medals])

# 确保 2028 年数据的列与训练数据的列一致（金牌数）
missing_cols_gold = set(X_train_gold.columns) - set(X_2028_gold.columns)
for col in missing_cols_gold:
    X_2028_gold[col] = 0
X_2028_gold = X_2028_gold[X_train_gold.columns]

# 对 2028 年预测数据进行标准化处理（金牌数）
X_2028_gold[numerical_columns_gold] = scaler_gold.transform(X_2028_gold[numerical_columns_gold])

# 进行预测（奖牌总数）
predictions_total_medals_2028 = model_total_medals.predict(X_2028_total_medals)
# 四舍五入为整数并将小于 0 的值替换为 0
predictions_total_medals_2028 = np.round(predictions_total_medals_2028).astype(int)
predictions_total_medals_2028 = np.where(predictions_total_medals_2028 < 0, 0, predictions_total_medals_2028)

# 进行预测（金牌数）
predictions_gold_2028 = model_gold.predict(X_2028_gold)
# 四舍五入为整数并将小于 0 的值替换为 0
predictions_gold_2028 = np.round(predictions_gold_2028).astype(int)
predictions_gold_2028 = np.where(predictions_gold_2028 < 0, 0, predictions_gold_2028)

# 构建预测区间（95% 置信区间）
def calculate_prediction_intervals(model, X, y, alpha=0.05):
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_error = np.std(residuals)
    lower_bound = y_pred - 1.96 * std_error
    upper_bound = y_pred + 1.96 * std_error
    return lower_bound, upper_bound

lower_bound_total_medals, upper_bound_total_medals = calculate_prediction_intervals(model_total_medals, X_2028_total_medals, predictions_total_medals_2028)
lower_bound_gold, upper_bound_gold = calculate_prediction_intervals(model_gold, X_2028_gold, predictions_gold_2028)

# 确保预测区间的值为正数并保留三位小数
lower_bound_total_medals = np.round(np.where(lower_bound_total_medals < 0, 0, lower_bound_total_medals), 3)
upper_bound_total_medals = np.round(np.where(upper_bound_total_medals < 0, 0, upper_bound_total_medals), 3)

lower_bound_gold = np.round(np.where(lower_bound_gold < 0, 0, lower_bound_gold), 3)
upper_bound_gold = np.round(np.where(upper_bound_gold < 0, 0, upper_bound_gold), 3)

# 创建 DataFrame
result_df = pd.DataFrame({
    'NOC': noc_list,
    'Predicted_Total_Medals_2028': predictions_total_medals_2028,
    'Lower_Bound_Total_Medals_2028': lower_bound_total_medals,
    'Upper_Bound_Total_Medals_2028': upper_bound_total_medals,
    'Predicted_Gold_2028': predictions_gold_2028,
    'Lower_Bound_Gold_2028': lower_bound_gold,
    'Upper_Bound_Gold_2028': upper_bound_gold
})

# 保存预测结果到 CSV 文件
result_df.to_csv('2028_medal_predictions_with_intervals.csv', index=False)

print("预测完成并保存到文件 '2028_medal_predictions_with_intervals.csv'")
