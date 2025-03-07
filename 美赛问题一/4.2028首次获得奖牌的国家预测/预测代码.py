import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置 matplotlib 后端为 tkagg
import matplotlib
matplotlib.use('tkagg')


# 加载数据
data = pd.read_csv(r"E:\1\2028首次获得奖牌的国家预测\never_won_medal_nocs_stats_new.csv")

# 计算 2016 - 2020 年参赛人数增长率
data['2016 - 2020参赛人数增长率(%)'] = ((data['参赛人数_2020'] - data['参赛人数_2016']) / data['参赛人数_2016'] * 100).fillna(0)
# 计算 2020 - 2024 年参赛人数增长率
data['2020 - 2024参赛人数增长率(%)'] = ((data['参赛人数_2024'] - data['参赛人数_2020']) / data['参赛人数_2020'] * 100).fillna(0)

# 由于数据中没有实际的获奖标记，我们这里进行模拟。
# 假设参赛人数增长率较高且连续参赛届数较多的国家有一定概率获奖，这里简单设定规则来模拟标记
# 参赛人数增长率大于 10% 且连续参赛届数大于 2 届的标记为 1（可能获奖），否则为 0
data['是否可能获奖'] = 0
data.loc[(data['2020 - 2024参赛人数增长率(%)'] > 10) & (data['最近连续参加届数'] > 2), '是否可能获奖'] = 1

# 准备特征和目标变量
features = ['参赛人数_2016', '参赛人数_2020', '参赛人数_2024', '2016 - 2020参赛人数增长率(%)', '2020 - 2024参赛人数增长率(%)', '最近连续参加届数']
X = data[features]
y = data['是否可能获奖']

# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用 GridSearchCV 进行参数调优
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
logreg = LogisticRegression(solver='liblinear', max_iter=1000)
grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_logreg = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_logreg.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy * 100:.2f}%')

# 预测所有国家在下一届奥运会获奖可能性
all_predictions = best_logreg.predict(X_scaled)

# 统计可能获得首枚奖牌的国家数量
potential_medal_countries = sum(all_predictions)

# 计算胜算（这里简单以可能获奖国家数量占总国家数量的比例作为胜算）
odds = potential_medal_countries / len(data)

print(f'预计在下一届奥运会中获得首枚奖牌的国家数量: {potential_medal_countries}')
print(f'预测胜算: {odds * 100:.2f}%')

# 可视化部分
labels = ['Potential to Win First Medal', 'Unlikely to Win First Medal']
sizes = [potential_medal_countries, len(data) - potential_medal_countries]
colors = ['lightgreen', 'salmon']
explode = (0.1, 0)  # 突出显示可能获奖的部分

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Prediction of Countries Winning First Olympic Medal in Next Olympics')
plt.show()