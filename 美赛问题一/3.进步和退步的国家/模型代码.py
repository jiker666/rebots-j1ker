import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 后端为 tkagg
matplotlib.use('tkagg')

# 加载2024年奥运奖牌榜数据
medal_table_2024 = pd.read_csv(r"C:\Users\Jiker\Downloads\2024_olympic_medal_table_again.csv")

# 加载2028年预测奖牌榜单数据
medal_table_2028 = pd.read_csv(r"E:\1\美赛问题一\2028奖牌榜预测\预测出来的2028奖牌榜单.csv")

# 定义一个函数进行模糊匹配并返回匹配度最高的NOC
def fuzzy_match_noc(noc, noc_list):
    match, score = process.extractOne(noc, noc_list)
    if score >= 80:  # 设定匹配度阈值，这里设为80，可以根据实际情况调整
        return match
    return noc

# 对2028年数据中的NOC进行模糊匹配，使其与2024年数据中的NOC尽可能一致
medal_table_2028['NOC'] = medal_table_2028['NOC'].apply(lambda x: fuzzy_match_noc(x, medal_table_2024['NOC'].tolist()))

# 合并2024年和2028年的数据，以NOC为连接键
merged_data = pd.merge(medal_table_2024, medal_table_2028, on='NOC', how='outer')

# 计算金牌数和奖牌总数的变化幅度
merged_data['Gold_change_rate'] = (merged_data['Predicted_Gold_2028'] - merged_data['Gold']) / (merged_data['Gold'] + 1e-8)
merged_data['Total_change_rate'] = (merged_data['Predicted_Total_Medals_2028'] - merged_data['Total']) / (merged_data['Total'] + 1e-8)

# 筛选出可能取得进步的国家
progress_countries = merged_data[(merged_data['Predicted_Gold_2028'] > merged_data['Gold']) |
                                 (merged_data['Predicted_Total_Medals_2028'] > merged_data['Total'])]
progress_countries = progress_countries.copy()
progress_countries['max_change_rate'] = progress_countries[['Gold_change_rate', 'Total_change_rate']].max(axis=1)
# 按最大变化率降序排序并取前5个
top_progress_countries = progress_countries.sort_values(by='max_change_rate', ascending=False).head(5)

# 筛选出表现可能不如2024年的国家
worse_countries = merged_data[(merged_data['Predicted_Gold_2028'] < merged_data['Gold']) |
                              (merged_data['Predicted_Total_Medals_2028'] < merged_data['Total'])]
worse_countries = worse_countries.copy()
worse_countries['min_change_rate'] = worse_countries[['Gold_change_rate', 'Total_change_rate']].min(axis=1)
# 按最小变化率升序排序并取前5个
top_worse_countries = worse_countries.sort_values(by='min_change_rate', ascending=True).head(5)

print('最显著可能取得进步的5个国家：', top_progress_countries['NOC'].tolist())
print('最显著表现可能不如2024年的5个国家：', top_worse_countries['NOC'].tolist())

# 可视化部分
# 创建一个包含两个子图的画布
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 绘制最显著可能取得进步的5个国家的柱状图
axes[0].bar(top_progress_countries['NOC'], top_progress_countries['max_change_rate'])
axes[0].set_title('Top 5 Countries with Most Significant Progress in 2028 Compared to 2024')
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Maximum Change Rate')

# 绘制最显著表现可能不如2024年的5个国家的柱状图
axes[1].bar(top_worse_countries['NOC'], top_worse_countries['min_change_rate'])
axes[1].set_title('Top 5 Countries with Most Significant Decline in 2028 Compared to 2024')
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Minimum Change Rate')

# 调整子图布局
plt.tight_layout()
# 显示图形
plt.show()