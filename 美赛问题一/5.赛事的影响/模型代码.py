import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 后端为 tkagg
matplotlib.use('tkagg')

# 加载 summerOly_athletes.csv 数据
df_athletes = pd.read_csv(r"E:\1\美赛\2025_Problem_C_Data\summerOly_athletes.csv")

# 加载 summerOly_programs.csv 数据
df_programs = pd.read_csv(r"E:\1\美赛\2025_Problem_C_Data\summerOly_programs.csv", encoding='latin - 1')

# 将 df_athletes 中的 Medal 列转换为数值，获得奖牌为 1，未获得为 0
df_athletes['Medal_Num'] = df_athletes['Medal'].apply(lambda x: 1 if x != 'No medal' else 0)

# 统计每个国家在每个体育项目上的获奖总数
country_sport_medals = df_athletes.groupby(['NOC', 'Sport'])['Medal_Num'].sum().reset_index()

# 合并 df_programs 和 country_sport_medals 数据，以 Sport 列为连接键
merged_data = pd.merge(df_programs, country_sport_medals, left_on='Sport', right_on='Sport', how='outer')

# 不同国家的重要体育项目分析
countries = merged_data['NOC'].dropna().unique()
important_events = {}

for country in countries:
    country_data = merged_data[merged_data['NOC'] == country]
    if not country_data.empty:
        max_medal_index = country_data['Medal_Num'].idxmax()
        important_event = country_data.loc[max_medal_index, 'Sport']
        important_events[country] = important_event

# 将结果转换为 DataFrame
result_df = pd.DataFrame(list(important_events.items()), columns=['Country', 'Most_Important_Sport'])

# 将结果保存为 CSV 文件
csv_path = 'summerOly_analysis_result.csv'
result_df.to_csv(csv_path, index=False)

# 可视化部分
plt.figure(figsize=(12, 8))
# 统计每个重要体育项目的国家数量
sport_counts = result_df['Most_Important_Sport'].value_counts()
bars = plt.bar(sport_counts.index, sport_counts.values)
plt.title('Number of Countries for Which Each Sport is Most Important')
plt.xlabel('Sport')
plt.ylabel('Number of Countries')
plt.xticks(rotation=90)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

plt.tight_layout()
plt.show()