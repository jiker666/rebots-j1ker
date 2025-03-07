import matplotlib
# 设置 matplotlib 后端为 tkagg
matplotlib.use('tkagg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载运动员数据
df_athletes = pd.read_csv(r"E:\1\美赛\2025_Problem_C_Data\summerOly_athletes.csv")

# 加载赛事数据
df_programs = pd.read_csv(r"E:\1\美赛\2025_Problem_C_Data\summerOly_programs.csv", encoding='latin - 1')

# 主办国信息
host_countries = {
    1896: 'GRE', 1900: 'FRA', 1904: 'USA', 1908: 'GBR',
    1912: 'SWE', 1920: 'BEL', 1924: 'FRA', 1928: 'NED',
    1932: 'USA', 1936: 'GER', 1948: 'GBR', 1952: 'FIN',
    1956: 'AUS', 1960: 'ITA', 1964: 'JPN', 1968: 'MEX', 1972: 'FRG',
    1976: 'CAN', 1980: 'URS', 1984: 'USA', 1988: 'KOR',
    1992: 'ESP', 1996: 'USA', 2000: 'AUS', 2004: 'GRE', 2008: 'CHN',
    2012: 'GBR', 2016: 'BRA', 2020: 'JPN', 2024: 'FRA', 2028: 'USA',
    2032: 'AUS'
}

# 添加主办国标识列
df_athletes['Is_Host'] = df_athletes['Year'].apply(lambda x: host_countries.get(x) if x in host_countries else None)
df_athletes['Is_Host'] = df_athletes.apply(lambda row: row['NOC'] == row['Is_Host'] if pd.notna(row['Is_Host']) else False, axis=1)

# 将 Medal 列转换为数值，获得奖牌为 1，未获得为 0
df_athletes['Medal_Num'] = df_athletes['Medal'].apply(lambda x: 1 if x != 'No medal' else 0)

# 统计主办国和非主办国在每个体育项目上的获奖总数
host_sport_medals = df_athletes[df_athletes['Is_Host']].groupby(['Sport'])['Medal_Num'].sum().reset_index(name='Host_Medals')
non_host_sport_medals = df_athletes[~df_athletes['Is_Host']].groupby(['Sport'])['Medal_Num'].sum().reset_index(name='Non_Host_Medals')

# 合并主办国和非主办国的奖牌数据
merged_medals = pd.merge(host_sport_medals, non_host_sport_medals, on='Sport', how='outer').fillna(0)

# 计算主办国奖牌占比
merged_medals['Host_Medal_Ratio'] = merged_medals['Host_Medals'] / (merged_medals['Host_Medals'] + merged_medals['Non_Host_Medals'])

# 筛选出奖牌数较多的前 10 个体育项目进行重点分析
top_sports = merged_medals.sort_values(by='Host_Medals', ascending=False).head(10)['Sport'].tolist()
top_data = merged_medals[merged_medals['Sport'].isin(top_sports)]

# 可视化：主办国和非主办国在热门赛事上的奖牌数对比
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(top_data))
plt.bar(index, top_data['Host_Medals'], bar_width, label='Host Country Medals')
plt.bar([i + bar_width for i in index], top_data['Non_Host_Medals'], bar_width, label='Non - Host Country Medals')
plt.xlabel('Sport')
plt.ylabel('Number of Medals')
plt.title('Medals Comparison in Top Sports between Host and Non - Host Countries')
plt.xticks([i + bar_width/2 for i in index], top_data['Sport'], rotation=45)
plt.legend()
plt.show()

# 分析主办国选择赛事的影响（以主办国奖牌占比为指标）
high_impact_sports = merged_medals[merged_medals['Host_Medal_Ratio'] > 0.5]['Sport'].tolist()
low_impact_sports = merged_medals[merged_medals['Host_Medal_Ratio'] < 0.2]['Sport'].tolist()

print("Host countries have a significant impact on these sports: ", high_impact_sports)
print("Host countries have a minor impact on these sports: ", low_impact_sports)

# 将结果保存到 CSV 文件
result_df = pd.DataFrame({
    'High_Impact_Sports': high_impact_sports,
    'Low_Impact_Sports': low_impact_sports[:len(high_impact_sports)] if len(low_impact_sports) > len(high_impact_sports) else low_impact_sports + [None] * (len(high_impact_sports) - len(low_impact_sports))
})
csv_path = 'host_country_sports_impact.csv'
result_df.to_csv(csv_path, index=False)

# 可视化：主办国奖牌占比分布15.py
plt.figure(figsize=(10, 6))
sns.histplot(merged_medals['Host_Medal_Ratio'], bins=20, kde=False)
plt.xlabel('Host Country Medal Ratio')
plt.ylabel('Number of Sports')
plt.title('Distribution of Host Country Medal Ratio in All Sports')
plt.show()