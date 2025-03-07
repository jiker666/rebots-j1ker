import pandas as pd

# 读取CSV文件
file_path = r"E:\1\美赛\2025_Problem_C_Data\summerOly_athletes.csv"
df = pd.read_csv(file_path)

# 检查数据结构，确保数据正确读取
print(df.head())

# 将奖牌列转换为数字：Gold = 1, Silver = 2, Bronze = 3, No medal = 0
df['Gold'] = df['Medal'].apply(lambda x: 1 if x == 'Gold' else 0)
df['Silver'] = df['Medal'].apply(lambda x: 1 if x == 'Silver' else 0)
df['Bronze'] = df['Medal'].apply(lambda x: 1 if x == 'Bronze' else 0)

# 主办国家的字典 (NOC代码)
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

def is_host_country(year, team, host_countries):
    if year in host_countries:
        return 1 if team == host_countries[year] else 0
    return 0

# 创建“是否为主办国”的列
df['Is_Host'] = df.apply(lambda row: is_host_country(row['Year'], row['NOC'], host_countries), axis=1)

# 按年份和NOC列（国家代码）汇总每届奥运会的金牌数、银牌数、铜牌数和参赛人数
df_grouped = df.groupby(['Year', 'NOC']).agg({
    'Gold': 'sum',  # 统计每个国家的金牌数
    'Silver': 'sum',  # 统计每个国家的银牌数
    'Bronze': 'sum',  # 统计每个国家的铜牌数
    'Name': 'count',  # 统计每个国家参赛人数
    'Is_Host': 'max'  # 标记主办方
}).reset_index()

# 计算总奖牌数
df_grouped['Total_medals'] = df_grouped['Gold'] + df_grouped['Silver'] + df_grouped['Bronze']

# 重命名列以便理解
df_grouped.rename(columns={'Name': 'Participants'}, inplace=True)

# 添加额外数据（各届奥运会的事件数、学科数、体育数）
additional_data = {
    'Year': [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'Total_events': [43, 97, 95, 110, 102, 156, 126, 109, 117, 129, 136, 149, 151, 150, 163, 172, 195, 198, 203, 221, 237, 257, 271, 300, 301, 302, 302, 306, 339, 329],
    'Total_disciplines': [10, 22, 18, 25, 18, 29, 23, 20, 20, 25, 23, 23, 23, 23, 25, 24, 28, 27, 27, 29, 31, 34, 37, 40, 40, 42, 40, 42, 50, 48],
    'Total_sports': [11, 20, 16, 22, 14, 22, 17, 14, 14, 19, 17, 17, 17, 17, 19, 18, 21, 21, 21, 21, 23, 25, 25, 27, 27, 27, 26, 28, 33, 32]
}

additional_df = pd.DataFrame(additional_data)

# 将额外的数据与现有的 df_grouped 数据框按年份进行合并
df_grouped = pd.merge(df_grouped, additional_df, on='Year', how='left')

# 计算4年前、8年前、12年前的各项数据
for col in ['Total_events', 'Total_disciplines', 'Total_sports']:
    # 使用shift()来计算各列在4、8、12年前的数据
    df_grouped[f'{col}_4_Years'] = df_grouped.groupby('NOC')[col].shift(4).fillna(0)
    df_grouped[f'{col}_8_Years'] = df_grouped.groupby('NOC')[col].shift(8).fillna(0)
    df_grouped[f'{col}_12_Years'] = df_grouped.groupby('NOC')[col].shift(12).fillna(0)

# 计算4年前、8年前、12年前的参赛人数、金牌和总奖牌数
for col in ['Participants', 'Gold', 'Total_medals']:
    df_grouped[f'{col}_4_Years'] = df_grouped.groupby('NOC')[col].shift(4).fillna(0)
    df_grouped[f'{col}_8_Years'] = df_grouped.groupby('NOC')[col].shift(8).fillna(0)
    df_grouped[f'{col}_12_Years'] = df_grouped.groupby('NOC')[col].shift(12).fillna(0)

# 排序
df_grouped = df_grouped.sort_values(by=['NOC', 'Year'])

# 将更新后的数据保存到新的文件
output_file_path_updated = r"E:\1\美赛\2025_Problem_C_Data\processed_athletes_data_with_noc_and_host.csv"
df_grouped.to_csv(output_file_path_updated, index=False)

print(f"处理后的文件已保存为：{output_file_path_updated}")
