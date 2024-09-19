import pandas as pd

# 读取txt文件
file_path = '/mnt/f/Code/VulCNN/VulCNN/data/myTest/results/3_1_1/4_epo200_bat32.txttest_table.txt'  # 替换为你的文件路径
with open(file_path, 'r') as f:
    lines = f.readlines()

# 初始化空列表来保存数据
data = []
columns = ['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC']

# 解析txt文件内容
for line in lines:
    # 忽略包含符号的行
    if not line.startswith("+") and not line.startswith("| typ"):
        # 去掉行首尾的空格，并按列分割
        line_data = line.strip().split("|")[1:-1]
        # 去掉每列数据两侧的空格
        line_data = [item.strip() for item in line_data]
        data.append(line_data)

# 创建DataFrame
df = pd.DataFrame(data, columns=columns)

# 将数据类型转换为合适的格式（数值列转换为float）
df['epo'] = pd.to_numeric(df['epo'])
df['loss'] = pd.to_numeric(df['loss'])
df['M_fpr'] = pd.to_numeric(df['M_fpr'])
df['M_fnr'] = pd.to_numeric(df['M_fnr'])
df['M_f1'] = pd.to_numeric(df['M_f1'])
df['W_fpr'] = pd.to_numeric(df['W_fpr'])
df['W_fnr'] = pd.to_numeric(df['W_fnr'])
df['W_f1'] = pd.to_numeric(df['W_f1'])
df['ACC'] = pd.to_numeric(df['ACC'])

# 计算每列的平均值
# mean_values = df.mean()
mean_values = df.select_dtypes(include=['float64', 'int64']).mean()


# 将DataFrame写入Excel
output_file = '/mnt/f/Code/VulCNN/VulCNN/data/myTest/results/3_1_1/4_epo200_bat32.txttest_table.xlsx'  # 替换为你要保存的文件名
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name='data', index=False)
    
    # 将平均值添加到新的一行
    mean_df = pd.DataFrame(mean_values).T
    mean_df.index = ['mean']  # 添加一个索引来标记平均值
    mean_df.to_excel(writer, sheet_name='data', startrow=len(df) + 1, index=True)
    
print(f"数据成功写入 {output_file}，并且计算了每一列的平均值。")
