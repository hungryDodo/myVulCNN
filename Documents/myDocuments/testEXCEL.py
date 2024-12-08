    
from prettytable import PrettyTable
import os
import pandas as pd
#TAG 每轮的训练表写入Excel文件中
def save_table_excel(file_path, table):
     # 将 PrettyTable 转换为 DataFrame
    data = [row for row in table.rows]
    headers = table.field_names
    df = pd.DataFrame(data, columns=headers)
    
    # 计算每列的平均值，跳过非数值列
    numeric_columns = df.columns[3:]  # 前三列是非数值列
    mean_values = df[numeric_columns].astype(float).mean()
    
    # 将平均值作为新行添加到 DataFrame 中
    mean_row = pd.Series(['Average', '', ''] + mean_values.tolist(), index=df.columns)
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # 保存到 Excel 文件
    with pd.ExcelWriter(file_path, mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

    print(f"Table data saved to {file_path}")
    


# 假设 save_table_excel 函数已经导入

# 创建一个模拟的 PrettyTable 对象
def create_test_table():
    table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
    
    # 添加一些测试数据
    table.add_row(['tra', '1', '0.4567', '0.123', '0.456', '0.789', '0.234', '0.567', '0.891', '0.345'])
    table.add_row(['tra', '2', '0.5678', '0.223', '0.556', '0.889', '0.334', '0.667', '0.991', '0.445'])
    table.add_row(['tra', '3', '0.6789', '0.323', '0.656', '0.989', '0.434', '0.767', '0.891', '0.545'])
    
    return table

def test_save_table_excel():
    # 创建测试的 PrettyTable
    test_table = create_test_table()
    
    # 定义保存路径
    file_path = 'test_train_results.xlsx'
    
    # 调用 save_table_excel 函数
    save_table_excel(file_path, test_table)
    
    # 检查文件是否已保存
    if os.path.exists(file_path):
        print(f"Test passed: File '{file_path}' has been saved successfully.")
    else:
        print(f"Test failed: File '{file_path}' was not saved.")
    
    # 打开并检查文件内容
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, sheet_name='Results')
        print("Saved Excel content:")
        print(df)
    
    # 删除测试文件
    # os.remove(file_path)

# 执行测试程序
test_save_table_excel()
