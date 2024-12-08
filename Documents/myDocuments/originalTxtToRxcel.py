import pandas as pd
import pandas as pd
import pickle

# 读取 .result 文件
with open('/mnt/f/Code/VulCNN/VulCNN/data/sard/results/FirstRunningResult/4_epo200_bat32.result', 'rb') as file:
     data = pickle.load(file)
    
# 使用 eval() 将字符串解析为 Python 字典
# data = eval(content)

# 创建一个存储所有结果的列表
all_epochs_data = []

for epoch, values in data.items():
    # 提取train和val的loss和score
    train_loss = values['train_loss']
    val_loss = values['val_loss']
    
    # 提取训练和验证的分数
    train_score = values['train_score']
    val_score = values['val_score']
    
    # 将每个 epoch 的数据整合为一个字典
    epoch_data = {
        'Epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_M_fpr': train_score['M_fpr'],
        'train_M_fnr': train_score['M_fnr'],
        'train_M_f1': train_score['M_f1'],
        'train_W_fpr': train_score['W_fpr'],
        'train_W_fnr': train_score['W_fnr'],
        'train_W_f1': train_score['W_f1'],
        'train_ACC': train_score['ACC'],
        'val_M_fpr': val_score['M_fpr'],
        'val_M_fnr': val_score['M_fnr'],
        'val_M_f1': val_score['M_f1'],
        'val_W_fpr': val_score['W_fpr'],
        'val_W_fnr': val_score['W_fnr'],
        'val_W_f1': val_score['W_f1'],
        'val_ACC': val_score['ACC'],
    }
    
    # 忽略 MCM（混淆矩阵），如果需要可以额外处理
    all_epochs_data.append(epoch_data)

df = pd.DataFrame(all_epochs_data)
# 计算 val 开头列的平均值
val_columns = [col for col in df.columns if col.startswith('val')]
val_means = df[val_columns].astype(float).mean()

# 将结果写入 Excel 文件
with pd.ExcelWriter('/mnt/f/Code/VulCNN/VulCNN/data/sard/results/FirstRunningResult/4_epo200_bat32.xlsx') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)
    
    # 创建一个新的 DataFrame 来保存平均值
    mean_df = pd.DataFrame(val_means, columns=['Mean'])
    mean_df.index.name = 'Metric'
    mean_df.to_excel(writer, sheet_name='Average Values')

print("数据已成功保存为 Excel 文件，并计算了 'val' 开头列的平均值")
