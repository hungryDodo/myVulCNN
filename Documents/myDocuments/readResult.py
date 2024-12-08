
import pickle

def load_data(filename):
    print("Begin to load data from:", filename)  # 打印加载数据的信息。
    with open(filename, 'rb') as f:  # 以二进制读取模式打开文件。
        data = pickle.load(f)  # 使用 pickle 库将数据从字节流反序列化。
    return data

# 示例用法
filename = '/mnt/f/Code/VulCNN/VulCNN/data/myTest/results/4_1_1/outputs/No-Vul/raw_000062516_goodB2G.pkl'
data = load_data(filename)

# 查看数据
print("Loaded data:", data)