import pickle
import numpy as np

# 创建一个简单的 numpy 数组
# array = np.array([1, 2, 3, 4, 5])

# # 序列化并保存到 .pkl 文件
# with open('test.pkl', 'wb') as f:
#     pickle.dump(array, f)

# 反序列化并从 .pkl 文件加载
with open('../data/myTest/outputs/Vul/CVE_raw_000121007_CWE78_OS_Command_Injection__wchar_t_console_system_68_bad.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
