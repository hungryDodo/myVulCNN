import argparse  # 导入 argparse 模块，用于解析命令行参数
from model import load_data, CNN_Classifier  # 从 model 模块导入 load_data 函数和 CNN_Classifier 类

# 定义一个函数，用于解析命令行选项和参数
def parse_options():
    # 创建一个 ArgumentParser 对象，用于处理命令行输入
    parser = argparse.ArgumentParser(description='VulCNN training.')
    # 添加参数 '-i' 或 '--input'，表示训练和测试数据文件所在的路径，必须是字符串类型且为必填参数
    parser.add_argument('-i', '--input', help='The dir path of train.pkl and test.pkl', type=str, required=True)
    # 解析命令行参数，并将结果存储到 args 变量中
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args

# 定义函数，用于获取 K 折交叉验证的数据集
def get_kfold_dataframe(pathname="./data/", item_num=0):
    # 确保路径以斜杠结尾，如果没有则加上
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    # 加载训练集的第 item_num 个数据，并将其存储到 train_df
    train_df = load_data(pathname + "train.pkl")[item_num]
    # 加载测试集的第 item_num 个数据，并将其存储到 eval_df
    eval_df = load_data(pathname + "test.pkl")[item_num]
    # 返回训练集和测试集的数据框
    return train_df, eval_df

# 主函数，管理整个训练流程
def main():
    # 解析命令行参数，获取输入路径
    args = parse_options()
    # 初始 item_num 设为 0，用于指定当前进行第几折验证
    item_num = 0
    # 设置隐藏层大小为 128
    hidden_size = 128
    # 设置数据路径为命令行参数提供的输入路径
    data_path = args.input
    # 循环执行 5 次 K 折交叉验证，每次处理不同的 item_num
    for item_num in range(5):
        # 加载当前 item_num 的训练集和测试集数据
        train_df, eval_df = get_kfold_dataframe(pathname=data_path, item_num=item_num)
        # 创建 CNN_Classifier 实例，设置结果保存路径、当前 item_num、训练 epoch 次数和隐藏层大小
        classifier = CNN_Classifier(result_save_path=data_path.replace("pkl", "results"), \
            item_num=item_num, epochs=200, hidden_size=hidden_size)
        # 准备训练和验证数据，将数据传递给模型进行预处理
        classifier.preparation(
            X_train=train_df['data'],  # 训练集的特征数据
            y_train=train_df['label'],  # 训练集的标签
            X_valid=eval_df['data'],  # 验证集的特征数据
            y_valid=eval_df['label'],  # 验证集的标签
        )
        # 开始训练模型
        classifier.train()

# 如果此脚本作为主程序运行，调用 main() 函数
if __name__ == "__main__":
    main()

