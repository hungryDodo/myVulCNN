import argparse
from model_1 import load_data, CNN_Classifier

# 解析命令行参数
def parse_options():
    parser = argparse.ArgumentParser(description='VulCNN training.')
    parser.add_argument('-i', '--input', help='The dir path of train.pkl and test.pkl', type=str, required=True)
    args = parser.parse_args()
    return args

# 获取K折交叉验证的DataFrame
def get_kfold_dataframe(pathname = "./data/", item_num = 0):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")[item_num]  # 加载训练数据
    eval_df = load_data(pathname + "test.pkl")[item_num]    # 加载测试/验证数据
    # print(train_df['data'].to_string())
    # test_df = eval_df.copy(deep=True)
    print("return data")
    return train_df, eval_df    # 返回训练和测试/验证DataFrame

def main():
    args = parse_options()
    item_num = 0
    hidden_size = 100
    # data_path = "/root/data/qm_data/vulcnn/data/pkl/sard"
    # data_path = "/root/data/qm_data/vulcnn/data/pkl/ffmped"
    # data_path = "/root/data/qm_data/vulcnn/data/pkl/qemu"
    # data_path = "/root/data/qm_data/vulcnn/data/pkl/sard-1"
    # data_path = "/root/data/qm_data/vulcnn/data/pkl/sard-2"
    data_path = args.input
    for item_num in range(5):
        train_df, eval_df = get_kfold_dataframe(pathname = data_path, item_num = item_num)
        classifier = CNN_Classifier(result_save_path = data_path.replace("pkl", "results"), \
            item_num = item_num, epochs=300, hidden_size = hidden_size)
        classifier.preparation(
            X_train=train_df['data'],
            y_train=train_df['label'],
            X_valid=eval_df['data'],
            y_valid=eval_df['label']
        )
        classifier.train()


if __name__ == "__main__":
    main()