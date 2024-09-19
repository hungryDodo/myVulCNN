import os  # 导入操作系统模块，用于文件和目录的操作。
import lap  # 导入 LAP 模块，用于线性分配问题的求解。
import torch  # 导入 PyTorch 库，用于深度学习模型的构建和训练。
import numpy  # 导入 NumPy 库，用于科学计算。
import pickle  # 导入 Pickle 库，用于对象的序列化和反序列化。
import numpy as np  # 导入 NumPy 库并重命名为 np，用于数组操作。
import pandas as pd #TAG 引入pandas将结果写入Excel中
from torch import nn  # 从 PyTorch 导入神经网络模块。
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条。
import torch.nn.functional as F  # 导入 PyTorch 的功能模块，用于神经网络的各种功能操作。
from prettytable import PrettyTable  # 导入 PrettyTable 库，用于打印格式化表格。
from torch.cuda.amp import autocast  # 导入 autocast，用于自动混合精度训练。
from torch.utils.data import Dataset  # 导入 PyTorch 的 Dataset 类，用于创建自定义数据集。
from torch.utils.data import DataLoader  # 导入 PyTorch 的 DataLoader 类，用于加载数据。
from sklearn.metrics import confusion_matrix  # 导入 sklearn 的混淆矩阵函数，用于计算分类模型的性能。
from transformers import AdamW, get_linear_schedule_with_warmup  # 从 transformers 库导入优化器和学习率调度器。
from sklearn.metrics import precision_recall_fscore_support  # 导入 sklearn 的精度、召回率、F1 分数函数。
from sklearn.metrics import multilabel_confusion_matrix  # 导入 sklearn 的多标签混淆矩阵函数。
from openpyxl import load_workbook





#将数据保存到指定的文件中，使用 pickle 库进行序列化
# def save_data(filename, data):
#     print("Begin to save data：", filename)  # 打印保存数据的信息。
#     f = open(filename, 'wb')  # 以二进制写入模式打开文件。
#     pickle.dump(data, f)  # 使用 pickle 库将数据序列化并保存到文件。
#     f.close()  # 关闭文件。
    
# TAG 刘昕给的保存代码
# def save_data2(file_path, data):
#     with open(file_path, 'w') as f:  # 使用 'w' 参数来覆盖写入
#         for epoch, metrics in data.items():
#             f.write(f'Epoch {epoch + 1}:\n')
#             for key, value in metrics.items():
#                 if isinstance(value, dict):  # 如果值是字典，进一步遍历
#                     f.write(f'{key}:\n')
#                     for sub_key, sub_value in value.items():
#                         f.write(f'  {sub_key}: {sub_value}\n')
#                 else:
#                     f.write(f'{key}: {value}\n')
#             f.write('\n')  # 在每个周期后添加空行

#TAG 我的保存代码，能够在写.result文件的同时写一个对应.txt文件以便方便读取
def save_data(filename, data):
    print("Begin to save data:", filename)  # 打印保存数据的信息。
    
    # 写入 .result 文件
    with open(filename, 'wb') as f:  # 以二进制写入模式打开文件。
        pickle.dump(data, f)  # 使用 pickle 库将数据序列化并保存到文件。

    # 写入 .txt 文件
    txt_filename = filename.replace('.result', '.txt')
    with open(txt_filename, 'w', encoding='utf-8') as f:  # 以文本写入模式打开文件。
        f.write(str(data))  # 将数据转换为字符串并写入文件。

    print("Data saved to both .result and .txt files.")
    
#TAG 每轮的训练表写入txt文件中
def save_table_data(filename, data):
     print("\033[1;31m将本轮epoch表格写入文件中保存！\033[0m")
    #  txt_filename = filename.replace('.result', '.txt')
     with open(filename, 'w', encoding='utf-8') as f:  # 以文本写入模式打开文件。
        f.write(str(data))  # 将数据转换为字符串并写入文件。

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

#从指定的文件中加载数据，使用 pickle 库进行反序列化
def load_data(filename):
    print("Begin to load data：", filename)  # 打印加载数据的信息。
    f = open(filename, 'rb')  # 以二进制读取模式打开文件。
    data = pickle.load(f)  # 使用 pickle 库从文件中反序列化数据。
    f.close()  # 关闭文件。
    return data  # 返回加载的数据。

#计算分类模型的准确率，使用线性分配算法重新排列混淆矩阵以最大化准确率。
def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)  # 计算混淆矩阵。
    
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 注释掉的代码，用于可视化混淆矩阵。
    
    def linear_assignment(cost_matrix):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)  # 使用 LAP 求解线性分配问题。
        return np.array([[y[i], i] for i in x if i >= 0])  # 返回线性分配的索引。
    
    def _make_cost_m(cm):
        s = np.max(cm)  # 找到混淆矩阵中的最大值。
        return (- cm + s)  # 返回转换后的代价矩阵。
    
    indexes = linear_assignment(_make_cost_m(cm))  # 计算线性分配的索引。
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]  # 对索引进行排序并提取第二列。
    cm2 = cm[:, js]  # 重新排列混淆矩阵的列。
    accuracy = np.trace(cm2) / np.sum(cm2)  # 计算准确率。
    return accuracy  # 返回准确率。

#计算分类模型的多种性能指标，包括准确率、精度、召回率、F1 分数等。
# def get_MCM_score(labels, predictions):
#     accuracy = get_accuracy(labels, predictions)  # 计算准确率。
#     precision, recall, f_score, true_sum, MCM = precision_recall_fscore_support(labels, predictions, average='macro')  # 计算精度、召回率、F1 分数和真实标签总数。
#     tn = MCM[:, 0, 0]  # 计算真阴性。
#     fp = MCM[:, 0, 1]  # 计算假阳性。
#     fn = MCM[:, 1, 0]  # 计算假阴性。
#     tp = MCM[:, 1, 1]  # 计算真阳性。
#     fpr_array = fp / (fp + tn)  # 计算假阳性率。
#     fnr_array = fn / (tp + fn)  # 计算假阴性率。
#     f1_array = 2 * tp / (2 * tp + fp + fn)  # 计算 F1 分数。
#     sum_array = fn + tp  # 计算正样本总数。
#     M_fpr = fpr_array.mean()  # 计算平均假阳性率。
#     M_fnr = fnr_array.mean()  # 计算平均假阴性率。
#     M_f1 = f1_array.mean()  # 计算平均 F1 分数。
#     W_fpr = (fpr_array * sum_array).sum() / sum(sum_array)  # 计算加权假阳性率。
#     W_fnr = (fnr_array * sum_array).sum() / sum(sum_array)  # 计算加权假阴性率。
#     W_f1 = (f1_array * sum_array).sum() / sum(sum_array)  # 计算加权 F1 分数。
#     return {
#         "M_fpr": format(M_fpr * 100, '.3f'),  # 返回平均假阳性率。
#         "M_fnr": format(M_fnr * 100, '.3f'),  # 返回平均假阴性率。
#         "M_f1": format(M_f1 * 100, '.3f'),  # 返回平均 F1 分数。
#         "W_fpr": format(W_fpr * 100, '.3f'),  # 返回加权假阳性率。
#         "W_fnr": format(W_fnr * 100, '.3f'),  # 返回加权假阴性率。
#         "W_f1": format(W_f1 * 100, '.3f'),  # 返回加权 F1 分数。
#         "ACC": format(accuracy * 100, '.3f'),  # 返回准确率。
#         "MCM": MCM  # 返回多标签混淆矩阵。
#     }


#TAG 刘昕给的修改后的代码
def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(labels, predictions, average='macro')
    print(f"\033[1;31m{accuracy}\033[0m")
    print(precision)
    print(recall)
    print(f_score)
    print(true_sum)
    MCM = multilabel_confusion_matrix(labels, predictions)
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum(sum_array)
    W_fnr = (fnr_array * sum_array).sum() / sum(sum_array)
    W_f1 = (f1_array * sum_array).sum() / sum(sum_array)
    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1": format(M_f1 * 100, '.3f'),
        "W_fpr": format(W_fpr * 100, '.3f'),
        "W_fnr": format(W_fnr * 100, '.3f'),
        "W_f1": format(W_f1 * 100, '.3f'),
        "ACC": format(accuracy * 100, '.3f'),
        "MCM": MCM
    }


#自定义数据集类，用于创建和处理文本数据及其标签
class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts  # 初始化文本数据。
        self.targets = targets  # 初始化目标标签。
        self.max_len = max_len  # 初始化最大序列长度。
        self.hidden_size = hidden_size  # 初始化隐藏层大小。

    def __len__(self):
        return len(self.texts)  # 返回数据集的大小。

    def __getitem__(self, idx):
        feature = self.texts[idx]  # 获取指定索引的文本特征。
        target = self.targets[idx]  # 获取指定索引的目标标签。
        vectors = numpy.zeros(shape=(3, self.max_len, self.hidden_size))  # 初始化特征向量矩阵。
        for j in range(3):  # 遍历每个特征。
            for i in range(min(len(feature[0]), self.max_len)):  # 遍历每个序列。
                vectors[j][i] = feature[j][i]  # 将特征赋值给特征向量矩阵。
        return {
            'vector': vectors,  # 返回特征向量矩阵。
            'targets': torch.tensor(target, dtype=torch.long)  # 返回目标标签。
        }

#TAG 加了注释的代码
class TextCNN(nn.Module):  # 定义一个继承自 nn.Module 的文本卷积神经网络类
    def __init__(self, hidden_size):  # 初始化方法，接受隐藏层大小作为参数
        super(TextCNN, self).__init__()  # 调用父类的初始化方法
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # 定义不同大小的卷积核
        self.num_filters = 32  # 定义每个卷积核大小的输出通道数
        classifier_dropout = 0.1  # 定义 dropout 的概率
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes]
        )  # 初始化多个卷积层，每个卷积层有不同的卷积核大小
        self.dropout = nn.Dropout(classifier_dropout)  # 定义 dropout 层，用于正则化
        num_classes = 2  # 定义分类任务的类别数
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)  # 初始化全连接层

    def conv_and_pool(self, x, conv):  # 定义卷积和池化的辅助函数
        x = F.relu(conv(x)).squeeze(3)  # 对输入进行卷积和 ReLU 激活，并去掉多余维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 进行一维最大池化，并去掉多余维度
        return x  # 返回池化后的特征

    def forward(self, x):  # 定义前向传播过程
        out = x.float()  # 确保输入是浮点数
        hidden_state = torch.cat(
            [self.conv_and_pool(out, conv) for conv in self.convs], 1
        )  # 对每个卷积层应用卷积和池化，将结果拼接在一起
        out = self.dropout(hidden_state)  # 应用 dropout
        out = self.fc(out)  # 通过全连接层得到最终输出
        return out, hidden_state  # 返回输出和中间特征

class CNN_Classifier():  # 定义 CNN 分类器类
    def __init__(self, max_len=100, n_classes=2, epochs=100, batch_size=32, learning_rate=0.001, \
                 result_save_path="/root/data/qm_data/vulcnn/data/results", item_num=0, hidden_size=128):
        self.model = TextCNN(hidden_size)  # 初始化 TextCNN 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为 GPU 或 CPU
        self.max_len = max_len  # 设置最大序列长度
        self.epochs = epochs  # 设置训练的轮数
        self.batch_size = batch_size  # 设置批次大小
        self.learning_rate = learning_rate  # 设置学习率
        self.model.to(self.device)  # 将模型移至设备
        self.hidden_size = hidden_size  # 设置隐藏层大小
        result_save_path = result_save_path + "/" if result_save_path[-1] != "/" else result_save_path  # 确保保存路径以斜杠结尾
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)  # 如果路径不存在则创建
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(batch_size) + ".result"  # 构建结果保存路径

    def preparation(self, X_train, y_train, X_valid, y_valid):  # 准备数据的方法
        # 创建数据集
        self.train_set = TraditionalDataset(X_train, y_train, self.max_len, self.hidden_size)  # 创建训练数据集
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.hidden_size)  # 创建验证数据集

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)  # 创建训练数据加载器
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)  # 创建验证数据加载器

        # 初始化优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)  # 使用 AdamW 优化器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )  # 使用线性学习率调度器
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)  # 定义交叉熵损失函数

    def fit(self):  # 训练模型的方法
        self.model = self.model.train()  # 设置模型为训练模式
        losses = []  # 初始化损失列表
        labels = []  # 初始化标签列表
        predictions = []  # 初始化预测列表
        scaler = torch.cuda.amp.GradScaler()  # 初始化自动混合精度缩放器
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))  # 创建进度条
        for i, data in progress_bar:  # 遍历训练数据加载器
            self.optimizer.zero_grad()  # 清除优化器的梯度
            vectors = data["vector"].to(self.device)  # 获取输入数据并移至设备
            targets = data["targets"].to(self.device)  # 获取目标标签并移至设备
            with autocast():  # 使用自动混合精度进行训练
                outputs, _ = self.model(vectors)  # 获取模型输出
                loss = self.loss_fn(outputs, targets)  # 计算损失
            scaler.scale(loss).backward()  # 反向传播
            scaler.step(self.optimizer)  # 更新模型参数
            scaler.update()  # 更新缩放器
            preds = torch.argmax(outputs, dim=1).flatten()  # 获取预测结果
            
            losses.append(loss.item())  # 将损失添加到列表
            predictions += list(np.array(preds.cpu()))  # 将预测结果添加到列表
            labels += list(np.array(targets.cpu()))  # 将真实标签添加到列表

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪

            self.scheduler.step()  # 更新学习率调度器
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}'
            )  # 更新进度条描述信息
        train_loss = np.mean(losses)  # 计算平均训练损失
        score_dict = get_MCM_score(labels, predictions)  # 计算多类混淆矩阵评分
        return train_loss, score_dict  # 返回训练损失和评分

    def eval(self):  # 评估模型的方法
        print("start evaluating...")  # 打印评估开始信息
        self.model = self.model.eval()  # 设置模型为评估模式
        losses = []  # 初始化损失列表
        pre = []  # 初始化预测结果列表
        label = []  # 初始化真实标签列表
        correct_predictions = 0  # 初始化正确预测数量
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))  # 创建进度条

        with torch.no_grad():  # 禁用梯度计算
            for _, data in progress_bar:  # 遍历验证数据加载器
                vectors = data["vector"].to(self.device)  # 获取输入数据并移至设备
                targets = data["targets"].to(self.device)  # 获取目标标签并移至设备
                outputs, _ = self.model(vectors)  # 获取模型输出
                loss = self.loss_fn(outputs, targets)  # 计算损失
                preds = torch.argmax(outputs, dim=1).flatten()  # 获取预测结果
                correct_predictions += torch.sum(preds == targets)  # 计算正确预测数量

                pre += list(np.array(preds.cpu()))  # 将预测结果添加到列表
                label += list(np.array(targets.cpu()))  # 将真实标签添加到列表
                
                losses.append(loss.item())  # 将损失添加到列表
                progress_bar.set_description(
                    f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}'
                )  # 更新进度条描述信息
        val_acc = correct_predictions.double() / len(self.valid_set)  # 计算验证集的准确率
        print("val_acc : ", val_acc)  # 打印验证集准确率
        score_dict = get_MCM_score(label, pre)  # 计算多类混淆矩阵评分
        val_loss = np.mean(losses)  # 计算平均验证损失
        return val_loss, score_dict  # 返回验证损失和评分

    def train(self):  # 训练和验证模型的方法
        learning_record_dict = {}  # 初始化学习记录字典
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])  # 初始化训练结果表
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])  # 初始化验证结果表
        for epoch in range(self.epochs):  # 遍历每个训练轮数
            print(f'Epoch {epoch + 1}/{self.epochs}')  # 打印当前轮数
            train_loss, train_score = self.fit()  # 调用 fit 方法进行训练
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])  # 将训练结果添加到表中
            print(train_table)  # 打印训练结果表

            val_loss, val_score = self.eval()  # 调用 eval 方法进行评估
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])  # 将验证结果添加到表中
            print(test_table)  # 打印验证结果表
            print("\n")  # 打印空行
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                                           "train_score": train_score, "val_score": val_score}  # 将学习记录添加到字典
            save_data(self.result_save_path, learning_record_dict)  # 保存学习记录到文件
            print("\n")  # 打印空行
            
        #TAG 一共有5次的epoch，前四次都是训练集的train.pkl，只有最后一次是测试集的test.pkl
        # save_table_data(self.result_save_path.replace('.result', '_tabel.txt'), train_table) 
        # save_table_data(self.result_save_path.replace('.result', '_tabel.txt'), test_table)
        # save_table_excel(self.result_save_path.replace('.result', '.xlsx'), train_table)
        # save_table_excel(self.result_save_path.replace('.result', '.xlsx'), test_table)
        
        save_table_data(self.result_save_path.replace('.result', '_train_tabel.txt'), train_table) 
        save_table_data(self.result_save_path.replace('.result', '_test_tabel.txt'), test_table)
        save_table_excel(self.result_save_path.replace('.result', '_train_tabel.xlsx'), train_table)
        save_table_excel(self.result_save_path.replace('.result', '_test_tabel.xlsx'), test_table)

