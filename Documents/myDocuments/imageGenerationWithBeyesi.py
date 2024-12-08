# TAG 加了注释的修改后的代码
import networkx as nx  # 导入 NetworkX 库并命名为 nx，用于创建和操作图结构。
import numpy as np  # 导入 NumPy 库并命名为 np，用于数值计算和数组操作。
import argparse  # 导入 argparse 库，用于解析命令行参数。
import os  # 导入 os 库，用于文件系统路径操作和文件检查。
import sent2vec  # 导入 sent2vec 库，用于将句子转换为向量嵌入。
import pickle  # 导入 pickle 库，用于序列化和反序列化 Python 对象。
import glob  # 导入 glob 库，用于查找符合特定模式的文件路径。
from multiprocessing import (
    Pool,
)  # 从 multiprocessing 模块中导入 Pool 类，用于并行处理。
from functools import (
    partial,
)  # 从 functools 模块中导入 partial 函数，用于固定某些函数参数的值。
from bayes_opt import BayesianOptimization  

def parse_options():
    # 创建命令行参数解析器，并为每个参数设置描述和选项。
    parser = argparse.ArgumentParser(description="Image-based Vulnerability Detection.")
    parser.add_argument(
        "-i", "--input", help="The path of a dir which consists of some dot_files"
    )  # 输入目录，包含.dot文件。
    parser.add_argument(
        "-o", "--out", help="The path of output.", required=True
    )  # 输出目录，存放处理后的文件，参数必需。
    parser.add_argument(
        "-m", "--model", help="The path of model.", required=True
    )  # 模型文件路径，参数必需。
    args = parser.parse_args()  # 解析命令行参数。
    return args  # 返回解析后的参数。


def graph_extraction(dot):
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot)
    except Exception as e:
        # 捕获并处理异常，打印详细的错误信息
        print(f"An exception occurred: {e}")
    # 使用 NetworkX 从.dot文件中读取图。
    # graph = nx.drawing.nx_pydot.read_dot(dot)

    # TAG 新增
    # graph = nx.drawing.nx_agraph.read_dot(dot)
    # BUG return none
    return graph  # 返回读取的图对象。


def sentence_embedding(sentence):
    # 使用 sent2vec_model 对句子进行嵌入处理，得到其向量表示。
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]  # 返回嵌入的第一个向量（假设结果是一个二维数组）。

#TAG 新增
def image_generation(dot, degree_weight, closeness_weight, katz_weight):  
    try:  
        # 提取图并计算中心性指标  
        pdg = graph_extraction(dot)  
        labels_dict = nx.get_node_attributes(pdg, "label")  
        labels_code = dict()  
        for label, all_code in labels_dict.items():  
            code = all_code[all_code.index(",") + 1 : -2].split("\\n")[0]  
            code = code.replace("static void", "void")  
            labels_code[label] = code  

        degree_cen_dict = nx.degree_centrality(pdg)  
        closeness_cen_dict = nx.closeness_centrality(pdg)  
        G = nx.DiGraph()  
        G.add_nodes_from(pdg.nodes())  
        G.add_edges_from(pdg.edges())  
        katz_cen_dict = nx.katz_centrality(G)  

        degree_channel = []  
        closeness_channel = []  
        katz_channel = []  

        for label, code in labels_code.items():  
            line_vec = sentence_embedding(code)  
            line_vec = np.array(line_vec)  

            degree_cen = degree_cen_dict[label]  
            degree_channel.append(degree_weight * degree_cen * line_vec)  

            closeness_cen = closeness_cen_dict[label]  
            closeness_channel.append(closeness_weight * closeness_cen * line_vec)  

            katz_cen = katz_cen_dict[label]  
            katz_channel.append(katz_weight * katz_cen * line_vec)  

        # 计算模型性能，例如准确率  
        accuracy = compute_accuracy(degree_channel, closeness_channel, katz_channel)  
        return accuracy  
    except:  
        return None  

#TAG 新增
def compute_accuracy(degree_channel, closeness_channel, katz_channel):  
    # 计算模型的准确率或任何您希望优化的指标  
    # 这里需要根据您的实际情况实现  
    # 作为占位符，这里返回一个随机数  
    accuracy = np.random.rand()  
    return accuracy  

#TAG 新增贝叶斯优化
def optimize_weights():  
    # 定义目标函数  
    def target_function(degree_weight, closeness_weight, katz_weight):  
        # 使用您的数据集或示例文件  
        dot_file = 'path_to_sample_dot_file.dot'  
        accuracy = image_generation(dot_file, degree_weight, closeness_weight, katz_weight)  
        if accuracy is None:  
            return 0  
        return accuracy  

    # 定义参数范围  
    pbounds = {  
        'degree_weight': (0.0, 5.0),  
        'closeness_weight': (0.0, 5.0),  
        'katz_weight': (0.0, 5.0)  
    }  

    optimizer = BayesianOptimization(  
        f=target_function,  
        pbounds=pbounds,  
        verbose=2,  
        random_state=1,  
    )  

    optimizer.maximize(  
        init_points=5,  
        n_iter=25,  
    )  

    print("最佳参数组合：", optimizer.max)  

def write_to_pkl(dot, out, existing_files):
    # 提取 .dot 文件的文件名。
    dot_name = dot.split("/")[-1].split(".dot")[0]
    # 如果该文件已经存在于 existing_files 中，则不进行处理。
    if dot_name in existing_files:
        return None
    else:
        print(dot_name)  # 打印文件名（用于跟踪处理进度）。
        # 调用 image_generation 函数生成图像通道。
        channels = image_generation(dot)
        if channels == None:
            # BUG 调试走了这里，说明通道生成失败了
            return None  # 如果通道生成失败，返回 None。
        else:
            # 解包生成的通道。
            (degree_channel, closeness_channel, katz_channel) = channels
            # 构造输出 .pkl 文件路径。
            out_pkl = out + dot_name + ".pkl"
            data = [
                degree_channel,
                closeness_channel,
                katz_channel,
            ]  # 准备要保存的数据。
            # 使用 pickle 序列化数据并保存到文件。
            with open(out_pkl, "wb") as f:
                pickle.dump(data, f)


def main():
    # 解析命令行参数。
    args = parse_options()
    dir_name = args.input  # 输入目录。
    out_path = args.out  # 输出目录。
    trained_model_path = args.model  # 模型文件路径。

    global sent2vec_model  # 声明全局变量 sent2vec_model。
    sent2vec_model = sent2vec.Sent2vecModel()  # 初始化 sent2vec 模型。
    sent2vec_model.load_model(trained_model_path)  # 加载预训练的模型。

    # 确保目录路径末尾包含 '/'。
    if dir_name[-1] == "/":
        dir_name = dir_name
    else:
        dir_name += "/"
    # 获取目录中所有 .dot 文件的列表。
    dotfiles = glob.glob(dir_name + "*.dot")

    # 确保输出路径末尾包含 '/'。
    if out_path[-1] == "/":
        out_path = out_path
    else:
        out_path += "/"

    # 如果输出路径不存在，则创建它。
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # BUG 调试结果 existing_files = []
    # 获取已存在的 .pkl 文件列表。
    existing_files = glob.glob(out_path + "/*.pkl")
    # 提取 .pkl 文件名（去掉后缀）。
    existing_files = [f.split(".pkl")[0] for f in existing_files]

    # 创建一个进程池以并行处理文件。
    pool = Pool(10)
    # 使用 pool.map 调用 write_to_pkl 函数处理每个 .dot 文件。
    pool.map(
        partial(write_to_pkl, out=out_path, existing_files=existing_files), dotfiles
    )

    # 释放 sent2vec 模型的共享内存。
    sent2vec_model.release_shared_mem(trained_model_path)
    
    #TAG 在主函数中调用优化函数  
    optimize_weights()  


if __name__ == "__main__":
    main()  # 调用 main 函数。
