
#INFO 替换katz中心性为pageRank中心性，不加通道注意力机制
#MARK  代码整体功能解释：
# 这段代码是一个基于图像的漏洞检测脚本。其功能包括从给定的 .dot 文件中提取图信息，计算图的中心性，使用 sent2vec 模型将节点标签嵌入为向量，结合图的中心性生成特征通道，然后将这些通道保存为 .pkl 文件格式。

# 主要步骤：
# 命令行参数解析：通过命令行指定输入目录、输出目录和模型路径。

# 图信息提取：使用 NetworkX 从 .dot 文件中读取程序依赖图（PDG）。

# 句子嵌入生成：通过 sent2vec 模型，将节点的代码标签转化为向量表示。

# 中心性计算：计算图的度中心性、接近中心性和 PageRank 中心性，并将这些值与嵌入向量结合生成特征通道。

# 并行处理与文件输出：利用多进程池加速文件处理，并将生成的通道数据保存为 .pkl 文件以便后续使用。


#TAG 加了注释的修改后的代码
import networkx as nx  # 导入 NetworkX 库并命名为 nx，用于创建和操作图结构。
import numpy as np  # 导入 NumPy 库并命名为 np，用于数值计算和数组操作。
import argparse  # 导入 argparse 库，用于解析命令行参数。
import os  # 导入 os 库，用于文件系统路径操作和文件检查。
import sent2vec  # 导入 sent2vec 库，用于将句子转换为向量嵌入。
import pickle  # 导入 pickle 库，用于序列化和反序列化 Python 对象。
import glob  # 导入 glob 库，用于查找符合特定模式的文件路径。
from multiprocessing import Pool  # 从 multiprocessing 模块中导入 Pool 类，用于并行处理。
from functools import partial  # 从 functools 模块中导入 partial 函数，用于固定某些函数参数的值。

def parse_options():
    # 创建命令行参数解析器，并为每个参数设置描述和选项。
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')  # 输入目录，包含.dot文件。
    parser.add_argument('-o', '--out', help='The path of output.', required=True)  # 输出目录，存放处理后的文件，参数必需。
    parser.add_argument('-m', '--model', help='The path of model.', required=True)  # 模型文件路径，参数必需。
    args = parser.parse_args()  # 解析命令行参数。
    return args  # 返回解析后的参数。

def graph_extraction(dot):
    try:
       graph = nx.drawing.nx_pydot.read_dot(dot)
    except Exception as e:
        # 捕获并处理异常，打印详细的错误信息
        print(f'An exception occurred: {e}')
    # 使用 NetworkX 从.dot文件中读取图。
    # graph = nx.drawing.nx_pydot.read_dot(dot)
    
    #TAG 新增
    # graph = nx.drawing.nx_agraph.read_dot(dot)
    #BUG return none
    return graph  # 返回读取的图对象。

def sentence_embedding(sentence):
    # 使用 sent2vec_model 对句子进行嵌入处理，得到其向量表示。
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]  # 返回嵌入的第一个向量（假设结果是一个二维数组）。

# def image_generation(dot):
#     try:
#         # 尝试从.dot文件中提取图。
#         #BUG 提取图发生了报错
#         pdg = graph_extraction(dot)
#         # 获取图中节点的属性（这里获取的是节点的标签）。
#         labels_dict = nx.get_node_attributes(pdg, 'label')
#         labels_code = dict()  # 初始化一个字典以存储节点标签和代码映射。
#         for label, all_code in labels_dict.items():
#             # 从标签字符串中提取代码片段并进行一些替换操作。
#             #BUG 源代码这一句可能会造成准确率ACC的错误
#             #DONE 解决，ACC错误原因是由于.dot文件生成为空，这里直接使用的作者已经生成的自己的.dot文件，没有自己生成
#             code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            
#             #MARK 来的issue中的修改----没有生效
#             # code = all_code[all_code.index(",") + 1:-1].split('\n')[0]
            
#             code = code.replace("static void", "void")
#             labels_code[label] = code  # 将处理后的代码映射到对应的节点标签。

#         # 打印节点标签与代码映射（调试用）。
#         #print(labels_code)
#         # 计算图的度中心性。
#         degree_cen_dict = nx.degree_centrality(pdg)
#         # 计算图的接近中心性。
#         closeness_cen_dict = nx.closeness_centrality(pdg)
#         # 计算图的谐波中心性（注释掉）。
#         #harmonic_cen_dict = nx.harmonic_centrality(pdg)

#         # 创建一个有向图 G 并添加节点和边。
#         G = nx.DiGraph()
#         G.add_nodes_from(pdg.nodes())
#         G.add_edges_from(pdg.edges())
#         # 计算图的Katz中心性。
#         katz_cen_dict = nx.katz_centrality(G)
        
#         # 打印中心性字典（调试用）。
#         # print(degree_cen_dict)
#         # print(closeness_cen_dict)
#         # print(harmonic_cen_dict)
#         # print(katz_cen_dict)

#         degree_channel = []  # 初始化度中心性通道列表。
#         closeness_channel = []  # 初始化接近中心性通道列表。
#         katz_channel = []  # 初始化Katz中心性通道列表。
        

#         for label, code in labels_code.items():
#             # 对节点代码进行句子嵌入处理，得到其向量。
#             line_vec = sentence_embedding(code)
#             line_vec = np.array(line_vec)  # 将嵌入向量转换为 NumPy 数组。

#             # 获取该节点的度中心性并与嵌入向量相乘，结果添加到相应通道。
#             degree_cen = degree_cen_dict[label]
#             degree_channel.append(degree_cen * line_vec)

#             # 获取该节点的接近中心性并与嵌入向量相乘，结果添加到相应通道。
#             closeness_cen = closeness_cen_dict[label]
#             closeness_channel.append(closeness_cen * line_vec)

#             # 获取该节点的Katz中心性并与嵌入向量相乘，结果添加到相应通道。
#             katz_cen = katz_cen_dict[label]
#             katz_channel.append(katz_cen * line_vec)

#         return (degree_channel, closeness_channel, katz_channel)  # 返回生成的三个通道。
#     except:
#         return None  # 处理失败时返回 None。


def image_generation(dot):
    try:
        # 尝试从.dot文件中提取图。
        pdg = graph_extraction(dot)
        # 获取图中节点的属性（这里获取的是节点的标签）。
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()  # 初始化一个字典以存储节点标签和代码映射。
        for label, all_code in labels_dict.items():
            # 从标签字符串中提取代码片段并进行一些替换操作。
            #DONE 解决，ACC错误原因是由于.dot文件生成为空，这里直接使用的作者已经生成的自己的.dot文件，没有自己生成
            code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            
            #MARK 来的issue中的修改----没有生效
            # code = all_code[all_code.index(",") + 1:-1].split('\n')[0]
            
            code = code.replace("static void", "void")
            labels_code[label] = code  # 将处理后的代码映射到对应的节点标签。

        # 打印节点标签与代码映射（调试用）。
        #print(labels_code)
        # 计算图的度中心性。
        degree_cen_dict = nx.degree_centrality(pdg)
        # 计算图的接近中心性。
        closeness_cen_dict = nx.closeness_centrality(pdg)
        # 计算图的谐波中心性（注释掉）。
        #harmonic_cen_dict = nx.harmonic_centrality(pdg)
        

        # 创建一个有向图 G 并添加节点和边。
        G = nx.DiGraph()
        G.add_nodes_from(pdg.nodes())
        G.add_edges_from(pdg.edges())
        # # 计算图的Katz中心性。
        # katz_cen_dict = nx.katz_centrality(G)
        
        #MARK 计算pageRank中心性
        try:  
            pageRank_cen_dict = nx.pagerank(G)  
        except Exception as e:  
            print(f"计算 PageRank 时发生错误：{e}")
        
        # print(f'\033[31m' + 'pageRank_cen_dict:' + pageRank_cen_dict + '\033[0m')
        
        # 打印中心性字典（调试用）。
        # print(degree_cen_dict)
        # print(closeness_cen_dict)
        # print(harmonic_cen_dict)
        # print(katz_cen_dict)

        degree_channel = []  # 初始化度中心性通道列表。
        closeness_channel = []  # 初始化接近中心性通道列表。
        # katz_channel = []  # 初始化Katz中心性通道列表。
        pageRank_channel = [] #MARK 初始化pageRank中心性通道列表

       

        for label, code in labels_code.items():
            # 对节点代码进行句子嵌入处理，得到其向量。
            line_vec = sentence_embedding(code)
            line_vec = np.array(line_vec)  # 将嵌入向量转换为 NumPy 数组。

            # 获取该节点的度中心性并与嵌入向量相乘，结果添加到相应通道。
            degree_cen = degree_cen_dict[label]
            degree_channel.append(degree_cen * line_vec)

            # 获取该节点的接近中心性并与嵌入向量相乘，结果添加到相应通道。
            closeness_cen = closeness_cen_dict[label]
            closeness_channel.append(closeness_cen * line_vec)

            # # 获取该节点的Katz中心性并与嵌入向量相乘，结果添加到相应通道。
            # katz_cen = katz_cen_dict[label]
            # katz_channel.append(katz_cen * line_vec)

            #MARK 获取该节点的PageRank中心性并与嵌入向量相乘，结果添加到相应通道。
            pageRank_cen = pageRank_cen_dict[label]
            pageRank_channel.append(pageRank_cen * line_vec)

        print('\033[31m' + 'pageRank_cen' + '\033[0m')
        return (degree_channel, closeness_channel, pageRank_channel)  # 返回生成的三个通道。
    except:
        print('\033[32m' + '绿色文本' + '\033[0m')
        return None  # 处理失败时返回 None。


def write_to_pkl(dot, out, existing_files):
    # 提取 .dot 文件的文件名。
    dot_name = dot.split('/')[-1].split('.dot')[0]
    # 如果该文件已经存在于 existing_files 中，则不进行处理。
    if dot_name in existing_files:
        return None
    else:
        print(dot_name)  # 打印文件名（用于跟踪处理进度）。
        # 调用 image_generation 函数生成图像通道。
        channels = image_generation(dot)
        if channels == None:
            #BUG 调试走了这里，说明通道生成失败了
            return None  # 如果通道生成失败，返回 None。
        else:
            #MARK 解包生成的通道。
            (degree_channel, closeness_channel, pageRank_channel) = channels
            # 构造输出 .pkl 文件路径。
            out_pkl = out + dot_name + '.pkl'
            data = [degree_channel, closeness_channel, pageRank_channel]  #MARK 准备要保存的数据。
            # 使用 pickle 序列化数据并保存到文件。
            with open(out_pkl, 'wb') as f:
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
    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    # 获取目录中所有 .dot 文件的列表。
    dotfiles = glob.glob(dir_name + '*.dot')

    # 确保输出路径末尾包含 '/'。
    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    # 如果输出路径不存在，则创建它。
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #BUG 调试结果 existing_files = []
    # 获取已存在的 .pkl 文件列表。
    existing_files = glob.glob(out_path + "/*.pkl")
    # 提取 .pkl 文件名（去掉后缀）。
    existing_files = [f.split('.pkl')[0] for f in existing_files]

    # 创建一个进程池以并行处理文件。
    pool = Pool(10)
    # 使用 pool.map 调用 write_to_pkl 函数处理每个 .dot 文件。
    pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files), dotfiles)

    # 释放 sent2vec 模型的共享内存。
    sent2vec_model.release_shared_mem(trained_model_path)

if __name__ == '__main__':
    main()  # 调用 main 函数。

