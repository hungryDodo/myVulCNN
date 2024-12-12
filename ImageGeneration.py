import networkx as nx
import numpy as np
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool, Value, Lock
from tqdm import tqdm


# 保证 centrality 输入参数有效
def validate_centrality(value):
    valid_characters = set('BCDEP')
    if not set(value).issubset(valid_characters) or len(value) != len(set(value)):
        raise argparse.ArgumentTypeError(
            f"Invalid centrality type '{value}'. Must be a subset of non-repeating letters from 'BCDEP'."
        )
    return value


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    # 增加一个控制 centrality 种类的选项, 默认使用5种 centrality
    parser.add_argument(
        '-c', '--centrality', help='The type of centrality to use.', default='DPCEB', type=validate_centrality
    )
    args = parser.parse_args()
    return args


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


def write_to_pkl(dot_file, output_dir, existing_files, centrality, progress_counter, lock):
    with lock:
        progress_counter.value += 1

    # 检查当前文件是否已经处理过了
    file_basename = os.path.splitext(os.path.basename(dot_file))[0]
    if file_basename in existing_files:
        print(f"Skipping {dot_file}, already processed.")
        return

    channels = image_generation(dot_file)
    if channels is None:
        return None

    degree_channel, closeness_channel, pageRank_channel = channels
    out_pkl = os.path.join(output_dir, f"{file_basename}.pkl")
    data = [degree_channel, closeness_channel, pageRank_channel]
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)


def validate_and_normalize_paths(input_path, output_path, model_path):
    input_dir = os.path.abspath(input_path) if input_path else None
    output_dir = os.path.abspath(output_path)
    model_file = os.path.abspath(model_path)

    if input_dir and not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    return input_dir, output_dir, model_file


def extract_pkl_filenames(output_dir):
    pkl_files = glob.glob(os.path.join(output_dir, "*.pkl"))
    filenames = [os.path.splitext(os.path.basename(file))[0] for file in pkl_files]
    return filenames


def main():
    args = parse_options()
    input_dir, output_dir, trained_model_path = validate_and_normalize_paths(args.input, args.out, args.model)
    
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)

    dot_files = glob.glob(os.path.join(input_dir, "*.dot")) if input_dir else []
    if not dot_files:
        print(f"No .dot files found in the directory: {input_dir}")
    pkl_filenames = extract_pkl_filenames(output_dir)

    progress_counter = Value('i', 0)
    lock = Lock()

    with tqdm(total=len(dot_files), desc="Processing files") as pbar:
        def update_progress(dot_file):
            with lock:
                pbar.set_postfix(current_file=os.path.basename(dot_file))
                pbar.update(1)
        
        pool = Pool(os.cpu_count() or 1)
        for dot_file in dot_files:
            pool.apply_async(
                write_to_pkl,
                args=(dot_file, output_dir, pkl_filenames, args.centrality, progress_counter, lock),
                callback=lambda _: update_progress(dot_file)
            )
        pool.close()
        pool.join()

    sent2vec_model.release_shared_mem(trained_model_path)


if __name__ == '__main__':
    main()
