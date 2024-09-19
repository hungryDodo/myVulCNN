import html
import re
import networkx as nx
import numpy as np
import argparse
import os
import pickle
import glob
from multiprocessing import Pool
from functools import partial
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import mutils




from sklearn.preprocessing import StandardScaler
def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 使用 split 函数只分割一次，并检查结果的长度是否为 2
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                key, value = parts
                # 对值进行进一步解析，如果值是字典，则解析为字典类型
                if value.startswith('{') and value.endswith('}'):
                    value = eval(value)  # 使用 eval 函数解析字符串为字典类型
                else:
                    value = int(value)  # 如果不是字典，则将值解析为整数
                # 将键值对添加到字典中
                result_dict[key] = value
            else:
                print(f"Ignoring invalid line: {line.strip()}")
    return result_dict

def load_word2vec_model(model_path):
    """加载 Word2Vec 模型"""
    model = Word2Vec.load(model_path)
    return model

def calculate_tfidf_for_corpus(directory_path):
    """计算目录中所有文本文件的TF-IDF分数"""
    # 使用正则表达式来定义分词器，匹配所有非空白字符序列
    token_pattern = r'\S+'

    # 初始化TF-IDF向量化器，使用自定义的token_pattern
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=token_pattern)

    # 存储文本内容的列表
    corpus = []
    # 存储文件名的列表
    filenames = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # 确保处理的是文本文件
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                corpus.append(text)
                filenames.append(filename)

    # 计算整个语料库的TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 初始化存储TF-IDF分数的字典
    tfidf_scores_dict = {}

    # 提取每个文档的TF-IDF分数，并以文件名为键存储在字典中
    feature_names = vectorizer.get_feature_names_out()
    for doc_idx, filename in enumerate(filenames):
        # 提取每个文档的TF-IDF向量并转换为字典，键为单词，值为TF-IDF分数
        doc_vector = tfidf_matrix[doc_idx].todense().A1
        word_scores = dict(zip(feature_names, doc_vector))
        # 过滤掉TF-IDF分数为0的项
        word_scores = {word: score for word, score in word_scores.items() if score > 0}
        front_name = filename.split('.')[0]
        tfidf_scores_dict[front_name] = word_scores

    return tfidf_scores_dict

def sentence_embedding(sentence,  model):
    words = mutils.tokenize_code_segment(sentence)
    embedding = np.zeros((model.vector_size,))  # 初始化嵌入向量
    weight_sum = 0  # 初始化权重和
    all_scores = 0.0

    for word in words:
        if word in model.wv:
            embedding += model.wv[word]
            weight_sum += 1


    if weight_sum > 0:
        embedding /= weight_sum  # 使用加权和的平均值作为句子嵌入

    return embedding


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    args = parser.parse_args()
    return args

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph

# def sentence_embedding(sentence):
#     emb = sent2vec_model.embed_sentence(sentence)
#     return emb[0]

def image_generation(dot, word2vec_model,  front_name,  flag2):
    try:
        pdg = graph_extraction(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        if flag2 == '1':
            for label, all_code in labels_dict.items():
                # code = all_code.split('code:')[1].split('\\n')[0]
                code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
                code = code.replace("static void", "void")
                labels_code[label] = code
        else:
            for label, all_code in labels_dict.items():
                # 使用正则表达式提取核心代码部分，忽略<SUB>和之后的内容

                code_match = re.search(r'<(.+?)<SUB>', all_code)
                if code_match:
                    # HTML实体解码
                    code = html.unescape(code_match.group(1))
                    # 移除HTML中的标签如 <operator> 和转义字符
                    code = re.sub(r'&lt;[^&gt;]+&gt;', '', code)
                    labels_code[label] = code.strip()


        #print(labels_code)
        degree_cen_dict = nx.degree_centrality(pdg)
        closeness_cen_dict = nx.closeness_centrality(pdg)
        #harmonic_cen_dict = nx.harmonic_centrality(pdg)

        G = nx.DiGraph()
        G.add_nodes_from(pdg.nodes())
        G.add_edges_from(pdg.edges())
        katz_cen_dict = nx.katz_centrality(G)

        degree_channel = []
        closeness_channel = []
        katz_channel = []

        for label, code in labels_code.items():

            line_vec = sentence_embedding(code, word2vec_model)  # 使用 Word2Vec 模型编码
            line_vec = np.array(line_vec)

            degree_cen = degree_cen_dict[label]
            degree_channel.append(degree_cen * line_vec)

            closeness_cen = closeness_cen_dict[label]
            closeness_channel.append(closeness_cen * line_vec)

            katz_cen = katz_cen_dict[label]
            katz_channel.append(katz_cen * line_vec)

        return (degree_channel, closeness_channel, katz_channel)
    except Exception as e:
        print("error:")
        print(e)
        return None

def write_to_pkl(dot, out, existing_files, word2vec_model, flag2):
    print("write_to_pkl")
    dot_name = dot.split('/')[-1].split('.dot')[0]
    front_name = dot_name.split('\\')[-1]
    if front_name in existing_files:
        return None

    else:
        print(front_name)
        channels = image_generation(dot, word2vec_model,  front_name, flag2)
        if channels == None:
            print("channels None")
            return None
        else:
            (degree_channel, closeness_channel, katz_channel, degree_channel2, closeness_channel2, katz_channel2) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [degree_channel, closeness_channel, katz_channel, degree_channel2, closeness_channel2, katz_channel2]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)
            print("write")




def main():
    args = parse_options()
    flag = args.input
    flag2 = args.out
    sub_name = "Vul"
    if flag != '1':
        sub_name = "No-Vul"
    dir_name = './data/sard/pdgs/' + sub_name + '/'
    dir_name2 = './data/sard/cfgs/' + sub_name + '/'
    out_path = './data/sard/outputs/cfgs/' + sub_name + '/'


    trained_model_path = './model/word2vec_model3.model'
    # 加载 Word2Vec 模型
    global word2vec_model
    word2vec_model = load_word2vec_model(trained_model_path)


    tfidf_scores_dict = {}

    if flag2 == 1:
        dotfiles = glob.glob(dir_name + '*.dot')
    else:
        dotfiles = glob.glob(dir_name2 + '*.dot')


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    existing_files = ['']

    pool = Pool(1)
    pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files, word2vec_model=word2vec_model, flag2=flag2), dotfiles)






if __name__ == '__main__':
    main()
