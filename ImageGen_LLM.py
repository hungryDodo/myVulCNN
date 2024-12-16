import os
import glob
import pickle
import argparse
import sent2vec
import datetime
import numpy as np
import networkx as nx
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from email_alarm import send_email

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Validate centrality input parameters
def validate_centrality(value):
    valid_characters = set('BCDKP')
    if not set(value).issubset(valid_characters) or len(value) != len(set(value)):
        raise argparse.ArgumentTypeError(
            f"Invalid centrality type '{value}'. Must be a subset of non-repeating letters from 'BCDKP'."
        )
    return value

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    parser.add_argument(
        '-c', '--centrality', help='The type of centrality to use.', default='BCDP', type=validate_centrality
    )
    args = parser.parse_args()
    return args

def save_info_file(output_dir, args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = f"# Run Information\n\n"
    info += "| **Parameter**       | **Value**                         |\n"
    info += "|---------------------|-----------------------------------|\n"
    info += f"| Generation Time     | {current_time}                   |\n"
    info += f"| Input Directory     | {args.input}                     |\n"
    info += f"| Output Directory    | {args.out}                       |\n"
    info += f"| Model Path          | {args.model}                     |\n"
    info += f"| Centrality Types    | {args.centrality}                |\n"

    info_file = os.path.join(output_dir, "info.md")
    if os.path.exists(info_file):
        with open(info_file, 'a') as f:
            f.write("\n" + "---" + "\n\n")
            f.write(info)
    else:
        with open(info_file, 'w') as f:
            f.write(info)

def sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"].squeeze(0)
    input_chunks = input_ids.split(512)
    chunk_embeddings = []
    for chunk in input_chunks:
        chunk_input = {"input_ids": chunk.unsqueeze(0)}
        outputs = model(**chunk_input)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1)
        chunk_embeddings.append(chunk_embedding)
    
    torch.cuda.synchronize("cuda")
    sentence_embedding = torch.stack(chunk_embeddings).mean(dim=0)
    emb = sentence_embedding.detach().cpu().numpy()
    del outputs
    return emb[0]

def image_generation(dot, centrality_types):
    try:
        pdg = nx.drawing.nx_pydot.read_dot(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        for label, all_code in labels_dict.items():
            code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            code = code.replace("static void", "void")
            labels_code[label] = code

        channels_dict = {
            'B': [],
            'C': [],
            'D': [],
            'K': [],
            'P': []
        }

        centrality_dicts = {}
        if 'B' in centrality_types:
            centrality_dicts['B'] = nx.betweenness_centrality(pdg)
        if 'C' in centrality_types:
            centrality_dicts['C'] = nx.closeness_centrality(pdg)
        if 'D' in centrality_types:
            centrality_dicts['D'] = nx.degree_centrality(pdg)
        if 'K' in centrality_types:
            G_k = nx.DiGraph()
            G_k.add_nodes_from(pdg.nodes())
            G_k.add_edges_from(pdg.edges())
            centrality_dicts['K'] = nx.katz_centrality(G_k)
        if 'P' in centrality_types:
            centrality_dicts['P'] = nx.pagerank(pdg)

        for label, code in labels_code.items():
            line_vec = sentence_embedding(code)
            line_vec = np.array(line_vec)

            for centrality_type in centrality_types:
                if centrality_type in centrality_dicts:
                    cen_value = centrality_dicts[centrality_type].get(label, 0)
                    channels_dict[centrality_type].append(cen_value * line_vec)

        return [channels_dict[ctype] for ctype in centrality_types]

    except Exception as e:
        print(f"\033[31mError: {e}\033[0m")
        return

def write_to_pkl(dot_file, output_dir, existing_files, centrality):
    file_basename = os.path.splitext(os.path.basename(dot_file))[0]
    if file_basename in existing_files:
        print(f"Skipping {dot_file}, already processed.")
        return

    channels = image_generation(dot_file, centrality)
    if channels is None:
        return None

    out_pkl = os.path.join(output_dir, f"{file_basename}.pkl")
    with open(out_pkl, 'wb') as f:
        pickle.dump(channels, f)

def validate_and_normalize_paths(input_path, output_path, model_path):
    input_dir = os.path.abspath(input_path) if input_path else None
    output_dir = os.path.abspath(output_path)
    model_file = os.path.abspath(model_path)

    if input_dir and not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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

    centrality_names = {
        'B': 'Betweenness Centrality',
        'C': 'Closeness Centrality',
        'D': 'Degree Centrality',
        'K': 'Katz Centrality',
        'P': 'PageRank Centrality'
    }
    table = PrettyTable()
    table.field_names = ["Short Name", "Full Name"]
    for ctype in args.centrality:
        table.add_row([ctype, centrality_names[ctype]])
    print("本次使用的中心性列表：")
    print(table)

    save_info_file(output_dir, args)

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base").to("cuda")

    dot_files = glob.glob(os.path.join(input_dir, "*.dot")) if input_dir else []
    if not dot_files:
        print(f"No .dot files found in the directory: {input_dir}")
    pkl_filenames = extract_pkl_filenames(output_dir)

    for dot_file in tqdm(dot_files, desc="Processing .dot files"):
        write_to_pkl(dot_file, output_dir, pkl_filenames, args.centrality)

if __name__ == '__main__':
    main()
    send_email("完成生成LLM vector了")
