import os
import networkx as nx

def load_dot_file(file_path):
    """Load a .dot file and return a NetworkX graph."""
    return nx.drawing.nx_pydot.read_dot(file_path)

def save_dot_file(graph, output_path):
    """Save a NetworkX graph to a .dot file."""
    nx.drawing.nx_pydot.write_dot(graph, output_path)

def combine_graphs(graph1, graph2):
    """Combine two graphs by merging their nodes and edges.
    If an edge exists in both directions between two nodes, add both directions.
    """
    combined_graph = nx.DiGraph()
    combined_graph.add_nodes_from(graph1.nodes(data=True))
    combined_graph.add_nodes_from(graph2.nodes(data=True))

    for u, v, data in graph1.edges(data=True):
        combined_graph.add_edge(u, v, **data)

    for u, v, data in graph2.edges(data=True):
        if combined_graph.has_edge(v, u):
            combined_graph.add_edge(u, v, **data)
        elif not combined_graph.has_edge(u, v):
            combined_graph.add_edge(u, v, **data)

    return combined_graph

def attach_graphs(graph1, graph2):
    """Attach graph2 to graph1 by creating new nodes and edges.
    If a node in graph1 exists in graph2, create a new node in graph1
    with labels from graph2 and connect the original node to the new node.
    """
    attached_graph = graph1.copy()

    for node, attribute in graph2.nodes(data=True):
        if node in graph1:
            new_node = f"{node}_attached"
            labels = []
			
			# Include the label from the original node in graph2 (node)
            if 'label' in attribute:
                labels.append(attribute['label'])
            
			# Use successors to get only nodes that `node` points to
            for target in graph2.successors(node):
                target_data = graph2.nodes[target]
                if 'label' in target_data:
                    labels.append(target_data['label'])
            # Add the new node with combined labels
            attached_graph.add_node(new_node, label=", ".join(labels))
            # Connect the original node to the new attached node
            attached_graph.add_edge(node, new_node)

    return attached_graph

def fuse_graphs(input_dir, fusion_mode, graph_types, output_dir):
    """Fuse graphs based on the specified mode and types, and save the result."""
    assert len(graph_types) == 2, "You must specify exactly two graph types to fuse."

    graph1_dir = os.path.join(input_dir, graph_types[0])
    graph2_dir = os.path.join(input_dir, graph_types[1])

    for sub_dir in ['Vul', 'No-Vul']:
        graph1_sub_dir = os.path.join(graph1_dir, sub_dir)
        graph2_sub_dir = os.path.join(graph2_dir, sub_dir)

        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in os.listdir(graph1_sub_dir):
            if file_name.endswith('.dot'):
                graph1_path = os.path.join(graph1_sub_dir, file_name)
                graph2_path = os.path.join(graph2_sub_dir, file_name)

                if not os.path.exists(graph2_path):
                    print(f"Warning: Missing {graph2_path}, skipping.")
                    continue

                graph1 = load_dot_file(graph1_path)
                graph2 = load_dot_file(graph2_path)

                if fusion_mode == 'combine':
                    fused_graph = combine_graphs(graph1, graph2)
                elif fusion_mode == 'attach':
                    fused_graph = attach_graphs(graph1, graph2)
                else:
                    raise ValueError(f"Unknown fusion mode: {fusion_mode}")

                output_path = os.path.join(output_sub_dir, file_name)
                save_dot_file(fused_graph, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fuse graphs from two types with specified mode.")
    parser.add_argument("-i", "--input_dir", type=str, help="Path to the input data/graph directory.")
    parser.add_argument("-f", "--fusion_mode", type=str, choices=['combine', 'attach'], help="Fusion mode.")
    parser.add_argument("-g", "--graph_types", nargs=2, type=str, help="Two graph types to fuse (e.g., pdg cfg).")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to save the fused graphs.")

    args = parser.parse_args()

    fuse_graphs(args.input_dir, args.fusion_mode, args.graph_types, args.output_dir)
