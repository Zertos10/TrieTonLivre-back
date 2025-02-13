import os
import networkx as nx
GRAPH_FILE_PATH = '../../../document_graph.graphml'
documentGraph = nx.Graph()
def load_graph():
    global documentGraph
    if os.path.exists(GRAPH_FILE_PATH):
        documentGraph = nx.read_graphml(GRAPH_FILE_PATH)
        print("Graph loading ")
        return documentGraph
def save_graph(graph:nx.Graph):
    global documentGraph
    nx.write_graphml(graph,GRAPH_FILE_PATH)
def get_graph() -> nx.Graph:
    global documentGraph
    if not documentGraph :
        return load_graph()
    return documentGraph