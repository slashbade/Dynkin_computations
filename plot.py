import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_graph(cartan_matrix):
    G = nx.MultiGraph()
    for i in range(cartan_matrix.shape[0]):
        G.add_node(i)
    for i in range(cartan_matrix.shape[0]):
        for j in range(i + 1, cartan_matrix.shape[0]):
            edge_num = int(cartan_matrix[i, j] * cartan_matrix[j, i])
            for _ in range(edge_num):
                
                G.add_edge(i, j)
    return G

def plot_graph(cartan_matrix):
    G = get_graph(cartan_matrix)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                ),
                                ),
                )
    plt.axis('off')
    plt.show()
    
