import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import math
import random
import copy
from scipy.sparse import csr_matrix
def create_align_graph(g, remove_rate):
    np.random.seed(0)


    edges = list(g.edges())

    remove_num = int(len(edges) * remove_rate)

    random.shuffle(edges)

    max_iters = 10000000

    new_g = copy.deepcopy(g)


    while remove_num and max_iters:
        candidate_edge = edges.pop()
        if new_g.degree[candidate_edge[0]] > 1 and new_g.degree[candidate_edge[1]] > 1:
            new_g.remove_edge(candidate_edge[0], candidate_edge[1])
            remove_num -= 1
        max_iters -= 1

    return new_g



def read_data(dataname, training_rate):
    pre_dir = "./dataset/"
    if dataname in ["phone-email", "ACM-DBLP", "foursquare-twitter","allmv-tmdb","offline-online"]:
        if dataname == "phone-email":
            pre_dir = pre_dir + "PE/"
        if dataname == "ACM-DBLP":
            pre_dir = pre_dir + "AD/"
        if dataname == "foursquare-twitter":
            pre_dir = pre_dir + "FT/"
        if dataname == "allmv-tmdb":
            pre_dir = pre_dir + "AT/"
        if dataname == "offline-online":
            pre_dir = pre_dir + "DB/"
        training_data_path = pre_dir + "{}_{}.npz".format(dataname, training_rate)
        data_bag = np.load(training_data_path)
        return data_bag

def get_hits(sim_o, test_pair, wrank=None, top_k=(1, 5, 10)):
    test_nodes1, test_nodes2 = test_pair[:, 0], test_pair[:, 1]
    sim_o = sim_o[test_nodes1,:][:, test_nodes2]
    l_n, r_n = sim_o.shape
    # sim_o = -Lvec.dot(Rvec.T)
    sim_o = -sim_o
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i, sim[i, j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank, -1), np.expand_dims(wrank, -1)], -1), axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(l_n):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(r_n):
        rank = sim[:, i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    # print('For each left:')
    # for i in range(len(top_lr)):
    #     print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    # print('MRR: %.3f' % (MRR_lr / l_n))
    # print('For each right:')
    # for i in range(len(top_rl)):
    #     print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    # print('MRR: %.3f' % (MRR_rl / r_n))

    res = {f"top {top_k[i]}": top_lr[i] / len(test_pair) for i in range(len(top_lr))}
    res['MRR'] = MRR_lr / l_n
    return res

def get_identity_emb(node_num, anchor_nodes):
    anchor_num = len(anchor_nodes)
    a = sp.lil_matrix((node_num, anchor_num))
    for i, anchor_node in enumerate(anchor_nodes):
        a[anchor_node, i] = 1.0
    return np.array(a.todense())

def get_rwr_scores(adj, anchor_nodes, iterations=100, p=0.85):
    trans_matrix = adj/adj.sum(axis=1)
    node_num = trans_matrix.shape[0]
    anchor_num = len(anchor_nodes)
    a = get_identity_emb(node_num, anchor_nodes)
    prev_R = np.ones((node_num, anchor_num)) / node_num

    for i in range(iterations):
        R = p * (trans_matrix @ prev_R) + (1 - p) * a
        diff = np.linalg.norm(R - prev_R, ord=1)
        if diff < 1e-6:
            break
        prev_R = R
    return R


def pbaa(S,p):
    S = S ** p
    S = normalize(S, axis=1, norm="l1")
    S = normalize(S, axis=0, norm="l1")
    return S



def run_avg_APNA(dataname, training_rate, use_attr, alpha, p, L):
    total_res = {"top 1":0,"top 5":0,"top 10":0,"MRR":0}
    for step in range(5):
        # read data
        data = read_data(dataname, 0.2)
        edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
        anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)


        total_links = np.concatenate((anchor_links,test_pairs),axis=0)
        anchor_links_num = int(total_links.shape[0] * training_rate)
        np.random.shuffle(total_links)
        anchor_links = total_links[:anchor_links_num,:]
        test_pairs = total_links[anchor_links_num:,:]


        anchor_nodes1, anchor_nodes2 = anchor_links[:, 0], anchor_links[:, 1]

        G1, G2 = nx.Graph(), nx.Graph()
        G1.add_edges_from(edge_index1)
        G2.add_edges_from(edge_index2)
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        for edge in G1.edges():
            G1[edge[0]][edge[1]]['weight'] = 1
        for edge in G2.edges():
            G2[edge[0]][edge[1]]['weight'] = 1

    
        adj1 = nx.to_numpy_array(G1, nodelist=list(range(len(G1))))
        adj2 = nx.to_numpy_array(G2, nodelist=list(range(len(G2))))




        if use_attr:
            x1 = data["x1"]
            x2 = data["x2"]
            H = cosine_similarity(x1, x2)

        else:

            rwr1 = get_rwr_scores(adj1, anchor_nodes1, iterations=100,p=0.85)
            rwr2 = get_rwr_scores(adj2, anchor_nodes2, iterations=100,p=0.85)

            H = cosine_similarity(rwr1, rwr2)




        S = np.ones((n1,n2))/(n1*n2)
        S[anchor_nodes1, anchor_nodes2] = 1.0


        res = S[:, :]

        for i in range(L):

            S = (adj1 @ S @ adj2) * (1 - alpha) + H * alpha
            S = S ** p
            S = normalize(S, axis=1, norm="l1")
            S = normalize(S, axis=0, norm="l1")
            print(i)

            res = res + S
        S = res ** p
        S = normalize(S, axis=1, norm="l1")
        S = normalize(S, axis=0, norm="l1")


        res = get_hits(S, test_pairs ,wrank=None, top_k=(1, 5, 10))
        print(f"The {step+1}-th result:",res)
        for j in [1,5,10]:
            total_res[f'top {j}'] = total_res[f'top {j}'] + res[f'top {j}']
        total_res["MRR"] = total_res["MRR"] + res["MRR"]
    print("\n")
    print("Average Results:")
    for j in [1, 5, 10]:
        print(f"top {j}:", total_res[f'top {j}']/5)
    print(f"MRR:", total_res["MRR"]/5)


def fast_run_avg_APNA(dataname, training_rate, use_attr, alpha, p, L):
    total_res = {"top 1": 0, "top 5": 0, "top 10": 0, "MRR": 0}
    for step in range(5):
        # read data
        data = read_data(dataname, 0.2)
        edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
        anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)

        total_links = np.concatenate((anchor_links, test_pairs), axis=0)
        anchor_links_num = int(total_links.shape[0] * training_rate)
        np.random.shuffle(total_links)
        anchor_links = total_links[:anchor_links_num, :]
        test_pairs = total_links[anchor_links_num:, :]

        anchor_nodes1, anchor_nodes2 = anchor_links[:, 0], anchor_links[:, 1]

        G1, G2 = nx.Graph(), nx.Graph()
        G1.add_edges_from(edge_index1)
        G2.add_edges_from(edge_index2)
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        for edge in G1.edges():
            G1[edge[0]][edge[1]]['weight'] = 1
        for edge in G2.edges():
            G2[edge[0]][edge[1]]['weight'] = 1

        adj1 = nx.to_numpy_array(G1, nodelist=list(range(len(G1))))
        adj2 = nx.to_numpy_array(G2, nodelist=list(range(len(G2))))

        sp_adj1 = csr_matrix(adj1)
        sp_adj2 = csr_matrix(adj2)

        if use_attr:
            x1 = data["x1"]
            x2 = data["x2"]
            H = cosine_similarity(x1, x2)

        else:

            rwr1 = get_rwr_scores(adj1, anchor_nodes1, iterations=100, p=0.85)
            rwr2 = get_rwr_scores(adj2, anchor_nodes2, iterations=100, p=0.85)

            H = cosine_similarity(rwr1, rwr2)

        S = np.ones((n1, n2)) / (n1 * n2)
        S[anchor_nodes1, anchor_nodes2] = 1.0

        res = S[:, :]

        for i in range(L):
            S = (sp_adj2 @ (sp_adj1 @S).T).T + H * alpha

            S = S ** p
            S = normalize(S, axis=1, norm="l1")
            S = normalize(S, axis=0, norm="l1")
            print(i)

            res = res + S
        S = res ** p
        S = normalize(S, axis=1, norm="l1")
        S = normalize(S, axis=0, norm="l1")

        res = get_hits(S, test_pairs, wrank=None, top_k=(1, 5, 10))
        print(f"The {step + 1}-th result:", res)
        for j in [1, 5, 10]:
            total_res[f'top {j}'] = total_res[f'top {j}'] + res[f'top {j}']
        total_res["MRR"] = total_res["MRR"] + res["MRR"]
    print("\n")
    print("Average Results:")
    for j in [1, 5, 10]:
        print(f"top {j}:", total_res[f'top {j}'] / 5)
    print(f"MRR:", total_res["MRR"] / 5)

if __name__ == '__main__':
    np.random.seed(2025)
    # foursquare-twitter "phone-email", "ACM-DBLP",allmv-tmdb offline-online,

    dataname = "phone-email"
    training_rate = 0.2
    use_attr= False
    alpha = 0.6 # 0.4
    L = 5
    p = 5

    s_time = time.time()
    run_avg_APNA(dataname, training_rate, use_attr, alpha, p, L)
    e_time = time.time()





