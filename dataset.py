# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/5/24
@Description: 
"""
import dgl
from dgl.sampling import RandomWalkNeighborSampler
from ogb.nodeproppred import Evaluator, DglNodePropPredDataset

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


def get_ogb_evaluator(dataset):
    """
    Get evaluator from OGB based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def add_reverse_edges(g):
    src_writes, dst_writes = g.all_edges(etype='writes')
    src_topic, dst_topic = g.all_edges(etype='has_topic')
    src_paper, dst_paper = g.all_edges(etype='cites')
    src_aff, dst_aff = g.all_edges(etype='affiliated_with')

    # add reverse edges
    new_g = dgl.heterograph({
        ("author", "writes", "paper"): (src_writes, dst_writes),
        ("paper", "writes_by", "author"): (dst_writes, src_writes),
        ("paper", "has_topic", "field"): (src_topic, dst_topic),
        ("field", "has_topic_by", "paper"): (dst_topic, src_topic),
        ("author", "affiliated_with", "institution"): (src_aff, dst_aff),
        ("institution", "affiliated_by", "author"): (dst_aff, src_aff),
        ("paper", "cites", "paper"): (src_paper, dst_paper),
        ("paper", "cites_by", "paper"): (dst_paper, src_paper)
    })

    # process features. We set feature to a variable now.
    # new_g.nodes["paper"].data['feat'] = g.nodes["paper"].data["feat"]
    return new_g


def load_dataset(name):
    dataset = DglNodePropPredDataset(name=name)
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    features = g.nodes["paper"].data["feat"]

    # MAG is a heterogeneous graph. The task is to make prediction for paper nodes.
    labels = labels["paper"]
    train_nid = train_nid["paper"]
    val_nid = val_nid["paper"]
    test_nid = test_nid["paper"]

    n_classes = dataset.num_classes
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)
    new_g = add_reverse_edges(g)
    del g

    return new_g, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator


# for test
if __name__ == '__main__':
    g, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset('ogbn-mag')
    seeds = range(10)
    # note: random walk may get same route(same edge), which will be removed in the finial graph.
    # The finial graph's edges may smaller than num_random_walks/num_neighbors.
    RW_sampler = RandomWalkNeighborSampler(G=g, num_traversals=1,
                                           termination_prob=0,
                                           num_random_walks=10,
                                           num_neighbors=10,
                                           metapath=['writes_by', 'writes'])
    frontier = RW_sampler(5)
    g = g.long()
    pap = dgl.metapath_reachable_graph(g, ['writes_by', 'writes'])  # num_nodes=736389, num_edges=65933339,
    # paiap = dgl.metapath_reachable_graph(g, ['writes_by', 'affiliated_with', 'affiliated_by', 'writes']) #OOM, edge nums about 21290605428.
    # print(paiap)
    # pfp = dgl.metapath_reachable_graph(g, ['has_topic', 'has_topic_by']) #OOM, it seems to be a fully connected graph.
