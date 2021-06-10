# -*- coding: utf-8 -*-
import dgl
import torch
from dgl.sampling import RandomWalkNeighborSampler

from dataset import load_dataset


class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(RandomWalkNeighborSampler(G=g,
                                                               num_traversals=1,
                                                               termination_prob=0,
                                                               num_random_walks=num_neighbors,
                                                               num_neighbors=num_neighbors,
                                                               metapath=metapath))

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


# for test
if __name__ == '__main__':
    new_g, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset('ogbn-mag')
    metapath_list = [  # ['writes_by', 'writes'],
        ['writes_by', 'affiliated_with', 'affiliated_by', 'writes'],
        # ['has_topic', 'has_topic_by']]
    ]
    num_neighbors = 10
    han_sampler = HANSampler(new_g, metapath_list, num_neighbors)
    block_list = han_sampler.sample_blocks(seeds=0)
    print(block_list)
