# -*- coding: utf-8 -*-
"""
HAN mini-batch training by RandomWalkSampler.
note: We use RandomWalkSampler to sample neighbors, it's hard to get all neighbors in valid or test,
We sampled twice as many neighbors during val/test than training.

PS: In ognb-mag, the performance is bad, accuracy is about 0.25.
    I try to use full meta-path based graph to train HAN,
    and I used 'PAP'/'PFP'/'PAIAP' to do dgl.metapath_reachable_graph(),
    but only 'PAP' can be constructed.
    The graph constructed by 'PFP' or 'PAIAP' is too dense, which will lead to OOM.

I try to train HAN on full 'PAP' based graph, the accuracy is about 0.29.

"""
import dgl
import numpy
import argparse
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from sampler import HANSampler
from utils import EarlyStopping
from model_hetero import HAN


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, metapath_list, num_neighbors, features, labels, val_nid, loss_fcn, batch_size):
    model.eval()

    han_valid_sampler = HANSampler(g, metapath_list, num_neighbors=num_neighbors * 2)
    dataloader = DataLoader(
        dataset=val_nid,
        batch_size=batch_size,
        collate_fn=han_valid_sampler.sample_blocks,
        shuffle=False,
        drop_last=False,
        num_workers=4)
    correct = total = 0
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for step, (seeds, blocks) in enumerate(dataloader):
            h_list = load_subtensors(blocks, features)
            blocks = [block.to(args['device']) for block in blocks]
            hs = [h.to(args['device']) for h in h_list]

            logits = model(blocks, hs)
            loss = loss_fcn(logits, labels[numpy.asarray(seeds)].to(args['device']))
            # get each predict label
            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            labels_batch = labels[numpy.asarray(seeds)].cpu().numpy()

            prediction_list.append(prediction)
            labels_list.append(labels_batch)

            correct += (prediction == labels_batch).sum()
            total += prediction.shape[0]

    total_prediction = numpy.concatenate(prediction_list)
    total_labels = numpy.concatenate(labels_list)
    micro_f1 = f1_score(total_labels, total_prediction, average='micro')
    macro_f1 = f1_score(total_labels, total_prediction, average='macro')
    accuracy = correct / total

    return loss, accuracy, micro_f1, macro_f1


def load_subtensors(blocks, features):
    h_list = []
    for block in blocks:
        input_nodes = block.srcdata[dgl.NID]
        h_list.append(features[input_nodes])
    return h_list


def main(args):
    # acm data
    if args['dataset'] == 'ACMRaw':
        from utils import load_data
        g, features, labels, n_classes, train_nid, val_nid, test_nid, train_mask, \
        val_mask, test_mask = load_data('ACMRaw')
        metapath_list = [['pa', 'ap'], ['pf', 'fp']]
    elif args['dataset'] == 'ogbn-mag':
        from dataset import load_dataset
        g, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset('ogbn-mag')
        metapath_list = [['writes_by', 'writes'],
                         ['cites', 'cites_by']
                         # ['writes_by', 'affiliated_with', 'affiliated_by', 'writes'],
                         # ['has_topic', 'has_topic_by']
                         ]
    else:
        raise NotImplementedError('Unsupported dataset {}'.format(args['dataset']))

    # Is it need to set different neighbors numbers for different meta-path based graph?
    num_neighbors = args['num_neighbors']
    han_sampler = HANSampler(g, metapath_list, num_neighbors)
    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid,
        batch_size=args['batch_size'],
        collate_fn=han_sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    model = HAN(num_metapath=len(metapath_list),
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=n_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total trainable params: {:d}".format(total_trainable_params))

    stopper = EarlyStopping(patience=args['patience'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        for step, (seeds, blocks) in enumerate(dataloader):
            h_list = load_subtensors(blocks, features)
            blocks = [block.to(args['device']) for block in blocks]
            hs = [h.to(args['device']) for h in h_list]

            logits = model(blocks, hs)
            loss = loss_fn(logits, labels[numpy.asarray(seeds)].to(args['device']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info in each batch
            train_acc, train_micro_f1, train_macro_f1 = score(logits, labels[numpy.asarray(seeds)])
            print(
                "Epoch {:d} | loss: {:.4f} | train_acc: {:.4f} | train_micro_f1: {:.4f} | train_macro_f1: {:.4f}".format(
                    epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1
                ))
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, metapath_list, num_neighbors, features,
                                                                 labels, val_nid, loss_fn, args['batch_size'])
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Val loss {:.4f} | Val Accuracy {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, metapath_list, num_neighbors, features,
                                                                 labels, test_nid, loss_fn, args['batch_size'])
    print('Test loss {:.4f} | Test Accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_acc, test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('mini-batch HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_neighbors', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_heads', type=list, default=[8])
    parser.add_argument('--hidden_units', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='ACMRaw')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args().__dict__
    # set_random_seed(args['seed'])

    main(args)
