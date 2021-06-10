
"""
train HAN use 'PAP' based full graph on CPU. Need more than 110G memory.
"""
import torch
import dgl
import argparse
from sklearn.metrics import f1_score

from dataset import load_dataset
from utils import EarlyStopping
from model_hetero import HAN
import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, val_nid, loss_fn):
    model.eval()
    blocks = []
    hs = []
    blocks.append(g)
    hs.append(features)
    logits = model(blocks, hs)

    loss = loss_fn(logits[val_nid], labels[val_nid])
    accuracy, micro_f1, macro_f1 = score(logits[val_nid], labels[val_nid])
    return loss, accuracy, micro_f1, macro_f1


def load_subtensors(blocks, features):
    h_list = []
    for block in blocks:
        input_nodes = block.srcdata[dgl.NID]
        h_list.append(features[input_nodes])
    return h_list


def main(args):
    # load data
    g, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset(args['dataset'])
    pap = dgl.metapath_reachable_graph(g, ['writes_by', 'writes'])
    # add self loop
    print("meta path based graph:")
    print(pap)
    pap = dgl.remove_self_loop(pap)
    print("remove self loop:")
    print(pap)
    pap = dgl.add_self_loop(pap)
    print("add self loop:")
    print(pap)

    model = HAN(num_metapath=1,
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=n_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        blocks = []
        hs = []
        pap = pap.to(args['device'])
        features = features.to(args['device'])
        blocks.append(pap)
        hs.append(features)
        logits = model(blocks, hs)
        loss = loss_fcn(logits, labels.to(args['device']))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits, labels)
        print(
            "Epoch {:d} | loss: {:.4f} | train_acc: {:.4f} | train_micro_f1: {:.4f} | train_macro_f1: {:.4f}".format(
                epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1
            ))
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, pap, features, labels, val_nid, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Val loss {:.4f} | Val Accuracy {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            val_loss.item(), val_acc, val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, pap, features,
                                                                 labels, test_nid, loss_fcn, args['batch_size'])
    print('Test loss {:.4f} | Test Accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_acc, test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("HAN on ogbn-mag used 'PAP' based graph")
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_heads', type=list, default=[8])
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='ogbn-mag')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args().__dict__

    main(args)