import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle

import ast

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/WN18RR/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=ast.literal_eval,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=ast.literal_eval, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=ast.literal_eval, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=ast.literal_eval, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/wn/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float, 
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args


args = parse_args()
# %%


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    print('get2hop:',args.get_2hop)
    print('use2hop:',args.use_2hop)
    print('partial2hop:',args.partial_2hop)
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)

print("in-out degree in ",args.data)
print("unique entities number:", len(Corpus_.unique_entities_set.keys()))

print("in_degree:")
print("{}/{}=".format(sum(Corpus_.in_d.values()),len(Corpus_.in_d.keys())))
print(sum(Corpus_.in_d.values())/len(Corpus_.in_d.keys()))
print("out_degree:")
print("{}/{}=".format(sum(Corpus_.out_d.values()),len(Corpus_.out_d.keys())))
print(sum(Corpus_.out_d.values())/len(Corpus_.out_d.keys()))

