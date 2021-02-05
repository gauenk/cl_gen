#!/usr/bin/python3.8

# -- python imports --
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path
import matplotlib
matplotlib.use('agg')

# -- project imports --
import settings
from attn.main import run_me

if __name__ == "__main__":
    # import torch.nn.functional as F

    # import torch
    # import torch.nn as nn
    
    # batch_size = 5
    # seq_len = 10
    # embed_dim = 24
    # value_dim = 12
    # num_heads = 1
    
    # query = torch.randn(seq_len, batch_size, embed_dim)
    # key = torch.randn(seq_len, batch_size, embed_dim)
    # value = torch.randn(seq_len, batch_size, value_dim)
    
    # mha = nn.MultiheadAttention(embed_dim, num_heads, kdim=embed_dim, vdim=value_dim)

    # proj_weights = mha.out_proj.weight.data

    # # print("proj_weights",proj_weights.shape)
    # # mha.out_proj.weight.data = torch.eye(proj_weights.shape[0])
    # # proj_bias = mha.out_proj.bias.data
    # # mha.out_proj.bias.data = torch.zeros_like(proj_bias)

    # attn_out, attn_out_weights = mha(query, key, value)

    # print(query.shape,key.shape,value.shape)
    # print(attn_out.shape)
    run_me()
