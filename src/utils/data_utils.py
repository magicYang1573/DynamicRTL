import torch
import os
import numpy as np

def read_file(file_name):
    f = open(file_name, "r")
    data = f.readlines()
    return data

def write_file(filename, dir, y):
    path = os.path.join(dir,filename)
    f = open(path, "w")
    for val in y:
        f.write(str(val[0]) + "\n")
    f.close()


def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x

def construct_node_feature(x, num_gate_types):
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    return x_torch
