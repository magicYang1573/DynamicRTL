'''
Utils for read CDFG in .dot format and transfer it into networkx graph and numpy format
The current CDFG_utils only includes get_op_to_index function.
The complete version of CDFG_utils will be open-sourced in the future.
Author: XXXX-1 XXXX-2
'''

import json

def get_op_to_index(path='operator_types/op_to_index.json'):
    with open(path, 'r') as f:
        op_to_index = json.load(f)
    return op_to_index