from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os, time
import GPUtil
from npz_parser import NpzParser
from model_arch import Model_default, Model_shared
from trainer import Trainer
import argparse
from models4cmp.models import GCN, GAT


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if not os.getcwd() == os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
    print('[INFO] Change working directory to the project root')
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/train', type=str)
    parser.add_argument('--graph_npz_name', default='graphs.npz', type=str)
    parser.add_argument('--label_npz_name', default='labels.npz', type=str)
    parser.add_argument('--distributed', action='store_true', help="If set, train in distributed mode")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_step', default=50, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--num_rounds', default=20, type=int, help="Number of rounds to GNN propogate")
    parser.add_argument('--train_seq_len', default=50, type=int)
    parser.add_argument('--eval_seq_len', default=50, type=int)
    parser.add_argument('--device', default='cuda', type=str, help="Device to use")
    parser.add_argument('--gpus', default='1', type=str, help="GPU IDs to use, example: 0,1,2,3")
    parser.add_argument('--model', default='default', type=str, help="choose the model to use")
    parser.add_argument('--exp_id', default='default', type=str, help="Experiment ID")
    parser.add_argument('--supervision', default='default', type=str, help="only_branch/only_tgl")

    args = parser.parse_args()
    return args

def select_device():
    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    availableGPU = -1
    for GPU in GPUs:
        if GPU.memoryFree > freeMemory:
            freeMemory = GPU.memoryFree
            availableGPU = GPU.id

    device = torch.device(f'cuda:{availableGPU}' if availableGPU != -1 else 'cpu')
    return device

model_factory = {
    'default': Model_default,
    'default_shared': Model_shared,
    'GCN': GCN,
    'GAT': GAT
}

if __name__ == '__main__':
    args = get_parse_args()
    
    circuit_path = os.path.join(args.data_dir, args.graph_npz_name)
    label_path = os.path.join(args.data_dir, args.label_npz_name)
    
    if args.model not in model_factory:
        raise ValueError('Model not supported')
    
    num_epochs = args.num_epochs

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu' or args.device == 'cuda':
        device = torch.device(args.device)
    else:
        device = select_device()
    
    print('[INFO] Parse Dataset')
    dataset = NpzParser(args.data_dir, circuit_path, label_path)
    train_dataset, val_dataset = dataset.get_dataset(split_with_design=True)
    
    
    print('[INFO] Create Model and Trainer')
    model = model_factory[args.model](num_rounds=args.num_rounds)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    trainer = Trainer(args, model, distributed=args.distributed, batch_size=args.batch_size, device=device, gpus=args.gpus, training_id=time_str)
    trainer.set_training_args(lr=args.lr, lr_step=args.lr_step)

    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset, train_seq_len=args.train_seq_len, eval_seq_len=args.eval_seq_len, supervision=args.supervision)
    
    print('[INFO] Finish Training')