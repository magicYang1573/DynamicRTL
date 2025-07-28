from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import time
import json, argparse
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from arch.mlp import MLP
from utils.utils import AverageMeter
from utils.logger import Logger

from utils.CDFG_utils import get_op_to_index

class Trainer():
    def __init__(self,
                 args,
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, num_workers=4, 
                 distributed = False,
                 gpus = None
                 ):
        super(Trainer, self).__init__()
        # Config
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Write exp config
        self.config_path = os.path.join(self.log_dir, 'config.txt')
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4)
        with open(self.config_path, 'w') as config_file:
            config_file.write(args_str)

        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'pretrain-log-{}.txt'.format(time_str))

        # CDFG utils
        self.op_to_index = get_op_to_index()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            if gpus is not None:
                gpus = gpus.split(',')
                gpus = [int(gpu) for gpu in gpus]
                self.device = 'cuda:%d' % gpus[self.local_rank]
            else:
                self.device = 'cuda:%d' % self.local_rank
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            if gpus is not None:
                gpus = gpus.split(',')
                gpus = [int(gpu) for gpu in gpus]
                if len(gpus):
                    self.device = gpus[0]
            print('Training in single device: ', self.device)
        

        # Loss and Optimizer

        # Loss 1: branch predict
        self.branch_loss = nn.CrossEntropyLoss().to(self.device)

        # Loss 2: assert predict
        # self.assert_loss = nn.CrossEntropyLoss().to(self.device)

        # Loss 3: toggle predict
        self.toggle_loss = nn.MSELoss().to(self.device)

        # # Loss 4: seq decode
        # self.seq_loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        
        if hasattr(model, 'to_device'):
            self.model.to_device()
        
        self.model_epoch = 0
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
        
    def set_training_args(self, lr=-1, lr_step=-1, device='null'):
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            
            self.branch_loss = self.branch_loss.to(self.device)
            # self.assert_loss = self.assert_loss.to(self.device)
            self.toggle_loss = self.toggle_loss.to(self.device)
            
            self.optimizer = self.optimizer
        
    def save(self, path):
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if os.path.exists(model_path):
            self.load(model_path)
            return True
        else:
            return False
    

    def run_batch(self, batch, seq_len):

        node_emb = self.model(batch, seq_len)

        # Task 1: Branch Hit Prediction
        # choose branch select signals of cond node
        cond_node_mask = batch.x[:, self.op_to_index['Cond']] == 1
        cond_node_id = cond_node_mask.nonzero().squeeze()
        cond_edge_mask = torch.zeros_like(batch.edge_index[1, :], dtype=torch.bool) 
        sel_node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=self.device)
        # print(cond_node_mask)
        # print(cond_node_mask.sum())
        # print(cond_node_id)
        if cond_node_mask.sum()>0:
            if len(cond_node_id.shape) == 0:
                cond_node_id = torch.unsqueeze(cond_node_id, 0)
            for val in cond_node_id:
                cond_edge_mask |= (batch.edge_index[1, :] == val)
            cond_edge_mask &= batch.edge_type == 1
            cond_sel_id = batch.edge_index[0, cond_edge_mask]
            cond_sel_id = torch.unique(cond_sel_id)
            sel_node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=self.device)
            sel_node_mask[cond_sel_id] = True

            branch_prob = self.model.pred_branch_hit(node_emb[sel_node_mask]).squeeze(1)
            gt_sim_res = batch.sim_res[sel_node_mask, :seq_len, :]
            gt_branch_hit = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float()
            # branch_loss = self.branch_loss(branch_prob, gt_branch_hit) / cond_node_mask.sum().float()
            branch_loss = self.branch_loss(branch_prob, gt_branch_hit)

            branch_pred_res = (branch_prob > 0.5)
            branch_acc = torch.eq(branch_pred_res, gt_branch_hit).float().mean()
        else:
            branch_loss = torch.tensor(0)
            branch_acc = torch.tensor(0)

        # # Task 2: Assertion Hit Prediction
        # # choose value nodes
        # input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
        # const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
        # value_node_mask = ~(input_node_mask | const_node_mask | sel_node_mask)
        # assert_hit_prob = self.model.pred_assert_zero(node_emb[value_node_mask]).squeeze(1)

        # if value_node_mask.sum(dim=0)>0:
        #     gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
        #     gt_assert = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float()
        #     assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / value_node_mask.sum().float()
        #     assert_pred_res = (assert_hit_prob > 0.5)
        #     assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()
        # else:
        #     assert_loss = torch.tensor(0)
        #     assert_acc = torch.tensor(0)

        # Task 3: Bit-level Toggle Rate Prediction
        # choose value and selection nodes
        # input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
        # const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
        # toggle_node_mask = ~(input_node_mask | const_node_mask )
        # toggle_rate = self.model.pred_toggle_rate(node_emb[toggle_node_mask]).squeeze(1)

        # if toggle_node_mask.sum(dim=0)>0:
        #     gt_sim_res = batch.sim_res[toggle_node_mask, :seq_len, :]
        #     reshape_gt_sim_res = gt_sim_res.view(gt_sim_res.size(0), -1)     # [node_number, seq_len*bit_width]
        #     gt_change_number = (reshape_gt_sim_res[:, 1:] != reshape_gt_sim_res[:, :-1]).sum(dim=1)
        #     gt_toggle_rate = gt_change_number / (seq_len * gt_sim_res.size(2))
        #     # toggle_loss = self.toggle_loss(toggle_rate, gt_toggle_rate) / toggle_node_mask.sum().float()
        #     toggle_loss = self.toggle_loss(toggle_rate, gt_toggle_rate)

        #     non_zero_mask = gt_toggle_rate != 0
        #     toggle_pred_error = torch.zeros_like(gt_toggle_rate)
        #     toggle_pred_error[non_zero_mask] =  torch.clamp(
        #         torch.abs(toggle_rate[non_zero_mask] - gt_toggle_rate[non_zero_mask]) / gt_toggle_rate[non_zero_mask],
        #         max=1.0
        #     )
        #     toggle_pred_error[~non_zero_mask] = 0.0
        #     toggle_pred_error = toggle_pred_error.mean()
        #     toggle_err = torch.tensor(1) -  toggle_pred_error
        # else:
        #     toggle_loss = torch.tensor(0)
        #     toggle_err = torch.tensor(0)

        # Task 4: Variable-level Transition Rate Prediction
        # choose value and selection nodes
        input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
        const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
        toggle_node_mask = ~(input_node_mask | const_node_mask )
        toggle_rate = self.model.pred_toggle_rate(node_emb[toggle_node_mask]).squeeze(1)

        if toggle_node_mask.sum(dim=0)>0:
            gt_sim_res = batch.sim_res[toggle_node_mask, :seq_len, :]
            gt_sim_res_decimal = self.binary_to_decimal(gt_sim_res)
            gt_change_number = (gt_sim_res_decimal[:, 1:] != gt_sim_res_decimal[:, :-1]).sum(dim=1)
            gt_toggle_rate = gt_change_number / seq_len
            # toggle_loss = self.toggle_loss(toggle_rate, gt_toggle_rate) / toggle_node_mask.sum().float()
            toggle_loss = self.toggle_loss(toggle_rate, gt_toggle_rate)
            toggle_pred_error = torch.abs(toggle_rate - gt_toggle_rate).mean()
        else:
            toggle_loss = torch.tensor(0)
            toggle_pred_error = torch.tensor(0)


        # Task 5: Seq Decoder Prediction
        # seq = self.model.pred_seq(node_emb, batch.sim_res.shape, batch.sim_res)
        # seq_decode_loss = self.seq_loss(seq, batch.sim_res.float())
        # seq_pred_res = (seq > 0.5)
        # seq_similarity = torch.eq(seq_pred_res, batch.sim_res).float().mean()

        loss_status = {
            'branch_loss': branch_loss,
            # 'assert_loss': assert_loss,
            'toggle_loss': toggle_loss
            # 'seq_decode_loss': seq_decode_loss
        }
        
        return branch_acc, toggle_pred_error, loss_status
    
    def binary_to_decimal(self, tensor):

        weights = 2**torch.arange(31, -1, -1).to(tensor.device)

        decimal_tensor = torch.matmul(tensor.float(), weights.float())
        return decimal_tensor

    def train(self, num_epoch, train_dataset, val_dataset, train_seq_len=10, eval_seq_len=10, supervision='default'):
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()

        branch_loss_stats = AverageMeter()
        branch_acc_stats = AverageMeter()
        # assert_loss_stats = AverageMeter()
        # assert_acc_stats = AverageMeter()
        toggle_loss_stats = AverageMeter()
        toggle_err_stats = AverageMeter()
        # seq_decode_loss_stats = AverageMeter()
        # seq_decode_similarity_stats = AverageMeter()
        
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                branch_loss_stats.reset()
                branch_acc_stats.reset()
                # assert_loss_stats.reset()
                # assert_acc_stats.reset()
                toggle_loss_stats.reset()
                toggle_err_stats.reset()

                if phase == 'train':
                    dataset = train_dataset
                    seq_len = train_seq_len
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    seq_len = eval_seq_len
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    branch_acc, toggle_err, loss_status = self.run_batch(batch, seq_len)
                    branch_loss = loss_status['branch_loss']
                    # assert_loss = loss_status['assert_loss']
                    toggle_loss = loss_status['toggle_loss']
                    # seq_loss = loss_status['seq_decode_loss']

                    # loss = assert_loss + seq_loss
                    # loss = assert_loss + branch_loss
                    # assert_loss = assert_loss * 100
                    toggle_loss = toggle_loss * 100000
                    # loss = toggle_loss + branch_loss + assert_loss

                    if supervision=='only_branch':
                        loss = branch_loss
                    elif supervision=='only_tgl':
                        loss = toggle_loss
                    else:
                        loss = toggle_loss + branch_loss

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)

                    # # assert loss
                    # assert_loss_stats.update(assert_loss.item())
                    # assert_acc_stats.update(assert_acc.item())

                    # branch loss
                    branch_loss_stats.update(branch_loss.item())
                    branch_acc_stats.update(branch_acc.item())

                    # toggle loss
                    toggle_loss_stats.update(toggle_loss.item())
                    toggle_err_stats.update(toggle_err.item())

                    # seq loss
                    # seq_decode_loss_stats.update(loss_status['seq_decode_loss'].item())
                    # seq_decode_similarity_stats.update(seq_similarity)

                    if self.local_rank == 0:
                        # Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} \n'.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        # # Bar.suffix += '|Branch Loss: {:.4f} |Assert Loss: {:.4f} |Decode Loss: {:.4f} \n'.format(0, assert_loss_stats.avg, seq_decode_loss_stats.avg)
                        # Bar.suffix += '        |Branch Loss: {:.4f} |Branch Acc: {:.2f}%% |Assert Loss: {:.4f} |Assert Acc: {:.2f}%% '.format(branch_loss_stats.avg, branch_acc_stats.avg*100, assert_loss_stats.avg, assert_acc_stats.avg*100)
                        # Bar.suffix += '|Net: {:.2f}s \n'.format(batch_time.avg)
                        # bar.next()
                        log_str = 'Epoch={} [{:}/{:}]|Tot: {total:} '.format(epoch, iter_id, len(dataset), total=bar.elapsed_td)
                        # log_str += '|Branch Loss: {:.4f} |Branch Acc: {:.2f}% |Assert Loss: {:.4f} |Assert Acc: {:.2f}% '.format(branch_loss_stats.avg, branch_acc_stats.avg*100, assert_loss_stats.avg, assert_acc_stats.avg*100)
                        # log_str += '|Branch Loss: {:.4f} | Acc: {:.2f}% || Assert Loss: {:.4f} | Acc: {:.2f}% || Toggle Loss: {:.4f} | Acc: {:.2f}%'.format(
                        #     branch_loss_stats.avg, branch_acc_stats.avg*100, assert_loss_stats.avg, assert_acc_stats.avg*100, toggle_loss_stats.avg, toggle_err_stats.avg*100
                        # )
                        log_str += '|Branch Loss: {:.4f} | Acc: {:.2f}% || Toggle Loss: {:.4f} | Err: {:.2f}%'.format(
                            branch_loss_stats.avg, branch_acc_stats.avg*100, toggle_loss_stats.avg, toggle_err_stats.avg*100
                        )
                        log_str += '|Net: {:.2f}s '.format(batch_time.avg)
                        print(log_str, flush=True)
                        self.logger.write(log_str + '\n')

                if phase == 'train' and self.model_epoch % 10 == 0:
                    self.save(os.path.join(self.log_dir, 'model_{:}.pth'.format(self.model_epoch)))
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                # if self.local_rank == 0:
                #     self.logger.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |ACC: {:.4f} |Net: {:.2f}s\n'.format(
                #         phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, acc_stats.avg, batch_time.avg))
                #     bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            