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
import numpy as np

from arch.mlp import MLP
from utils.utils import AverageMeter
from utils.logger import Logger

from utils.CDFG_utils import get_op_to_index

class DownstreamTrainer():
    def __init__(self,
                 args,
                 pretrain_model,
                 downstream_model,
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, num_workers=4, 
                 distributed = False,
                 gpus = None
                 ):
        super(DownstreamTrainer, self).__init__()
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
        self.args = args
        self.config_path = os.path.join(self.log_dir, 'config.txt')
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4)
        with open(self.config_path, 'w') as config_file:
            config_file.write(args_str)

        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'downstream-log-{}.txt'.format(time_str))
        self.pred_result_path = os.path.join(self.log_dir, 'downstream-result-{}.txt'.format(time_str))

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

        # # Loss 1: specific assert predict
        self.assert_loss = nn.CrossEntropyLoss().to(self.device)

        # # Loss 2: power predict
        self.power_loss = nn.MSELoss().to(self.device)

        # # Loss 3: area predict
        self.area_loss = nn.MSELoss().to(self.device)

        # # Loss 4: slack predict
        self.slack_loss = nn.MSELoss().to(self.device)

        
        self.model_epoch = 0
        # Downstream Model
        self.pretrain_model = pretrain_model.to(self.device)
        self.downstream_model = downstream_model.to(self.device)

        
        # from itertools import chain
        # self.optimizer = torch.optim.Adam(chain(self.assert_mlp_1.parameters(), self.assert_mlp_2.parameters()), lr=self.lr)
        self.optimizer = torch.optim.Adam(downstream_model.parameters(), lr=self.lr)

        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)

        # epoch pred result record
        self.epoch_pred_result = []
        self.epoch_gt_result = []

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
            self.downstream_model = self.downstream_model.to(self.device)
            self.reg_loss = self.reg_loss.to(self.device)
            self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
        
    def save(self, path):
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.downstream_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    

    def run_batch(self, batch, seq_len):
        
        node_emb = self.pretrain_model(batch, seq_len)

        if self.args.downstream_task=='assertion_1':
            loss, acc, _ = self.pred_assert_one_variable(batch, node_emb, seq_len, self.args.assertion)
        elif self.args.downstream_task=='assertion_2':
            loss, acc, _ = self.pred_assert_two_variable(batch, node_emb, seq_len, self.args.assertion)
        elif self.args.downstream_task=='power':
            loss, acc, _ = self.pred_power(batch, node_emb, self.args.power_model)
        elif self.args.downstream_task=='area':
            loss, acc, _ = self.pred_area(batch, node_emb, self.args.power_model)
        elif self.args.downstream_task=='slack':
            loss, acc, _ = self.pred_slack(batch, node_emb, self.args.power_model)

        return acc, loss


    ###################################
    # 1> downstream single value assert
    ###################################
    def pred_assert_one_variable(self, batch, node_emb, seq_len, assert_type):
        allowed_assert_types = ['v!=0', 'v>2', 'v>4', 'v!=1', 'v!=2', 'v!=4'] 
        if assert_type not in allowed_assert_types:
            raise ValueError(f"Invalid assert_type: {assert_type}. Must be one of {allowed_assert_types}.")
        
        sel_node_mask, input_node_mask, const_node_mask, value_node_mask = self.get_node_mask(batch)

        assert_hit_prob = self.downstream_model.predict_downstream_value_assert(node_emb[value_node_mask]).squeeze(1)

        if value_node_mask.sum(dim=0)>0:
            gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
            gt_sim_res_decimal = self.binary_to_decimal(gt_sim_res)

            if assert_type=='v!=0':
                gt_assert = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float() 
            
            elif assert_type=='v>2':
                gt_assert = torch.any(torch.lt(gt_sim_res_decimal, 2), dim=1).float()

            elif assert_type=='v>4':
                gt_assert = torch.any(torch.lt(gt_sim_res_decimal, 4), dim=1).float()
            
            elif assert_type=='v!=1':
                gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 1), dim=1).float()

            elif assert_type=='v!=2':
                gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 2), dim=1).float()

            elif assert_type=='v!=4':
                gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 4), dim=1).float()

            stas_freq = gt_assert.mean()

            assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / value_node_mask.sum().float()
            assert_pred_res = (assert_hit_prob > 0.5)
            assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()
        else:
            assert_loss = torch.tensor(0)
            assert_acc = torch.tensor(0)

        return assert_loss, assert_acc, stas_freq

    ####################################
    # 2> downstream two value cmp assert
    ####################################
    # need to train with batch_size = 1
    def pred_assert_two_variable(self, batch, node_emb, seq_len, assert_type):
        allowed_assert_types = ['a<b', 'a&b==0', 'a|b!=0', 'a!=b']
        if assert_type not in allowed_assert_types:
            raise ValueError(f"Invalid assert_type: {assert_type}. Must be one of {allowed_assert_types}.")

        sel_node_mask, input_node_mask, const_node_mask, value_node_mask = self.get_node_mask(batch)
        
        if value_node_mask.sum(dim=0)>0:
            gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
            gt_sim_res_decimal = self.binary_to_decimal(gt_sim_res)

            n = gt_sim_res_decimal.size(0)
            i, j = torch.triu_indices(n, n, offset=1)       
            cnt = i.numel()
            assert_hit_prob = self.downstream_model.predict_downstream_cmp_assert(node_emb[value_node_mask][i], node_emb[value_node_mask][j]).squeeze(1)

            if assert_type=='a<b':
                gt_assert = torch.gt(gt_sim_res_decimal[i], gt_sim_res_decimal[j]).any(dim=1).float().to(self.device)

            if assert_type=='a&b==0':
                cal_res = gt_sim_res_decimal[i].to(torch.int64) & gt_sim_res_decimal[j].to(torch.int64)
                gt_assert = torch.gt(cal_res,0).any(dim=1).float().to(self.device)

            if assert_type=='a|b!=0':
                cal_res = gt_sim_res_decimal[i].to(torch.int64) | gt_sim_res_decimal[j].to(torch.int64)
                gt_assert = torch.eq(cal_res,0).any(dim=1).float().to(self.device)
            
            if assert_type=='a!=b':
                gt_assert = torch.eq(gt_sim_res_decimal[i], gt_sim_res_decimal[j]).any(dim=1).float().to(self.device)

            stas_freq = gt_assert.mean()

            assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / torch.tensor(cnt).float()
            assert_pred_res = (assert_hit_prob > 0.5)
            assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()
        
        return assert_loss, assert_acc, stas_freq

    ####################################
    # 3> downstream power prediction
    ####################################
    def pred_power(self, batch, node_emb, power_model):
        pred_power = self.downstream_model.predict_downstream_power(batch, node_emb, power_model)
        gt_power = batch.power
        
        power_loss = self.power_loss(pred_power, gt_power)

        # the same MAPE metric as MasterRTL
        power_pred_error =  torch.clamp(
            torch.abs(pred_power - gt_power) / gt_power,
            max=1.0
        )

        power_err_avg = power_pred_error.mean()
        power_acc = torch.tensor(1) - power_err_avg

        self.epoch_pred_result = self.epoch_pred_result + pred_power.tolist()
        self.epoch_gt_result = self.epoch_gt_result + gt_power.tolist()

        return power_loss, power_acc, 0

    ####################################
    # 4> downstream area prediction (also use power GNN model)
    ####################################
    def pred_area(self, batch, node_emb, power_model):
        pred_area = self.downstream_model.predict_downstream_power(batch, node_emb, power_model)
        gt_area = batch.area
        
        area_loss = self.area_loss(pred_area, gt_area)

        # the same MAPE metric as MasterRTL
        area_pred_error =  torch.clamp(
            torch.abs(gt_area - pred_area) / gt_area,
            max=1.0
        )

        area_err_avg = area_pred_error.mean()
        area_acc = torch.tensor(1) - area_err_avg

        return area_loss, area_acc, 0

    ####################################
    # 5> downstream slack prediction (also use power GNN model)
    ####################################
    def pred_slack(self, batch, node_emb, power_model):
        pred_slack = self.downstream_model.predict_downstream_power(batch, node_emb, power_model)
        gt_slack = batch.slack
        
        slack_loss = self.slack_loss(pred_slack, gt_slack)

        # the same MAPE metric as MasterRTL
        slack_pred_error =  torch.clamp(
            torch.abs(gt_slack - pred_slack) / gt_slack,
            max=1.0
        )

        slack_err_avg = slack_pred_error.mean()
        slack_acc = torch.tensor(1) - slack_err_avg

        return slack_loss, slack_acc, 0


    def get_node_mask(self, batch):
        cond_node_mask = batch.x[:, self.op_to_index['Cond']] == 1
        cond_node_id = cond_node_mask.nonzero().squeeze()
        cond_edge_mask = torch.zeros_like(batch.edge_index[1, :], dtype=torch.bool) 
        sel_node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=self.device)
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
        
        input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
        const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
        value_node_mask = ~(input_node_mask | const_node_mask | sel_node_mask)

        return sel_node_mask, input_node_mask, const_node_mask, value_node_mask


    def binary_to_decimal(self, tensor):

        weights = 2**torch.arange(31, -1, -1).to(tensor.device)

        decimal_tensor = torch.matmul(tensor.float(), weights.float())
        return decimal_tensor

    def train(self, num_epoch, train_dataset, val_dataset, train_seq_len=10, eval_seq_len=10):
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

        loss_stats = AverageMeter()
        acc_stats = AverageMeter()

        
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
            # for phase in ['val']:
                loss_stats.reset()
                acc_stats.reset()

                if phase == 'train':
                    dataset = train_dataset
                    seq_len = train_seq_len
                    self.downstream_model.train()
                    self.downstream_model.to(self.device)
                else:
                    dataset = val_dataset
                    seq_len = eval_seq_len
                    self.downstream_model.eval()
                    self.downstream_model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                
                self.epoch_pred_result = []
                self.epoch_gt_result = []

                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    acc, loss = self.run_batch(batch, seq_len)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)

                    loss_stats.update(loss.item())
                    acc_stats.update(acc.item())

                    if self.local_rank == 0:
                        # Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} \n'.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        # # Bar.suffix += '|Branch Loss: {:.4f} |Assert Loss: {:.4f} |Decode Loss: {:.4f} \n'.format(0, assert_loss_stats.avg, seq_decode_loss_stats.avg)
                        # Bar.suffix += '        |Branch Loss: {:.4f} |Branch Acc: {:.2f}%% |Assert Loss: {:.4f} |Assert Acc: {:.2f}%% '.format(branch_loss_stats.avg, branch_acc_stats.avg*100, assert_loss_stats.avg, assert_acc_stats.avg*100)
                        # Bar.suffix += '|Net: {:.2f}s \n'.format(batch_time.avg)
                        # bar.next()
                        log_str = 'Epoch={} [{:}/{:}]|Tot: {total:} '.format(epoch, iter_id, len(dataset), total=bar.elapsed_td)
                        log_str += '|Loss: {:.4f} |Acc: {:.2f}% '.format(loss_stats.avg, acc_stats.avg*100)
                        log_str += '|Net: {:.2f}s '.format(batch_time.avg)
                        print(log_str, flush=True)
                        self.logger.write(log_str + '\n')

                if phase=='val' and self.args.downstream_task=='power':
                    y_true = np.array(self.epoch_gt_result)
                    y_pred = np.array(self.epoch_pred_result)
                    correlation_matrix = np.corrcoef(y_pred, y_true)
                    correlation_coefficient = correlation_matrix[0, 1]

                    y_mean = np.mean(y_true)
                    numerator = np.sum((y_true - y_pred) ** 2)
                    denominator = np.sum((y_true - y_mean) ** 2)
                    rrse = np.sqrt(numerator / denominator)
                    # rrse = np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))
                    log_str = f'R = {correlation_coefficient};  '
                    log_str += f'RRSE = {rrse}'
                    print(log_str, flush=True)
                    self.logger.write(log_str + '\n')

                    with open(self.pred_result_path, 'a') as f:  # 使用追加模式 'a'
                        f.write(f"Epoch: {self.model_epoch}\n")
                        f.write("y_true: " + np.array2string(y_true, separator=', ', threshold=np.inf, edgeitems=np.inf) + '\n')
                        f.write("y_pred: " + np.array2string(y_pred, separator=', ', threshold=np.inf, edgeitems=np.inf) + '\n')
                        f.write('-' * 50 + '\n')  # 添加分隔线以便区分不同 epoch 的记录

                if phase == 'train' and (self.model_epoch+1) % 10 == 0:
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
            

    # def run_batch_old(self, batch, seq_len):

    #     node_emb = self.model(batch, seq_len)
    #     # print(node_emb)

    #     # Task 1: Branch Hit Prediction
    #     # choose branch select signals of cond node
    #     cond_node_mask = batch.x[:, self.op_to_index['Cond']] == 1
    #     cond_node_id = cond_node_mask.nonzero().squeeze()
    #     cond_edge_mask = torch.zeros_like(batch.edge_index[1, :], dtype=torch.bool) 
    #     sel_node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=self.device)
    #     # # print(cond_node_mask)
    #     # # print(cond_node_mask.sum())
    #     # # print(cond_node_id)
    #     if cond_node_mask.sum()>0:
    #         if len(cond_node_id.shape) == 0:
    #             cond_node_id = torch.unsqueeze(cond_node_id, 0)
    #         for val in cond_node_id:
    #             cond_edge_mask |= (batch.edge_index[1, :] == val)
    #         cond_edge_mask &= batch.edge_type == 1
    #         cond_sel_id = batch.edge_index[0, cond_edge_mask]
    #         cond_sel_id = torch.unique(cond_sel_id)
    #         sel_node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=self.device)
    #         sel_node_mask[cond_sel_id] = True

    #     #     branch_prob = self.model.pred_branch_hit(node_emb[sel_node_mask]).squeeze(1)
    #     #     gt_sim_res = batch.sim_res[sel_node_mask, :seq_len, :]
    #     #     gt_branch_hit = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float()
    #     #     branch_loss = self.branch_loss(branch_prob, gt_branch_hit) / cond_node_mask.sum().float()
    #     #     branch_pred_res = (branch_prob > 0.5)
    #     #     branch_acc = torch.eq(branch_pred_res, gt_branch_hit).float().mean()
    #     # else:
    #     #     branch_loss = torch.tensor(0)
    #     #     branch_acc = torch.tensor(0)

    #     # Task 2: Assertion Hit Prediction
    #     # choose value nodes
    #     # input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
    #     # const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
    #     # value_node_mask = ~(input_node_mask | const_node_mask | sel_node_mask)
    #     # assert_hit_prob = self.model.pred_assert_zero(node_emb[value_node_mask]).squeeze(1)

    #     # if value_node_mask.sum(dim=0)>0:
    #     #     gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
    #     #     gt_assert = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float()
    #     #     assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / value_node_mask.sum().float()
    #     #     assert_pred_res = (assert_hit_prob > 0.5)
    #     #     assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()
    #     # else:
    #     #     assert_loss = torch.tensor(0)
    #     #     assert_acc = torch.tensor(0)
        
    #     # Task 3: Seq Decoder Prediction
    #     # seq = self.model.pred_seq(node_emb, batch.sim_res.shape, batch.sim_res)
    #     # seq_decode_loss = self.seq_loss(seq, batch.sim_res.float())
    #     # seq_pred_res = (seq > 0.5)
    #     # seq_similarity = torch.eq(seq_pred_res, batch.sim_res).float().mean()

        
    #     input_node_mask = batch.x[:, self.op_to_index['Input']] == 1
    #     const_node_mask = batch.x[:, self.op_to_index['Const']] == 1
    #     value_node_mask = ~(input_node_mask | const_node_mask | sel_node_mask)

    #     ###################################
    #     # 1> downstream single value assert
    #     ###################################
    #     assert_hit_prob = self.predict_downstream_value_assert(node_emb[value_node_mask]).squeeze(1)

    #     if value_node_mask.sum(dim=0)>0:
    #         gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
    #         gt_sim_res_decimal = self.binary_to_decimal(gt_sim_res)

    #         # assert value != 0
    #         # gt_assert = gt_sim_res.view(gt_sim_res.size(0), -1).sum(dim=1).clamp(min=0, max=1).float() 
            
    #         # # assert value < 4 (judge if has value>4)
    #         # gt_assert = torch.any(torch.gt(gt_sim_res_decimal, 2), dim=1).float()
    #         # print(gt_assert)
    #         # print(gt_assert.mean())
            

    #         # # assert value < 16 (judge if has value>16)
    #         gt_assert = torch.any(torch.gt(gt_sim_res_decimal, 16), dim=1).float()

    #         # assert value !=1
    #         # gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 1), dim=1).float()

    #         # assert value !=2
    #         # gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 2), dim=1).float()

    #         # assert value !=4
    #         # gt_assert = torch.any(torch.eq(gt_sim_res_decimal, 2), dim=1).float()
    #         stas_freq = gt_assert.mean()
            
    #         assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / value_node_mask.sum().float()
    #         assert_pred_res = (assert_hit_prob > 0.5)
    #         assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()
    #     else:
    #         assert_loss = torch.tensor(0)
    #         assert_acc = torch.tensor(1)
        
    #     ####################################
    #     # 2> downstream two value cmp assert
    #     ####################################
    #     # need to train with batch_size = 1

    #     # assert a > b
    #     if value_node_mask.sum(dim=0)>0:
    #         gt_sim_res = batch.sim_res[value_node_mask, :seq_len, :]
    #         gt_sim_res_decimal = self.binary_to_decimal(gt_sim_res)

    #         n = gt_sim_res_decimal.size(0)
    #         i, j = torch.triu_indices(n, n, offset=1)       
    #         cnt = i.numel()
    #         assert_hit_prob = self.predict_downstream_cmp_assert(node_emb[value_node_mask][i], node_emb[value_node_mask][j]).squeeze(1)

    #         # a < b
    #         # gt_assert = torch.gt(gt_sim_res_decimal[i], gt_sim_res_decimal[j]).any(dim=1).float().to(self.device)

    #         # print(gt_sim_res_decimal[i].size())
            

    #         # print(cal_res.size())
    #         # a & b == 0
    #         # cal_res = gt_sim_res_decimal[i].to(torch.int64) & gt_sim_res_decimal[j].to(torch.int64)
    #         # gt_assert = torch.gt(cal_res,0).any(dim=1).float().to(self.device)

    #         # a | b != 0
    #         # cal_res = gt_sim_res_decimal[i].to(torch.int64) | gt_sim_res_decimal[j].to(torch.int64)
    #         # gt_assert = torch.eq(cal_res,0).any(dim=1).float().to(self.device)

    #         # a != b
    #         gt_assert = torch.eq(gt_sim_res_decimal[i], gt_sim_res_decimal[j]).any(dim=1).float().to(self.device)

    #         stas_freq = gt_assert.mean()
    #         # print(gt_assert.size())

    #         # print(assert_hit_prob)
    #         # print(gt_assert)
    #         # print(assert_hit_prob.size())
    #         # print(gt_assert.size())
    #         # print(torch.tensor(cnt).float())
    #         assert_loss = self.assert_loss(assert_hit_prob, gt_assert) / torch.tensor(cnt).float()
    #         # print(assert_loss)
    #         assert_pred_res = (assert_hit_prob > 0.5)
    #         assert_acc = torch.eq(assert_pred_res, gt_assert).float().mean()

    #     loss_status = {
    #         # 'branch_loss': branch_loss,
    #         'assert_loss': assert_loss
    #         # 'seq_decode_loss': seq_decode_loss
    #     }


    #     return assert_acc, loss_status, stas_freq