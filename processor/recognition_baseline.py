#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import os
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from tqdm import tqdm

from .processor import Processor
from .utils.losses import SupConLoss, SimDistillLoss, CLIPLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        if self.model.pretrained_path and self.arg.phase == 'train':
            self.model.load_pretrained_model()
        self.loss = nn.CrossEntropyLoss()
        self.supcl = SupConLoss(temperature=0.07)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.loss_distill = lambda x, y: kl_loss(F.softmax(x, dim=1), F.log_softmax(y, dim=1))
        self.loss_ddm = lambda output, label_ddm: -torch.mean(torch.sum(torch.log(output) * label_ddm, dim=1))
        self.save_log = {}
        self.loss_sim = SimDistillLoss(norm='none')
        self.loss_clip = CLIPLoss(self.arg.clip_loss_type)
        self.loss_fd_distill = nn.MSELoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Mix':
            # skl contains encoder and classifier
            self.optimizer_skl = optim.Adam(
                self.model.encoder.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            # self.optimizer_txt = optim.SGD(
            #     self.model.text_encoder.parameters(),
            #     lr=self.arg.base_lr,
            #     momentum=0.9,
            #     nesterov=self.arg.nesterov,
            #     weight_decay=self.arg.weight_decay)

            # self.optimizer_txt = optim.Adam(
            #     self.model.text_encoder.parameters(),
            #     lr=self.arg.base_lr,
            #     weight_decay=self.arg.weight_decay)

            self.optimizer_cls = optim.Adam(
                self.model.classifier.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            self.optimizer = [self.optimizer_skl, self.optimizer_cls]
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.lr_decay_type == 'step':
            if self.meta_info['epoch'] < self.arg.warmup_epoch:
                lr = self.warmup(warmup_epoch=self.arg.warmup_epoch)
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            if self.arg.optimizer == 'Mix':
                for param_group in self.optimizer[0].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizer[1].param_groups:
                    param_group['lr'] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.lr = lr
        elif self.arg.lr_decay_type == 'cosine':
            if self.meta_info['epoch'] < self.arg.warmup_epoch:
                lr = self.warmup(warmup_epoch=self.arg.warmup_epoch)
            else:
                lr = self.cosine_annealing(self.arg.base_lr, eta_min=self.arg.end_cosine_lr,
                                           warmup_epoch=self.arg.warmup_epoch)
            if self.arg.optimizer == 'Mix':
                for param_group in self.optimizer[0].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizer[1].param_groups:
                    param_group['lr'] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.lr = lr
        elif self.arg.lr_decay_type == 'constant':
            if self.arg.optimizer == 'Mix':
                for param_group in self.optimizer[0].param_groups:
                    param_group['lr'] = self.arg.base_lr
                for param_group in self.optimizer[1].param_groups:
                    param_group['lr'] = self.arg.base_lr
                self.lr = self.arg.base_lr
            self.lr = self.arg.base_lr
        else:
            self.lr = self.arg.base_lr

    def cosine_annealing(self, x, warmup_epoch=0, eta_min=0.):
        """Cosine annealing scheduler
        """
        return eta_min + (x - eta_min) * (1. + np.cos(
            np.pi * (self.meta_info['epoch'] - warmup_epoch) / (self.arg.num_epoch - warmup_epoch))) / 2

    def warmup(self, warmup_epoch=5):
        """Cosine annealing scheduler
        """
        lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / warmup_epoch
        return lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        assert self.arg.scl_epoch <= self.arg.clip_epoch <= self.arg.num_epoch
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_cls_value = []
        loss_clip_value = []
        loss_scl_value = []
        loss_froster_value = []
        loss_cls_value_Spt = []
        loss_clip_value_Spt = []
        loss_scl_value_Spt = []
        loss_froster_value_Spt = []

        for data, label, support_data, support_label in tqdm(loader):
            # get data
            xs, xse = data[0].float().to(self.dev), data[1].float().to(self.dev)
            xt, xte = support_data[0].float().to(self.dev), support_data[1].float().to(self.dev)
            label = label.long().to(self.dev)
            support_label = support_label.long().to(self.dev)
            # print(support_label)

            # forward
            cls, cls_e, emb_norm, emb_e_norm, y_txt, y_txt_froster, y_txt_stu = self.model(True, xs, xse, label)
            cls_spt, cls_e_spt, emb_norm_spt, emb_e_norm_spt, y_txt_spt, y_txt_froster_spt, y_txt_stu_t = self.model(False, xt, xte, support_label)
            # print(cls_t.shape)
            # print(cls_s.shape)
            loss_cls = self.arg.w_cls * (self.loss(cls, label) + self.loss(cls_e, label))
            loss_cls_Spt = self.arg.w_cls * (self.loss(cls_spt, support_label) + self.loss(cls_e_spt, support_label))

            loss = (loss_cls + loss_cls_Spt)

            # backward
            if isinstance(self.optimizer, list):
                self.optimizer[0].zero_grad()
                self.optimizer[1].zero_grad()
                loss.backward()
                self.optimizer[0].step()
                self.optimizer[1].step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_cls'] = loss_cls.data.item()
            self.iter_info['loss_cls_Spt'] = loss_cls_Spt.data.item()
            loss_value.append(self.iter_info['loss'])
            loss_cls_value.append(self.iter_info['loss_cls'])
            loss_cls_value_Spt.append(self.iter_info['loss_cls_Spt'])
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_loss_cls'] = np.mean(loss_cls_value)
        self.epoch_info['mean_loss_cls_Spt'] = np.mean(loss_cls_value_Spt)
        self.epoch_info['lr'] = self.lr
        self.show_epoch_info()
        self.io.print_timer()
        self.save_log[f'train_epoch{self.meta_info["epoch"] + 1}'] = self.epoch_info
        # self.save_log[f'sm_epoch{self.meta_info["epoch"] + 1}'] = sm.cpu().numpy()
        if self.arg.bool_save_checkpoint and (self.meta_info['epoch'] + 1) % self.arg.save_interval == 0:
            self.save_checkpoint()
        if (self.meta_info['epoch'] + 1) > self.arg.num_epoch - 5 and (
                self.meta_info['epoch'] + 1) % self.arg.eval_interval != 0:
            self.test()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loader_s = self.data_loader['support_test']
        loss_value = []
        result_frag = []
        label_frag = []
        corr_1 = 0
        num = 0
        corr_1_txt = 0

        for data, label in tqdm(loader):

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output, _ = self.model(True, data)
                # output_s2t, _, index_s2t, index_t2s = self.model(data, test='s2t')
            indices_1 = output.topk(1, dim=-1)[1]
            # indices_1_txt = txt_sim.topk(1, dim=-1)[1]
            # print(indices_1_txt)
            loss_cls = self.loss(output, label)
            loss_value.append(loss_cls.data.item())
            for i in range(len(label)):
                if label[i] in indices_1[i]:
                    corr_1 += 1
                num += 1

        loss_value_s = []
        corr_1_s = 0
        num_s = 0
        corr_1_txt_s = 0
        for data, label in tqdm(loader_s):
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            with torch.no_grad():
                output, _ = self.model(False, data)
            indices_1 = output.topk(1, dim=-1)[1]
            for i in range(len(label)):
                if label[i] in indices_1[i]:
                    corr_1_s += 1
                num_s += 1
            loss_cls_s = self.loss(output, label)
            loss_value_s.append(loss_cls_s.data.item())
        top1 = float(corr_1) / num * 100
        self.io.print_log('\tAccuracy (target): {:.2f}%'.format(top1))
        self.io.print_log('\tLoss (target): {:.4f}'.format(np.mean(loss_value)))
        top1_s = float(corr_1_s) / num_s * 100
        self.io.print_log('\tAccuracy (support): {:.2f}%'.format(top1_s))
        self.io.print_log('\tLoss (support): {:.4f}'.format(np.mean(loss_value_s)))
        top1_avg = (top1 + top1_s) / 2
        self.io.print_log('\tAccuracy (average): {:.2f}%'.format(top1_avg))
        top1_sum = float(corr_1 + corr_1_s) / (num + num_s) * 100
        self.io.print_log('\tAccuracy (sum): {:.2f}%'.format(top1_sum))
        self.save_log[f'test_epoch{self.meta_info["epoch"] + 1}_support'] = {
            'accuracy_sum': top1_sum,
            'accuracy_s': top1_s,
            'accuracy_t': top1,
            'accuracy_avg': top1_avg,
            'loss_t': np.mean(loss_value),
            'loss_s': np.mean(loss_value_s),
        }
        if self.arg.phase == 'train':
            if top1_sum > self.best_val_acc:
                self.best_val_acc = top1_sum
                self.io.save_model(self.model, 'best_model.pt')
                self.io.print_log('Best model saved.')

        if self.meta_info['epoch'] + 1 == self.arg.num_epoch and self.arg.phase == 'train':
            self.io.save_pkl(self.save_log, 'log.pkl')

    def extract_feature(self):
        self.model.eval()
        with torch.no_grad():
            sm_woP, sm_wP, label_names = self.model.action_alignment(view_weight=[1., 0.])
            sm_woP_feats, sm_wP_feats, label_names = self.model.action_alignment(view_weight=[0., 1.])
            sm_woP_fusion, sm_wP_fusion, label_names = self.model.action_alignment(view_weight=[0.5, 0.5])
        self.save_log[f'sm_woP'] = sm_woP.cpu().numpy()
        self.save_log[f'sm_wP'] = sm_wP.cpu().numpy()
        self.save_log[f'label_names'] = label_names
        self.save_log[f'sm_woP_feats'] = sm_woP_feats.cpu().numpy()
        self.save_log[f'sm_wP_feats'] = sm_wP_feats.cpu().numpy()
        self.save_log[f'sm_woP_fusion'] = sm_woP_fusion.cpu().numpy()
        self.save_log[f'sm_wP_fusion'] = sm_wP_fusion.cpu().numpy()
        # save log with pickle
        self.io.save_pkl(self.save_log, 'log_sm.pkl')

        exit()

        self.model.eval()
        loader = self.data_loader['test']
        num_data = len(loader.dataset)
        features = np.zeros((num_data, 512))
        # features = np.zeros((num_data, 51))
        pred = np.zeros(num_data)
        labels = np.zeros(num_data)
        ptr = 0

        for data, label in tqdm(loader):
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                cls, output = self.model.extract_feature(data)
            pred[ptr:ptr + data.size(0)] = cls.topk(1, dim=-1)[1].view(-1).data.cpu().numpy()
            features[ptr:ptr + data.size(0)] = output.data.cpu().numpy()
            labels[ptr:ptr + data.size(0)] = label.data.cpu().numpy()
            ptr += data.size(0)

        # np.save(os.path.join(self.arg.work_dir, self.arg.extract_feature_name), features)
        # np.save(os.path.join(self.arg.work_dir, self.arg.extract_label_name), labels)
        np.save(os.path.join(self.arg.work_dir, 'feature.npy'), features)
        np.save(os.path.join(self.arg.work_dir, 'label.npy'), labels)
        np.save(os.path.join(self.arg.work_dir, 'pred.npy'), pred)
        self.extract_feature_center(features, labels)
        print('Features are saved in {}/{}'.format(self.arg.work_dir, self.arg.extract_feature_name))

    def extract_feature_center(self, feat, label):
        # feat = feat.numpy()
        # label = label.numpy()

        num_class = len(np.unique(label))
        centers = np.zeros((num_class, feat.shape[1]))
        for i in range(num_class):
            centers[i] = np.mean(feat[label == i], axis=0)
        np.save(os.path.join(self.arg.work_dir, 'centers.npy'), centers)
        return centers

    def resume(self):
        if self.arg.resume:
            if self.arg.resume_epoch > 0:
                pass
            else:
                if os.path.exists(os.path.join(self.arg.work_dir, 'checkpoint')):
                    for name in os.listdir(os.path.join(self.arg.work_dir, 'checkpoint')):
                        if name.endswith('.cp'):
                            self.arg.resume_epoch = max(self.arg.resume_epoch,
                                                        int(name.split('epoch')[-1].split('.cp')[0]))
        if self.arg.resume_epoch > 0:
            checkpoint = torch.load(os.path.join(self.arg.work_dir, 'checkpoint',
                                                 f'epoch{self.arg.resume_epoch}.cp'))
            # temp_dict = clean_state_dict(checkpoint['model'], 'netD')
            # self.model.load_state_dict(temp_dict, strict=False)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.meta_info['epoch'] = checkpoint['epoch']
            self.meta_info['iter'] = checkpoint['iter']
            self.io.print_log(f'Load checkpoint from epoch {self.arg.resume_epoch}')
            self.arg.start_epoch = self.arg.resume_epoch

    def save_checkpoint(self):
        if not os.path.exists(os.path.join(self.arg.work_dir, 'checkpoint')):
            os.makedirs(os.path.join(self.arg.work_dir, 'checkpoint'))
        epoch = self.meta_info['epoch']
        torch.save({
            'epoch': self.meta_info['epoch'],
            'iter': self.meta_info['iter'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.arg.work_dir, 'checkpoint', f'epoch{epoch + 1}.cp'))
        self.io.print_log('Save checkpoint at {}'.format(
            os.path.join(self.arg.work_dir, 'checkpoint', f'epoch{epoch + 1}.cp')))

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))
        if self.arg.support_test_feeder_args:
            self.data_loader['support_test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.support_test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay_type', type=str, default='constant', help='lr_decay_type')
        parser.add_argument('--end_cosine_lr', type=float, default=0.00001, help='')

        parser.add_argument('--extract_feature_name', type=str, default='train.npy', help='extract_feature_name')
        parser.add_argument('--extract_label_name', type=str, default='train_label.npy', help='extract_label_name')
        parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup_epoch')
        # parser.add_argument('--end_lr', type=float, default=0.00001, help='end learning rate')
        # endregion yapf: enable
        # parser.add_argument('--end_lr', type=float, default=0.00001, help='end learning rate')
        parser.add_argument('--bool_save_checkpoint', type=str2bool, default=False)
        parser.add_argument('--bool_save_best', type=str2bool, default=True)
        parser.add_argument('--resume', type=str2bool, default=True)
        parser.add_argument('--resume_epoch', type=int, default=-1)

        parser.add_argument('--w_cls', type=float, default=0.5, help='loss weights')
        parser.add_argument('--w_clip', type=float, default=0.5, help='loss weights')
        parser.add_argument('--w_scl', type=float, default=0.5, help='loss weights')
        parser.add_argument('--w_froster', type=float, default=0.5, help='loss weights')
        parser.add_argument('--w_dann', type=float, default=0.5, help='loss weights')
        parser.add_argument('--clip_epoch', type=int, default=50)
        parser.add_argument('--weighted_clip', type=str2bool, default=False)
        parser.add_argument('--scl_epoch', type=int, default=25)

        parser.add_argument('--support_test_feeder_args', action=DictAction, default=dict())
        parser.add_argument('--num_classes_target', type=int, default=60)
        parser.add_argument('--clip_loss_type', type=str, default='ce')


        return parser
