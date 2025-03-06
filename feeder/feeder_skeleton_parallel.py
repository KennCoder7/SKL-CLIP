from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os, pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm


try:
    from tools import (random_choose, f_random_rot, valid_crop_resize, f_random_shear, rotate_to_vertical,
                       rotate_to_face_front, rotate_around_y, flip_skeleton_ntu, normalize_skeleton_to_fixed_distance,
                       rotate_to_vertical_per_frame, rotate_to_face_front_per_frame,
                       f_random_shift, f_random_move, f_random_scale, f_temporal_crop, f_padding_none,
                       random_time_flip, random_spatial_flip, gaus_noise, gaus_filter, axis_mask)
except:
    from .tools import (random_choose, f_random_rot, valid_crop_resize, f_random_shear, rotate_to_vertical,
                        rotate_to_face_front, rotate_around_y, flip_skeleton_ntu, normalize_skeleton_to_fixed_distance,
                        rotate_to_vertical_per_frame, rotate_to_face_front_per_frame,
                        f_random_shift, f_random_move, f_random_scale, f_temporal_crop, f_padding_none,
                        random_time_flip, random_spatial_flip, gaus_noise, gaus_filter, axis_mask)


class Feeder(Dataset):
    def __init__(self, dataset, data_path, support_dataset=None, support_data_path=None,
                 label_path=None, p_interval=1, split='train', random_choose=False,
                 random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=-1, debug=False, use_mmap=False,
                 bone=False, vel=False, fold=1, unpaired_cross_dataset_path=None, resize_crop=True, random_crop=False,
                 xy=False, coordinate_transfer_type=-1, random_shear=False, temporal_crop=False,
                 vertical=False, fixed_direction=False, fixed_rot=None, random_scale=False, padding_none=False,
                 fixed_distance=False, get_pairs=False, strong_aug_method=None, preprocess=True, align_st=False,
                 ada=False, segment='none', segment_length=32, segment_overlap=0.5, backup=False,
                 mini=-1, fusion_dataset=False, unseen_list=None, unseen=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.std_map = None
        self.mean_map = None
        self.dataset = dataset
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.random_shear = random_shear
        self.bone = bone
        self.vel = vel
        self.fold = fold
        self.unpaired_cross_dataset_path = unpaired_cross_dataset_path
        self.resize_crop = resize_crop
        self.random_crop = random_crop
        self.xy = xy
        self.coordinate_transfer_type = coordinate_transfer_type
        self.support_data_path = support_data_path
        self.support_dataset = support_dataset
        self.vertical = vertical
        self.fixed_direction = fixed_direction
        self.fixed_rot = fixed_rot
        self.skeleton_info = {}
        self.random_scale = random_scale
        self.temporal_crop = temporal_crop
        self.padding_none = padding_none
        self.fixed_distance = fixed_distance
        self.get_pairs = get_pairs
        self.strong_aug_method = strong_aug_method
        self.preprocess = preprocess
        self.align_st = align_st
        self.ada = ada
        self.segment = segment
        self.segment_length = segment_length
        assert self.segment in ['none', 'all', 'target', 'support']
        assert (window_size - segment_length) % (segment_length * segment_overlap) == 0
        self.segment_overlap = segment_overlap
        self.segment_stride = int(segment_length * segment_overlap)
        self.mini = mini
        self.fusion_dataset = fusion_dataset

        if self.preprocess:
            file_name = self.data_path.split('/')[-1]
            preprocessed_file_name = "preprocessed_" + file_name
            preprocessed_file_path = self.data_path.replace(file_name, preprocessed_file_name)
            if os.path.exists(preprocessed_file_path):
                self.data_path = preprocessed_file_path
                self.preprocess = False
                print(f'load preprocessed data from {preprocessed_file_path}')

        if dataset == 'ntu':
            self.data, self.label = self.load_data_ntu()
        elif dataset == 'ntu2posetics3d':
            self.data, self.label = self.load_data_ntu(transform='posetics3d')
        elif dataset == 'ntu2kinetics':
            self.data, self.label = self.load_data_ntu(transform='kinetics')
        elif dataset == 'ntu2sbu':
            self.data, self.label = self.load_data_ntu(transform='sbu')
            # self.pairs = ntu_pairs
        elif dataset == 'sbu':
            self.data, self.label = self.load_data_sbu()
            # self.pairs = sbu_pairs
        elif dataset == 'kinetics':
            self.data, self.label = self.load_data_kinetics()
            # self.pairs = kinetics_pairs
        elif dataset == 'posetics2d' or dataset == 'posetics3d':
            self.data, self.label = self.load_data_posetics(axis='2d' if dataset == 'posetics2d' else '3d')
        elif dataset == 'posetics':
            self.data, self.label = self.load_data_posetics(axis='5d')
        elif dataset == 'posetics2dto3d':
            self.data, self.label = self.load_data_posetics(axis='2dto3d')
        elif dataset == 'preprocessed_posetics3d':
            self.data, self.label = self.load_data_posetics(axis='preprocessed')
        elif dataset == 'pkummd':
            self.data, self.label = self.load_data_pkummd()
        elif dataset == 'orgbd' or dataset == 'msrda':
            self.data, self.label = self.load_data_orgbd_msrda()
        else:
            raise NotImplementedError('dataset not supported: {}'.format(dataset))
        print(f'{split} Dataset: {dataset}, data shape: {self.data.shape}, label shape: {self.label.shape}')

        self.unseen = unseen
        self.unseen_list = unseen_list
        self._unseen_dataset()

        if support_dataset is not None:
            if preprocess:
                support_file_name = support_data_path.split('/')[-1]
                preprocessed_support_file_name = "preprocessed_" + support_file_name
                preprocessed_support_file_path = support_data_path.replace(support_file_name,
                                                                         preprocessed_support_file_name)
                if os.path.exists(preprocessed_support_file_path):
                    support_data_path = preprocessed_support_file_path
                    self.preprocess = False
                    print(f'load preprocessed support data from {preprocessed_support_file_path}')
                else:
                    self.preprocess = True

        if support_dataset == 'ntu':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_ntu()
        elif support_dataset == 'sbu':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_sbu()
        elif support_dataset == 'kinetics':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_kinetics()
        elif support_dataset == 'ntu2kinetics':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_ntu(transform='kinetics')
        elif support_dataset == 'posetics2d' or support_dataset == 'posetics3d':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_posetics(
                axis='2d' if support_dataset == 'posetics2d' else '3d')
        elif support_dataset == 'posetics':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_posetics(axis='5d')
        elif support_dataset == 'posetics2dto3d':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_posetics(axis='2dto3d')
        elif support_dataset == 'preprocessed_posetics3d':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_posetics(axis='preprocessed')
        elif support_dataset == 'pkummd':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_pkummd()
        elif support_dataset == 'orgbd' or support_dataset == 'msrda':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_orgbd_msrda()
        elif support_dataset == 'ntu2sbu':
            self.data_path = support_data_path
            self.support_data, self.support_label = self.load_data_ntu(transform='sbu')
        # elif support_dataset == 'kinetics':
        #     self.data_path = support_data_path
        #     self.support_data, self.support_label = self.load_data_kinetics()
        if support_dataset is not None:
            print(f'Support dataset: {support_dataset}, data shape: {self.support_data.shape}, '
                  f'label shape: {self.support_label.shape}')
            if align_st:
                self.align_N()

        if debug:
            self.data = self.data[:1000]
            self.label = self.label[:1000]
            if support_dataset is not None:
                self.support_data = self.support_data[:1000]
                self.support_label = self.support_label[:1000]

        if split == 'train' and backup:
            self.data_backup = self.data.copy()
            self.label_backup = self.label.copy()
            if support_dataset is not None:
                self.support_data_backup = self.support_data.copy()
                self.support_label_backup = self.support_label.copy()
        self.backup = backup
        self.mini_dataset()
        self._fusion_dataset()

    def mini_dataset(self):
        if self.mini == -1:
            return
        else:
            mini_classes = np.arange(self.mini)
            self.data = self.data[self.label[:] < self.mini]
            self.label = self.label[self.label[:] < self.mini]
            if self.support_dataset is not None:
                self.support_data = self.support_data[self.support_label[:] < self.mini]
                self.support_label = self.support_label[self.support_label[:] < self.mini]

    def _fusion_dataset(self):
        if self.fusion_dataset:
            assert self.support_dataset is not None
            num_target_classes = len(np.unique(self.label))
            self.data = np.concatenate((self.data, self.support_data), axis=0)
            self.label = np.concatenate((self.label, self.support_label+num_target_classes), axis=0)
            self.support_data = None
            self.support_label = None
            print(f'Fusion dataset: data shape: {self.data.shape}, label shape: {self.label.shape}')
            print(f'Fusion dataset: num classes: {len(np.unique(self.label))}')

    def _unseen_dataset(self):
        if self.unseen_list is not None:
            print(f'Unseen_list: {self.unseen_list}')
            seen_list = list(set(np.unique(self.label)) - set(self.unseen_list))
            unseen_data = []
            unseen_label = []
            seen_data = []
            seen_label = []
            for i, l in tqdm(enumerate(self.label)):
                if l in self.unseen_list:
                    unseen_data.append(self.data[i])
                    # unseen_label.append(self.label[i])
                    unseen_label.append(self.unseen_list.index(l))
                else:
                    seen_data.append(self.data[i])
                    seen_label.append(self.label[i])
                    # seen_label.append(seen_list.index(l))
            unseen_data = np.stack(unseen_data, axis=0)
            unseen_label = np.stack(unseen_label, axis=0)
            seen_data = np.stack(seen_data, axis=0)
            seen_label = np.stack(seen_label, axis=0)
            # for i in self.unseen_list:
            #     unseen_data.append(self.data[self.label == i])
            #     unseen_label.append(self.label[self.label == i])
            # unseen_data = np.concatenate(unseen_data, axis=0)
            # unseen_label = np.concatenate(unseen_label, axis=0)
            # seen_data = []
            # seen_label = []
            # for i in range(len(self.unseen_list)):
            #     seen_data.append(self.data[self.label != self.unseen_list[i]])
            #     seen_label.append(self.label[self.label != self.unseen_list[i]])
            # seen_data = np.concatenate(seen_data, axis=0)
            # seen_label = np.concatenate(seen_label, axis=0)
            if self.unseen:
                self.data = unseen_data
                self.label = unseen_label
                print(f'Unseen dataset: data shape: {self.data.shape}, label shape: {self.label.shape}, '
                      f'classes num: {len(np.unique(self.label))}')
            else:
                self.data = seen_data
                self.label = seen_label
                print(f'Seen dataset: data shape: {self.data.shape}, label shape: {self.label.shape}, '
                      f'classes num: {len(np.unique(self.label))}')

    def reset_data(self):
        assert self.backup
        self.data = self.data_backup.copy()
        self.label = self.label_backup.copy()
        if self.support_dataset is not None:
            self.support_data = self.support_data_backup.copy()
            self.support_label = self.support_label_backup.copy()

    def load_data_ntu(self, transform=None):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['x_train']
            try:
                label = np.where(npz_data['y_train'] > 0)[1]
                # sample_name = ['train_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_train']
        elif self.split == 'test':
            data = npz_data['x_test']
            try:
                label = np.where(npz_data['y_test'] > 0)[1]
                # sample_name = ['test_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')
        try:
            N, T, _ = data.shape
            data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
        except:
            pass
        self.skeleton_info['top'] = 20  # chest
        self.skeleton_info['bottom'] = 0  # pelvis
        self.skeleton_info['left'] = 4  # left shoulder
        self.skeleton_info['right'] = 8  # right shoulder
        self.skeleton_info['thigh_left_top'] = 12
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 16
        self.skeleton_info['thigh_right_bottom'] = 17
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 14
        self.skeleton_info['calf_right_top'] = 17
        self.skeleton_info['calf_right_bottom'] = 18

        if transform == 'posetics3d':
            N, C, T, V, M = data.shape
            posetics_data = np.zeros((N, 3, T, 17, 2), dtype=data.dtype)
            posetics_data[:, :, :, 0, :] = data[:, :, :, 3, :]
            posetics_data[:, :, :, 1, :] = data[:, :, :, 2, :]
            posetics_data[:, :, :, 2, :] = data[:, :, :, 20, :]
            posetics_data[:, :, :, 3, :] = data[:, :, :, 1, :]
            posetics_data[:, :, :, 4, :] = data[:, :, :, 0, :]
            posetics_data[:, :, :, 5, :] = data[:, :, :, 4, :]
            posetics_data[:, :, :, 6, :] = data[:, :, :, 8, :]
            posetics_data[:, :, :, 7, :] = data[:, :, :, 5, :]
            posetics_data[:, :, :, 8, :] = data[:, :, :, 9, :]
            posetics_data[:, :, :, 9, :] = data[:, :, :, 7, :]
            posetics_data[:, :, :, 10, :] = data[:, :, :, 11, :]
            posetics_data[:, :, :, 11, :] = data[:, :, :, 12, :]
            posetics_data[:, :, :, 12, :] = data[:, :, :, 16, :]
            posetics_data[:, :, :, 13, :] = data[:, :, :, 13, :]
            posetics_data[:, :, :, 14, :] = data[:, :, :, 17, :]
            posetics_data[:, :, :, 15, :] = data[:, :, :, 14, :]
            posetics_data[:, :, :, 16, :] = data[:, :, :, 18, :]
            data = posetics_data
            self.skeleton_info['top'] = 2
            self.skeleton_info['bottom'] = 4
            self.skeleton_info['left'] = 5
            self.skeleton_info['right'] = 6
        elif transform == 'kinetics':
            N, C, T, V, M = data.shape
            # data = data[:, :2, :, :, :]
            kinetics_data = np.zeros((N, 3, T, 19, 2), dtype=data.dtype)
            kinetics_data[:, :, :, 0, :] = data[:, :, :, 2, :]
            kinetics_data[:, :, :, 1, :] = data[:, :, :, 20, :]
            kinetics_data[:, :, :, 2, :] = data[:, :, :, 8, :]
            kinetics_data[:, :, :, 3, :] = data[:, :, :, 9, :]
            kinetics_data[:, :, :, 4, :] = data[:, :, :, 10, :]
            kinetics_data[:, :, :, 5, :] = data[:, :, :, 4, :]
            kinetics_data[:, :, :, 6, :] = data[:, :, :, 5, :]
            kinetics_data[:, :, :, 7, :] = data[:, :, :, 6, :]
            kinetics_data[:, :, :, 8, :] = data[:, :, :, 16, :]
            kinetics_data[:, :, :, 9, :] = data[:, :, :, 17, :]
            kinetics_data[:, :, :, 10, :] = data[:, :, :, 18, :]
            kinetics_data[:, :, :, 11, :] = data[:, :, :, 12, :]
            kinetics_data[:, :, :, 12, :] = data[:, :, :, 13, :]
            kinetics_data[:, :, :, 13, :] = data[:, :, :, 14, :]
            kinetics_data[:, :, :, 14, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 15, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 16, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 17, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 18, :] = data[:, :, :, 0, :]
            kinetics_data[:, 2, :, :, :] = 0
            # kinetics_data[:, [1, 2], :, :, :] = kinetics_data[:, [2, 1], :, :, :]
            self.skeleton_info['top'] = 1
            self.skeleton_info['bottom'] = 18
            self.skeleton_info['left'] = 2
            self.skeleton_info['right'] = 5
            data = kinetics_data
        elif transform == 'sbu':
            N, C, T, V, M = data.shape
            sbu_data = np.zeros((N, 3, T, 15, 2), dtype=data.dtype)
            sbu_data[:, :, :, 0, :] = data[:, :, :, 3, :]
            sbu_data[:, :, :, 1, :] = data[:, :, :, 2, :]
            sbu_data[:, :, :, 2, :] = data[:, :, :, 1, :]
            sbu_data[:, :, :, 3, :] = data[:, :, :, 4, :]
            sbu_data[:, :, :, 4, :] = data[:, :, :, 5, :]
            sbu_data[:, :, :, 5, :] = data[:, :, :, 7, :]
            sbu_data[:, :, :, 6, :] = data[:, :, :, 8, :]
            sbu_data[:, :, :, 7, :] = data[:, :, :, 9, :]
            sbu_data[:, :, :, 8, :] = data[:, :, :, 11, :]
            sbu_data[:, :, :, 9, :] = data[:, :, :, 12, :]
            sbu_data[:, :, :, 10, :] = data[:, :, :, 13, :]
            sbu_data[:, :, :, 11, :] = data[:, :, :, 14, :]
            sbu_data[:, :, :, 12, :] = data[:, :, :, 16, :]
            sbu_data[:, :, :, 13, :] = data[:, :, :, 17, :]
            sbu_data[:, :, :, 14, :] = data[:, :, :, 18, :]
            self.skeleton_info['top'] = 1
            self.skeleton_info['bottom'] = 2
            self.skeleton_info['left'] = 3
            self.skeleton_info['right'] = 6
            data = sbu_data

        if self.preprocess:
            data = self.pre_process(data)
        if transform == 'kinetics':
            data = data[:, :, :, :15, :]
        return data, label

    def load_data_orgbd_msrda(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['data']
            label = npz_data['label']
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, V, C = data.shape
        data = data.transpose(0, 3, 1, 2)  # N C T V
        # data = data.reshape((N, C, T, V, 1))  # N C T V M
        data_new = np.zeros((N, 3, T, V, 2), dtype=np.float32)
        data_new[:, :, :, :, 0] = data
        data_new[:, :, :, :, 1] = data
        data = data_new

        self.skeleton_info['top'] = 2  # chest
        self.skeleton_info['bottom'] = 0  # pelvis
        self.skeleton_info['left'] = 4  # left shoulder
        self.skeleton_info['right'] = 8  # right shoulder

        if self.preprocess:
            data = self.pre_process(data)
        return data, label
    def load_data_pkummd(self):
        # data: N C T V M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['x_train']
            try:
                label = np.where(npz_data['y_train'] > 0)[1]
                # sample_name = ['train_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_train']
        elif self.split == 'test':
            data = npz_data['x_test']
            try:
                label = np.where(npz_data['y_test'] > 0)[1]
                # sample_name = ['test_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')

        if self.xy:
            data = data[:, :2, :, :, :]
        self.skeleton_info['top'] = 20
        self.skeleton_info['bottom'] = 0
        self.skeleton_info['left'] = 4
        self.skeleton_info['right'] = 8
        self.skeleton_info['thigh_left_top'] = 12
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 16
        self.skeleton_info['thigh_right_bottom'] = 17
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 14
        self.skeleton_info['calf_right_top'] = 17
        self.skeleton_info['calf_right_bottom'] = 18
        if self.preprocess:
            data = self.pre_process(data)
        return data, label

    def load_data_kinetics(self):
        # data: N C T V M
        if self.split == 'train':
            phase = 'train'
        elif self.split == 'n12':
            phase = 'n12'
        else:
            phase = 'val'
        data_path = self.data_path + f'/{phase}_data.npy'
        label_path = self.data_path + f'/{phase}_label.npy'
        if self.use_mmap:
            data = np.load(data_path, mmap_mode='r')
        else:
            data = np.load(data_path)
        label = np.load(label_path)

        # if self.xy:
        #     xy_data = data[:, :2, :, :, :]
        #     data = xy_data
        # else:
        #     data_ = np.zeros_like(data)
        #     data_[:, 0] = data[:, 0]
        #     data_[:, 1] = data[:, 1]
        #     data = data_

        data_ = np.zeros_like(data)
        data_[:, 0] = data[:, 0]
        data_[:, 1] = data[:, 1]
        data = data_


        # data[:, :, :, 15, :] = (data[:, :, :, 8, :] + data[:, :, :, 11, :]) / 2.
        # data = data[:, :, :, :16, :]

        # self.skeleton_info['top'] = 1
        # self.skeleton_info['bottom'] = 15
        # self.skeleton_info['left'] = 5  # left shoulder
        # self.skeleton_info['right'] = 4  # right shoulder
        # if self.preprocess:
        #     self.vertical = False
        #     self.fixed_direction = False
        #     data = self.pre_process(data)
        #     # data = data[:, :2, :, :, :]
        data[:, :, :, 14, :] = (data[:, :, :, 14, :] + data[:, :, :, 15, :]) / 2.
        data = data[:, :, :, :15, :]
        if self.preprocess:
            self.vertical = False
            self.fixed_direction = False
            self.coordinate_transfer_type = -1
            data = self.pre_process(data)
            # data[:, 0] = - data[:, 0]
            data[:, 1] = - data[:, 1]
        return data, label

    def load_data_posetics(self, axis='3d'):
        if axis == 'preprocessed':
            if self.use_mmap:
                npz_data = np.load(self.data_path, mmap_mode='r')
            else:
                npz_data = np.load(self.data_path)
            if self.split == 'train':
                data = npz_data['x_train']
                try:
                    label = np.where(npz_data['y_train'] > 0)[1]
                    # sample_name = ['train_' + str(i) for i in range(len(data))]
                except:
                    label = npz_data['y_train']
            elif self.split == 'test':
                data = npz_data['x_test']
                try:
                    label = np.where(npz_data['y_test'] > 0)[1]
                    # sample_name = ['test_' + str(i) for i in range(len(data))]
                except:
                    label = npz_data['y_test']
            else:
                raise NotImplementedError('data split only supports train/test')
            return data, label
        # data: N C T V M
        if self.split == 'train':
            phase = 'train'
        elif self.split == 'n12':
            phase = 'n12'
        else:
            phase = 'val'
        data_path = self.data_path + f'/{phase}_data.npy'
        label_path = self.data_path + f'/{phase}_label.npy'
        if self.use_mmap:
            data = np.load(data_path, mmap_mode='r')
        else:
            data = np.load(data_path)
        label = np.load(label_path)

        if axis == '2d':
            xy_data = data[:, :2, :, :, :]
            data = xy_data
            assert data.shape[1] == 2
        elif axis == '3d':
            xyz_data = data[:, 2:, :, :, :]
            data = xyz_data
            assert data.shape[1] == 3
        elif axis == '5d':
            xyz_data = data[:, :, :, :, :]
            data = xyz_data
            assert data.shape[1] == 5
        elif axis == '2dto3d':
            xy_data = data[:, :2, :, :, :]
            N, C, T, V, M = xy_data.shape
            xyz_data = np.zeros((N, 3, T, V, M))
            xyz_data[:, 0] = -xy_data[:, 0]
            xyz_data[:, 1] = -xy_data[:, 1]
            data = xyz_data
            assert data.shape[1] == 3
        else:
            raise NotImplementedError('axis not supported: {}'.format(axis))
        self.skeleton_info['top'] = 2
        self.skeleton_info['bottom'] = 4
        self.skeleton_info['left'] = 5
        self.skeleton_info['right'] = 6
        self.skeleton_info['thigh_left_top'] = 11
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 12
        self.skeleton_info['thigh_right_bottom'] = 14
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 15
        self.skeleton_info['calf_right_top'] = 14
        self.skeleton_info['calf_right_bottom'] = 16
        if self.preprocess:
            data = self.pre_process(data)
        return data, label

    def load_data_sbu(self):
        root_folder = self.data_path
        fold = self.fold
        # phase = self.phase
        # fold = 0
        phase = 'train' if self.split == 'train' else 'eval'
        folds = {'train': list({1, 2, 3, 4, 5} - {fold}), 'eval': [fold]}

        for i in folds[phase]:
            data_path = '{}/fold{}_data.npy'.format(root_folder, i)
            if os.path.exists(data_path):
                fold = np.load(data_path)
                if i == folds[phase][0]:
                    data = fold
                else:
                    data = np.concatenate((data, fold), axis=0)
            else:
                raise ValueError()

            label_path = '{}/fold{}_label.pkl'.format(root_folder, i)
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label_ = pickle.load(f, encoding='latin1')
                    if i == folds[phase][0]:
                        label = label_
                    else:
                        label = np.concatenate((label, label_))
            else:
                raise ValueError()

        if self.xy:
            data = data[:, :2, :, :, :]

        self.skeleton_info['top'] = 1
        self.skeleton_info['bottom'] = 2
        self.skeleton_info['left'] = 3
        self.skeleton_info['right'] = 6
        if self.preprocess:
            data = self.pre_process(data)

        return data, label

    def pre_process(self, data):
        N, C, T, V, M = data.shape
        print(f'preprocess dataset {self.dataset} with normalization {self.normalization}, '
              f'coordinate_transfer {self.coordinate_transfer_type}, vertical {self.vertical}, '
              f'fixed_direction {self.fixed_direction}, fixed_rot {self.fixed_rot}, '
              f'padding_none {self.padding_none}')
        if self.padding_none:
            pp_data = f_padding_none(data)
        else:
            pp_data = data.copy()
        for i, data_numpy in enumerate(tqdm(pp_data)):
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
            data_numpy = data_numpy[:, :valid_frame_num, :, :]
            if self.coordinate_transfer_type != -1:
                data_numpy = self.coordinate_transfer(data_numpy)
            if self.vertical:
                data_numpy = rotate_to_vertical(data_numpy, self.skeleton_info)
            if self.fixed_direction:
                data_numpy = rotate_to_face_front(data_numpy, self.skeleton_info)
            if self.fixed_rot is not None:
                data_numpy = rotate_around_y(data_numpy, self.fixed_rot)
            if self.fixed_distance:
                data_numpy = normalize_skeleton_to_fixed_distance(data_numpy, self.skeleton_info)
            pp_data[i, :, :valid_frame_num, :, :] = data_numpy
        if self.normalization == 0:
            self.mean_map = pp_data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
            self.std_map = pp_data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M,
                                                                       C * V)).std(axis=0).reshape((C, 1, V, 1))
            pp_data = (pp_data - self.mean_map) / self.std_map
        return pp_data

    def coordinate_transfer(self, data_numpy):  # change here
        C, T, V, M = data_numpy.shape
        if self.coordinate_transfer_type == 2:
            #  take joints 3  of each person, as origins of each person
            origin_target = data_numpy[:, :, 1, :]
            new = data_numpy - origin_target[:, :, None, :]
        elif self.coordinate_transfer_type == 0:
            #  take joints 3  of first person, as origins of each person
            origin_target = data_numpy[:, :, 1, 0]
            new = data_numpy - origin_target[:, :, None, None]
        elif self.coordinate_transfer_type == 1:
            #  take joints 3  of second person, as origins of each person
            origin_target = data_numpy[:, :, 1, 1]
            new = data_numpy - origin_target[:, :, None, None]
            new = data_numpy - origin_target
        elif self.coordinate_transfer_type == 3:
            #  take mean joints  of first person, as origins of each person
            origin_target = data_numpy[:, :, :, 0].mean(axis=1, keepdims=True).mean(axis=3, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 4:
            #  take mean joints  of first person, as origins of each person
            origin_target = data_numpy[:, :, :, 0].mean(axis=2, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 5:
            #  take axis distance of first person, as origins of each person
            origin_target = data_numpy[:, :, :, 0].mean(axis=0, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 6:
            #  take mean joints  of first person at mean frame, as origins of each person
            origin_target = data_numpy[:, :, :, 0].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 7:
            #  take mean joints of each person at mean frame, as origins of each person
            origin_target = data_numpy[:, :, :, :].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_target[:, :, :, :]
        elif self.coordinate_transfer_type == 8:
            #  take mean joints of first person at first frame, as origins of each person
            origin_target = data_numpy[:, :1, :, 0].mean(axis=2, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 9:
            #  take mean joints of first person at first 5 frames, as origins of each person
            origin_target = data_numpy[:, :5, :, 0].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_target[:, :, :, None]
        elif self.coordinate_transfer_type == 88:
            #  take bottom joints of first person at first frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_target = data_numpy[:, :1, bottom, 0]
            new = data_numpy - origin_target[:, :, None, None]
        elif self.coordinate_transfer_type == 888:
            #  take bottom joints of first person at first frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_target = data_numpy[:, :1, bottom, 0]
            new = np.zeros_like(data_numpy)
            for m in range(M):
                if np.sum(data_numpy[:, :, :, m].sum(0).sum(-1) != 0) > 0:
                    new[:, :, :, m] = data_numpy[:, :, :, m] - origin_target[:, :, None]
        elif self.coordinate_transfer_type == 87:
            #  take top joints of first person at first frame, as origins of each person
            top = self.skeleton_info['top']
            origin_target = data_numpy[:, :1, top, 0]
            new = data_numpy - origin_target[:, :, None, None]
        elif self.coordinate_transfer_type == 80:
            #  take bottom joints of first person at each frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_target = data_numpy[:, :, bottom, 0]
            new = data_numpy - origin_target[:, :, None, None]
        elif self.coordinate_transfer_type == 86:
            #  take bottom joints of each person at each frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_target = data_numpy[:, :, bottom, :]
            new = data_numpy - origin_target[:, :, None, :]
        elif self.coordinate_transfer_type == -1:
            new = data_numpy
            # print('no origin transfer')
        else:
            raise NotImplementedError(
                'coordinate_transfer_type not supported: {}'.format(self.coordinate_transfer_type))
        return new

    def _augment(self, data_numpy):
        if self.resize_crop:
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
            p_interval = self.p_interval
            data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, self.window_size)
        if self.random_crop:
            data_numpy = random_choose(data_numpy, self.window_size)
        if self.random_scale:
            data_numpy = f_random_scale(data_numpy)
        if self.random_shear:
            r = self.random_shear if isinstance(self.random_shear, float) else 0.3
            data_numpy = f_random_shear(data_numpy, r)
        if self.random_rot:
            r = self.random_rot if isinstance(self.random_rot, float) else 0.3
            data_numpy = f_random_rot(data_numpy, r)
        if self.random_shift:
            data_numpy = f_random_shift(data_numpy)
        if self.random_move:
            data_numpy = f_random_move(data_numpy)
        if self.temporal_crop:
            ratio = self.temporal_crop if isinstance(self.temporal_crop, float) else 6
            data_numpy = f_temporal_crop(data_numpy, ratio)
        return data_numpy

    def _strong_aug(self, data_numpy, dataset):
        if '0' in self.strong_aug_method:
            data_numpy = f_temporal_crop(data_numpy, 6)
        if '1' in self.strong_aug_method:
            data_numpy = random_spatial_flip(data_numpy, dataset=dataset)
        if '2' in self.strong_aug_method:
            data_numpy = f_random_rot(data_numpy, r=0.3)
        if '3' in self.strong_aug_method:
            data_numpy = gaus_noise(data_numpy)
        if '4' in self.strong_aug_method:
            data_numpy = gaus_filter(data_numpy)
        if '5' in self.strong_aug_method:
            data_numpy = axis_mask(data_numpy)
        if '6' in self.strong_aug_method:
            data_numpy = random_time_flip(data_numpy)

        return data_numpy

    def _segment(self, data_numpy, domain):
        if self.segment == 'all' or domain == self.segment:
            N, C, T, V, M = data_numpy.shape
            S = (T - self.segment_length) // self.segment_stride + 1
            T_ = self.segment_length
            data_numpy_segment = np.zeros((N, S, C, T_, V, M))
            # segment
            for i in range(S):
                data_numpy_segment[:, i] = data_numpy[:, :, i * self.segment_stride:i * self.segment_stride + T_, :, :]
        return data_numpy

    def transform_modality(self, data_numpy):
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        return data_numpy

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        data_numpy = self._augment(data_numpy)
        # data_numpy = self._segment(data_numpy, 'target')

        if self.get_pairs:  # v2
            data_numpy_2 = self.data[index]
            data_numpy_2 = np.array(data_numpy_2)
            data_numpy_2 = self._augment(data_numpy_2)
            # data_numpy_2 = self._strong_aug(data_numpy_2, self.dataset)
            # data_numpy_2 = self._segment(data_numpy_2, 'target')
            data_numpy = (data_numpy, data_numpy_2)

        if self.fusion_dataset:
            return data_numpy, label

        if self.support_data_path is not None:
            support_index = self.index_s2t(index)
            support_data_numpy = self.support_data[support_index]
            support_label = self.support_label[support_index]
            support_data_numpy = np.array(support_data_numpy)
            support_data_numpy = self._augment(support_data_numpy)
            support_data_numpy = self._strong_aug(support_data_numpy, self.support_dataset)
            # support_data_numpy = self._segment(support_data_numpy, 'support')
            if self.get_pairs:
                support_data_numpy_2 = self.support_data[support_index]
                support_data_numpy_2 = np.array(support_data_numpy_2)
                support_data_numpy_2 = self._augment(support_data_numpy_2)
                support_data_numpy_2 = self._strong_aug(support_data_numpy_2, self.support_dataset)
                # support_data_numpy_2 = self._segment(support_data_numpy_2, 'support')
                support_data_numpy = (support_data_numpy, support_data_numpy_2)
            return data_numpy, label, support_data_numpy, support_label
        else:
            return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def index_s2t(self, index):
        if not self.align_st:
            support_index = random.randint(0, len(self.support_label) - 1)
        else:
            support_index = index
        return support_index

    def align_N(self):
        N_target = len(self.label)
        N_support = len(self.support_label)
        if N_target > N_support:
            roundtimes = N_target // N_support
            tail = N_target - N_support * roundtimes
            tail_index = random.sample(range(N_support), tail)
            self.support_data = np.concatenate(
                (np.tile(self.support_data, (roundtimes, 1, 1, 1, 1)),
                 self.support_data[tail_index, :, :, :, :]), axis=0)
            self.support_label = np.concatenate((np.tile(self.support_label, roundtimes),
                                                self.support_label[tail_index]), axis=0)
            print(f'target {N_target} > support {N_support}, padding support dataset to {len(self.support_label)}')
        else:
            roundtimes = N_support // N_target
            tail = N_support - N_target * roundtimes
            tail_index = random.sample(range(N_target), tail)
            self.data = np.concatenate(
                (np.tile(self.data, (roundtimes, 1, 1, 1, 1)),
                 self.data[tail_index, :, :, :, :]), axis=0)
            self.label = np.concatenate((np.tile(self.label, roundtimes),
                                         self.label[tail_index]), axis=0)
            print(f'target {N_target} < support {N_support}, padding target dataset to {len(self.label)}')
        assert len(self.label) == len(self.support_label)
        assert len(self.data) == len(self.support_data)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


ntu_pairs = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
    (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
    (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)
)
sbu_pairs = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6),
    (6, 7), (7, 8), (2, 9), (9, 10), (10, 11),
    (2, 12), (12, 13), (13, 14)
)
kinetics_pairs = (
    (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
    (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
)
posetics_pairs = (
    (0, 1), (1, 2), (2, 3), (2, 5), (2, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (3, 4),
    (4, 11), (4, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
)

if __name__ == '__main__':
    feeder = Feeder('ntu',
                    '/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2kinetics/NTU60_CV.npz',
                    split='test', coordinate_transfer_type=0, normalization=1)
    feeder.get_mean_map()
    print(feeder.mean_map.shape, feeder.std_map.shape, feeder.min_map.shape, feeder.max_map.shape,
          feeder.mean_mean.shape)
