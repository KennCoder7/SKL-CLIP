import os
import numpy as np
from feeder_skeleton import Feeder
import pandas as pd
import torch

if __name__ == '__main__':
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2kinetics/NTU60_N12.npz',
    #                    dataset='ntu', window_size=128, use_mmap=True, split='train', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # preprocess_data = dataloder.data
    # label = dataloder.label
    # print(preprocess_data.shape)
    # np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2kinetics/preprocessed_NTU60_N12.npz',
    #          x_train=preprocess_data, y_train=label)


    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/Posetics/posetics2ntu',
    #                    dataset='posetics3d', window_size=128, use_mmap=True, split='train', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=False, fixed_direction=False, fixed_rot=None, padding_none=True)
    # preprocess_data = dataloder.data
    # label = dataloder.label
    #
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/Posetics/posetics2ntu',
    #                    dataset='posetics3d', window_size=128, use_mmap=True, split='test', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=False, fixed_direction=False, fixed_rot=None, padding_none=True)
    # preprocess_data_val = dataloder.data
    # label_val = dataloder.label
    #
    # np.savez('/data/wangkun/project/datasets/Posetics/posetics2ntu/preprocessed_posetics2ntu.npz',
    #          x_train=preprocess_data, y_train=label, x_test=preprocess_data_val, y_test=label_val)

    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/NTU_dataset/ntu/NTU60_CS.npz',
    #                    dataset='ntu', window_size=128, use_mmap=True, split='train', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # preprocess_data = dataloder.data
    # label = dataloder.label
    # print(preprocess_data.shape)
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/NTU_dataset/ntu/NTU60_CS.npz',
    #                    dataset='ntu', window_size=128, use_mmap=True, split='test', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # test_preprocess_data = dataloder.data
    # test_label = dataloder.label
    # print(test_preprocess_data.shape)
    # np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/preprocessed_NTU60_CS.npz',
    #          x_train=preprocess_data, y_train=label, x_test=test_preprocess_data, y_test=test_label)
    #
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/NTU_dataset/ntu/NTU60_CV.npz',
    #                    dataset='ntu', window_size=128, use_mmap=True, split='train', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # preprocess_data = dataloder.data
    # label = dataloder.label
    # print(preprocess_data.shape)
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/NTU_dataset/ntu/NTU60_CV.npz',
    #                    dataset='ntu', window_size=128, use_mmap=True, split='test', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # test_preprocess_data = dataloder.data
    # test_label = dataloder.label
    # print(test_preprocess_data.shape)
    # np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/preprocessed_NTU60_CV.npz',
    #          x_train=preprocess_data, y_train=label, x_test=test_preprocess_data, y_test=test_label)
    #
    # total_data = np.concatenate((preprocess_data, test_preprocess_data), axis=0)
    # total_label = np.concatenate((label, test_label), axis=0)
    # print(total_data.shape, total_label.shape)
    # np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/preprocessed_NTU60_ALL.npz',
    #          x_train=total_data, y_train=total_label)

    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CS.npz',
    #                    dataset='pkummd', window_size=128, use_mmap=True, split='train', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # preprocess_data = dataloder.data
    # label = dataloder.label
    # print(preprocess_data.shape)
    # dataloder = Feeder(data_path='/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CS.npz',
    #                    dataset='pkummd', window_size=128, use_mmap=True, split='test', p_interval=[1],
    #                    normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
    #                    vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    # test_preprocess_data = dataloder.data
    # test_label = dataloder.label
    # print(test_preprocess_data.shape)
    # np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/preprocessed_PKUp2_CS.npz',
    #          x_train=preprocess_data, y_train=label, x_test=test_preprocess_data, y_test=test_label)

    dataloder = Feeder(data_path='/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz',
                       dataset='pkummd', window_size=128, use_mmap=True, split='train', p_interval=[1],
                       normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
                       vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    preprocess_data = dataloder.data
    label = dataloder.label
    print(preprocess_data.shape)
    dataloder = Feeder(data_path='/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz',
                       dataset='pkummd', window_size=128, use_mmap=True, split='test', p_interval=[1],
                       normalization=-1, coordinate_transfer_type=888, random_shear=False, random_rot=False,
                       vertical=True, fixed_direction=True, fixed_rot=None, padding_none=True)
    test_preprocess_data = dataloder.data
    test_label = dataloder.label
    print(test_preprocess_data.shape)
    np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/preprocessed_PKUp2_CV.npz',
             x_train=preprocess_data, y_train=label, x_test=test_preprocess_data, y_test=test_label)