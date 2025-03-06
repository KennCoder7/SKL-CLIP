import numpy as np
import pickle

if __name__ == '__main__':
    # cs_data_train = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xsub/train_data.npy')
    # cs_data_val = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xsub/val_data.npy')
    #
    # with open('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xsub/train_label.pkl', 'rb') as f:
    #     _, cs_label_train = pickle.load(f)
    # with open('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xsub/val_label.pkl', 'rb') as f:
    #     _, cs_label_val = pickle.load(f)
    #
    # np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CS.npz',
    #          x_train=cs_data_train, y_train=cs_label_train, x_test=cs_data_val, y_test=cs_label_val)
    #
    # cv_data_train = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xview/train_data.npy')
    # cv_data_val = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xview/val_data.npy')
    #
    # with open('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xview/train_label.pkl', 'rb') as f:
    #     _, cv_label_train = pickle.load(f)
    # with open('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/xview/val_label.pkl', 'rb') as f:
    #     _, cv_label_val = pickle.load(f)
    #
    # np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz',
    #          x_train=cv_data_train, y_train=cv_label_train, x_test=cv_data_val, y_test=cv_label_val)


    # delete nan
    # data = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CS.npz')
    # x_train = data['x_train']
    # y_train = data['y_train']
    # x_test = data['x_test']
    # y_test = data['y_test']
    # x_train_new = []
    # y_train_new = []
    # for i in range(len(x_train)):
    #     if x_train[i].sum() != 0:
    #         x_train_new.append(x_train[i])
    #         y_train_new.append(y_train[i])
    # x_test_new = []
    # y_test_new = []
    # for i in range(len(x_test)):
    #     if x_test[i].sum() != 0:
    #         x_test_new.append(x_test[i])
    #         y_test_new.append(y_test[i])
    # x_train_new = np.array(x_train_new)
    # y_train_new = np.array(y_train_new)
    # x_test_new = np.array(x_test_new)
    # y_test_new = np.array(y_test_new)
    # np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CS.npz',
    #          x_train=x_train_new, y_train=y_train_new, x_test=x_test_new, y_test=y_test_new)
    #
    # data = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz')
    # x_train = data['x_train']
    # y_train = data['y_train']
    # x_test = data['x_test']
    # y_test = data['y_test']
    # x_train_new = []
    # y_train_new = []
    # for i in range(len(x_train)):
    #     if x_train[i].sum() != 0:
    #         x_train_new.append(x_train[i])
    #         y_train_new.append(y_train[i])
    # x_test_new = []
    # y_test_new = []
    # for i in range(len(x_test)):
    #     if x_test[i].sum() != 0:
    #         x_test_new.append(x_test[i])
    #         y_test_new.append(y_test[i])
    # x_train_new = np.array(x_train_new)
    # y_train_new = np.array(y_train_new)
    # x_test_new = np.array(x_test_new)
    # y_test_new = np.array(y_test_new)
    # np.savez('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz',
    #          x_train=x_train_new, y_train=y_train_new, x_test=x_test_new, y_test=y_test_new)

    data = np.load('/data/wangkun/project/datasets/PKUMMD_2/pku_part2/PKUp2_CV.npz')
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    num_classes = np.unique(y_train).shape[0]
    print(num_classes)