import numpy as np
import pickle
import os
from tqdm import tqdm

ntu2posetics = []
ntu_label_chosen = ['drink water', 'eat meal', 'brush teeth', 'brush hair', 'clapping', 'reading', 'kicking something',
                    'writing', 'shake head', 'punch or slap', 'hugging', 'shaking hands']
ntu_label_list = np.load('/data/wangkun/project/datasets/NTU_dataset/label_list.npy')
posetics_chosen = [['drinking (p)'],
                   ['eating (p)'],
                   ['brushing teeth'],
                   ['hair (p)'],
                   ['clapping'],
                   ['reading (p)'],
                   ['kicking (p)'],
                   ['writing'],
                   ['shaking head'],
                   ['punching person (boxing)', 'slapping'],
                   ['hugging'],
                   ['shaking hands']]
# with open('/data/wangkun/project/datasets/Posetics/posetics400_raw/classes.txt', 'rb') as f:
#     posetics_label_list = pickle.load(f)
# posetics_label_list = np.load('/data/wangkun/project/datasets/Posetics/posetics400_raw/classes.txt')
with open('/data/wangkun/project/datasets/Posetics/posetics400_raw/classes.txt', 'r') as f:
    posetics_label_list_txt = f.read()
posetics_label_list = posetics_label_list_txt.split('\n')
posetics_label_list = np.array(posetics_label_list)
print(posetics_label_list)
posetics2ntu = []


def split_ntu_for_cross_dataset(which, which_name):
    chosen = which
    save_path = f'/data/wangkun/project/datasets/NTU_dataset/ntu/{which_name}'
    for set in ['CS', 'CV']:
        npz_data = np.load(f'/data/wangkun/project/datasets/NTU_dataset/ntu/NTU60_{set}.npz', mmap_mode='r')
        data_train = npz_data['x_train']
        label_train = np.where(npz_data['y_train'] > 0)[1]
        data_test = npz_data['x_test']
        label_test = np.where(npz_data['y_test'] > 0)[1]
        print(data_train.shape, label_train.shape, data_test.shape, label_test.shape)

        # start split
        new_data_train = []
        new_label_train = []
        new_data_test = []
        new_label_test = []
        for i in range(len(label_train)):
            if label_train[i] in chosen:
                new_data_train.append(data_train[i])
                new_label_train.append(chosen.index(label_train[i]))
        for i in range(len(label_test)):
            if label_test[i] in chosen:
                new_data_test.append(data_test[i])
                new_label_test.append(chosen.index(label_test[i]))
        new_data_train = np.array(new_data_train)
        new_label_train = np.array(new_label_train)
        new_data_test = np.array(new_data_test)
        new_label_test = np.array(new_label_test)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(f'{save_path}/NTU60_{set}.npz',
                 x_train=new_data_train, y_train=new_label_train,
                 x_test=new_data_test, y_test=new_label_test)
        new_label_list = []
        for i in chosen:
            new_label_list.append(ntu_label_list[i])
        new_label_list = np.array(new_label_list)
        print(new_label_list)
        np.save(f'{save_path}/label_list.npy', new_label_list)
        print(f'label_list.npy saved.')


def get_length(data):
    length = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
    return length


def split_posetics_for_cross_dataset(which, which_name):
    which_dict = {}
    for index, i in enumerate(which):
        if type(i) != list:
            which_dict[i] = index
        else:
            for j in i:
                which_dict[j] = index
    print(which_dict)

    save_path = f'/data/wangkun/project/datasets/Posetics/{which_name}'
    for set in ['train', 'val']:
        data = np.load(f'/data/wangkun/project/datasets/Posetics/posetics/{set}_data_joint.npy', mmap_mode='r')
        label_path = f'/data/wangkun/project/datasets/Posetics/posetics/{set}_label.pkl'
        with open(label_path, 'rb') as f:
            _, label = pickle.load(f)
        print(data.shape, len(label))
        new_data = []
        new_label = []
        for i in tqdm(range(len(label))):
            if label[i] in which_dict:
                if get_length(data[i, :, :, :, :2]) != 0:
                    new_data.append(data[i, :, :, :, :2])  # 2 persons
                    new_label.append(which_dict[label[i]])
        new_data = np.array(new_data)
        new_label = np.array(new_label)
        print(new_data.shape, new_label.shape)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f'{save_path}/{set}_data.npy', new_data)
        np.save(f'{save_path}/{set}_label.npy', new_label)
        print(f'{set}_data.npy saved. length: {len(new_data)}')


if __name__ == '__main__':
    # print(ntu_label_list.tolist())
    for i in ntu_label_chosen:
        # print(i)
        # print(ntu_label_list.tolist().index(i))
        ntu2posetics.append(ntu_label_list.tolist().index(i))

    # print(posetics_label_list.tolist())
    for i in posetics_chosen:
        if len(i) == 1:
            # print(i)
            # print(posetics_label_list.tolist().index(i[0]))
            posetics2ntu.append(posetics_label_list.tolist().index(i[0]))
        else:
            temp = []
            for j in i:
                # print(j)
                # print(posetics_label_list.tolist().index(j))
                temp.append(posetics_label_list.tolist().index(j))
            posetics2ntu.append(temp)

    print(ntu2posetics)
    print(ntu_label_chosen)
    print(posetics2ntu)
    print(posetics_chosen)
    which_dict = {}
    for index, i in enumerate(posetics2ntu):
        if type(i) != list:
            which_dict[i] = index
        else:
            for j in i:
                which_dict[j] = index
    print(which_dict)

    # split_ntu_for_cross_dataset(ntu2posetics, 'ntu2posetics')
    # npz_data = np.load(f'/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2posetics/NTU60_CS.npz', mmap_mode='r')
    # data_train = npz_data['x_train']
    # label_train = npz_data['y_train']
    # data_test = npz_data['x_test']
    # label_test = npz_data['y_test']
    # print(data_train.shape, label_train.shape, data_test.shape, label_test.shape)
    # fusion_data = np.concatenate((data_train, data_test), axis=0)
    # fusion_label = np.concatenate((label_train, label_test), axis=0)
    # print(fusion_data.shape, fusion_label.shape)
    # np.savez(f'/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2posetics/NTU60_N12.npz',
    #          x_train=fusion_data, y_train=fusion_label)

    # split_posetics_for_cross_dataset(posetics2ntu, 'posetics2ntu')
    # data_train = np.load(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/train_data.npy', mmap_mode='r')
    # label_train = np.load(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/train_label.npy')
    # data_val = np.load(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/val_data.npy', mmap_mode='r')
    # label_val = np.load(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/val_label.npy')
    # print(data_train.shape, label_train.shape, data_val.shape, label_val.shape)
    # fusion_data = np.concatenate((data_train, data_val), axis=0)
    # fusion_label = np.concatenate((label_train, label_val), axis=0)
    # print(fusion_data.shape, fusion_label.shape)
    # np.save(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/n12_data.npy', fusion_data)
    # np.save(f'/data/wangkun/project/datasets/Posetics/posetics2ntu/n12_label.npy', fusion_label)
