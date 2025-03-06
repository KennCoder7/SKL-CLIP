import numpy as np
from .nacl_utils import calc_topo_weights_with_components_idx, calc_knn_graph
from scipy.stats import mode
import torch


def remove_outlier(data, feats, label, pred, num_classes, k_cc=3):

    train_gt_labels = label.cpu().numpy().astype(np.int64).tolist()
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    train_pred_labels = pred.astype(np.int64).tolist()
    train_labels_onehot = np.eye(num_classes)[train_gt_labels]
    # train_labels_onehot = torch.from_numpy(train_labels_onehot).float().to(feats.device)
    ntrain = len(train_gt_labels)
    _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(ntrain, train_labels_onehot,
                                                                feats,
                                                                train_gt_labels, train_pred_labels, k=k_cc,
                                                                use_log=False, cp_opt=3, nclass=num_classes)
    # --- update largest connected component ---
    big_comp = set()
    cur_big_comp = list(set(range(ntrain)) - set(idx_of_comp_idx2))
    big_comp = big_comp.union(set(cur_big_comp))
    # --- remove outliers in largest connected component ---
    big_com_idx = list(big_comp)

    feats_big_comp = feats[big_com_idx]
    labels_big_comp = np.array(train_gt_labels)[big_com_idx]

    knnG_list = calc_knn_graph(feats_big_comp, k=k_cc)

    knnG_list = np.array(knnG_list)
    knnG_shape = knnG_list.shape
    knn_labels = labels_big_comp[knnG_list.ravel()]
    knn_labels = np.reshape(knn_labels, knnG_shape)

    majority, counts = mode(knn_labels, axis=1)
    majority = majority.ravel()
    # counts = counts.ravel()
    non_outlier_idx = np.where(majority == labels_big_comp)[0]
    outlier_idx = np.where(majority != labels_big_comp)[0]
    outlier_idx = np.array(list(big_comp))[outlier_idx]
    print(f'num of outliers: {len(outlier_idx)}')
    big_comp = np.array(list(big_comp))[non_outlier_idx]
    big_comp = set(big_comp.tolist())
    conf_label = [train_gt_labels[index] for index in big_comp]

    conf_data = [data[index] for index in big_comp]
    conf_data = np.array(conf_data)
    conf_label = np.array(conf_label)
    return conf_data, conf_label