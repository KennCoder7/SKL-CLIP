import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class SupConLoss(torch.nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR
#     From: https://github.com/HobbitLong/SupContrast"""
#
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature
#
#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#
#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)
#
#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)
#
#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()
#
#         return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning with weighted positive pairs."""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model, with optional weights for contrastive pairs.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i.
            weights: weight matrix of shape [bsz, bsz], weights_{i,j} gives the
                weight of the positive pair between sample i and sample j.

        Returns:
            A loss scalar.
        """

        device = features.device

        features = F.normalize(features, dim=2)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print(mask.shape)
        # Optionally add weights to positive pairs
        if weights is not None:
            weights = weights.detach().float()
            weights = weights.repeat(anchor_count, contrast_count)
            if weights.shape != mask.shape:
                raise ValueError("Weights shape must match mask shape.")
            weights = weights.to(device)
        else:
            weights = torch.ones_like(mask).to(device)  # Default: uniform weight

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute weighted mean of log-likelihood over positive pairs
        mean_log_prob_pos = (weights * mask * log_prob).sum(1) / (weights * mask).sum(1)

        # Loss calculation
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SimDistillLoss(torch.nn.Module):
    def __init__(self, norm='None'):
        super(SimDistillLoss, self).__init__()
        self.sim = F.cosine_similarity
        self.norm = norm

    def forward(self, view, simY):
        N_views = view.shape[1]
        loss = 0
        for i in range(N_views):
            sim = self.sim(view[:, i, :].unsqueeze(1), view[:, i, :].unsqueeze(0), dim=2)
            if self.norm == 'softmax':
                sim = F.softmax(sim, dim=1)
                simY = F.softmax(simY, dim=1)
            elif self.norm == 'norm':
                sim = normalize_similarity(sim)
                simY = normalize_similarity(simY)
            loss += F.mse_loss(sim, simY)

        return loss


def normalize_similarity(similarity_matrix):
    # 计算矩阵的最小值和最大值
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()

    # Min-Max归一化
    normalized_similarity = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized_similarity


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        # print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


class CLIPLoss(nn.Module):
    def __init__(self, loss_type='ce', logit_scale=1.):
        super(CLIPLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'ce':
            self.logit_scale = logit_scale
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'kl':
            self.logit_scale = logit_scale
            self.criterion = KLLoss()
        elif loss_type == 'cl':
            self.logit_scale = 64.
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise NotImplementedError

    def forward(self, skl, txt, labels=None):
        if self.loss_type == 'ce':
            ground_truth = torch.arange(len(skl)).to(skl.device)
            # skl = F.normalize(skl, dim=-1)
            # txt = F.normalize(txt, dim=-1)
            # skl = skl / skl.norm(dim=-1, keepdim=True)
            # txt = txt / txt.norm(dim=-1, keepdim=True)
        elif self.loss_type == 'kl':
            ground_truth = torch.tensor(gen_label(labels)).float().to(skl.device)
            skl = skl / skl.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        elif self.loss_type == 'cl':
            target = torch.zeros(skl.shape[0], dtype=torch.long).to(skl.device)
            skl = skl.type(torch.float16)
            txt = txt.type(torch.float16)
            skl = skl / skl.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            logits1 = self.logit_scale * (skl @ txt.t())
            logits2 = self.logit_scale * (txt @ skl.t())
            loss1 = self.criterion(logits1, target)
            loss2 = self.criterion(logits2, target)
            loss = (loss1 + loss2) / 2.
            return loss
        else:
            raise NotImplementedError

        skl = skl.type(torch.float16)
        txt = txt.type(torch.float16)
        logits_per_skl = self.logit_scale * skl @ txt.t()
        logits_per_txt = self.logit_scale * txt @ skl.t()
        loss_skl = self.criterion(logits_per_skl, ground_truth)
        loss_txt = self.criterion(logits_per_txt, ground_truth)
        return (loss_skl + loss_txt) / 2.


# def gen_label(labels):
#     num = len(labels)
#     gt = np.zeros(shape=(num, num))
#     for i, label in enumerate(labels):
#         for k in range(num):
#             if labels[k] == label:
#                 gt[i, k] = 1
#     return gt


def gen_label(labels):
    # 转换为 NumPy 数组（如果不是的话）
    labels = np.array(labels.cpu())
    num = len(labels)

    # 使用广播和布尔索引生成标签矩阵
    gt = (labels[:, None] == labels).astype(int)  # (num, num) 的布尔数组转为整数

    return gt
# def create_logits(x1, x2, logit_scale):
#     x1 = x1 / x1.norm(dim=-1, keepdim=True)
#     x2 = x2 / x2.norm(dim=-1, keepdim=True)
#
#     # cosine similarity as logits
#     logits_per_x1 = logit_scale * x1 @ x2.t()
#     logits_per_x2 = logit_scale * x2 @ x1.t()
#
#     # shape = [global_batch_size, global_batch_size]
#     return logits_per_x1, logits_per_x2
#
# def loss_clip_(logits_per_skele, logits_per_text, ground_truth):
#     loss_skele = KLLoss()(logits_per_skele, ground_truth)
#     loss_text = KLLoss()(logits_per_text, ground_truth)
#     return (loss_skele + loss_text) / 2


def weighted_cross_entropy_loss(predictions, targets, confidences):
    """
    计算加权交叉熵损失，置信度越高的样本对损失的贡献越大。

    Args:
        predictions (torch.Tensor): 模型的预测输出，形状为 (batch_size, num_classes)。
        targets (torch.Tensor): 伪标签，形状为 (batch_size,)。
        confidences (torch.Tensor): 每个样本的置信度，形状为 (batch_size,)。

    Returns:
        torch.Tensor: 加权的交叉熵损失标量值。
    """
    # 计算标准交叉熵损失
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')  # 计算每个样本的交叉熵损失，不进行平均

    # 加权交叉熵损失
    weighted_loss = confidences * ce_loss  # 置信度作为每个样本的权重
    return weighted_loss.mean()  # 返回加权损失的平均值