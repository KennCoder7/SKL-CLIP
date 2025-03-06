import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


def text_prompt(classes_lst):
    # text_aug = [f"{{}}", f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}",
    #             f"{{}}, an action", f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}",
    #             f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
    #             f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
    #             f"The man is {{}}", f"The woman is {{}}", f"a skeleton of action {{}}", f"{{}}, a skeleton of action",
    #             f"Skeleton classification of {{}}", f"{{}}, a skeleton", f"{{}}, a skeleton of human action",
    #             ]
    # text_aug = [f"{{}}",
    #             f"Recognize the action: {{}}",
    #             f"A skeleton-based action recognition task: {{}}",
    #             f"The human skeleton is performing {{}}",
    #             f"Skeleton action of {{}}",
    #             f"Classify the skeleton action: {{}}",
    #             f"The human skeleton does {{}}",
    #             f"A video showing the skeleton performing {{}}",
    #             f"Human skeleton is performing an action: {{}}",
    #             f"Watch the skeleton performing {{}}",
    #             f"Can you identify the action of this skeleton: {{}}?",
    #             f"The skeleton shows the action {{}}",
    #             f"Skeleton video of {{}}",
    #             f"Recognizing a skeleton action: {{}}",
    #             f"A skeleton-based action video of {{}}",
    #             f"This skeleton is {{}}",
    #             f"The skeleton movement indicates {{}}",
    #             f"Watch this skeleton perform {{}}", ]
    text_aug = [f"{{}}",]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in classes_lst])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


class Model(nn.Module):
    def __init__(self,
                 arch,
                 pretrain,
                 clip_root=None,
                 device='cuda'
                 ):
        super(Model, self).__init__()
        self.model, _ = clip.load(arch, download_root=clip_root)
        if pretrain:
            if os.path.isfile(pretrain):
                print(("=> loading checkpoint '{}'".format(pretrain)))
                checkpoint = torch.load(pretrain)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint
            else:
                print(("=> no checkpoint found at '{}'".format(pretrain)))
        self.text_encoder = TextCLIP(self.model).to(device)
        self.device = device

    def forward(self, text):
        self.classes, self.num_text_aug, self.text_dict = text_prompt(text)
        text_inputs = self.classes.to(self.device)
        text_features = self.model.encode_text(text_inputs)  # bs*num_text_aug, 512
        text_features = text_features.view(-1, self.num_text_aug, text_features.size(1))  # bs, num_text_aug, 512
        return text_features


# def entity_alignment(A, B):
#     # 假设视图数目可以不同，但dimA和dimB相同，直接对dimA和dimB对齐。
#     # 如果dimA 和 dimB不同，可以通过线性层或PCA等手段来对齐维度。
#     if A.size(-1) != B.size(-1):
#         raise ValueError("Dimensionality of A and B should match for comparison.")
#
#     # Step 1: 聚合视图 (mean pooling)，将viewsA 和 viewsB 维度平均
#     A_aggregated = A.mean(dim=1)  # A: (nA, dimA)
#     B_aggregated = B.mean(dim=1)  # B: (nB, dimB)
#     # A_aggregated = A[:, 0]  # A: (nA, dimA)
#     # B_aggregated = B[:, 0]  # B: (nB, dimB)
#
#     # Step 2: 计算相似度 (余弦相似度)
#     # 使用 PyTorch 的 `cosine_similarity` 函数
#     # 首先将 A 和 B 中的实体扩展为可广播的形状以进行两两相似度计算
#     A_expanded = A_aggregated.unsqueeze(1)  # A_expanded: (nA, 1, dimA)
#     B_expanded = B_aggregated.unsqueeze(0)  # B_expanded: (1, nB, dimB)
#
#     # 计算每对实体的余弦相似度，dim=2 是沿着最后一个维度计算
#     similarity_matrix = F.cosine_similarity(A_expanded, B_expanded, dim=2)  # (nA, nB)
#
#     # softmax
#     # similarity_matrix = F.softmax(similarity_matrix, dim=1)
#
#     return similarity_matrix

def entity_alignment(A, B, method='mean', view_weight=None):
    """
    输入:
    - A: (nA, views, dim) 的张量，代表A中实体的特征
    - B: (nB, views, dim) 的张量，代表B中实体的特征
    - method: 聚合方式，默认为 'mean'，也可以选择 'sum'

    输出:
    - similarity_matrix: A中每个实体与B中每个实体的相似度矩阵
    """

    nA, viewsA, dimA = A.shape
    nB, viewsB, dimB = B.shape

    # 保证A和B有相同的views数
    assert viewsA == viewsB, "A and B must have the same number of views"

    if view_weight is None:
        view_weight = np.ones(viewsA)

    # 初始化相似度矩阵
    similarity_matrix = torch.zeros((nA, nB)).to(A.device)

    # 对每个视角进行相似度计算
    for v in range(viewsA):
        # 获取第v个视角的特征
        A_view = A[:, v, :]  # A 中第 v 个视角的特征 (nA, dim)
        B_view = B[:, v, :]  # B 中第 v 个视角的特征 (nB, dim)

        # 计算当前视角下A与B的相似度
        # similarity = torch.matmul(A_view, B_view.T)  # (nA, nB)
        similarity = F.cosine_similarity(A_view.unsqueeze(1), B_view.unsqueeze(0), dim=2)  # (nA, nB)

        # 将相似度累加到总相似度矩阵中
        similarity_matrix += view_weight[v] * similarity

    # 根据选择的聚合方法对视角进行聚合
    if method == 'mean':
        similarity_matrix /= viewsA  # 对所有视角求平均
    elif method == 'sum':
        pass  # 已经累加过，无需再操作

    # 选择Softmax归一化
    # similarity_matrix = F.softmax(similarity_matrix, dim=1)

    # 0-1归一化
    similarity_matrix = normalize_similarity(similarity_matrix)

    return similarity_matrix


def normalize_similarity(similarity_matrix):
    # 计算矩阵的最小值和最大值
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()

    # Min-Max归一化
    normalized_similarity = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized_similarity

def plot_similarity_matrix(softmax_similarity, nameA, nameB,
                           title='Similarity Matrix',
                           xlabel='Entities in B',
                           ylabel='Entities in A',
                           figsize=(12, 12)):
    # 设置绘图的大小
    plt.figure(figsize=figsize)

    # 使用seaborn的heatmap绘制相似度矩阵，annot=True显示每个格子的数值
    sns.heatmap(softmax_similarity, xticklabels=nameB, yticklabels=nameA, cmap='viridis', annot=True, fmt='.2f')

    # 设置轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 设置标题
    plt.title(title)

    # 为了避免标签被切割，自动调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()


def ntu12ANDposetics12():
    ntu_label_chosen = ['drink water', 'eat meal', 'brush teeth', 'brush hair', 'clapping', 'reading',
                        'kicking something',
                        'writing', 'shake head', 'punch or slap', 'hugging', 'shaking hands']
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
    # combine inhomogeneous part
    posetics_chosen = [i[0] if len(i) == 1 else i[0] + ' & ' + i[1] for i in posetics_chosen]
    # replace '(p)' with '' in posetics_chosen
    # posetics_chosen = [i.replace(' (p)', '') for i in posetics_chosen]
    # print(posetics_chosen)
    # exit()

    # model = Model('ViT-B/16', pretrain='/data/wangkun/project/ZSL/szsl/models/kinetic400/vit-b-16-32f.pt')
    model = Model('ViT-B/32', pretrain=None)
    model.eval()

    input = ntu_label_chosen
    output_ntu = model(input)
    print(output_ntu.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2kinetics/true_label.npz',
             label_embedding=output_ntu.cpu().detach().numpy(), label_name=np.array(ntu_label_chosen))

    input = posetics_chosen
    output_p = model(input)
    print(output_p.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/Posetics/posetics2ntu/true_label.npz',
             label_embedding=output_p.cpu().detach().numpy(), label_name=np.array(posetics_chosen))
    similarity_matrix = entity_alignment(output_ntu, output_p).to('cpu').detach().numpy()
    # print(similarity_matrix)
    plot_similarity_matrix(similarity_matrix, ntu_label_chosen, posetics_chosen,
                           title='Similarity Matrix between NTU12 and Posetics12',
                           xlabel='Posetics12',
                           ylabel='NTU12')


def ntu51ANDpku51():
    ntu_label_chosen = [
        'nod head or bow',
        'brush hair',
        'brush teeth',
        'check time from watch',
        'cheer up',
        'clapping',
        'cross hands in front',
        'drink water',
        'drop',
        'eat meal',
        'falling down',
        'giving object',
        'hand waving',
        'shaking hands',
        'hopping',
        'hugging',
        'jump up',
        'kicking',
        'kicking something',
        'phone call',
        'pat on back',
        'pick up',
        'play with phone or tablet',
        'point finger',
        'point at something',
        'punch or slap',
        'pushing',
        'put on hat or cap',
        'touch pocket',
        'reading',
        'rub two hands',
        'salute',
        'sit down',
        'stand up',
        'take off hat or cap',
        'take off glasses',
        'take off jacket',
        'reach into pocket',
        'taking photo',
        'tear up paper',
        'throw',
        'back pain',
        'chest pain',
        'headache',
        'neck pain',
        'type on keyboard',
        'fan self',
        'put on jacket',
        'put on glasses',
        'wipe face',
        'writing'
    ]
    pku_label = np.load('/data/wangkun/project/datasets/PKUMMD/pku_part1/label_list.npy', allow_pickle=True)


    model = Model('ViT-B/16', pretrain='/data/wangkun/project/ZSL/szsl/models/kinetic400/vit-b-16-32f.pt')
    # model = Model('ViT-B/32', pretrain=None)
    model.eval()

    input = ntu_label_chosen
    output_ntu = []
    with torch.no_grad():
        output_ntu = model(input)
    print(output_ntu.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2pkummd/true_label.npz',
             label_embedding=output_ntu.cpu().detach().numpy(), label_name=np.array(ntu_label_chosen))

    input = pku_label
    with torch.no_grad():
        output_p = model(input)
    # output_p = model(input)
    print(output_p.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/PKUMMD/pku_part1/true_label.npz',
             label_embedding=output_p.cpu().detach().numpy(), label_name=np.array(pku_label))
    similarity_matrix = entity_alignment(output_ntu, output_p).to('cpu').detach().numpy()
    # print(similarity_matrix)
    plot_similarity_matrix(similarity_matrix, ntu_label_chosen, pku_label,
                           title='Similarity Matrix between NTU51 and Pku51',
                           xlabel='Pku51',
                           ylabel='NTU51',
                           figsize=(24, 24))


def ntu60ANDpku51():
    ntu_label_chosen = np.load('/data/wangkun/project/datasets/NTU_dataset/label_list.npy', allow_pickle=True)
    ntu_label_chosen = ntu_label_chosen[:60]
    pku_label = np.load('/data/wangkun/project/datasets/PKUMMD/pku_part1/label_list.npy', allow_pickle=True)


    model = Model('ViT-B/16', pretrain='/data/wangkun/project/ZSL/szsl/models/kinetic400/vit-b-16-32f.pt')
    # model = Model('ViT-B/32', pretrain=None)
    model.eval()

    input = ntu_label_chosen
    output_ntu = []
    with torch.no_grad():
        output_ntu = model(input)
    print(output_ntu.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/NTU_dataset/ntu/true_label.npz',
             label_embedding=output_ntu.cpu().detach().numpy(), label_name=np.array(ntu_label_chosen))

    input = pku_label
    with torch.no_grad():
        output_p = model(input)
    # output_p = model(input)
    print(output_p.shape)
    print(input)
    np.savez('/data/wangkun/project/datasets/PKUMMD/pku_part1/true_label.npz',
             label_embedding=output_p.cpu().detach().numpy(), label_name=np.array(pku_label))
    similarity_matrix = entity_alignment(output_ntu, output_p).to('cpu').detach().numpy()
    # print(similarity_matrix)
    plot_similarity_matrix(similarity_matrix, ntu_label_chosen, pku_label,
                           title='Similarity Matrix between NTU60 and Pku51',
                           xlabel='Pku51',
                           ylabel='NTU60',
                           figsize=(24, 24))

if __name__ == '__main__':
    # ntu12ANDposetics12()
    # ntu51ANDpku51()
    ntu60ANDpku51()
