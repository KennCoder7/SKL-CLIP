import numpy as np
import torch
import torch.nn as nn
import os
import clip
from torch.nn import functional as F
from torchlight import import_class
from .utils.action_alignment import action_alignment
from .utils.motion_description import *
from .utils.DANN import GradientReversalLayer

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        # print(model.logit_scale.exp()) # 62.5014
        # exit()
        del model.visual
        self.clip = model
        # del self.clip.visual
        # print(self.clip.dtype)
        # exit()

    def forward(self, text):
        x = self.clip.token_embedding(text).type(torch.float16)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.type(torch.float16)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(torch.float16)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        # detect NaN
        if torch.isnan(x).any():
            print('NaN detected in text encoder')
            exit()

        return x


class TextEncoder(nn.Module):
    def __init__(self,
                 target_classes_lst,
                 support_classes_lst,
                 freeze_text_encoder=False,
                 unfreeze_layer=[],
                 text_prompt_type='fixed',
                 arch='ViT-B/16',
                 pretrain='datasets/vit-b-16-32f.pt',
                 clip_root='datasets/clip',
                 device='cuda',
                 froster=False,
                 unseen_class=None,
                 extra_support_txt=None,
                 ):
        super(TextEncoder, self).__init__()
        CLIP, _ = clip.load(arch, download_root=clip_root)
        # CLIP, _ = clip.available_models()[arch]

        if pretrain:
            if os.path.isfile(pretrain):
                print(("=> loading checkpoint '{}'".format(pretrain)))
                checkpoint = torch.load(pretrain)
                CLIP.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint
            else:
                print(("=> no checkpoint found at '{}'".format(pretrain)))
        # self.model = CLIP.encode_text
        self.model = TextCLIP(CLIP)
        del CLIP
        if len(unfreeze_layer) > 0:
            freeze_text_encoder = True
            # unfreeze = ['clip.positional_embedding', 'clip.text_projection', 'clip.token_embedding.weight',
            #             'clip.ln_final.weight', 'clip.ln_final.bias']
            unfreeze = ['clip.text_projection', 'clip.ln_final.weight', 'clip.ln_final.bias']
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if len(unfreeze_layer) > 0 and name in unfreeze:
                    param.requires_grad = True
                    print('unfreeze', name)
                if len(name.split('.')) > 3 and name.split('.')[3] in unfreeze_layer:
                    param.requires_grad = True
                    print('unfreeze', name)

        self.froster = froster
        # assert froster
        if froster:
            assert not freeze_text_encoder or len(unfreeze_layer) > 0
            CLIP_, _ = clip.load(arch, download_root=clip_root)
            # pretrain = '/data/wangkun/project/ZSL/szsl/models/kinetic400/vit-b-16-32f.pt'
            if pretrain:
                if os.path.isfile(pretrain):
                    print(("=> loading checkpoint '{}'".format(pretrain)))
                    checkpoint = torch.load(pretrain)
                    CLIP_.load_state_dict(checkpoint['model_state_dict'])
                    del checkpoint
                else:
                    print(("=> no checkpoint found at '{}'".format(pretrain)))
            self.freeze_model = TextCLIP(CLIP_)
            for param in self.freeze_model.parameters():
                param.requires_grad = False
            del CLIP_
            self.mlp_froster = nn.Sequential(nn.Linear(512, 512),
                                             nn.GELU(),
                                             nn.Linear(512, 512),
                                             )
            # set the weights and bias of mlp_froster to torch.float16
            for param in self.mlp_froster.parameters():
                param.data = param.data.type(torch.float16)

        self.device = device

        self.text_prompt_type = text_prompt_type
        self.unseen_class = unseen_class
        if unseen_class is not None:
            print(f"Unseen class: {unseen_class}")
            self.classes_gzsl, self.num_text_aug_gzsl, self.text_dict_gzsl = self.text_prompt(target_classes_lst)
            self.gszl_num_class = len(target_classes_lst)
            unseen_target_classes_lst = [c for i, c in enumerate(target_classes_lst) if i in unseen_class]
            # target_classes_lst = [c for i, c in enumerate(target_classes_lst) if i not in unseen_class]
            self.unseen_num_class = len(unseen_target_classes_lst)
            self.classes_unseen, self.num_text_aug_unseen, self.text_dict_unseen = self.text_prompt(
                unseen_target_classes_lst)

        self.target_num_class = len(target_classes_lst)
        self.support_num_class = len(support_classes_lst)
        self.target_classes_lst = target_classes_lst
        self.support_classes_lst = support_classes_lst
        if extra_support_txt:
            support_classes_lst = [f'{extra_support_txt} {c}' for c in support_classes_lst]
        self.classes_target, self.num_text_aug_target, self.text_dict_target = self.text_prompt(target_classes_lst)
        self.classes_support, self.num_text_aug_support, self.text_dict_support = self.text_prompt(support_classes_lst)

        self.embedding_text_classes()

    def text_prompt(self, classes_lst, classes_desc=None, classes_shape=None):
        if self.text_prompt_type == 'fixed':
            text_aug = [f"{{}}", f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}",
                        f"{{}}, an action", f"{{}} this is an action", f"{{}}, a video of action",
                        f"Playing action of {{}}",
                        f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                        f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                        f"The man is {{}}", f"The woman is {{}}"
                        ]
            # text_aug = [f"{{}}", f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}",
            #             f"{{}}, an action", f"{{}} this is an action", f"{{}}, a video of action",
            #             f"Playing action of {{}}",
            #             f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
            #             f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
            #             f"The man is {{}}", f"The woman is {{}}", f"a skeleton of action {{}}",
            #             f"a skeleton motion of {{}} ", f"a skeleton data of {{}} ",
            #             f"a skeleton pose of {{}} ", f"the pose estimation of {{}} ", f"a pose of action {{}} ",
            #             ]
            # text_aug = [f"a photo of action {{}}",]
            text_dict = {}
            num_text_aug = len(text_aug)

            for ii, txt in enumerate(text_aug):
                text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in classes_lst])

            classes = torch.cat([v for k, v in text_dict.items()])

            return classes, num_text_aug, text_dict

        elif self.text_prompt_type == 'learned':
            raise ValueError(f"Text prompt type {self.text_prompt_type} is not supported.")
        else:
            raise ValueError(f"Text prompt type {self.text_prompt_type} is not supported.")

    def forward(self, label, Trg_Spt):
        if Trg_Spt:
            text_dict = self.text_dict_target
            num_text_aug = self.num_text_aug_target
        else:
            text_dict = self.text_dict_support
            num_text_aug = self.num_text_aug_support
        text_id = torch.randint(0, num_text_aug, (label.size(0),)).numpy()  # b
        text = torch.stack([text_dict[text_id[i]][label[i]] for i in range(label.size(0))]).to(
            label.device)  # b, 1, 77
        text = text.view(-1, text.size(-1))  # b, 77
        # text = self.text_classes[random_aug + label * self.num_text_aug]
        text_output = self.model(text)  # b, 512
        text_output_froster = self.freeze_model(text) if self.froster else None
        text_output_stu = text_output + 0.1 * self.mlp_froster(text_output) if self.froster else None
        return text_output, text_output_froster, text_output_stu

    def _embedding_text_classes(self, Trg_Spt, unseen=False, gzsl=False):
        assert self.eval()
        if Trg_Spt:
            if unseen:
                assert self.unseen_class is not None
                if gzsl:
                    text = self.classes_gzsl.to(self.device)
                    num_class = self.gszl_num_class
                else:
                    text = self.classes_unseen.to(self.device)
                    num_class = self.unseen_num_class
            else:
                text = self.classes_target.to(self.device)
                num_class = self.target_num_class
        else:
            text = self.classes_support.to(self.device)
            num_class = self.support_num_class
        with torch.no_grad():
            text_batch = []
            for i in range(len(text) // num_class):
                input_text = text[i * num_class:(i + 1) * num_class]
                output_text = self.model(input_text)
                text_batch.append(output_text)
            text = torch.cat(text_batch, dim=0)
        return text

    def embedding_text_classes(self):
        self.classes_target_embedding = self._embedding_text_classes(Trg_Spt=True)
        self.classes_support_embedding = self._embedding_text_classes(Trg_Spt=False)
        if self.unseen_class is not None:
            self.classes_unseen_embedding = self._embedding_text_classes(Trg_Spt=True, unseen=True)
            self.classes_gzsl_embedding = self._embedding_text_classes(Trg_Spt=True, unseen=False, gzsl=True)

    def sim_eval(self, feats, Trg_Spt, unseen=False, gzsl=False):
        if Trg_Spt:
            if unseen:
                assert self.unseen_class is not None
                if gzsl:
                    text = self.classes_gzsl_embedding.to(self.device)
                    num_text_aug = self.num_text_aug_gzsl
                else:
                    text = self.classes_unseen_embedding.to(self.device)
                    num_text_aug = self.num_text_aug_unseen
            else:
                text = self.classes_target_embedding.to(self.device)
                num_text_aug = self.num_text_aug_target
        else:
            text = self.classes_support_embedding.to(self.device)
            num_text_aug = self.num_text_aug_support
        # feats = F.normalize(feats, dim=1)
        feats /= feats.norm(dim=-1, keepdim=True)
        text /= text.norm(dim=-1, keepdim=True)
        feats = feats.type(torch.float16)
        text = text.type(torch.float16)
        similarity = (100.0 * feats @ text.T)
        similarity = similarity.view(feats.size(0), num_text_aug, -1).softmax(dim=-1)
        similarity = similarity.mean(dim=1, keepdim=False)
        logits_clip = similarity
        return logits_clip


class Model(nn.Module):
    def __init__(self,
                 encoder_args,
                 target_label_info,
                 support_label_info,
                 support_encoder_args=None,
                 shared_encoder=True,
                 pretrained_path=None,
                 device='cuda',
                 latent_dim=512,
                 feature_dim=256 * 2,
                 extra_fc=False,
                 extra_data_view=True,
                 extra_txt_view=True,
                 extra_txt_data_view=False,
                 freeze_text_encoder=False,
                 unfreeze_text_layer=[],
                 froster=True,
                 baseline=False,
                 clip_pretrain='datasets/vit-b-16-32f.pt',
                 text_prompt_type='fixed',
                 extra_support_txt=None,
                 unseen_class=None,
                 dann_lambda=0.1,
                 shared_cls=False,
                 ):
        super(Model, self).__init__()
        self.device = device

        # skeleton encoder
        backbone = import_class(encoder_args.pop('backbone'))
        if support_encoder_args is None:
            support_encoder_args = encoder_args.copy()
        self.encoder = nn.ModuleDict({
            'target': backbone(**encoder_args),
        })
        self.encoder['support'] = self.encoder['target'] if shared_encoder else backbone(**support_encoder_args)

        # text encoder
        if target_label_info == 'ntu60':
            target_label_info = ntu60_classes_names
        elif target_label_info == 'pku51':
            target_label_info = pku51_classes_names
        elif target_label_info == 'ntu51':
            target_label_info = ntu51_classes_names
        elif target_label_info == 'ntu2sbu':
            target_label_info = ntu2sbu_classes_names
        elif target_label_info == 'ntu2pku':
            target_label_info = pku51_classes_names
        elif target_label_info == 'ntu2kinetics':
            target_label_info = ntu2kinetics_classes_names
        else:
            raise ValueError(f"Target label info {target_label_info} is not supported.")
        if support_label_info == 'pku51':
            support_label_info = pku51_classes_names
        elif support_label_info == 'ntu51':
            support_label_info = ntu51_classes_names
        elif support_label_info == 'ntu60':
            support_label_info = ntu60_classes_names
        elif support_label_info == 'pku51_rename':
            support_label_info = pku51_classes_rename
        elif support_label_info == 'ntu60_rename':
            support_label_info = ntu60_classes_rename
        elif support_label_info == 'pku51_desc':
            support_label_info = pku51_classes_description
        elif support_label_info == 'ntu60_desc':
            support_label_info = ntu60_classes_description
        elif support_label_info == 'ntu2sbu':
            support_label_info = ntu2sbu_classes_names
        elif support_label_info == 'ntu2pku':
            support_label_info = pku51_classes_names
        elif support_label_info == 'ntu2kinetics':
            support_label_info = ntu2kinetics_classes_names
        else:
            raise ValueError(f"Support label info {support_label_info} is not supported.")
        self.target_label_name = target_label_info
        self.support_label_name = support_label_info
        if baseline:
            assert not extra_data_view
            assert not extra_txt_view
        else:
            self.text_encoder = TextEncoder(self.target_label_name,
                                            self.support_label_name,
                                            freeze_text_encoder,
                                            unfreeze_text_layer,
                                            froster=froster,
                                            device=device,
                                            pretrain=clip_pretrain,
                                            text_prompt_type=text_prompt_type,
                                            unseen_class=unseen_class,
                                            extra_support_txt=extra_support_txt,
                                            )

        # print(self.text_encoder)
        # exit()
        # classifier
        self.target_num_class = len(self.target_label_name)
        # if unseen_class is not None:
        #     self.target_num_class -= len(unseen_class)
        self.support_num_class = len(self.support_label_name)
        # self.num_class = len(self.label_name)
        self.classifier = nn.ModuleDict({
            'target': nn.Linear(feature_dim, self.target_num_class),
            'support': nn.Linear(feature_dim, self.support_num_class),
            'domain': nn.Linear(latent_dim, 2),
        })
        if shared_cls:
            assert self.target_num_class == self.support_num_class
            self.classifier['support'] = self.classifier['target']
        self.extra_fc = extra_fc
        if self.extra_fc:
            self.fc = nn.Sequential(nn.Linear(feature_dim, latent_dim),
                                    nn.ReLU(),
                                    )

        self.latent_dim = latent_dim
        self.pretrained_path = pretrained_path

        self.extra_data_view = extra_data_view
        self.extra_txt_view = extra_txt_view
        self.extra_txt_data_view = extra_txt_data_view
        self.unseen_class = unseen_class

        self.grl = GradientReversalLayer(dann_lambda)

    def load_pretrained_model(self, pretrained_path=None):
        if pretrained_path or self.pretrained_path:
            pretrained_path = pretrained_path if pretrained_path else self.pretrained_path
            if type(pretrained_path) is dict:
                self.encoder.load_pretrained_model(pretrained_path['target'])
                self.encoder.load_pretrained_model(pretrained_path['support'])
            else:
                self.encoder.load_pretrained_model(pretrained_path)
                self.encoder.load_pretrained_model(pretrained_path)
            return True
        return False

    def action_alignment(self):
        assert self.eval()
        with torch.no_grad():
            text_classes_target = self.text_encoder.classes_target_embedding
            text_classes_support = self.text_encoder.classes_support_embedding
            text_classes_target = text_classes_target.reshape(-1, self.target_num_class, 512).mean(dim=0).unsqueeze(1)
            text_classes_support = text_classes_support.reshape(-1, self.support_num_class, 512).mean(dim=0).unsqueeze(1)
            similarity_matrix_with_prompts = action_alignment(text_classes_target,
                                                              text_classes_support,
                                                              method='mean',
                                                              norm=False)
        return similarity_matrix_with_prompts, self.target_label_name, self.support_label_name

    def _share_params(self):
        for name, layer in self.encoder.named_children():
            # print(name)
            if name in self.partial_shared_params:
                setattr(self.encoder, name, layer)
                # print('share', name)

    def forward(self, Trg_Spt, x, xe=None, y=None, unseen=False, txt=True, gzsl=False):
        N, C, T, V, M = x.shape
        if Trg_Spt:
            classifier = self.classifier['target']
            encoder = self.encoder['target']
        else:
            classifier = self.classifier['support']
            encoder = self.encoder['support']
        if xe is None:
            _, feats = encoder(x)
            cls = classifier(feats)
            feats = self.fc(feats) if self.extra_fc else feats
            if self.extra_txt_view:
                txt_sim = self.text_encoder.sim_eval(feats, Trg_Spt, unseen, gzsl)
            else:
                txt_sim = None
            return cls, txt_sim

        if self.extra_txt_view and txt:
            if self.extra_txt_data_view:
                assert self.extra_data_view
                y = torch.cat([y, y], dim=0)
            y_txt, y_txt_froster, y_txt_stu = self.text_encoder(y, Trg_Spt)
        else:
            y_txt, y_txt_froster, y_txt_stu = None, None, None

        _, emb = encoder(x)
        cls = classifier(emb)
        emb_norm = emb if not self.extra_fc else self.fc(emb)
        cls_domain = self.classifier['domain'](self.grl(emb_norm))

        if self.extra_data_view:
            _, emb_e = encoder(xe)
            cls_e = classifier(emb_e)
            emb_e_norm = emb_e if not self.extra_fc else self.fc(emb_e)
            cls_e_domain = self.classifier['domain'](self.grl(emb_e_norm))
        else:
            emb_e_norm = emb_norm
            cls_e_domain = cls_domain
            cls_e = cls

        return cls, cls_e, emb_norm, emb_e_norm, y_txt, y_txt_froster, y_txt_stu, cls_domain, cls_e_domain

    def extract_feature(self, xs, xt=None):
        with torch.no_grad():
            if xt is None:
                cls, emb = self.encoder(xs)
                return cls, emb
            cls_s, emb_s = self.encoder(xs)
            emb_s_norm = F.normalize(emb_s, dim=1)
            cls_t, emb_t = self.encoder(xt)
            emb_t_norm = F.normalize(emb_t, dim=1)
        return cls_s, cls_t, emb_s_norm, emb_t_norm


if __name__ == '__main__':
    class_name = 'hit with object'
    token = clip.tokenize(class_name)
    print(token)
