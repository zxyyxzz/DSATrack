from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.grm.utils import combine_tokens, recover_tokens

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # For original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_x = None
        self.pos_embed_z = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_sep_seg = False

        self.memory_bank = []

    def finetune_track(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # For template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def gen_masked_tokens(self, tokens, indices, alpha=0.4):
        drop_indices = [i for i in range(144) if i not in indices]
        tokens = tokens.copy()
        tokens[drop_indices] = alpha * tokens[drop_indices] + (1 - alpha) * 255
        #for memory visulization
        # new_tokens = tokens[keep_indices]
        # return tokens, new_tokens
        return tokens

    def recover_image(self, args):
        # image: (C, 196, 16, 16)
        # tokens, new_tokens = args
        tokens = args
        image = tokens.reshape(1, 12, 12, 16, 16, 3).swapaxes(2, 3).reshape(1, 192, 192, 3)
        #for memory visulization
        # return image, new_tokens.reshape(1, 12, 12, 16, 16, 3).swapaxes(2, 3).reshape(1, 192, 192, 3)
        return image

    def update_memory(self, z_clone, x, x_clone, templates_feat_len, search_feat_len, templates_num, prev_decision,
                      policy, templates_feat_len_org, B):
        for i, blk in enumerate(self.blocks):
            x, prev_decision, policy, topk_mem_index, _ = blk(x=x,
                                               templates_feat_len=templates_feat_len,
                                               search_feat_len=search_feat_len,
                                               templates_num=templates_num,
                                               prev_decision=prev_decision,
                                               policy=policy,
                                               dm=True,
                                               train_num=templates_num,
                                               templates_feat_len_org=templates_feat_len_org)

            if i == self.first_dm_layer and topk_mem_index is not None:
                topk_mem = []
                for j in range(B):
                    topk_mem_b = z_clone[j:j + 1, templates_feat_len_org:][:, topk_mem_index[j]]
                    topk_mem.append(topk_mem_b)
                topk_mem = torch.cat(topk_mem, dim=0)

                z = torch.cat((z_clone[:, :templates_feat_len_org], topk_mem), dim=1)
                x = combine_tokens(z, x_clone, mode=self.cat_mode)
                x = self.pos_drop(x)
                if not self.is_training:
                    del self.memory_bank[:]
                    self.memory_bank.append(z)
            if i == self.first_dm_layer:
                return x


    def forward_features(self, z, x, template_mask, search_feat_len, templates_feat_len_org, threshold, tgt_type, train_num, template_show, update=False, track_query=None, token_type="add", remove_rate_cur_epoch=1.0):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if z != None:
            if z.dim() !=5:
                z = z.unsqueeze(1)
            t, _, c, ZH, ZW = z.shape
            z = z.view(-1, c, ZH, ZW)
            z = self.patch_embed(z)
            z += self.pos_embed_z
            z1 = z.view(t, B, -1, z.shape[-1]).permute(1, 0, 2, 3)
            z = z1.reshape(B, -1, z.shape[-1])
            if not self.is_training:
                self.memory_bank.append(z)
        if not self.is_training:
            z = torch.cat(self.memory_bank, dim=1)
            t = z.size(1) // templates_feat_len_org
        z_clone = z.clone()
        templates_feat_len = z.shape[1]

        if self.add_cls_token:
            if token_type == "concat":
                new_query = self.cls_token.expand(B, -1, -1)
                query = new_query if track_query is None else torch.cat([new_query, track_query], dim=1)
                query = query + self.cls_pos_embed
            elif token_type == "add":
                query = self.cls_token if track_query is None else track_query + self.cls_token   # self.cls_token is init query
                query = query.expand(B, -1, -1)  # copy B times
                query = query + self.cls_pos_embed
        
        x = self.patch_embed(x)
        x += self.pos_embed_x
        x_clone = x.clone()
        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)

        x = self.pos_drop(x)

        templates_decisions = []
        class_scores = []
        prev_decision = torch.ones(B, templates_feat_len, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, 1, x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)

        T = 3
        new_templates_feat_len = templates_feat_len_org * T
        #memory updating
        if templates_feat_len > new_templates_feat_len:
            x = self.update_memory(z_clone, x, x_clone, templates_feat_len, search_feat_len, t, prev_decision,
                      policy, templates_feat_len_org, B)

        #ViT
        prev_decision = torch.ones(B, new_templates_feat_len, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, 1, x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.remove_layers and self.is_training:
                remove_rate = remove_rate_cur_epoch
            else:
                remove_rate = 1.0
            x, prev_decision, policy, topk_mem_index, mask_show = blk(x=x, templates_feat_len=new_templates_feat_len, search_feat_len=search_feat_len, templates_num=T,
                               prev_decision=prev_decision, policy=policy, templates_feat_len_org=templates_feat_len_org, remove_rate=remove_rate)
            if self.is_training and i in self.using_att_weights[0]:
                templates_decisions.append(prev_decision[:, :, 0])

        x = recover_tokens(x, mode=self.cat_mode)
        return self.norm(x), templates_decisions, class_scores

    def forward_features_separate(self, z, x, search=False):
        x = self.patch_embed(x)
        x += self.pos_embed_x
        z = self.patch_embed(z)
        z += self.pos_embed_z
        if not search:
            x = z
        x_clone = x.clone()

        x = self.pos_drop(x)

        templates_decisions = []
        search_decisions = []

        for i, blk in enumerate(self.blocks):
            x, prev_decision, templates_decision, search_decision, policy, topk_mem_index, prev_search_decision, mask_show = blk(x=x, template_mask=None, search_feat_len=None, templates_feat_len=None,
                              threshold=None, tgt_type=None, prev_decision=None, policy=None, prev_search_decision=None)

        return self.norm(x), templates_decisions, search_decisions

    def forward_features_combine(self, z, x):
        x = self.patch_embed(x)
        x += self.pos_embed_x
        z = self.patch_embed(z)
        z += self.pos_embed_z
        x = combine_tokens(z, x, mode=self.cat_mode)
        x_clone = x.clone()


        x = self.pos_drop(x)

        templates_decisions = []
        search_decisions = []

        for i, blk in enumerate(self.blocks):
            x, prev_decision, templates_decision, search_decision, policy, topk_mem_index, prev_search_decision, mask_show = blk(x=x, template_mask=None, search_feat_len=None, templates_feat_len=None,
                              threshold=None, tgt_type=None, prev_decision=None, policy=None, prev_search_decision=None)

        return self.norm(x), templates_decisions, search_decisions

    def forward(self, z, x, template_mask, search_feat_len, templates_feat_len, threshold, tgt_type, train_num, template_show, mode=None, update=False, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.

        Args:
            z (torch.Tensor): Template feature, [B, C, H_z, W_z].
            x (torch.Tensor): Search region feature, [B, C, H_x, W_x].

        Returns:
            x (torch.Tensor): Merged template and search region feature, [B, L_z+L_x, C].
            attn : None.
        """

        if mode=='separate':
            x, templates_decision, search_decision = self.forward_features_separate(z, x, kwargs['search'])
        elif mode=='combine':
            x, templates_decision, search_decision = self.forward_features_combine(z, x)
        else:
            x, templates_decision, class_scores = self.forward_features(z, x, template_mask, search_feat_len, templates_feat_len, threshold, tgt_type, train_num, template_show, update=update, track_query=kwargs['track_query'])
        return x, templates_decision, class_scores