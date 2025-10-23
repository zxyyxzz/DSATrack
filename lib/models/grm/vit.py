"""
Vision Transformer (ViT) in PyTorch.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import named_apply, adapt_input_conv
from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.grm.base_backbone import BaseBackbone
from lib.utils.blocks import Encoder4Dgnn
keep_index = None

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0., residual=False, feature_size_z=64, feature_size_x=256, relevance_attn=False, layer=None, using_att_weights=False, keep_num=None, early=True, norm=False,training=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relevance_attn = relevance_attn
        self.using_att_weights = using_att_weights
        self.norm = norm
        self.keep_num = keep_num
        self.early = early
        self.layer = layer

        self.is_training = training
        if self.relevance_attn:
            if self.using_att_weights:
                self.cross_predict_temp = nn.Sequential(
                    nn.Linear(self.num_heads, 384),
                    nn.GELU(),
                    nn.Linear(384, 192),
                    nn.GELU(),
                    nn.Linear(192, 2),
                    nn.LogSoftmax(dim=-1)
                )

                if self.norm:
                    self.norm_temp = nn.BatchNorm1d(self.num_heads)
            else:
                self.cross_predict_temp = nn.Sequential(
                    nn.Linear(dim*2, 384),
                    nn.GELU(),
                    nn.Linear(384, 192),
                    nn.GELU(),
                    nn.Linear(192, 2),
                )
            self.consensus_4D = Encoder4Dgnn(  # Encoder for conv_5
                corr_levels=(self.num_heads, self.num_heads),
                layer_num=1,
                feature_size_z=feature_size_z,
                feature_size_x=feature_size_x,
                residual=residual
            )

    def softmax_with_policy(self, attn, args, eps=1e-6, dm=False, new_N=None):

        B, H, N, N = attn.size()
        if dm:
            N=new_N

        if self.relevance_attn:
            query_templates_decision, query_search_decision, key_templates_decision, key_search_decision = args

            group1 = query_templates_decision.reshape(B, 1, N, 1) @ key_templates_decision.reshape(B, 1, 1, N)


            group2 = query_search_decision.reshape(B, 1, N, 1) @ key_search_decision.reshape(B, 1, 1, N)

            attn_policy = group1 + group2
            if dm:
                return None, attn_policy

        else:
            attn_policy = args
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye

        #For stable training
        max_att, _ = torch.max(attn, dim=-1, keepdim=True)
        attn = attn - max_att
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att), attn_policy

    def batch_index_select(self, x, idx):
        if len(x.size()) == 3:
            B, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
            return out
        elif len(x.size()) == 2:
            B, N = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        elif len(x.size()) == 4:
            B, H, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.permute(0, 2, 1, 3).reshape(B * N, H, C)[idx.reshape(-1)].reshape(B, N_new, H, C).permute(0, 2, 1, 3)
            return out

        else:
            raise NotImplementedError

    def forward(self, x, z_size, x_size, t, prev_decision=None, policy=None, noNorm_x=None, dm=False, train_num=1, templates_feat_len_org=64):
        B, N, C = x.shape
        decision = prev_decision
        filter_x = None
        topk_mem_index = None
        mask_show = None
        global keep_index

        if self.relevance_attn:
            z0_size = int(z_size / t)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn_pre = attn
            if self.is_training:
                attn_pre = attn*policy
            if self.using_att_weights:
                attn_zx = attn_pre[:, :, :-x_size, -x_size:]
                if self.is_training:
                    attn_zx = attn_zx.reshape(-1, self.num_heads, t, z0_size, x_size)                 
                    attn_zx = self.consensus_4D(attn_zx)
                    attn_zx = attn_zx.reshape(-1, self.num_heads, z_size, x_size)
                else:
                    t = attn_zx.size(2) // templates_feat_len_org
                    attn_zx_cur = attn_zx[:, :, :t * templates_feat_len_org, :]
                    attn_zx_cur = attn_zx_cur.reshape(-1, self.num_heads, t, z0_size, x_size)
                    attn_zx_cur = self.consensus_4D(attn_zx_cur)
                    attn_zx_cur = attn_zx_cur.reshape(-1, self.num_heads, t * templates_feat_len_org, x_size)
                    attn_zx = torch.cat((attn_zx_cur, attn_zx[:, :, t * templates_feat_len_org:, :]), dim=2)
                tgt_rep = attn_zx.mean(-1)
                if self.norm:
                    tgt_rep = self.norm_temp(tgt_rep).permute(0, 2, 1)
                else:
                    tgt_rep = tgt_rep.permute(0, 2, 1)

                relevance = self.cross_predict_temp(tgt_rep)

            if self.is_training:
                cross_prediction = relevance
                cross_decision = F.gumbel_softmax(cross_prediction, hard=True)
                cross_decision = cross_decision[:, :, 0:1] 
                score = cross_prediction[:, :, 0]

                search_decision = torch.ones(B, x_size, 1, dtype=x.dtype,
                                             device=x.device)
                blank_search_decision = torch.zeros(B, x_size, 1, dtype=x.dtype,
                                                    device=x.device)

                blank_templates_decision = torch.zeros(B, N - x_size, 1, dtype=x.dtype,
                                                       device=x.device)

                if self.layer == 3 and dm:
                    cross_decision = torch.ones(B, N - x_size, 1, dtype=x.dtype,
                                             device=x.device)
                    blank_templates_decision = torch.zeros(B, N - x_size, 1, dtype=x.dtype,
                                                       device=x.device)
                else:
                    cross_decision = cross_decision * prev_decision

                query_templates_decision = torch.cat((cross_decision, blank_search_decision), dim=1)
                query_search_decision = torch.cat((blank_templates_decision, search_decision), dim=1)

                key_templates_decision = torch.cat((cross_decision, search_decision), dim=1)
                key_search_decision = torch.cat((cross_decision, search_decision), dim=1)

                qk_decisions = [query_templates_decision, query_search_decision, key_templates_decision,
                                key_search_decision]

                decision = cross_decision
                attn, policy = self.softmax_with_policy(attn, qk_decisions, dm=dm, new_N=train_num*(templates_feat_len_org)+x_size)
                # get topk_mem
                if self.layer == 3 and dm:
                    deform_mem_score = score[:, templates_feat_len_org:]
                    topk_mem_index = torch.argsort(deform_mem_score, dim=1, descending=True)[:, :templates_feat_len_org*2]

                    return x, filter_x, decision, policy, topk_mem_index, None
                # attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
            else:
                # During inference
                cross_prediction = relevance
                score = cross_prediction[:, :, 0]

                #get topk_mem
                if self.layer == 3 and dm:
                    dynamic_score = score[:, templates_feat_len_org:]
                    topk_mem_index = torch.argsort(dynamic_score, dim=1, descending=True)[:, :templates_feat_len_org*2]
                    return x, filter_x, decision, policy, topk_mem_index, None


                token_num = score.shape[1]
                if token_num >= self.keep_num:

                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :self.keep_num]

                    templates_attn = torch.gather(attn, dim=2, index=keep_policy.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, N).expand(-1, self.num_heads, -1, -1))# B, NH, KN, N
                    templates_search_attn = templates_attn[:, :, :, -x_size:]
                    templates_templates_attn = torch.gather(templates_attn, dim=3, index=keep_policy.unsqueeze(1).unsqueeze(2).expand(-1, -1, templates_attn.shape[2], -1).expand(-1, self.num_heads, -1, -1))
                    templates_attn = torch.cat((templates_templates_attn, templates_search_attn), dim=-1)

                    search_attn = attn[:, :, -x_size:]
                    search_templates_attn = torch.gather(search_attn, dim=3, index=keep_policy.unsqueeze(1).unsqueeze(2).expand(-1, -1, search_attn.shape[2], -1).expand(-1, self.num_heads, -1, -1))# B, NH, SN, KN
                    search_attn = torch.cat((search_templates_attn, search_attn[:, :, :, -x_size:]), dim=-1)

                    attn = torch.cat((templates_attn, search_attn), dim=2)

                    templates_x = self.batch_index_select(noNorm_x, keep_policy)
                    filter_x = torch.cat((templates_x, noNorm_x[:, -x_size:, :]), dim=1)

                    templates_v = self.batch_index_select(v, keep_policy)
                    v = torch.cat((templates_v, v[:, :, -x_size:]), dim=2)
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, v.shape[2], C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                else:
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)


        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.is_training:
                attn, policy = self.softmax_with_policy(attn, policy)
            else:
                attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x, filter_x, decision, policy, topk_mem_index, mask_show


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual=False, feature_size_z=64, feature_size_x=256,
                 relevance_attn=False, using_att_weights=False, keep_num=None, layer=None, training=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              residual=residual, feature_size_z=feature_size_z, feature_size_x=feature_size_x,
                              relevance_attn=relevance_attn, layer=layer, using_att_weights=using_att_weights, keep_num=keep_num, training=training)
        # Note: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, templates_feat_len, search_feat_len, templates_num, prev_decision, policy, dm=False, train_num=1, templates_feat_len_org=64):
        feat, filter_x, decision, policy, topk_mem_index, mask_show = self.attn(self.norm1(x), templates_feat_len, search_feat_len, templates_num,
                                   prev_decision=prev_decision, policy=policy, noNorm_x=x, dm=dm, train_num=train_num, templates_feat_len_org=templates_feat_len_org)
        if filter_x is not None:
           x = filter_x
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, decision, policy, topk_mem_index, mask_show


class VisionTransformer(BaseBackbone):
    """
    Vision Transformer.
    A PyTorch impl of : 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.
        (https://arxiv.org/abs/2010.11929)
    Includes distillation token & head support for 'DeiT: Data-efficient Image Transformers'.
        (https://arxiv.org/abs/2012.12877)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', residual=False, feature_size_z=64, feature_size_x=256,
                 relevance_attn=None, using_att_weights=None, training=True, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): Input image size.
            patch_size (int, tuple): Patch size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classes for classification head.
            embed_dim (int): Embedding dimension.
            depth (int): Depth of transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): Enable bias for qkv if True.
            representation_size (Optional[int]): Enable and set representation layer (pre-logits) to this value if set.
            distilled (bool): Model includes a distillation token and head as in DeiT models.
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            embed_layer (nn.Module): Patch embedding layer.
            norm_layer: (nn.Module): Normalization layer.
            weight_init: (str): Weight init scheme.
        """

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.using_att_weights = using_att_weights
        self.relevance_attn = relevance_attn
        self.is_training = training
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule

        relevance_attn_layer = [True if i in relevance_attn[0] else False for i in range(depth)]
        using_att_weights = [True if i in self.using_att_weights[0] else False for i in range(depth)]
        keep_num = relevance_attn[1]

        print('relevance_attn', relevance_attn_layer)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                residual=residual, feature_size_z=feature_size_z, feature_size_x=feature_size_x,
                relevance_attn=relevance_attn_layer[i], using_att_weights=using_att_weights[i], keep_num=keep_num[i], layer=i, training=self.is_training)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # Leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # This fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """
    ViT weight initialization.

    When called without n, head_bias, jax_impl args it will behave exactly the same
    as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl.
    """

    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # Note: conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """
    Load weights from .npz checkpoints for official Google Brain Flax implementation.
    """

    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # Hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # Resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    # NOTE classifier head has been removed in our model
    # if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
    #     model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
    #     model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('resized position embedding %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # Backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """
    Convert patch embedding weight from manual patchify + linear proj to conv.
    """

    out_dict = {}
    if 'model' in state_dict:
        # For DeiT models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('ERROR: features_only not implemented for Vision Transformer models')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            print('load pretrained model from ' + pretrained)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """

    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """
     ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model