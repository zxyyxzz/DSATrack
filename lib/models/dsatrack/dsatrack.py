"""
Basic DSATrack model.
"""

import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head, build_iou_head
from lib.models.dsatrack.vit import vit_base_patch16_224, vit_tiny_patch16_224, deit_tiny_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
import math
from lib.models.dsatrack.vit import resize_pos_embed

class DSATrack(nn.Module):
    """
    This is the base class for DSATrack.
    """

    def __init__(self, transformer, box_head, iou_head, head_type='CORNER', token_len=1, tgt_type='allmax', train_num=1):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.iou_head = iou_head
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_len_z = int(self.feat_len_s/4)
        self.tgt_type = tgt_type
        self.train_num = train_num

        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor, search: torch.Tensor, template_mask=None, threshold=0., search_proposals=None, template_show=None, update=False):
        assert isinstance(search, list), "The type of search is not List"
        out_dict = []
        for i in range(len(search)):
            # if self.training:
            x, templates_decisions, class_scores = self.backbone(z=template, x=search[i], template_mask=template_mask, search_feat_len=self.feat_len_s, templates_feat_len=self.feat_len_z,
                                    threshold=threshold, tgt_type=self.tgt_type, train_num=self.train_num, template_show=template_show, update=update, track_query=self.track_query)
            # else:
            #     x, templates_decisions, class_scores = self.backbone(z=template.copy(), x=search[i], template_mask=template_mask, search_feat_len=self.feat_len_s, templates_feat_len=self.feat_len_z,
            #                             threshold=threshold, tgt_type=self.tgt_type, train_num=self.train_num, template_show=template_show, update=update, track_query=self.track_query)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)

            # att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)
            out['templates_decision'] = templates_decisions
            out['class_scores'] = class_scores
            out['iou_pred'] = None

            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: Output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C).
        """

        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == 'CORNER':
            # Run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
            return out
        elif self.head_type == 'CENTER':
            # Run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_dsatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('DSATrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth' or 'DSATrack' in cfg.MODEL.PRETRAIN_FILE:
            backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, residual=cfg.MODEL.BACKBONE.RESIDUAL,
                                                 feature_size_z=cfg.MODEL.BACKBONE.FEATURE_SIZE_Z, feature_size_x=cfg.MODEL.BACKBONE.FEATURE_SIZE_X,
                                                 relevance_attn=cfg.MODEL.BACKBONE.RELEVANCE_ATTN,
                                                 using_att_weights=cfg.MODEL.BACKBONE.USING_ATT_WEIGHTS,
                                                 training=training)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny':
        if cfg.MODEL.PRETRAIN_FILE == 'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz':
            backbone = vit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, residual=cfg.MODEL.BACKBONE.RESIDUAL,
                                                 feature_size_z=cfg.MODEL.BACKBONE.FEATURE_SIZE_Z, feature_size_x=cfg.MODEL.BACKBONE.FEATURE_SIZE_X,
                                                 relevance_attn=cfg.MODEL.BACKBONE.RELEVANCE_ATTN,
                                                 using_att_weights=cfg.MODEL.BACKBONE.USING_ATT_WEIGHTS,
                                                 training=training)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'deit_tiny':
        if cfg.MODEL.PRETRAIN_FILE == 'deit_tiny_patch16_224-a1311bcf.pth':
            backbone = deit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, residual=cfg.MODEL.BACKBONE.RESIDUAL,
                                                 feature_size_z=cfg.MODEL.BACKBONE.FEATURE_SIZE_Z, feature_size_x=cfg.MODEL.BACKBONE.FEATURE_SIZE_X,
                                                 relevance_attn=cfg.MODEL.BACKBONE.RELEVANCE_ATTN,
                                                 using_att_weights=cfg.MODEL.BACKBONE.USING_ATT_WEIGHTS,
                                                 training=training)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    # iou_head = build_iou_head(hidden_dim)
    iou_head = None
    model = DSATrack(
        backbone,
        box_head,
        iou_head,
        head_type=cfg.MODEL.HEAD.TYPE,
        tgt_type=cfg.MODEL.TGT_TYPE,
        train_num=cfg.DATA.TEMPLATE.TRAIN_NUMBER
    )

    if 'DSATrack' in cfg.MODEL.PRETRAIN_FILE and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location='cpu')
        # update state dict
        new_state_dict = checkpoint['net'].copy()
        old_pos_embed_x = new_state_dict['backbone.pos_embed_x']
        old_pos_embed_z = new_state_dict['backbone.pos_embed_z']
        if model.backbone.pos_embed_x.shape != old_pos_embed_x.shape or model.backbone.pos_embed_z.shape != old_pos_embed_z.shape:
            if model.backbone.pos_embed_x.shape != old_pos_embed_x.shape:
                new_pos_embed_x = resize_pos_embed(old_pos_embed_x, model.backbone.pos_embed_x, 0)
                new_state_dict['backbone.pos_embed_x'] = new_pos_embed_x
            if model.backbone.pos_embed_z.shape != old_pos_embed_z.shape:
                new_pos_embed_z = resize_pos_embed(old_pos_embed_z, model.backbone.pos_embed_z, 0)
                new_state_dict['backbone.pos_embed_z'] = new_pos_embed_z
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('load pretrained model from ' + cfg.MODEL.PRETRAIN_FILE)
    return model
