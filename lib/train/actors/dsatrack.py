import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ...utils.heapmap_utils import generate_heatmap
from ...utils.mask_utils import generate_mask_cond


class DSATrackActor(BaseActor):
    """
    Actor for training DSATrack models.
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None, drop_ratio=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg
        self.drop_ratio = drop_ratio

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # Currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1

        template_list = []
        search_list = []
        # for i in range(self.settings.num_template):
        #     template_img_i = data['template_images'][i].view(-1,
        #                                                      *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
        #     template_list.append(template_img_i)
        # search_proposals = data['search_proposals']
        search_proposals = None

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)

        # box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
        #                                 data['template_anno'][0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=data['template_images'], search=search_list, search_proposals=search_proposals)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, out_pred_score=None, return_status=True, entropy=False):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda() # 定义 0 tensor，并指定GPU设备
        # GT gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)

        for i in range(len(pred_dict)):
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError('ERROR: network outputs is NAN! stop training')
            num_queries = pred_boxes.size(1)
            # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            # (B,4) --> (B,1,4) --> (B,N,4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                            max=1.0)
            # Compute GIoU and IoU
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # Compute L1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # Compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)


            # pred_dict['iou_pred'] = None
            if pred_dict[i]['iou_pred'] is not None:
                iou_pred = pred_dict[i]['iou_pred']
                iou_loss = self.objective['iou'](iou_pred, gt_dict['proposal_iou'])
            # weighted sum
                loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+ self.loss_weight[
                    'iou'] * iou_loss
            else:
                loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
                    'focal'] * location_loss
            # Weighted sum
            # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            #     'focal'] * location_loss
            if len(pred_dict[i]['class_scores']) != 0:
                class_loss = self.objective['focal'](torch.cat(pred_dict[i]['class_scores'], dim=2).mean(2, keepdim=True).permute(0, 3, 1, 2), gt_dict['template_FeatureMasks'].permute(1, 0, 2 ,3).flatten(1).unsqueeze(1).unsqueeze(-1))
                loss = class_loss
            drop_loss = 0

            if pred_dict[i]['templates_decision'] != [] and len(pred_dict[i]['class_scores'])==0:
                for j, template_score in enumerate(pred_dict[i]['templates_decision']):

                    template_drp_ratio = template_score.mean(1)
                    # print(f'template_drp_ratio{i}', template_drp_ratio)
                    # search_drp_ratio = search_score.mean(1)
                    # print(f'search_drp_ratio{i}', template_drp_ratio)
                    drop_loss = drop_loss + ((template_drp_ratio - self.drop_ratio[0][j]) ** 2).mean()
                    # drop_loss = drop_loss + ((search_drp_ratio - self.drop_ratio[1][i]) ** 2).mean()

                loss = loss+drop_loss
                
            total_loss += loss

            if return_status:
                # Status for log
                status = {}
                mean_iou = iou.detach().mean()
                if pred_dict[i]['templates_decision'] != [] and len(pred_dict[i]['class_scores'])==0:
                    # print('???')
                    status = {f'Ls/total': loss.item(),
                            f'Ls/giou': giou_loss.item(),
                            f'Ls/l1': l1_loss.item(),
                            f'Ls/loc': location_loss.item(),
                            f'Ls/drp': drop_loss.item(),
                            # 'Ls/iou': iou_loss.item(),
                            # 'Ls/cls':class_loss.item(),
                            f'IoU': mean_iou.item()}
                elif len(pred_dict[i]['class_scores']) != 0:
                    status = {'Ls/total': loss.item(),
                            'Ls/class': class_loss.item()}

                elif pred_dict[i]['iou_pred'] is not None:
                    print('xxx')
                    status = {'Ls/total': loss.item(),
                            'Ls/giou': giou_loss.item(),
                            'Ls/l1': l1_loss.item(),
                            'Ls/loc': location_loss.item(),
                            'Ls/iou': iou_loss.item(),
                            'IoU': mean_iou.item()}
                else:
                    status = {'Ls/total': loss.item(),
                            'Ls/giou': giou_loss.item(),
                            'Ls/l1': l1_loss.item(),
                            'Ls/loc': location_loss.item(),
                            'IoU': mean_iou.item()}
                total_status.update(status)
        
        if return_status:                    
            return total_loss, total_status
        else:
            return total_loss
