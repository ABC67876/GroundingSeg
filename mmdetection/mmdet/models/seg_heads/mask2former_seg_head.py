# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union, List, Optional
import io
from collections import defaultdict


from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import ConvUpsample
from ..utils import interpolate_as
from .base_semantic_head import BaseSemanticHead

from mmengine.structures import PixelData

# copied from detr
from .util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
from torchvision.ops.boxes import nms

import matplotlib.pyplot as plt  

from matplotlib.colors import ListedColormap
from matplotlib.image import imread


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def _expand1(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self):
        super().__init__()

        self.lay7 = torch.nn.Conv2d(2, 2, 3, padding=1)
        self.ln7 = torch.nn.BatchNorm2d(2)

        # self.out_lay = torch.nn.Conv2d(inter_dims[5], 1, 1, 3, padding=1)
        self.out_lay = torch.nn.Conv2d(2, 1, 3, 1, padding=1)

        self.adapter5 = torch.nn.Conv2d(1, 1, 3, 1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        image_input: Tensor,
        outputs_mask: Tensor,
    ):
        cur_fpn = self.adapter5(image_input)
        num_decoders, num_queries, h, w = outputs_mask.shape
        x = outputs_mask.reshape(num_decoders * num_queries, 1, h, w)

        cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))

        x = torch.cat(
            [
                cur_fpn,
                F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest"),
            ],
            1,
        )
        x = self.lay7(x)
        x = self.ln7(x)
        x = F.relu(x)

        x = self.out_lay(x)  # .unsqueeze(2)
        x = x.reshape(num_decoders, num_queries, 1, x.shape[-2], x.shape[-1])
        L, C, N, H, W = x.shape  # N = 1
        background = torch.zeros(L, 1, N, H, W).cuda()
        background.requires_grad = False
        x = torch.cat((x, background), 1)

        # loss里面先用了sigmoid，所以这里不用softmax

        return x  # (L,C+1,1,H,W)


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    inputs = F.softmax(inputs, dim=0)
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() * 30


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    inputs = F.softmax(inputs, dim=0)
    inputs = inputs.flatten(1)

    targets = targets.flatten(1)  # 奇怪，detr抄来的，但报错，加了这一句

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()#sum()


@MODELS.register_module()
class Mask2FormerSegHead(BaseSemanticHead):
    """
    modified from detr segmentation
    """

    def __init__(self, embed_dims, nhead, num_queries) -> None:
        super().__init__(
            num_classes=1,
        )
        # 目前根本不用父类的东西，所以随便写的初始化

        hidden_dim, nheads = embed_dims, nhead

        # self.mask_head = MaskHeadSmallConv()
        self.num_queries = num_queries
        self.mask_embed = MLP(256, 256, 256, 3)
        # self.mask_embed = nn.Identity()

        self.deconv_swinoutput = nn.ConvTranspose2d(
            in_channels=96, out_channels=96, kernel_size=2, stride=2
        )
        self.conv_swinoutput = nn.Conv2d(
            in_channels=96, out_channels=128, kernel_size=3, padding=1
        )
        self.bn_swinoutput = nn.BatchNorm2d(128)
        self.conv_decoder_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.bn_decoder_1 = nn.BatchNorm2d(128)
        self.conv_decoder_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.bn_decoder_2 = nn.BatchNorm2d(128)
        self.deconv_decoder = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2
        )
        self.conv_combined_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.bn_combined_1 = nn.BatchNorm2d(128)
        self.conv_combined_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.bn_combined_2 = nn.BatchNorm2d(128)
        self.deconv_combined = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.conv_pic_1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=1
        )
        self.bn_pic_1 = nn.BatchNorm2d(8)
        self.conv_pic_2 = nn.Conv2d(
            in_channels=8, out_channels=64, kernel_size=3, padding=1
        )
        self.bn_pic_2 = nn.BatchNorm2d(64)
        self.conv_output_1 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1
        )
        self.bn_output_1 = nn.BatchNorm2d(64)
        self.conv_output_2 = nn.Conv2d(
            in_channels=64, out_channels=8, kernel_size=3, padding=1
        )
        self.bn_output_2 = nn.BatchNorm2d(8)
        self.conv_output_3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(
            self,
            batch_input_datalist,#([hs[-1:, batch : batch + 1, real_indices, :], src_prj[batch : batch + 1], mask_feature[batch : batch + 1], image_input[batch : batch + 1]])
        ):  # , dict_for_debug
            num_queries_batch = [_[0].shape[2] for _ in batch_input_datalist]
            bs = len(num_queries_batch)

            # num_decoders = hs.shape[0]#=1

            mask_embed = self.mask_embed(torch.cat([batch_input_datalist[i][0][0,0] for i in range(bs)],dim=0))

            outputs_mask = []
            _=0
            for i in range(bs):
                outputs_mask.append(
                    torch.einsum(
                        "qc,chw->qchw", mask_embed[_:_+num_queries_batch[i]], batch_input_datalist[i][2][0]
                    )# [num_queries, 256, h/4, w/4]
                )
                _+=num_queries_batch[i]
            outputs_mask=torch.cat(outputs_mask,dim=0)

            res_swin_output = self.deconv_swinoutput(torch.cat([batch_input_datalist[i][1] for i in range(bs)],dim=0))
            res_swin_output = self.conv_swinoutput(res_swin_output)
            res_swin_output = F.relu(self.bn_swinoutput(res_swin_output))

            pic = self.conv_pic_1(torch.cat([batch_input_datalist[i][3] for i in range(bs)],dim=0))
            pic = F.relu(self.bn_pic_1(pic))
            pic = self.conv_pic_2(pic)
            pic = F.relu(self.bn_pic_2(pic))

            res_swin_output_list=[]
            pic_list=[]
            for i in range(bs):
                for _ in range(num_queries_batch[i]):
                    res_swin_output_list.append(res_swin_output[i:i+1])
                    pic_list.append(pic[i:i+1])
            res_swin_output=torch.cat(res_swin_output_list,dim=0)
            pic = torch.cat(pic_list,dim=0)

            decoder_upsample = self.conv_decoder_1(
                outputs_mask
            )
            decoder_upsample = F.relu(self.bn_decoder_1(decoder_upsample))
            decoder_upsample = self.conv_decoder_2(decoder_upsample)
            decoder_upsample = F.relu(self.bn_decoder_2(decoder_upsample))
            decoder_upsample = self.deconv_decoder(decoder_upsample)

            decoder_upsample = torch.cat([decoder_upsample, res_swin_output], 1)
            decoder_upsample = self.conv_combined_1(decoder_upsample)
            decoder_upsample = F.relu(self.bn_combined_1(decoder_upsample))
            decoder_upsample = self.conv_combined_2(decoder_upsample)
            decoder_upsample = F.relu(self.bn_combined_2(decoder_upsample))
            decoder_upsample = self.deconv_combined(decoder_upsample)

            decoder_upsample = torch.cat(
                [decoder_upsample[:, :, : pic.shape[-2], : pic.shape[-1]], pic], 1
            )
            decoder_upsample = self.conv_output_1(decoder_upsample)
            decoder_upsample = F.relu(self.bn_output_1(decoder_upsample))
            decoder_upsample = self.conv_output_2(decoder_upsample)
            decoder_upsample = F.relu(self.bn_output_2(decoder_upsample))
            decoder_upsample = self.conv_output_3(decoder_upsample)

            x = decoder_upsample.reshape(
                1,
                -1, # sum of querie numbers
                1,
                decoder_upsample.shape[-2],
                decoder_upsample.shape[-1],
            )
            L, C, N, H, W = x.shape  # N = 1
            background = torch.zeros(L, 1, N, H, W).cuda()
            background.requires_grad = False

            _=0
            x_batch=[]
            for i in range(bs):
                x_batch.append(torch.cat((x[:,_:_+num_queries_batch[i]], background), 1))
                _+=num_queries_batch[i]
            return x_batch

    def forward_no_batch(
        self,
        hs,
        src_prj,
        mask_feature,
        image_input,
    ):  # , dict_for_debug
        hs = hs[:, :, :, :]
        num_decoders = hs.shape[0]
        num_queries = hs.shape[2]

        mask_embed = self.mask_embed(hs[:, 0])  # [num_decoders, num_queries, 256]
        outputs_mask = torch.einsum(
            "bqc,chw->bqchw", mask_embed, mask_feature[0]
        )  # [num_decoders, num_queries, 256, h/4, w/4]

        res_swin_output = self.deconv_swinoutput(src_prj)
        res_swin_output = self.conv_swinoutput(res_swin_output)
        res_swin_output = F.relu(self.bn_swinoutput(res_swin_output))

        decoder_upsample = self.conv_decoder_1(
            outputs_mask.reshape(
                num_decoders * num_queries,
                256,
                outputs_mask.shape[-2],
                outputs_mask.shape[-1],
            )
        )
        decoder_upsample = F.relu(self.bn_decoder_1(decoder_upsample))
        decoder_upsample = self.conv_decoder_2(decoder_upsample)
        decoder_upsample = F.relu(self.bn_decoder_2(decoder_upsample))
        decoder_upsample = self.deconv_decoder(decoder_upsample)
        res_swin_output = _expand(
            res_swin_output, decoder_upsample.size(0) // res_swin_output.size(0)
        )
        decoder_upsample = torch.cat([decoder_upsample, res_swin_output], 1)
        decoder_upsample = self.conv_combined_1(decoder_upsample)
        decoder_upsample = F.relu(self.bn_combined_1(decoder_upsample))
        decoder_upsample = self.conv_combined_2(decoder_upsample)
        decoder_upsample = F.relu(self.bn_combined_2(decoder_upsample))
        decoder_upsample = self.deconv_combined(decoder_upsample)
        pic = self.conv_pic_1(image_input)
        pic = F.relu(self.bn_pic_1(pic))
        pic = self.conv_pic_2(pic)
        pic = F.relu(self.bn_pic_2(pic))
        pic = _expand(pic, decoder_upsample.size(0) // pic.size(0))
        decoder_upsample = torch.cat(
            [decoder_upsample[:, :, : pic.shape[-2], : pic.shape[-1]], pic], 1
        )
        decoder_upsample = self.conv_output_1(decoder_upsample)
        decoder_upsample = F.relu(self.bn_output_1(decoder_upsample))
        decoder_upsample = self.conv_output_2(decoder_upsample)
        decoder_upsample = F.relu(self.bn_output_2(decoder_upsample))
        decoder_upsample = self.conv_output_3(decoder_upsample)

        x = decoder_upsample.reshape(
            num_decoders,
            num_queries,
            1,
            decoder_upsample.shape[-2],
            decoder_upsample.shape[-1],
        )
        L, C, N, H, W = x.shape  # N = 1
        background = torch.zeros(L, 1, N, H, W).cuda()
        background.requires_grad = False
        x = torch.cat((x, background), 1)
        return x

        # return self.mask_head(image_input,outputs_mask,)  # (L,C+1,1,H,W)

    def forward_previous(
        self,
        hs,
        memory,
        src_proj,
        features,
        mask,
        attention_weights,
        sampling_locations,
        spatial_shapes,
        level_start_index,
        out_1st_stage,
        mask_feature,
        image_input,
        bbox_map,
    ):  # , dict_for_debug
        hs = hs[:, :, :, :]
        bs = features[-1].shape[0]
        num_decoders = hs.shape[0]
        num_queries = hs.shape[2]

        mask_embed = self.mask_embed(hs[:, 0])  # [num_decoders, num_queries, 256]
        outputs_mask = torch.einsum(
            "bqc,chw->bqhw", mask_embed, mask_feature[0]
        )  # [num_decoders, num_queries, h, w]

        return self.mask_head(
            image_input,
            outputs_mask,
        )  # (L,C+1,1,H,W)

    def loss(
        self,
        hs: Tensor,
        memory: Tensor,
        features: List[Tensor],
        mask: Tensor,
        mask_result: Tensor,
        deformable_positions_dict: dict,
        out_1st_stage: Tensor,
        src_prj: Tensor,
        mask_feature: Tensor,
        # dict_for_debug=dict(),  # for debug
        batch_data_samples: SampleList,
        detection_results,
        image_input,
    ) -> dict:
        # print(batch_data_samples[0].gt_sem_seg.sem_seg.shape[-2:])
        # print(batch_data_samples[1].gt_sem_seg.sem_seg.shape[-2:])

        # seg_preds = self(hs, memory, None, features, mask)

        loss_sum_focal = torch.zeros(1, requires_grad=True).to(hs.device)
        loss_sum_dice = torch.zeros(1, requires_grad=True).to(hs.device)
        loss_ce = torch.zeros(1, requires_grad=True).to(hs.device)
        loss_sum_dice_aux = torch.zeros(1, requires_grad=True).to(hs.device)
        loss_ce_aux = torch.zeros(1, requires_grad=True).to(hs.device)
        cross_entropy_loss = nn.CrossEntropyLoss().to(hs.device)

        batch_input_datalist=[]
        batch_indices_list = []
        for batch in range(len(detection_results)):
            # num_class = len(batch_data_samples[batch].text)
            # img_shape = batch_data_samples[batch].img_shape
            # mask_ = mask[batch]
            mask_result_ = mask_result[batch]
            indices = (detection_results[batch].scores > 0.2).nonzero()[:, 0]

            # to make sure that labels are unique
            _boxes = detection_results[batch].bboxes[indices]*0.0 + detection_results[batch].labels[indices][..., None]
            _boxes[:,2:]+=0.5
            indices = nms(
                boxes=_boxes,
                scores=detection_results[batch].scores[indices],
                iou_threshold=0.5,
            )

            # if indices.sum() == 0: # 这样写是错的，如果有一个值为0的indice
            if indices.shape[0] == 0:
                batch_indices_list.append([])
                continue
            """
            indices = nms(
                boxes=detection_results[batch].bboxes[indices]
                + 1000.0 * detection_results[batch].labels[indices][..., None],
                scores=detection_results[batch].scores[indices],
                iou_threshold=0.5,
            )

            indices = nms(
                boxes=detection_results[batch].bboxes[indices],
                scores=detection_results[batch].scores[indices],
                iou_threshold=0.99,
            )
            #"""

            real_indices = detection_results[batch].indices[
                indices
            ]  # 经历topk之前的indices

            # labels = detection_results[batch].labels[indices]
            # bboxes = detection_results[batch].bboxes[indices].int()
            # gt_bboxes = batch_data_samples[batch].gt_instances.bboxes
            # gt_labels = batch_data_samples[batch].gt_instances.labels
            # 注意gt bboxes和labels并没有和bboxes和labels进行对应

            valid_h = mask_result_.shape[0] - mask_result_[:, 0].sum().item()
            valid_w = mask_result_.shape[1] - mask_result_[0, :].sum().item()

            """seg_pred = self(
                # hs[:, batch : batch + 1, real_indices, :],# 这是有aux loss的版本
                hs[-1:, batch : batch + 1, real_indices, :],#.detach(),
                src_prj[batch : batch + 1],#.detach(),
                mask_feature[batch : batch + 1],
                image_input[batch : batch + 1],#.detach(),
            )"""
            batch_input_datalist.append([hs[-1:, batch : batch + 1, real_indices, :], src_prj[batch : batch + 1], mask_feature[batch : batch + 1], image_input[batch : batch + 1]])
            batch_indices_list.append(indices)
        seg_pred_batch = self(batch_input_datalist)

        _valid_cnt=0
        for batch in range(len(detection_results)):
            gt_pred = batch_data_samples[batch].gt_sem_seg.sem_seg
            indices=batch_indices_list[batch]
            if indices==[]:
                continue
            real_indices = detection_results[batch].indices[
                indices
            ]  # 经历topk之前的indices
            labels = detection_results[batch].labels[indices]

            seg_pred = seg_pred_batch[_valid_cnt]
            _valid_cnt+=1
            seg_pred = seg_pred[
                :, :, 0, : gt_pred.shape[-2], : gt_pred.shape[-1]
            ]  # dim1包含背景 (L,C+1,1,H,W)

            gt_onehot = torch.zeros_like(seg_pred[0]).to(seg_pred.device)
            _4label_index_list = []
            for i in range(seg_pred.shape[1] - 1):
                if (
                    "contain_relations_new_label"
                    in batch_data_samples[batch].gt_sem_seg
                    and labels[i].item()
                    in batch_data_samples[batch].gt_sem_seg.contain_relations_new_label
                ):
                    _4label_index_list.append(i)
                    for sub_label in batch_data_samples[
                        batch
                    ].gt_sem_seg.contain_relations_new_label[labels[i].item()]:
                        gt_onehot[i][gt_pred[0] == sub_label] = 1
                else:
                    gt_onehot[i][gt_pred[0] == labels[i]] = 1
            gt_onehot[-1][gt_pred[0] == 255] = 1

            # tmp
            """
            pred_onehot = torch.zeros_like(seg_pred[0]).to(seg_pred.device)
            psuedo_max = seg_pred[0].max(dim=0)[0]
            for i in range(seg_pred.shape[1]):
                pred_onehot[i][seg_pred[0][i] == psuedo_max] = 1
            # ((pred_onehot[i].bool())&(gt_onehot[i].bool())).float().sum()/(1e-7+((pred_onehot[i].bool())|(gt_onehot[i].bool())).float().sum())
            
            # for checking
            if "contain_relations_new_label" in batch_data_samples[batch].gt_sem_seg:
                _contain_dict = batch_data_samples[batch].gt_sem_seg.contain_relations_new_label
            else:
                _contain_dict = None
            for i in range(seg_pred.shape[1] - 1):
                if ((pred_onehot[i].bool())&(gt_onehot[i].bool())).float().sum()/(1e-7+((pred_onehot[i].bool())|(gt_onehot[i].bool())).float().sum()).item() < 0.5:
                    iou = ((pred_onehot[i].bool())&(gt_onehot[i].bool())).float().sum()/(1e-7+((pred_onehot[i].bool())|(gt_onehot[i].bool())).float().sum()).item()
                    _label = labels[i].item()
                    if  _contain_dict == None:
                        _name = batch_data_samples[batch].text[_label]
                        continue
                    _gt_bbox = batch_data_samples[batch].gt_instances.bboxes[labels[i].item()]
                    _pred_bbox = detection_results[batch].bboxes[indices][i]
            #"""
            # end tmp

            # loss_sum_focal += sigmoid_focal_loss(seg_pred, gt_onehot)
            
            # loss_sum_dice += dice_loss(seg_pred[-1], gt_onehot)

            # prob_dist = F.softmax(seg_pred, dim=0)
            # loss = F.nll_loss(torch.log(pred_softmax + 1e-5), data_item.segment, reduction="none")
            prob_dist = seg_pred[-1].unsqueeze(0)  # to (N, C, h,w)

            _index1 = [i for i in range(prob_dist.shape[1]) if i not in _4label_index_list]

            ce_loss = cross_entropy_loss(prob_dist[:, _index1], gt_onehot.unsqueeze(0)[:, _index1])
            diceloss = dice_loss(seg_pred[-1, _index1], gt_onehot[_index1])

            if _4label_index_list!=[]:
                _4label_index_list.append(_index1[-1])
                ce_loss += cross_entropy_loss(prob_dist[:, _4label_index_list], gt_onehot.unsqueeze(0)[:, _4label_index_list])
                diceloss += dice_loss(seg_pred[-1, _4label_index_list], gt_onehot[_4label_index_list])
            # ce_loss = cross_entropy_loss(prob_dist, gt_onehot.unsqueeze(0))
            loss_ce += ce_loss
            loss_sum_dice += diceloss
            # 因为显存问题，所以不计算auxillary loss
        return dict(
            ce_loss=loss_ce / len(detection_results) * 20,
            loss_seg_dice=loss_sum_dice / len(detection_results) * 20,
        )
        """
            # auxillary loss
            for l in range(seg_pred.shape[0] - 1):
                loss_sum_dice_aux += dice_loss(seg_pred[l], gt_onehot)
                prob_dist = seg_pred[l].unsqueeze(0)
                ce_loss = cross_entropy_loss(prob_dist, gt_onehot.unsqueeze(0))
                loss_ce_aux += ce_loss
        if loss_sum_dice_aux == 0:
            return dict(
                ce_loss=loss_ce,
                loss_seg_dice=loss_sum_dice,
                loss_ce_aux=loss_ce_aux,
                loss_seg_dice_aux=loss_sum_dice_aux,
            )
        return dict(
            ce_loss=loss_ce,
            loss_seg_dice=loss_sum_dice,
            loss_ce_aux=loss_ce_aux / (seg_pred.shape[0] - 1),
            loss_seg_dice_aux=loss_sum_dice_aux / (seg_pred.shape[0] - 1),
        )
        #"""
        # return dict(loss_seg_focal=loss_sum_focal, loss_seg_dice=loss_sum_dice)
        # return dict(loss_seg_dice=loss_sum_dice)

    def predict(
        self,
        hs: Tensor,
        memory: Tensor,
        features: List[Tensor],
        mask: Tensor,
        mask_result: Tensor,
        deformable_positions_dict: dict,
        out_1st_stage: Tensor,
        src_prj: Tensor,
        mask_feature: Tensor,
        # dict_for_debug=dict(),  # for debug
        batch_data_samples: SampleList,
        detection_results,
        image_input,
    ) -> List[Tensor]:
        seg_pred_list = []
        for batch in range(len(detection_results)):
            # num_class = len(batch_data_samples[batch].text)
            # img_shape = batch_data_samples[batch].img_shape
            # mask_ = mask[batch]
            h, w = batch_data_samples[batch].ori_shape

            mask_result_ = mask_result[batch]
            indices = (detection_results[batch].scores > 0.2).nonzero()[:, 0]

            # to make sure that labels are unique
            _boxes = detection_results[batch].bboxes[indices]*0.0 + detection_results[batch].labels[indices][..., None]
            _boxes[:,2:]+=0.5
            indices = nms(
                boxes=_boxes,
                scores=detection_results[batch].scores[indices],
                iou_threshold=0.5,
            )

            if indices.shape[0] == 0:
                seg_pred_list.append(
                    PixelData(
                        sem_seg=(torch.ones([h, w]) * 255)
                        .long()
                        .to(mask_result.device)
                        .unsqueeze(0),
                        metainfo=dict(ignore_index=255),
                    )
                )
                continue
            """
            indices = nms(
                boxes=detection_results[batch].bboxes[indices],
                scores=detection_results[batch].scores[indices],
                iou_threshold=0.99,
            )
            #"""

            real_indices = detection_results[batch].indices[
                indices
            ]  # 经历topk之前的indices

            labels = detection_results[batch].labels[indices]
            bboxes = detection_results[batch].bboxes[indices].int()

            valid_h = mask_result_.shape[0] - mask_result_[:, 0].sum().item()
            valid_w = mask_result_.shape[1] - mask_result_[0, :].sum().item()

            bbox_map = torch.zeros(
                (bboxes.shape[0], image_input.shape[-2], image_input.shape[-1]),
                dtype=torch.float32,
            ).cuda()
            for i in range(bboxes.shape[0]):
                bbox_map[
                    i, bboxes[i][1] : bboxes[i][3], bboxes[i][0] : bboxes[i][2]
                ] = 1

            seg_pred = self.forward_no_batch(
                # hs[:, batch : batch + 1, real_indices, :],# 这是有aux loss的版本
                hs[-1:, batch : batch + 1, real_indices, :],
                src_prj[batch : batch + 1],
                mask_feature[batch : batch + 1],
                image_input[batch : batch + 1],
            )

            """
            seg_pred = seg_pred[:, :, :valid_h, :valid_w]
            seg_pred = F.interpolate(
                seg_pred,
                # size=gt_pred.shape[-2:],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]
            # src_masks = src_masks[:, 0].flatten(1)
            """
            seg_pred = seg_pred[-1, :, 0, :h, :w]  # dim1最后一个维度是背景

            output_demo = torch.zeros([h, w]).long().to(seg_pred.device)

            output = 255 * (torch.ones([h, w])).long().to(seg_pred.device)

            psuedo_max = seg_pred.max(dim=0)[0]
            # psuedo_background = psuedo_max.mean()

            for i in range(seg_pred.shape[0] - 1):
                output[seg_pred[i] == psuedo_max] = labels[i]
                output_demo[seg_pred[i] == psuedo_max] = labels[i]+1
            # output[seg_pred[-1] == psuedo_max] = 255
            # output[psuedo_max < 0.3 * psuedo_background] = 255
            # output[psuedo_max < -0.5] = 255

            """
            >>> metainfo = dict(
            ...     img_id=random.randint(0, 100),
            ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
            >>> image = np.random.randint(0, 255, (4, 20, 40))
            >>> featmap = torch.randint(0, 255, (10, 20, 40))
            >>> pixel_data = PixelData(metainfo=metainfo,
            ...                        image=image,
            ...                        featmap=featmap)
            """


            seg_pred_list.append(
                PixelData(sem_seg=output.unsqueeze(0), metainfo=dict(ignore_index=255))
            )
            

            name_ = batch_data_samples[batch].img_path.split('/')[-1]
            # pic_ = imread(batch_data_samples[batch].img_path)
            # gt_ = np.array(((batch_data_samples[batch].gt_sem_seg.sem_seg[0])+1).cpu()%256)
            pred_ = np.array(output_demo.cpu())
            
            
            """custom_colors = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800080', '#808000', '#008000','#C0C0C0']
            cmap = ListedColormap(custom_colors) 
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(pic_)
            img1 = ax2.imshow(gt_+0.5, cmap=cmap, vmin=0, vmax=11)
            img2 = ax3.imshow(pred_+0.5, cmap=cmap, vmin=0, vmax=11)
            fig.colorbar(img1, ax=ax2, shrink=0.5, aspect=5)  
            fig.colorbar(img2, ax=ax3, shrink=0.5, aspect=5)
            fig.savefig('preddebug/'+name_+'.png')"""

        return seg_pred_list

        """
        # copy from panoptic fpn
        seg_preds = self.forward(hs)["seg_preds"]
        seg_preds = F.interpolate(
            seg_preds,
            size=batch_img_metas[0]["batch_input_shape"],
            mode="bilinear",
            align_corners=False,
        )
        seg_preds = [seg_preds[i] for i in range(len(batch_img_metas))]

        if rescale:
            seg_pred_list = []
            for i in range(len(batch_img_metas)):
                h, w = batch_img_metas[i]["img_shape"]
                seg_pred = seg_preds[i][:, :h, :w]

                h, w = batch_img_metas[i]["ori_shape"]
                seg_pred = F.interpolate(
                    seg_pred[None], size=(h, w), mode="bilinear", align_corners=False
                )[0]
                seg_pred_list.append(seg_pred)
        else:
            seg_pred_list = seg_preds

        return seg_pred_list
        """
