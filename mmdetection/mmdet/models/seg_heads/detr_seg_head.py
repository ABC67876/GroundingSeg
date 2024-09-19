# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union, List, Optional
import io
from collections import defaultdict

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


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def _expand1(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 32,
        ]
        # self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.lay1 = torch.nn.Conv2d(dim + 1, dim, 3, padding=1)  # for bbox
        self.gn1 = torch.nn.GroupNorm(3, dim)  # previously, group = 8
        self.lay2 = torch.nn.Conv2d(
            dim * 8 + 1, inter_dims[1], 3, padding=1
        )  # n_heads = 8
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1] + 8 + 1, inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2] + 8 + 1, inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3] + 8 + 1, inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.lay6 = torch.nn.Conv2d(inter_dims[4] + 1, inter_dims[5], 3, padding=1)
        self.gn6 = torch.nn.GroupNorm(8, inter_dims[5])
        self.lay7 = torch.nn.Conv2d(inter_dims[5] * 2 + 1, inter_dims[5], 3, padding=1)
        self.gn7 = torch.nn.GroupNorm(8, inter_dims[5])

        # self.out_lay = torch.nn.Conv2d(inter_dims[5], 1, 1, 3, padding=1)
        self.out_lay = torch.nn.Conv2d(inter_dims[5], 1, 3, 1, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        self.adapter4 = torch.nn.Conv2d(96, inter_dims[4], 1)  # FIXEME 这里暂时写死的
        self.adapter5 = torch.nn.Conv2d(1, inter_dims[5], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Tensor,
        # bbox_mask: Tensor,
        fpns: List[Tensor],
        attention_maps_level_list: List[Tensor],
        out_1st_stage: Tensor,
        image_input: Tensor,
        bbox_map: Tensor,
    ):
        # 把attention_maps_level_list和bbox_map的query维度上增加一条代表background的
        """
        for i in range(len(attention_maps_level_list)):
            attention_maps_level=attention_maps_level_list[i]
            attention_maps_level_list[i]=torch.cat([attention_maps_level,torch.zeros_like(attention_maps_level[:,:1].to(x.device))],1)
        bbox_map=torch.cat([bbox_map,torch.ones_like(bbox_map[:1]).to(x.device)],0)
        """
        # FIXME: a small trick: move num_head to bs dimension.
        x = _expand1(x, attention_maps_level_list[3].shape[1]).permute(
            1, 0, 2, 3, 4
        )  # [1, num head, C, H, W] -> [num query, num head, C, H, W] -> [num head, num query, C, H, W]
        attn_map = (
            attention_maps_level_list[3][0].permute(1, 0, 2, 3).unsqueeze(2)
        )  # [num query, num head, H, W] ->  [num head, num query, H, W] ->  [num head, num query, 1, H, W]
        x = (
            torch.cat([x, attn_map], 2)
            .permute(1, 0, 2, 3, 4)
            .reshape(x.shape[0] * x.shape[1], -1, x.shape[3], x.shape[4])
        )  # [num query*num head, C+1, H, W]

        # x = self.lay1(x)
        bbox_1 = (
            _expand(
                F.interpolate(bbox_map.unsqueeze(0), size=x.shape[-2:], mode="nearest"),
                attn_map.shape[0],
            )
            .permute(1, 0, 2, 3)
            .reshape(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # bbox map, interpolate to the same size as x, reshape to [num query*num head, 1, H, W]
        x = self.lay1(torch.cat([x, bbox_1], 1))
        x = self.gn1(x)
        x = F.relu(x)

        x = x.reshape(
            attn_map.shape[1], attn_map.shape[0] * x.shape[1], x.shape[2], x.shape[3]
        )  # [num query, num head*C', H', W']

        # x = self.lay2(x)
        bbox_2 = F.interpolate(bbox_map.unsqueeze(1), size=x.shape[-2:], mode="nearest")
        x = self.lay2(torch.cat([x, bbox_2], 1))
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = torch.cat([x, attention_maps_level_list[2].flatten(0, 1)], 1)
        # x = self.lay3(x)
        bbox_3 = F.interpolate(bbox_map.unsqueeze(1), size=x.shape[-2:], mode="nearest")
        x = self.lay3(torch.cat([x, bbox_3], 1))
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = torch.cat([x, attention_maps_level_list[1].flatten(0, 1)], 1)
        # x = self.lay4(x)
        bbox_4 = F.interpolate(bbox_map.unsqueeze(1), size=x.shape[-2:], mode="nearest")
        x = self.lay4(torch.cat([x, bbox_4], 1))
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = torch.cat([x, attention_maps_level_list[0].flatten(0, 1)], 1)
        # x = self.lay5(x)
        bbox_5 = F.interpolate(bbox_map.unsqueeze(1), size=x.shape[-2:], mode="nearest")
        x = self.lay5(torch.cat([x, bbox_5], 1))
        x = self.gn5(x)
        x = F.relu(x)

        cur_fpn = self.adapter4(out_1st_stage)
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # x = torch.cat([x, attention_maps_level_list[0].flatten(0, 1)], 1)
        # x = self.lay6(x)
        bbox_6 = F.interpolate(bbox_map.unsqueeze(1), size=x.shape[-2:], mode="nearest")
        x = self.lay6(torch.cat([x, bbox_6], 1))
        x = self.gn6(x)
        x = F.relu(x)

        cur_fpn = self.adapter5(image_input)
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = torch.cat(
            [
                cur_fpn,
                F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest"),
                bbox_map.unsqueeze(1),
            ],
            1,
        )
        x = self.lay7(x)
        x = self.gn7(x)
        x = F.relu(x)

        x = self.out_lay(x).permute(1, 0, 2, 3)
        N, C, H, W = x.shape  # N = 1
        background = torch.zeros(N, 1, H, W).cuda()
        background.requires_grad = False
        x = torch.cat((x, background), 1)
        """
        # 将形状重排为 (N*H*W, C+1)，以便应用 softmax
        logits_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C + 1)
        logits_softmax = F.softmax(logits_reshaped, dim=1)
        logits_softmax = logits_softmax.view(N, H, W, C + 1).permute(0, 3, 1, 2)
        # 现在 logits_softmax 的形状为 (N, C, H, W)，表示每个像素点在 C 类别上的概率分布
        """
        # loss里面先用了sigmoid，所以这里不用softmax

        return x.permute(1, 0, 2, 3)  # (C+1,1,H,W)


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias
        )
        qh = q.view(
            q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads
        )
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


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
    return loss.mean()


@MODELS.register_module()
class DetrSegHead(BaseSemanticHead):
    """
    modified from detr segmentation
    """

    def __init__(self, embed_dims, nhead, num_queries) -> None:
        super().__init__(
            num_classes=1,
        )
        # 目前根本不用父类的东西，所以随便写的初始化

        hidden_dim, nheads = embed_dims, nhead
        self.bbox_attention = MHAttentionMap(
            hidden_dim, hidden_dim, nheads, dropout=0.0
        )
        self.mask_head = MaskHeadSmallConv(
            # hidden_dim + nheads, [1024, 512, 256], hidden_dim
            # hidden_dim + nheads,
            hidden_dim // nheads + 1,
            [256, 256, 256],
            hidden_dim,
        )
        self.num_queries = num_queries

    def forward(
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
        image_input,
        bbox_map,
    ):  # , dict_for_debug
        hs = hs[:, :, -self.num_queries :, :]
        bs = features[-1].shape[0]

        # modified from multi_scale_deformable_attn_pytorch
        (
            _,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            coords_len,
        ) = sampling_locations.shape
        assert coords_len == 2
        attention_weights = attention_weights.reshape(-1, num_points)
        sampling_locations = sampling_locations.reshape(-1, num_points, coords_len)

        sampling_locations = torch.clamp(
            torch.round(sampling_locations * 100), 0, 99
        ).long()  # [0,99]

        attention_maps = torch.zeros(
            (attention_weights.shape[0], 100, 100), dtype=torch.float32
        ).cuda()

        #########################################################################
        # the following code is generated by GPT3.5 and debugged, output is same
        # with the handwritting loop code below.
        # FIXME: the whole process needs dicussion. about interpolation.
        # Extract indices for each dimension
        indices_0 = torch.arange(attention_weights.shape[0]).view(-1, 1).cuda()
        indices_1 = sampling_locations[:, :, 0].long()
        indices_0 = indices_0.expand_as(indices_1)
        attention_indices = torch.stack(
            [indices_0, indices_1, sampling_locations[:, :, 1].long()], dim=-1
        )

        # Use advanced indexing to update attention_maps
        for point in range(num_points):
            attention_maps[
                attention_indices[:, point, 0],
                attention_indices[:, point, 1],
                attention_indices[:, point, 2],
            ] += attention_weights[:, point]

        ##########################################################################
        # I AM STUPID ! I AM STUPID !
        """
        for idx_0 in range(attention_weights.shape[0]):
            for idx_point in range(num_points):
                attention_maps[
                    idx_0,
                    sampling_locations[idx_0, idx_point, 0],
                    sampling_locations[idx_0, idx_point, 1],
                ] += attention_weights[idx_0, idx_point]
        """
        ##########################################################################

        attention_maps = attention_maps.reshape(
            bs, num_queries, num_heads, num_levels, 100, 100
        )
        attention_maps_level_list = []
        for level, (H_, W_) in enumerate(spatial_shapes):
            resized = F.interpolate(
                attention_maps[:, :, :, level, :, :].reshape(
                    bs * num_queries, num_heads, 100, 100
                ),
                size=[H_, W_],
                mode="area",
            ).reshape(bs, num_queries, num_heads, H_, W_)
            resized *= 10000.0 / (H_ * W_)
            attention_maps_level_list.append(resized)

        # mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        # FIXME h_boxes takes the last one computed, keep this in mind
        """
        bbox_mask = self.bbox_attention(
            hs[-1], memory, mask=mask
        )  # torch.Size([8(batch), 900, 8(heads), 14, 14])
        """

        # FIXME src proj isn't the raw features!
        # Fixed
        # src_proj = features[3]

        src_proj = (
            src_proj[:, level_start_index[3] :, :, :]
            .permute(0, 2, 3, 1)
            .reshape(
                bs,
                num_heads,
                src_proj.shape[-1],
                spatial_shapes[3][0],
                spatial_shapes[3][1],
            )
        )
        """
        import matplotlib.pyplot as plt
        plt.imshow(tmp[0,0,0], cmap='gray')
        <matplotlib.image.AxesImage object at 0x7fc425181c30>
        plt.imshow(tmp[0,1,0], cmap='gray')
        <matplotlib.image.AxesImage object at 0x7fc426dde710>
        plt.imshow(tmp[0,3,0], cmap='gray')
        <matplotlib.image.AxesImage object at 0x7fc426d8bdf0>
        plt.imshow(tmp[0,4,0], cmap='gray')
        <matplotlib.image.AxesImage object at 0x7fc426d50c10>
        """

        seg_masks = self.mask_head(
            src_proj,
            # bbox_mask,
            [features[2], features[1], features[0]],
            attention_maps_level_list,
            out_1st_stage,
            image_input,
            bbox_map,
        )
        outputs_seg_masks = seg_masks.view(
            bs,
            hs.shape[2] + 1,  # self.num_queries here is modified to hs.shape[2]
            seg_masks.shape[-2],
            seg_masks.shape[-1],
        )

        return outputs_seg_masks

    def loss(
        self,
        hs: Tensor,
        memory: Tensor,
        features: List[Tensor],
        mask: Tensor,
        mask_result: Tensor,
        deformable_positions_dict: dict,
        out_1st_stage: Tensor,
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
        loss = torch.zeros(1, requires_grad=True).to(hs.device)
        cross_entropy_loss = nn.CrossEntropyLoss().to(hs.device)

        # 改成了只喂进去有用的query，简单起见，就不用batch进行forward了
        for batch in range(len(detection_results)):
            # num_class = len(batch_data_samples[batch].text)
            # img_shape = batch_data_samples[batch].img_shape
            # mask_ = mask[batch]
            mask_result_ = mask_result[batch]
            indices = (detection_results[batch].scores > 0.5).nonzero()[:, 0]
            # if indices.sum() == 0: # 这样写是错的，如果有一个值为0的indice
            if indices.shape[0] == 0:
                continue

            real_indices = detection_results[batch].indices[
                indices
            ]  # 经历topk之前的indices

            labels = detection_results[batch].labels[indices]
            bboxes = detection_results[batch].bboxes[indices].int()
            # gt_bboxes = batch_data_samples[batch].gt_instances.bboxes
            # gt_labels = batch_data_samples[batch].gt_instances.labels
            # 注意gt bboxes和labels并没有和bboxes和labels进行对应
            gt_pred = batch_data_samples[batch].gt_sem_seg.sem_seg

            valid_h = mask_result_.shape[0] - mask_result_[:, 0].sum().item()
            valid_w = mask_result_.shape[1] - mask_result_[0, :].sum().item()

            bbox_map = torch.zeros(
                (bboxes.shape[0], image_input.shape[-2], image_input.shape[-1]),
                dtype=torch.float32,
            ).to(hs.device)
            for i in range(bboxes.shape[0]):
                bbox_map[
                    i, bboxes[i][1] : bboxes[i][3], bboxes[i][0] : bboxes[i][2]
                ] = 1

            seg_pred = self(
                hs[:, batch : batch + 1, real_indices, :],
                memory[batch : batch + 1, :, :, :],
                deformable_positions_dict["src_proj"][batch : batch + 1],
                [feature[batch : batch + 1, :, :, :] for feature in features],
                mask[batch : batch + 1, :, :],
                deformable_positions_dict["attention_weights"][
                    batch : batch + 1, real_indices
                ],
                deformable_positions_dict["sampling_locations"][
                    batch : batch + 1, real_indices
                ],
                deformable_positions_dict["spatial_shapes"],
                deformable_positions_dict["level_start_index"],
                out_1st_stage[batch : batch + 1],
                image_input[batch : batch + 1],
                bbox_map,
                # dict_for_debug,  # for debug
            )

            """
            seg_pred = seg_pred[:, :, :valid_h, :valid_w]
            seg_pred = F.interpolate(
                seg_pred,
                size=gt_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            # src_masks = src_masks[:, 0].flatten(1)
            """
            seg_pred = seg_pred[
                0, :, : gt_pred.shape[-2], : gt_pred.shape[-1]
            ]  # dim1包含背景

            gt_onehot = torch.zeros_like(seg_pred).to(seg_pred.device)
            for i in range(seg_pred.shape[0] - 1):
                gt_onehot[i][gt_pred[0] == labels[i]] = 1
            gt_onehot[-1][gt_pred[0] == 255] = 1

            # loss_sum_focal += sigmoid_focal_loss(seg_pred, gt_onehot)
            loss_sum_dice += dice_loss(seg_pred, gt_onehot)

            # prob_dist = F.softmax(seg_pred, dim=0)
            # loss = F.nll_loss(torch.log(pred_softmax + 1e-5), data_item.segment, reduction="none")
            prob_dist = seg_pred.unsqueeze(0)  # to (N, C, h,w)
            ce_loss = cross_entropy_loss(prob_dist, gt_onehot.unsqueeze(0))
            loss += ce_loss
        return dict(ce_loss=loss, loss_seg_dice=loss_sum_dice)
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
            indices = (detection_results[batch].scores > 0.5).nonzero()[:, 0]

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

            seg_pred = self(
                hs[:, batch : batch + 1, real_indices, :],
                memory[batch : batch + 1, :, :, :],
                deformable_positions_dict["src_proj"][batch : batch + 1],
                [feature[batch : batch + 1, :, :, :] for feature in features],
                mask[batch : batch + 1, :, :],
                deformable_positions_dict["attention_weights"][
                    batch : batch + 1, real_indices
                ],
                deformable_positions_dict["sampling_locations"][
                    batch : batch + 1, real_indices
                ],
                deformable_positions_dict["spatial_shapes"],
                deformable_positions_dict["level_start_index"],
                out_1st_stage[batch : batch + 1],
                image_input[batch : batch + 1],
                bbox_map,
                # dict_for_debug,  # for debug
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
            seg_pred = seg_pred[0, :, :h, :w]  # dim1最后一个维度是背景

            output = 255 * (torch.ones([h, w])).long().to(seg_pred.device)

            psuedo_max = seg_pred.max(dim=0)[0]
            # psuedo_background = psuedo_max.mean()

            for i in range(seg_pred.shape[0] - 1):
                output[seg_pred[i] == psuedo_max] = labels[i]
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
