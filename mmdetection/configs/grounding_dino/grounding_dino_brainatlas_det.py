_base_ = "grounding_dino_swin-t_finetune_16xb2_1x_coco.py"

load_from = "/home/lihua/Desktop/projects/mmd/mmdetection/biomedbert/new_mixdata_epoch_17.pth"

data_root = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/brain-atlas-test-nofiltered/"

class_name = (
    "Left Lateral Ventricle",
    "Right Lateral Ventricle",
    "Left Insula",
    "Right Insula",
    "Left Parietal Lobe",
    "Right Parietal Lobe",
    "Left Frontal Lobe",
    "Right Frontal Lobe",
    "Left Basal Ganglia",
    "Right Basal Ganglia",
    "Left Cingulate Gyrus",
    "Right Cingulate Gyrus",
    "Brain Stem",
    "Left Temporal Lobe",
    "Right Temporal Lobe",
    "Left Thalamus",
    "Right Thalamus",
    "Left Cerebellum",
    "Right Cerebellum",
    "Left Occipital Lobe",
    "Right Occipital Lobe",
    "Left Hippocampus",
    "Right Hippocampus",
    "Left Amygdala",
    "Right Amygdala",
    "3rd Ventricle",
)
"""class_name = (
    "Left Lateral Ventricle",
    "Right Lateral Ventricle",
    "Brain Stem",
    "Left Thalamus",
    "Right Thalamus",
    "Left Hippocampus",
    "Right Hippocampus",
    "Left Amygdala",
    "Right Amygdala",
    "3rd Ventricle",
)
"""
import random


def generate_unique_coordinates(N):
    # 创建一个空集合来存储rgb
    coordinates_set = set()

    # 当集合中的坐标数量小于N时，继续生成坐标
    while len(coordinates_set) < N:
        # 随机生成X, Y, Z坐标作为rgb
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        z = random.randint(0, 255)

        # 将坐标添加到集合中
        coordinates_set.add((x, y, z))

    # 将集合转换为列表并返回
    return list(coordinates_set)


num_classes = len(class_name)
# metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
metainfo = dict(classes=class_name, palette=generate_unique_coordinates(num_classes))
"""
    palette=[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (128, 0, 128),
        (255, 165, 0),
        (255, 192, 203),
        (0, 128, 128),
        (139, 69, 19),
        (0, 255, 255),
        (230, 230, 250),
        (128, 0, 0),
    ],
)"""

model = dict(
    _delete_ = True,
    type="GroundingDINOMask2former",
    is_seg=True,
    is_det=True,  # 本来用的is seg，后来改了代码，不想大改，额外打了个补丁is det
    num_queries=300,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[47.08, 47.08, 47.08],  # TODO
        std=[47.52, 47.52, 47.52],
        bgr_to_rgb=False,
        pad_mask=False,
    ),
    language_model=dict(
        type="BertModel",
        name="bert-base-uncased",
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=["[CLS]", "[SEP]", "+"],
        add_pooling_layer=False,
        max_tokens=256,
        use_biomedclip=True,
    ),
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=False,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    pixel_decoder=dict(
        type="ModifiedMSDeformAttnPixelDecoder",
        in_channels=[96, 256, 256, 256, 256],
        strides=[2, 4, 8, 16, 32],
        num_outs=4,
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=dict(type="ReLU"),
        encoder=dict(  # DeformableDetrTransformerEncoder
            num_layers=6,
            layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiScaleDeformableAttention
                    embed_dims=256,
                    num_heads=8,
                    num_levels=4,
                    num_points=4,
                    dropout=0.0,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
        ),
        positional_encoding=dict(num_feats=128, normalize=True),
    ),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    before_seg_cfg=dict(
        num_layers=3,
        num_cp=3,
        cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=1, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type="GroundingDINOHead",
        num_classes=256,  # 仅影响dino里的denoising，要用label embedding
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),  # 2.0 in DeformDETR
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # segmentation
    seg_head=dict(
        type="Mask2FormerSegHead",
        embed_dims=256,
        nhead=8,
        num_queries=900,
    ),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="BinaryFocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=300),
)

##################################################################################
# dataset settings
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.0),  # previously 0.5
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "text",
            "custom_entities",
            "text_aux",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "text",
            "custom_entities",
            "text_aux",
        ),
    ),
]

# grounding dino.py的predict的rescale需要和这里保持一致
##################################################################################


train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        _delete_=True,
        type="ConcatDataset",
        datasets=[
            dict(
                # type="RepeatDataset",
                # times=1,
                type="ClassBalancedDataset",
                oversample_thr=0.9,
                dataset=dict(
                    data_root=data_root,
                    metainfo=metainfo,
                    ann_file="train/annotations_without_background_detection.json",
                    data_prefix=dict(img="train/"),
                    pipeline=train_pipeline,
                    type="CocoDataset",
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    backend_args=None,
                    return_classes=True,
                ),
            ),
        ],
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        type="ClassBalancedDataset",
        oversample_thr=0.9,
        dataset=dict(
            metainfo=metainfo,
            data_root=data_root,
            ann_file="valid/annotations_without_background_detection.json",
            data_prefix=dict(img="valid/"),
            pipeline=test_pipeline,
            type="CocoDataset",
            test_mode=True,
            backend_args=None,
            return_classes=True,
        ),
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        metainfo=metainfo,
        data_root=data_root,
        ann_file="test/annotations_without_background_detection.json",
        data_prefix=dict(img="test/"),
        pipeline=test_pipeline,
        type="CocoDataset",
        test_mode=True,
        backend_args=None,
        return_classes=True,
    ),
)

val_evaluator = dict(
    ann_file=data_root + "valid/annotations_without_background_detection.json"
)
test_evaluator = dict(
    ann_file=data_root + "test/annotations_without_background_detection.json"
)

max_epoch = 30

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best="auto", rule="greater"),
    logger=dict(type="LoggerHook", interval=5),
)
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[9],#[19,29],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.0),
            "language_model": dict(lr_mult=0),
        }
    ),
)
"""            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.0),
            "language_model": dict(lr_mult=0),
            "neck": dict(lr_mult=0),
            "bbox_head": dict(lr_mult=1.0),
            "positional_encoding": dict(lr_mult=0),
            "encoder": dict(lr_mult=0),
            "decoder.layers.0": dict(lr_mult=0),
            "decoder.layers.1": dict(lr_mult=0),
            "decoder.layers.2": dict(lr_mult=0.1),
            "decoder.layers.3": dict(lr_mult=0.1),
            "query_embedding": dict(lr_mult=0),
            "memory_trans_fc": dict(lr_mult=0),
            "memory_trans_norm": dict(lr_mult=0),
            "text_feat_map": dict(lr_mult=0),
            "pixel_decoder": dict(lr_mult=0),
            "seg_head": dict(lr_mult=0),
            "FuseBeforeSeg.layers.0": dict(lr_mult=0),
            "FuseBeforeSeg.layers.1": dict(lr_mult=0),
        }"""


auto_scale_lr = dict(base_batch_size=8)
