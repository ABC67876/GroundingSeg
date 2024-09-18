_base_ = "grounding_dino_swin-t_finetune_16xb2_1x_coco.py"

# load_from = "/home/lihua/Desktop/projects/mmd/mmdetection/biomedbert/withoutrecon/det/adni33/best_coco_bbox_mAP_epoch_4.pth"

data_root_adni = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/adni-35-test/"
data_root_oasis = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/oasis-35-test/"
class_name = (
    "Left Cerebral White Matter",
    "Left Cerebral Cortex",
    "Left Lateral Ventricle",
    "Left Inf Lat Ventricle",
    "Left Cerebellum White Matter",
    "Left Cerebellum Cortex",
    "Left Thalamus",
    "Left Caudate",
    "Left Putamen",
    "Left Pallidum",
    "3rd Ventricle",
    "4th Ventricle",
    "Brain Stem",
    "Left Hippocampus",
    "Left Amygdala",
    "Left Accumbens",
    "Left Ventral DC",
    "Left Vessel",
    "Left Choroid Plexus",
    "Right Cerebral White Matter",
    "Right Cerebral Cortex",
    "Right Lateral Ventricle",
    "Right Inf Lat Ventricle",
    "Right Cerebellum White Matter",
    "Right Cerebellum Cortex",
    "Right Thalamus",
    "Right Caudate",
    "Right Putamen",
    "Right Pallidum",
    "Right Hippocampus",
    "Right Amygdala",
    "Right Accumbens",
    "Right Ventral DC",
    "Right Vessel",
    "Right Choroid Plexus",
)

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

# Mean brightness per channel:
# {'R': 39.1393717890213, 'G': 39.1393717890213, 'B': 39.1393717890213}
# Variance per channel:
# {'R': 2094.208085656205, 'G': 2094.208085656205, 'B': 2094.208085656205}

model = dict(
    type="GroundingDINOMask2former",
    is_seg=True,
    is_det=True,  # 本来用的is seg，后来改了代码，不想大改，额外打了个补丁is det
    num_queries=300,
    # bbox_head=dict(num_classes=num_classes, contrastive_cfg=dict(max_text_len=512)),  # 1200
    language_model=dict(max_tokens=256, use_biomedclip=True),
    # dn_cfg=dict(group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=50)),
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[39.139, 39.139, 39.139],
        std=[45.763, 45.763, 45.763],
        bgr_to_rgb=False,
        pad_mask=False,
    ),
    bbox_head=dict(num_classes=256),#num_classes),
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
    batch_size=32,
    num_workers=16,
    dataset=dict(
        _delete_=True,
        type="ConcatDataset",
        datasets=[
            dict(
                type="RepeatDataset",
                times=1,
                dataset=dict(
                    data_root=data_root_adni,
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
            dict(
                type="RepeatDataset",
                times=3,
                dataset=dict(
                    data_root=data_root_oasis,
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
    batch_size=16,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root_adni,
        ann_file="valid/annotations_without_background_detection.json",
        data_prefix=dict(img="valid/"),
        pipeline=test_pipeline,
        type="CocoDataset",
        test_mode=True,
        backend_args=None,
        return_classes=True,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root_adni,
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
    ann_file=data_root_adni + "valid/annotations_without_background_detection.json"
)
test_evaluator = dict(
    ann_file=data_root_adni + "test/annotations_without_background_detection.json"
)

max_epoch = 40

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=20, save_best="auto", rule="greater"),
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
        milestones=[5, 10, 15, 20],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    # optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0),
        }
    ),
)

auto_scale_lr = dict(base_batch_size=8)
