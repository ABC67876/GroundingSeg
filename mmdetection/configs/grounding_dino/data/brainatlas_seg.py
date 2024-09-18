brainatlas_data_root = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/brain-atlas-test-nofiltered/"
# brainatlas_data_root = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/brain-atlas-test-seen/"
brainatlas_backend_args = None

brainatlas_class_name = (
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
"""
brainatlas_class_name = (
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
)"""
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


brainatlas_num_classes = len(brainatlas_class_name)
# brainatlas_metainfo = dict(classes=brainatlas_class_name, palette=[(220, 20, 60)])
brainatlas_metainfo = dict(
    classes=brainatlas_class_name,
    thing_classes=brainatlas_class_name,
    stuff_classes=(),
    palette=generate_unique_coordinates(brainatlas_num_classes),
)

brainatlas_train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=brainatlas_backend_args),
    dict(
        type="LoadPanopticAnnotations", backend_args=brainatlas_backend_args
    ),  # previously for detection: dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.0),
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
"""
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
#"""


brainatlas_test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=brainatlas_backend_args),
    # dict(type="FixScaleResize", scale=(800, 1333), keep_ratio=True),
    dict(type="LoadPanopticAnnotations", backend_args=brainatlas_backend_args),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",  # grounding dino seg.py的predict的rescale需要和这里保持一致
            "text",
            "custom_entities",
            "text_aux",
        ),
    ),
]


brainatlas_train_dataset = dict(
    # type="RepeatDataset",
    # times=1,
    type="ClassBalancedDataset",
    oversample_thr=0.00000000000000000000009,
    dataset=dict(
        type="CocoPanopticDataset",
        metainfo=brainatlas_metainfo,
        data_root=brainatlas_data_root,
        ann_file="train/annotations_without_background.json",
        data_prefix=dict(img="train/", seg="train/annotations_without_background/"),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=brainatlas_train_pipeline,
        return_classes=True,
        backend_args=brainatlas_backend_args,
    ),
),

brainatlas_valid_dataset = dict(
    type="ClassBalancedDataset",
    oversample_thr=0.00000000000000000000000000000009,
    dataset=dict(
        type="CocoPanopticDataset",
        metainfo=brainatlas_metainfo,
        data_root=brainatlas_data_root,
        ann_file="valid/annotations_without_background.json",
        data_prefix=dict(img="valid/", seg="valid/annotations_without_background/"),
        test_mode=True,
        pipeline=brainatlas_test_pipeline,
        return_classes=True,
        backend_args=brainatlas_backend_args,
    ),
),

brainatlas_test_dataset = dict(
    type="CocoPanopticDataset",
    metainfo=brainatlas_metainfo,
    data_root=brainatlas_data_root,
    ann_file="test/annotations_without_background.json",
    data_prefix=dict(img="test/", seg="test/annotations_without_background/"),
    test_mode=True,
    pipeline=brainatlas_test_pipeline,
    return_classes=True,
    backend_args=brainatlas_backend_args,
),

