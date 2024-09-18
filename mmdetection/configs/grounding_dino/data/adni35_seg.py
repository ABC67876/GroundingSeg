adni35_data_root = "/home/lihua/Desktop/GLIP-main/GLIP-main/DATASET/adni-35-test/"
adni35_backend_args = None
adni35_class_name = (
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


adni35_num_classes = len(adni35_class_name)
# adni35_metainfo = dict(classes=adni35_class_name, palette=[(220, 20, 60)])
adni35_metainfo = dict(
    classes=adni35_class_name,
    thing_classes=adni35_class_name,
    stuff_classes=(),
    palette=generate_unique_coordinates(adni35_num_classes),
)

adni35_train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=adni35_backend_args),
    dict(
        type="LoadPanopticAnnotations", backend_args=adni35_backend_args
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


adni35_test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=adni35_backend_args),
    # dict(type="FixScaleResize", scale=(800, 1333), keep_ratio=True),
    dict(type="LoadPanopticAnnotations", backend_args=adni35_backend_args),
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


adni35_train_dataset = dict(
    type="RepeatDataset",
    times=1,
    dataset=dict(
        type="CocoPanopticDataset",
        metainfo=adni35_metainfo,
        data_root=adni35_data_root,
        ann_file="train/annotations_without_background.json",
        data_prefix=dict(img="train/", seg="train/annotations_without_background/"),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=adni35_train_pipeline,
        return_classes=True,
        backend_args=adni35_backend_args,
    ),
),

adni35_valid_dataset = dict(
    type="CocoPanopticDataset",
    metainfo=adni35_metainfo,
    data_root=adni35_data_root,
    ann_file="valid/annotations_without_background.json",
    data_prefix=dict(img="valid/", seg="valid/annotations_without_background/"),
    test_mode=True,
    pipeline=adni35_test_pipeline,
    return_classes=True,
    backend_args=adni35_backend_args,
),

adni35_test_dataset = dict(
    type="CocoPanopticDataset",
    metainfo=adni35_metainfo,
    data_root=adni35_data_root,
    ann_file="test/annotations_without_background.json",
    data_prefix=dict(img="test/", seg="test/annotations_without_background/"),
    test_mode=True,
    pipeline=adni35_test_pipeline,
    return_classes=True,
    backend_args=adni35_backend_args,
),

