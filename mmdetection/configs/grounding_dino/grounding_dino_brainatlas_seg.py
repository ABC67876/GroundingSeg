_base_ = [
    # "./data/adni33_seg.py",
    "./data/adni35_seg.py",
    "./data/oasis35_seg.py",
    "./data/brainatlas_seg.py",
    # "./data/oasis4_seg.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
lang_model_name = (
    "bert-base-uncased"  # 先初始化为这个。实际上load之后会被覆盖为biomedclip
)

# load_from = "/home/lihua/Desktop/projects/mmd/mmdetection/biomedbert/new_mixdata_epoch_17.pth"
# load_from = "new-brainatlas/det_unfreeze_1e-4/best_coco_bbox_mAP_epoch_10.pth"
load_from = "newnew-brainatlas/det/best_coco_bbox_mAP_epoch_7.pth"

# load_from = "biomedbert/mask2formerv3/seg/epoch_3.pth"

model = dict(
    type="GroundingDINOMask2former",
    is_seg=True,
    is_det=False,  # 本来用的is seg，后来改了代码，不想大改，额外打了个补丁is det
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
        name=lang_model_name,
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

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    # sampler=dict(_delete_=True,type="CustomSampleSizeSampler",ratio_mode=True,dataset_size=[-1, -1, -1],),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            _base_.brainatlas_train_dataset,
        ],
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            _base_.brainatlas_test_dataset,
        ],
    ),
)


val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            _base_.brainatlas_valid_dataset,
        ],
    ),
)

val_evaluator = dict(
    type="CocoPanopticMetric",
    ann_file=_base_.brainatlas_data_root + "valid/annotations_without_background.json",
    seg_prefix=_base_.brainatlas_data_root+ "valid/annotations_without_background/",
)
test_evaluator = dict(
    type="CocoPanopticMetric",
    ann_file=_base_.brainatlas_data_root + "test/annotations_without_background.json",
    seg_prefix=_base_.brainatlas_data_root + "test/annotations_without_background/",
)

max_epoch = 40

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
        milestones=[18],
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
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0),
        }
    ),
)
"""            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0),
            "neck": dict(lr_mult=0.1),
            "bbox_head": dict(lr_mult=0.1),
            "positional_encoding": dict(lr_mult=0),
            "encoder": dict(lr_mult=0.1),
            "decoder.layers.0": dict(lr_mult=0.1),
            "decoder.layers.1": dict(lr_mult=0.1),
            "decoder.layers.2": dict(lr_mult=0.1),
            "decoder.layers.3": dict(lr_mult=0.1),
            "query_embedding": dict(lr_mult=0),
            "memory_trans_fc": dict(lr_mult=0),
            "memory_trans_norm": dict(lr_mult=0),
            "text_feat_map": dict(lr_mult=0.1),
            "pixel_decoder": dict(lr_mult=1.0),
            "seg_head": dict(lr_mult=1.0),
            "FuseBeforeSeg.layers.0": dict(lr_mult=0.1),
            "FuseBeforeSeg.layers.1": dict(lr_mult=0.1),"""


auto_scale_lr = dict(base_batch_size=1)  # 实际上没用
