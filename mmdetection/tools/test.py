# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

from open_clip import create_model_from_pretrained, get_tokenizer
import numpy as np


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args(
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth".split(" ")
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_oasis_35_mask2former.py biomedbert/withoutrecon/seg/merged/epoch_22.pth".split(
        
        # "configs/grounding_dino/grounding_dino_brainatlas_seg.py new-brainatlas/seg_unfreeze_1e-4/seen-nofilter-balanced/best_coco_panoptic_PQ_epoch_14.pth".split( # new-brainatlas/seg/epoch_11.pth".split(
        # "configs/grounding_dino/grounding_dino_brainatlas_seg.py newnew-brainatlas/seg/best_coco_panoptic_PQ_epoch_19.pth".split(
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_33_small_mask2former.py biomedbert/mask2formerv3/seg/epoch_3.pth".split(
        "configs/grounding_dino/grounding_dino_brainatlas_seg.py /home/lihua/Desktop/projects/mmd/mmdetection/newnew-brainatlas/seg/best_coco_panoptic_PQ_epoch_19.pth".split(

        # "configs/grounding_dino/grounding_dino_brainptm_seg.py brainptm/seg-bs16/epoch_2.pth".split(
        # "configs/grounding_dino/mask2former_detect_brainptm.py brainptm/seg-bs16/epoch_2.pth".split(
        ###############################################################################################
        # "configs/grounding_dino/mask2former.py latest_model_weights.pth".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_oasis_4_mask2former_detect.py biomedbert/gpt/det/epoch_7.pth".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_oasis_4_mask2former_detect.py biomedbert/withoutrecon/det/merged/epoch_7.pth".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_oasis_35_mask2former.py /home/lihua/Desktop/projects/mmd/mmdetection/oasis35/mask2formerv2/best_coco_panoptic_PQ_epoch_6.pth".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_detect_larger.py /home/lihua/Desktop/projects/mmd/mmdetection/adni_all_det/latest_model_weights.pth --out ./result.pkl".split(  # latest_model_weights.pth --out ./result.pkl".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_modified_from_cat.py /home/lihua/Desktop/projects/mmd/mmdetection/adni-seg-detectpretrain/epoch_20.pth".split(
            # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_modified_from_cat_seg.py /home/lihua/Desktop/projects/mmd/mmdetection/adni-seg/epoch_150.pth".split(
            " "
        )
    )
    # 如果要用monaimetrics，需要mmdetection/mmdet/evaluation/metrics/coco_panoptic_metric.py中的def _compute_batch_pq_stats(self, data_samples: Sequence[dict], is_new_metrics)中的is_new_metrics改为True
    # 还要把miniconda3/envs/dino/lib/python3.10/site-packages/mmengine/evaluator/metric.py中monai的部分取消注释
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        if "tta_model" not in cfg:
            warnings.warn(
                "Cannot find ``tta_model`` in config, " "we will set it as default."
            )
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            warnings.warn(
                "Cannot find ``tta_pipeline`` in config, " "we will set it as default."
            )
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg["dataset"]
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
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
                            ),
                        )
                    ],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # replace bert with biomedclip bert
    # """
    biomedclip, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    # tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    device = next(runner.model.parameters()).device
    runner.model.language_model.language_backbone.body.model = (
        biomedclip.text.transformer.to(device)
    )
    runner.model.language_model.training = False
    # runner.model.language_model.tokenizer = tokenizer.tokenizer
    # """

    # start testing
    runner.test()
    import pickle

    # with open("/home/lihua/Desktop/GLIP-main/GLIP-main/new_pred.pkl", "wb") as f:
    #    pickle.dump(runner.model.save_results_list, f)
    for label in runner.model.save_results_list[0].keys():
        ious = np.array(runner.model.save_results_list[0][label])
        print("label={},iou mean={}".format(label, ious.mean()))
    print(
        "overall mean={}".format(np.array(runner.model.save_results_list[0][0]).mean())
    )


if __name__ == "__main__":
    main()
