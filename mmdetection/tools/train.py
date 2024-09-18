# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

from open_clip import create_model_from_pretrained, get_tokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--auto-scale-lr", action="store_true", help="enable automatically scaling LR."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If specify checkpoint path, resume from it, while if not "
        "specify, try to auto resume from the latest checkpoint "
        "in the work directory.",
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args( 
        # "configs/grounding_dino/mask2former.py --work-dir biomedbert/gpt/seg --resume biomedbert/gpt/seg/epoch_1.pth".split()
        # "configs/grounding_dino/mask2former.py --resume biomedbert/gpt/seg_fixed/epoch_2_.pth".split()
        # "configs/grounding_dino/mask2former.py --work-dir biomedbert/mask2formerv3_fixed/seg_adni_no_aux --resume biomedbert/mask2formerv3_fixed/seg_adni_no_aux/epoch_3.pth".split()
        "configs/grounding_dino/grounding_dino_brainatlas_seg.py".split() #--work-dir new-brainptm/seg".split()
        # "configs/grounding_dino/grounding_dino_brainatlas_seg.py --work-dir newnew-brainatlas/seg --resume newnew-brainatlas/seg/epoch_2.pth".split()
        # "configs/grounding_dino/brainatlas_seg.py".split()
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_33_small_mask2former_detect.py --work-dir biomedbert/gpt/det".split()
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_adni_33_small_mask2former.py --work-dir biomedbert/withoutrecon/seg/adni33 --resume /home/lihua/Desktop/projects/mmd/mmdetection/biomedbert/withoutrecon/seg/adni33/epoch_8.pth".split()
        # "configs/grounding_dino/mask2former.py --work-dir biomedbert/withoutrecon/seg/merged --resume biomedbert/withoutrecon/seg/merged/epoch_3.pth".split()
        # "configs/grounding_dino/grounding_dino_swin-t_finetune_oasis_35_mask2former.py --work-dir oasis35/mask2formerv2".split()
    )
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
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

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                " configuration file."
            )

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # replace bert with biomedclip bert
    """
    if cfg.model.language_model.use_biomedclip:
        biomedclip, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        # tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        device = next(runner.model.parameters()).device
        runner.model.language_model.language_backbone.body.model = (
            biomedclip.text.transformer.to(device)
        )
        # runner.model.language_model.tokenizer = tokenizer.tokenizer
    # """

    for name, param in runner.model.named_parameters():
        if "language_model" in name:
        # if "bbox_head" not in name and "decoder." not in name[:8] and "pixel_decoder" not in name and "seg_head" not in name and "FuseBeforeSeg" not in name:
            param.requires_grad = False
            
    # start training
    runner.train()


if __name__ == "__main__":
    main()

"""
Q&A:Do I need to reinstall mmdet after some code modifications

If you follow the best practice and install mmdet with pip install -e ., 
any local modifications made to the code will take effect without reinstallation.
"""
