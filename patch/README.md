1. To use monaimetrics，set is_new_metrics=True in def _compute_batch_pq_stats(self, data_samples: Sequence[dict], is_new_metrics) inside mmdetection/mmdet/evaluation/metrics/coco_panoptic_metric.py, and uncomment content related to monai in envs/NAME/lib/python3.10/site-packages/mmengine/evaluator/metric.py
2. During development, we directly modify the following files for convenience. Replace the original files with the provided ones:
envs/NAME/lib/python3.10/site-packages/panopticapi/evaluation.py  envs/NAME/lib/python3.10/site-packages/mmcv/ops/multi_scale_deform_attn.py
