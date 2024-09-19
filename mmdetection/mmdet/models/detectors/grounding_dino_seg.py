# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerEncoder,
)
from .dino import DINO
from .glip import create_positive_map, create_positive_map_label_to_token, run_ner
import matplotlib.pyplot as plt
import numpy as np


@MODELS.register_module()
class GroundingDINOSeg(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, language_model, seg_head, is_seg, *args, **kwargs) -> None:
        self.language_model_cfg = language_model
        self._special_tokens = ". "
        self.seg_cfg = seg_head
        self.is_seg = is_seg
        self.save_results_list = []
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True,
        )

        # segment head
        # if self.is_seg:
        if True:
            self.seg_head = MODELS.build(self.seg_cfg)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def get_tokens_and_prompts(
        self, original_caption: Union[str, list, tuple], custom_entities: bool = False
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))

            caption_string = ""
            tokens_positive = []
            for idx, word in enumerate(original_caption):
                tokens_positive.append(
                    [[len(caption_string), len(caption_string) + len(word)]]
                )
                caption_string += word
                caption_string += self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding="max_length" if self.language_model.pad_to_max else "longest",
                return_tensors="pt",
            )
            entities = original_caption
        else:
            if not original_caption.endswith("."):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding="max_length" if self.language_model.pad_to_max else "longest",
                return_tensors="pt",
            )
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1
        )
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self, original_caption: Union[str, list, tuple], custom_entities: bool = False
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        (
            tokenized,
            caption_string,
            tokens_positive,
            entities,
        ) = self.get_tokens_and_prompts(original_caption, custom_entities)
        positive_map_label_to_token, positive_map = self.get_positive_map(
            tokenized, tokens_positive
        )
        return positive_map_label_to_token, caption_string, positive_map, entities

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        bs = img_feats[0].shape[0]

        last_layer_index = encoder_inputs_dict["level_start_index"][-1]
        last_layer_shape = encoder_inputs_dict["spatial_shapes"][-1]

        first_layer_end_index = encoder_inputs_dict["level_start_index"][1]
        first_layer_shape = encoder_inputs_dict["spatial_shapes"][0]

        if encoder_inputs_dict["feat_mask"] != None:
            feature_mask_last = encoder_inputs_dict["feat_mask"][
                :, last_layer_index:
            ].reshape(-1, last_layer_shape[0], last_layer_shape[1])

            feature_mask_first = encoder_inputs_dict["feat_mask"][
                :, :first_layer_end_index
            ].reshape(-1, first_layer_shape[0], first_layer_shape[1])
        else:
            # 有时候batch内图片尺寸相同，mask为none
            feature_mask_last = torch.full(
                (bs, last_layer_shape[0], last_layer_shape[1]), False, dtype=torch.bool
            ).to(last_layer_shape.device)
            feature_mask_first = torch.full(
                (bs, first_layer_shape[0], first_layer_shape[1]),
                False,
                dtype=torch.bool,
            ).to(last_layer_shape.device)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict
        )
        # memory: shape(8,?,256) memory_text: shape(8,96,256)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)

        dict_for_debug = dict()
        """
        dict_for_debug["mask"] = encoder_inputs_dict["feat_mask"]
        dict_for_debug["index"] = encoder_inputs_dict["level_start_index"]
        dict_for_debug["shape"] = encoder_inputs_dict["spatial_shapes"]
        dict_for_debug["memory"] = encoder_outputs_dict["memory"]
        """

        decoder_outputs_dict, deformable_positions_dict = self.forward_decoder(
            **decoder_inputs_dict
        )
        head_inputs_dict.update(decoder_outputs_dict)
        return (
            head_inputs_dict,
            encoder_outputs_dict["memory"][:, last_layer_index:, :]
            .permute(0, 2, 1)
            .reshape(bs, -1, last_layer_shape[0], last_layer_shape[1]),
            feature_mask_last,
            feature_mask_first,
            deformable_positions_dict,
            # dict_for_debug,  # for debug only
        )

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: Dict,
    ) -> Dict:
        text_token_mask = text_dict["text_token_mask"]
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict["embedded"],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["masks"],
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
        )
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory, memory_text, text_token_mask
        )
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]

        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        # append text_feats to head_inputs_dict
        head_inputs_dict["memory_text"] = memory_text
        head_inputs_dict["text_token_mask"] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def detect_pred(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> Union[dict, list]:
        text_prompts = [data_samples.text for data_samples in batch_data_samples]
        if "custom_entities" in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompts[0], custom_entities)
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt, custom_entities)
                for text_prompt in text_prompts
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts
        )

        # extract text feats
        text_dict = self.language_model(list(text_prompts))
        # text feature map layer
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            data_samples.token_positive_map = token_positive_maps[i]

        # image feature extraction
        visual_features, out_1st_stage = self.extract_feat(batch_inputs, is_seg=True)

        (
            head_inputs_dict,
            memory,
            feature_mask,
            feature_mask_first,
            deformable_positions_dict,
            # dict_for_debug,  # for debug
        ) = self.forward_transformer(visual_features, text_dict, batch_data_samples)

        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )

        # inside loss: forward(self, hs, memory, src_proj, features)
        seg_input_dict = dict()
        seg_input_dict["hs"] = head_inputs_dict["hidden_states"]
        seg_input_dict["memory"] = memory
        seg_input_dict["features"] = visual_features
        seg_input_dict["mask"] = feature_mask
        seg_input_dict["mask_result"] = feature_mask_first
        seg_input_dict["deformable_positions_dict"] = deformable_positions_dict
        seg_input_dict["out_1st_stage"] = out_1st_stage
        # seg_input_dict["dict_for_debug"] = dict_for_debug  # for debug

        return seg_input_dict, batch_data_samples, results_list, entities

    def detect_unfreeze(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> Union[dict, list]:
        text_prompts = [data_samples.text for data_samples in batch_data_samples]

        # 下面主要是为了计算positive_maps，给detect原始的loss用
        gt_labels = [
            data_samples.gt_instances.labels for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                text_prompts[0], True
            )
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                (
                    tokenized,
                    caption_string,
                    tokens_positive,
                    _,
                ) = self.get_tokens_and_prompts(text_prompt, True)
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)
                text_prompts = [
                    data_samples.text for data_samples in batch_data_samples
                ]

        # 下面的为了计算token_positive_maps，给pred用
        if "custom_entities" in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompts[0], custom_entities)
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt, custom_entities)
                for text_prompt in text_prompts
            ]
        token_positive_maps, _, _, _ = zip(*_positive_maps_and_prompts)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            # 这项来自于detect_pred：
            data_samples.token_positive_map = token_positive_maps[i]
            # 这些项来自于detect原始的loss：
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict["text_token_mask"][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(
                0
            ).repeat(len(positive_map), 1)

        # image feature extraction
        visual_features, out_1st_stage = self.extract_feat(batch_inputs, is_seg=True)

        (
            head_inputs_dict,
            memory,
            feature_mask,
            feature_mask_first,
            deformable_positions_dict,
            # dict_for_debug,  # for debug
        ) = self.forward_transformer(visual_features, text_dict, batch_data_samples)

        losses, results_list = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples, need_results=True
        )

        # inside loss: forward(self, hs, memory, src_proj, features)
        num_denoising = head_inputs_dict["dn_meta"]["num_denoising_queries"]
        seg_input_dict = dict()
        seg_input_dict["hs"] = head_inputs_dict["hidden_states"][
            :, :, num_denoising:, :
        ]
        seg_input_dict["memory"] = memory
        seg_input_dict["features"] = visual_features
        seg_input_dict["mask"] = feature_mask
        seg_input_dict["mask_result"] = feature_mask_first

        deformable_positions_dict["attention_weights"] = deformable_positions_dict[
            "attention_weights"
        ][:, num_denoising:]
        deformable_positions_dict["sampling_locations"] = deformable_positions_dict[
            "sampling_locations"
        ][:, num_denoising:]

        seg_input_dict["deformable_positions_dict"] = deformable_positions_dict
        seg_input_dict["out_1st_stage"] = out_1st_stage
        # seg_input_dict["dict_for_debug"] = dict_for_debug  # for debug

        return seg_input_dict, batch_data_samples, results_list, losses

    """
    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = False,
    ) -> Union[dict, list]:
        if self.is_seg:
            return self.loss_seg(batch_inputs, batch_data_samples, rescale)

        text_prompts = [data_samples.text for data_samples in batch_data_samples]

        gt_labels = [
            data_samples.gt_instances.labels for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                text_prompts[0], True
            )
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                (
                    tokenized,
                    caption_string,
                    tokens_positive,
                    _,
                ) = self.get_tokens_and_prompts(text_prompt, True)
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict["text_token_mask"][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(
                0
            ).repeat(len(positive_map), 1)

        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict, _, _, _ = self.forward_transformer(
            visual_features, text_dict, batch_data_samples
        )

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = False):
        if self.is_seg:
            return self.predict_seg(batch_inputs, batch_data_samples, rescale)

        text_prompts = [data_samples.text for data_samples in batch_data_samples]
        if "custom_entities" in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompts[0], custom_entities)
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt, custom_entities)
                for text_prompt in text_prompts
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts
        )
        # extract text feats
        text_dict = self.language_model(list(text_prompts))
        # text feature map layer
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            data_samples.token_positive_map = token_positive_maps[i]

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        head_inputs_dict, _, _, _ = self.forward_transformer(
            visual_feats, text_dict, batch_data_samples
        )
        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )
        for data_sample, pred_instances, entity in zip(
            batch_data_samples, results_list, entities
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        warnings.warn(
                            "The unexpected output indicates an issue with "
                            "named entity recognition. You can try "
                            "setting custom_entities=True and running "
                            "again to see if it helps."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
    """

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = False,
    ) -> Union[dict, list]:
        # losses = dict()
        # losses["testloss"] = torch.zeros(1, requires_grad=True).to(batch_inputs.device)
        # return losses

        losses = dict()

        freeze_det = False
        if freeze_det:
            with torch.no_grad():
                self.training = False
                seg_input_dict, batch_data_samples, results_list = self.detect_pred(
                    batch_inputs, batch_data_samples, rescale
                )
                # seg_input_dict["out_1st_stage"] = out_1st_stage
                self.training = True
        else:
            seg_input_dict, batch_data_samples, results_list, det_losses = (
                self.detect_unfreeze(batch_inputs, batch_data_samples, rescale)
            )
            losses.update(det_losses)

        seg_loss = self.seg_head.loss(
            **seg_input_dict,
            batch_data_samples=batch_data_samples,
            detection_results=results_list,
            image_input=batch_inputs[:, 0:1],  # rgb相同所以取一个
        )
        losses.update(seg_loss)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = False):
        # assert rescale

        seg_input_dict, batch_data_samples, results_list, entities = self.detect_pred(
            batch_inputs, batch_data_samples, rescale
        )

        seg_list = self.seg_head.predict(
            **seg_input_dict,
            batch_data_samples=batch_data_samples,
            detection_results=results_list,
            image_input=batch_inputs[:, 0:1],  # rgb相同所以取一个
        )
        # the 'pred_panoptic_seg' has a key ``sem_seg``, which is a tensor of shape (1, h, w).

        colors = np.random.rand(30, 3)
        colors[0] = [0, 0, 0]

        for data_sample, pred_instances, entity, seg in zip(
            batch_data_samples, results_list, entities, seg_list
        ):
            # """
            pred = ((seg.sem_seg[0]) + 1) % 256
            gt = ((data_sample.gt_sem_seg.sem_seg[0]) + 1) % 256

            tmp = dict()
            tmp["pred"] = pred.detach().cpu().numpy()
            tmp["gt"] = gt.detach().cpu().numpy()
            tmp["img_path"] = data_sample.img_path

            self.save_results_list.append(tmp)

            """
            segmentationpred = np.zeros(
                (pred.shape[0], pred.shape[1], 3), dtype=np.uint8
            )
            segmentationgt = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for label in range(30):
                segmentationpred[pred.detach().cpu() == label] = colors[label] * 255
                segmentationgt[gt.detach().cpu() == label] = colors[label] * 255

            plt.figure(figsize=(8, 8))
            plt.imshow(segmentationpred)
            plt.savefig(
                "./debug/output/{}_pred.png".format(
                    data_sample.img_path.split("/")[-1][:-4]
                )
            )
            # plt.show()
            plt.clf()
            plt.imshow(segmentationgt)
            plt.savefig(
                "./debug/output/{}_gt.png".format(
                    data_sample.img_path.split("/")[-1][:-4]
                )
            )
            # plt.show()
            plt.close()
            """

            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        warnings.warn(
                            "The unexpected output indicates an issue with "
                            "named entity recognition. You can try "
                            "setting custom_entities=True and running "
                            "again to see if it helps."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
            data_sample.pred_panoptic_seg = seg
        return batch_data_samples
        # the return value format is identical to panoptic_two_stage_segmentor in which semantic head is panoptic_fpn_head
