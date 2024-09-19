# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from mmengine.structures import InstanceData, PixelData
from mmdet.structures.bbox import BaseBoxes
from mmdet.structures.mask import BitmapMasks

from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerEncoder,
    FuseBeforeSeg,
)
from .dino import DINO
from .glip import create_positive_map, create_positive_map_label_to_token, run_ner
import matplotlib.pyplot as plt
import numpy as np

from torchvision.ops.boxes import box_iou

from mmdet.registry import MODELS
import re

def find_patterns(s):   
    pattern = r'<(\d+),(\d+)>(.*?)</\1,\2>'
    matches = re.findall(pattern, s)  
    positions = []  
    for match in matches:
        n1, n2, content = match  
        start_pos = s.find(f'<{n1},{n2}>{content}</{n1},{n2}>')  
        positions.append((start_pos, start_pos + len(f'<{n1},{n2}>{content}</{n1},{n2}>'),[int(n1),int(n2)]))
    
    pattern = r'<(\d+)>(.*?)</\1>'
    matches = re.findall(pattern, s)
    for match in matches:
        n, content = match  
        start_pos = s.find(f'<{n}>{content}</{n}>')  
        positions.append((start_pos, start_pos + len(f'<{n}>{content}</{n}>'),[int(n)]))
      
    return positions


# reconstruction starts
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.is_deconv = is_deconv

        if self.is_deconv:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, mid_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                # nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        enc_out5_channels,
        enc_out4_channels,
        enc_out3_channels,
        enc_out2_channels,
        enc_out1_channels,
        out_channels,
    ):
        super(UNetDecoder, self).__init__()

        self.dec_block0 = DecoderBlock(enc_out5_channels + enc_out4_channels, 512, 256)
        self.dec_block1 = DecoderBlock(256 + enc_out3_channels, 512, 256)
        self.dec_block2 = DecoderBlock(256 + enc_out2_channels, 512, 256)
        self.dec_block3 = DecoderBlock(256 + enc_out1_channels, 256, 128)
        self.dec_block4 = DecoderBlock(128, 64, out_channels)

        self.final_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, enc_out5, enc_out4, enc_out3, enc_out2, enc_out1, images):
        # upsample enc_out5 and concat with enc_out4
        """
        x = F.interpolate(enc_out5, size=enc_out4.shape[-2:], mode="nearest")
        x = torch.cat([x, enc_out4], dim=1)
        x = self.dec_block0(x)

        # upsample and concat with enc_out3
        x = F.interpolate(x, size=enc_out3.shape[-2:], mode="nearest")
        x = torch.cat([x, enc_out3], dim=1)
        x = self.dec_block1(x)

        # upsample and concat with enc_out2
        x = F.interpolate(x, size=enc_out2.shape[-2:], mode="nearest")
        x = torch.cat([x, enc_out2], dim=1)
        x = self.dec_block2(x)

        # upsample and concat with enc_out1
        x = F.interpolate(x, size=enc_out1.shape[-2:], mode="nearest")
        x = torch.cat([x, enc_out1], dim=1)
        x = self.dec_block3(x)

        # final upsampling
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dec_block4(x)

        # final convolution to match the original image channels
        x = F.interpolate(x, size=images.shape[-2:], mode="nearest")
        x = self.final_layer(x)
        #"""
        x = self.dec_block3(enc_out1)
        x = F.interpolate(x, size=images.shape[-2:], mode="nearest")
        x = self.dec_block4(x)
        x = self.final_layer(x)

        return x


# reconstruction ends


@MODELS.register_module()
class GroundingDINOMask2former(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(
        # self, language_model, seg_head, is_seg, is_det, *args, **kwargs
        self,
        language_model,
        is_seg,
        is_det,
        *args,
        **kwargs,
    ) -> None:
        self.language_model_cfg = language_model
        self._special_tokens = "+ "# ". "
        if not is_det or True:
            self.seg_cfg = kwargs["seg_head"]
            del kwargs["seg_head"]
            self.before_seg_cfg = kwargs["before_seg_cfg"]
            del kwargs["before_seg_cfg"]

        self.is_seg = is_seg
        self.is_det = is_det
        self.save_results_list = []

        self.pixel_decoder = None
        self.pixel_decoder_cfg = kwargs["pixel_decoder"]
        del kwargs["pixel_decoder"]

        self.recon_head = None

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

        self.pixel_decoder = MODELS.build(self.pixel_decoder_cfg)
        self.pixel_decoder.postional_encoding = self.positional_encoding

        # segment head
        # if self.is_seg:
        if not self.is_det or True:
            self.seg_head = MODELS.build(self.seg_cfg)
            self.FuseBeforeSeg = FuseBeforeSeg(**self.before_seg_cfg)

        # self.recon_head = UNetDecoder(256, 256, 256, 256, 256, 1)
        # self.recon_head = UNetDecoder(256, 256, 256, 256, 0, 1)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def get_tokens_and_prompts(
        self, original_caption: Union[str, list, tuple], custom_entities: bool = False, aux_text: str = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                raise NotImplementedError
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))

            caption_string = ""
            tokens_positive = []
            if not self.training or not True: # if use brain atlas or brainPTM, set aux_text None here
                aux_text = "None"
            if aux_text != "None":
                num_of_prompts = len(aux_text)
                probabilities = [1.0/(num_of_prompts+5)] * num_of_prompts
                probabilities.append(1.0-num_of_prompts/(num_of_prompts+5.0))
                options = [i for i in range(num_of_prompts+1)] 
                choice = np.random.choice(options, p=probabilities)
                if choice==num_of_prompts:
                    aux_text = "None"
                else:
                    aux_text = aux_text[choice]
            entities=[]
            if aux_text == "None":
                if not True: # brain atlas
                    names = [
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
                    ]         
                    _texts = [
                        "Left Lateral Ventricle",
                        "Right Lateral Ventricle",
                        "Left Insula within the Left Cerebral Cortex.",
                        "Right Insula within the Right Cerebral Cortex.",
                        "Left Parietal Lobe within the Left Cerebral Cortex.",
                        "Right Parietal Lobe within the Right Cerebral Cortex.",
                        "Left Frontal Lobe within the Left Cerebral Cortex.",
                        "Right Frontal Lobe within the Right Cerebral Cortex.",
                        "Left Basal Ganglia consists of Left Caudate, Left Accumbens, Left Putamen, Left Pallidum.",
                        "Right Basal Ganglia consists of Right Caudate, Right Accumbens, Right Putamen, Right Pallidum.",
                        "Left Cingulate Gyrus within the Left Cerebral Cortex.",
                        "Right Cingulate Gyrus within the Right Cerebral Cortex.",
                        "Brain Stem",
                        "Left Temporal Lobe within the Left Cerebral Cortex.",
                        "Right Temporal Lobe within the Right Cerebral Cortex.",
                        "Left Thalamus",
                        "Right Thalamus",
                        "Left Cerebellum consists of Left Cerebellum White Matter, Left Cerebellum Cortex.",
                        "Right Cerebellum consists of Right Cerebellum White Matter, Right Cerebellum Cortex.",
                        "Left Occipital Lobe within the Left Cerebral Cortex.",
                        "Right Occipital Lobe within the Right Cerebral Cortex.",
                        "Left Hippocampus",
                        "Right Hippocampus",
                        "Left Amygdala",
                        "Right Amygdala",
                        "3rd Ventricle",
                    ]
                    # names = [names[i-1] for i in [1,2,13,16,17,22,23,24,25,26]]
                    # _texts = [_texts[i-1] for i in [1,2,13,16,17,22,23,24,25,26]]
                    for _, words in enumerate(_texts):
                        caption_string += words
                        caption_string += self._special_tokens
                    for name in names:
                        """if len(tokens_positive)+1 not in [1,2,13,16,17,22,23,24,25,26]:
                            tokens_positive.append([[tokens_positive[-1][0][1]+2, tokens_positive[-1][0][1]+3]])
                            continue"""
                        _ = caption_string.find(name)
                        tokens_positive.append(
                            [[_, _ + len(name)]]
                        )
                elif not True:#brain PTM
                    names = [
                        "Left Optic Radiation",
                        "Right Optic Radiation",
                        "Left Corticospinal Tract",
                        "Right Corticospinal Tract",
                    ]
                    _texts = [
                        "Left Optic Radiation bewteen the Left Thalamus and Left Cerebral Cortex.",
                        "Right Optic Radiation bewteen the Right Thalamus and Right Cerebral Cortex.",
                        "Left Corticospinal Tract bewteen the Left Cerebral Cortex and Brain Stem.",
                        "Right Corticospinal Tract bewteen the Right Cerebral Cortex and Brain Stem.",
                    ]
                    for words in _texts:
                        caption_string += words
                        caption_string += self._special_tokens
                    for name in names:
                        _ = caption_string.find(name)
                        tokens_positive.append(
                            [[_, _ + len(name)]]
                        )

                else:
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
            else:
                # if not aux_text.endswith("."):
                if True:
                    aux_text = aux_text + self._special_tokens
                """for idx, word in enumerate(original_caption):
                    start = aux_text.lower().find(word.lower())
                    if start==-1:
                        tokens_positive.append(
                            [[-1, -1]]
                        )
                    else:
                        tokens_positive.append(
                            [[start, start + len(word)]]
                        )
                caption_string = aux_text"""
                results = find_patterns(aux_text)
                _labels = []
                for result in results:
                    start, end, ns = result
                    entities.append(aux_text[start:end].split('>')[1].split('<')[0])

                    if all(x <= 35 for x in ns):  
                        if len(ns)<=1:
                            _labels.append(ns[0])
                        else:
                            _labels.append(ns)
                    elif len(ns)==1 and ns[0]>35:
                        _labels.append(ns[0])
                    else:
                        raise RuntimeError
                cleaned_aux_text = re.sub('<[^>]*>', '', aux_text)
                for entity in entities:
                    start = cleaned_aux_text.lower().find(entity.lower())
                    tokens_positive.append([[start, start + len(entity)]])
                caption_string = cleaned_aux_text

            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding="max_length" if self.language_model.pad_to_max else "longest",
                return_tensors="pt",
            )
            if entities==[]:
                entities = original_caption
            else: # 为了返回不一样的类型，以供外面的代码判断是否采用的是aux_text；并且把额外的信息附进来
                entities = [entities, _labels]
        else: # not used
            raise NotImplementedError
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
        out_1st_stage: Tensor = None,
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
        # """

        if not self.is_det:
            input_list = [out_1st_stage]
            current = 0
            for level in range(len(encoder_inputs_dict["spatial_shapes"])):
                pxs = (
                    encoder_inputs_dict["spatial_shapes"][level][0]
                    * encoder_inputs_dict["spatial_shapes"][level][1]
                )
                input_list.append(
                    encoder_outputs_dict["memory"][:, current : current + pxs, :]
                    .permute(0, 2, 1)
                    .reshape(
                        bs,
                        -1,
                        encoder_inputs_dict["spatial_shapes"][level][0],
                        encoder_inputs_dict["spatial_shapes"][level][1],
                    )
                )
                current += pxs
            mask_feature, multi_scale_features = self.pixel_decoder(input_list)#([x.detach() for x in input_list])
            """mask_feature_ = self.FuseBeforeSeg(
                query=encoder_outputs_dict["memory"][:,:encoder_inputs_dict["level_start_index"][1].item()],
                query_pos=encoder_inputs_dict['feat_pos'][:,:encoder_inputs_dict["level_start_index"][1].item()],
                key_padding_mask=encoder_outputs_dict["memory_mask"][:,:encoder_inputs_dict["level_start_index"][1].item()],
                spatial_shapes=encoder_inputs_dict["spatial_shapes"][:1,:],
                level_start_index=encoder_inputs_dict["level_start_index"][:1],
                valid_ratios=encoder_inputs_dict["valid_ratios"][:, :1],
                memory_text=encoder_outputs_dict["memory_text"],
                text_attention_mask=encoder_outputs_dict["text_token_mask"],
            )"""

            _aa,_bb = self.pre_transformer([mask_feature],batch_data_samples)
            mask_feature = self.FuseBeforeSeg(
                query=mask_feature.view(mask_feature.shape[0],mask_feature.shape[1],-1).permute(0,2,1),#.detach(),
                query_pos=_aa['feat_pos'],#.detach(),
                key_padding_mask=_aa["feat_mask"],
                spatial_shapes=_aa["spatial_shapes"],
                level_start_index=_aa["level_start_index"],
                valid_ratios=_aa["valid_ratios"],
                memory_text=encoder_outputs_dict["memory_text"],#.detach(),
                text_attention_mask=~encoder_outputs_dict["text_token_mask"],
            ).permute(0,2,1).reshape(bs, -1, _aa["spatial_shapes"][0][0], _aa["spatial_shapes"][0][1])
        else:
            mask_feature=None
        # this is for the previous design that detection also use features after pixel decoder:
        """encoder_outputs_dict["memory"] = torch.cat(
            [
                feature.reshape(bs, 256, -1)
                for feature in reversed(multi_scale_features[0:4])
            ],
            dim=2,
        ).permute(0, 2, 1)
        """
        # and actually the layer of 1/8 resolution is not used in segmentation, so return None instead
        
        """
        mask_feature = nn.functional.relu(
            self.pixel_decoder.lateral_convs[0](out_1st_stage)
        )
        mask_feature = self.pixel_decoder.mask_feature(mask_feature)
        # """

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
            None,# encoder_outputs_dict["memory"][:, last_layer_index:, :].permute(0, 2, 1).reshape(bs, -1, last_layer_shape[0], last_layer_shape[1]),
            feature_mask_last,
            feature_mask_first,
            deformable_positions_dict,
            mask_feature,
            None,# multi_scale_features,
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
            # Important! text token mask is ~ here
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
        # text_auxs = [data_samples.text_aux for data_samples in batch_data_samples]
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
        visual_features, out_1st_stage, src_prj = self.extract_feat(
            batch_inputs, is_seg=True
        )

        (
            head_inputs_dict,
            memory,
            feature_mask,
            feature_mask_first,
            deformable_positions_dict,
            mask_feature,
            multi_scale_features,
            # dict_for_debug,  # for debug
        ) = self.forward_transformer(
            visual_features, text_dict, batch_data_samples, out_1st_stage
        )

        # start reconstruction
        """
        reconstructed = self.recon_head(
            multi_scale_features[0],
            multi_scale_features[1],
            multi_scale_features[2],
            multi_scale_features[3],
            mask_feature,
            batch_inputs,
        )

        for batch in range(len(batch_inputs)):
            plt.figure(figsize=(8, 8))
            plt.imshow(reconstructed[batch].detach().cpu().numpy().squeeze(),cmap='gray')
            plt.rcParams["image.cmap"] = "gray"
            plt.savefig(
                "./debug/output/recon/{}_recon.png".format(
                    batch_data_samples[batch].img_path.split("/")[-1][:-4]
                )
            )
            # plt.show()
            plt.clf()
            plt.imshow(batch_inputs[batch, :1].detach().cpu().numpy().squeeze(),cmap='gray')
            plt.rcParams["image.cmap"] = "gray"
            plt.savefig(
                "./debug/output/recon/{}_origin.png".format(
                    batch_data_samples[batch].img_path.split("/")[-1][:-4]
                )
            )
        # """
        # end reconstruction

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
        seg_input_dict["mask_feature"] = mask_feature
        seg_input_dict["src_prj"] = src_prj
        # seg_input_dict["dict_for_debug"] = dict_for_debug  # for debug

        return seg_input_dict, batch_data_samples, results_list, entities

    def detect_unfreeze(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> Union[dict, list]:
        text_prompts = [data_samples.text for data_samples in batch_data_samples]

        text_auxs = [data_samples.text_aux for data_samples in batch_data_samples]

        # 下面主要是为了计算positive_maps，给detect原始的loss用
        gt_labels = [
            data_samples.gt_instances.labels for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []
        # if len(set(text_prompts)) == 1:
        if False: # 按照新的设计，如果全是之前的拼接label，就用true的来减少计算量；如果读入的有aux text，就用false的
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                text_prompts[0], True, text_auxs[0]
            )
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)

            # 下面的为了计算token_positive_maps，给pred用
            positive_map_label_to_token, _ = self.get_positive_map(
                tokenized, tokens_positive
            )
            token_positive_maps = [positive_map_label_to_token] * len(batch_inputs)
        else:
            token_positive_maps = []
            for batch_idx, (text_prompt, gt_label, text_aux) in enumerate(zip(text_prompts, gt_labels, text_auxs)):
                (
                    tokenized,
                    caption_string,
                    tokens_positive,
                    entities,
                ) = self.get_tokens_and_prompts(text_prompt, True, text_aux)
                if not isinstance(entities, list): # 原本是tuple；如果用了aux_text，就是list
                    new_tokens_positive = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
                    text_prompts = [
                        data_samples.text for data_samples in batch_data_samples
                    ]
                    # 下面的为了计算token_positive_maps，给pred用
                    positive_map_label_to_token, _ = self.get_positive_map(
                        tokenized, tokens_positive
                    )
                    token_positive_maps.append(positive_map_label_to_token)
                else:
                    entities, _labels = entities
                    positive_map_label_to_token, positive_map = self.get_positive_map(
                        tokenized, tokens_positive
                    )
                    # check labels from aux_text
                    label_map = dict()
                    consistence = {
                        36: [2, 21],
                        37: [6, 7, 8, 9, 10, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32],
                        38: [1, 5, 13, 17, 20, 24, 33],
                        39: [3, 4, 11, 12, 18, 19, 22, 23, 34, 35],
                    }
                    label_aux = []
                    _cnt = 0
                    for idx,_label in enumerate(_labels):
                        if not isinstance(_label, list):
                            _label = [_label]
                        label_aux+=_label
                        label_map[idx] = [_-1 for _ in _label]
                    # use new label to express contain relations:
                    contain_relations_new_label = dict()
                    for _label in label_aux:
                        if _label>35:
                            _keys=[key for key, val in label_map.items() if val[0] == _label-1]
                            if len(_keys)>1:
                                raise RuntimeError
                            _sub_labels = [_-1 for _ in label_aux if _ in consistence[_label]]
                            if len(_sub_labels)==0:
                                raise RuntimeError
                            label_map[_keys[0]] = _sub_labels
                            # bug fixed. this will result in empty dict item if sub labels are combined:
                            # contain_relations_new_label[_keys[0]] = [key for key, val in label_map.items() if len(val)==1 and val[0] in _sub_labels]
                            contain_relations_new_label[_keys[0]] = [key for key, val in label_map.items() if key!=_keys[0] and val[0] in _sub_labels]
                            if contain_relations_new_label[_keys[0]]==[]:
                                raise RuntimeError

                        elif _label-1 not in gt_label:
                            raise RuntimeError
                        else:
                            _cnt+=1
                    if _cnt!=len(gt_label):
                        raise RuntimeError
                    # finish checking labels from aux_text
                    # batch_data_samples[batch_idx].gt_sem_seg
                    # batch_data_samples[batch_idx].gt_instances

                    # combine the old labels innto new labels
                    # batch_data_samples[batch_idx].gt_instances is totally modified
                    # batch_data_samples[batch_idx].gt_sem_seg.sem_seg.contain_relations_new_label is for the high level label mapping for gt_sem_seg
                    instance_data = InstanceData()
                    if not self.is_det:
                        prev_gt_masks = batch_data_samples[batch_idx].gt_sem_seg.sem_seg.cpu().numpy()[0]
                    prev_gt_labels = batch_data_samples[batch_idx].gt_instances.labels.cpu().numpy()
                    new_gt_masks = []
                    device = batch_data_samples[batch_idx].gt_instances.bboxes[0].device
                    new_bbox = np.zeros((len(label_map),4))
                    new_bbox[:,:2] = 9999
                    new_bbox[:,2:] = -999
                    for idx, new_label in enumerate(label_map):
                        if not self.is_det:
                            new_gt_masks.append(np.isin(prev_gt_masks, label_map[new_label]).astype(np.uint8))
                        for _idx in np.where(np.isin(prev_gt_labels,label_map[new_label]))[0]:
                            new_bbox[idx] = [min(new_bbox[idx][0],batch_data_samples[batch_idx].gt_instances.bboxes[_idx][0].item()),
                                        min(new_bbox[idx][1],batch_data_samples[batch_idx].gt_instances.bboxes[_idx][1].item()),
                                        max(new_bbox[idx][2],batch_data_samples[batch_idx].gt_instances.bboxes[_idx][2].item()),
                                        max(new_bbox[idx][3],batch_data_samples[batch_idx].gt_instances.bboxes[_idx][3].item())]
                    instance_data["bboxes"] = torch.tensor(new_bbox,dtype=torch.float32).to(device)
                    instance_data["labels"] = torch.tensor(list(range(len(label_map)))).to(device)
                    if not self.is_det:
                        h,w = batch_data_samples[batch_idx].img_shape
                        instance_data["masks"] = BitmapMasks(new_gt_masks, h, w)
                        mapping = {255:255}
                        for new_label,old_labels in label_map.items():
                            if new_label not in contain_relations_new_label:
                                for old_label in old_labels:
                                    mapping[old_label] = new_label
                        
                        pixeldata=PixelData(metainfo={"contain_relations_new_label":contain_relations_new_label})
                        pixeldata.sem_seg = torch.tensor(np.vectorize(mapping.get)(batch_data_samples[batch_idx].gt_sem_seg.sem_seg.cpu().numpy()),dtype=torch.int32).to(device)
                        del batch_data_samples[batch_idx].gt_sem_seg
                        batch_data_samples[batch_idx].gt_sem_seg = pixeldata
                    del batch_data_samples[batch_idx].gt_instances
                    batch_data_samples[batch_idx].gt_instances = instance_data


                    # check:
                    """
                    for _ in range(batch_data_samples[batch_idx].gt_instances.masks.masks.shape[0]):
                        if _ in contain_relations_new_label:
                            continue
                        mask1 = batch_data_samples[batch_idx].gt_instances.masks.masks[_]
                        mask2 = np.where(batch_data_samples[batch_idx].gt_sem_seg.sem_seg.cpu().numpy()==_,1,0)[0]
                        if not np.all(mask1==mask2):
                            raise RuntimeError
                    """

                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
                    token_positive_maps.append(positive_map_label_to_token)

        """
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
        # """

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
        visual_features, out_1st_stage, src_prj = self.extract_feat(
            batch_inputs, is_seg=True
        )

        (
            head_inputs_dict,
            memory,
            feature_mask,
            feature_mask_first,
            deformable_positions_dict,
            mask_feature,
            multi_scale_features,
            # dict_for_debug,  # for debug
        ) = self.forward_transformer(
            visual_features, text_dict, batch_data_samples, out_1st_stage
        )

        # start reconstruction
        """
        reconstructed = self.recon_head(
            multi_scale_features[0],
            multi_scale_features[1],
            multi_scale_features[2],
            multi_scale_features[3],
            mask_feature,
            batch_inputs,
        )
        # """
        # end reconstruction

        losses, results_list = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples, need_results=True
        )
        """
        losses["reconstruction_loss"] = (
            F.mse_loss(reconstructed, batch_inputs[:, :1]) * 3
        )
        # """

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
        seg_input_dict["mask_feature"] = mask_feature
        seg_input_dict["src_prj"] = src_prj
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

        losses = dict()
        # losses["testloss"] = torch.zeros(1, requires_grad=True).to(batch_inputs.device)
        # return losses

        freeze_det = False
        if freeze_det:
            # currently only used in inference, because the aux text is not suitable for detect_pred
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
        
        if not True:# Calculate IoU of bboxes
            for data_sample, pred_instances in zip(batch_data_samples, results_list):
                if 'oasis-4' not in data_sample.img_path and False:
                    continue
                gt_labels = data_sample.gt_instances.labels
                gt_bboxes = data_sample.gt_instances.bboxes
                pred_labels = pred_instances.labels
                pred_bboxes = pred_instances.bboxes
                indices = (pred_instances.scores > 0.3).nonzero()[:, 0]
                pred_labels = pred_labels[indices]
                pred_bboxes = pred_bboxes[indices]
                for i in range(len(gt_labels)):
                    label = gt_labels[i].item()
                    iou=0
                    if label in pred_labels:
                        idx = torch.where(pred_labels==label)[0][0].item()
                        iou = box_iou(gt_bboxes[i:i+1], pred_bboxes[idx:idx+1]).item()
                    if self.save_results_list==[]:
                        self.save_results_list.append(dict())
                        self.save_results_list[0][0]=[]
                    if label+1 not in self.save_results_list[0]:
                        self.save_results_list[0][label+1] = []
                    self.save_results_list[0][label+1].append(iou)
                    self.save_results_list[0][0].append(iou)

        if self.is_det:
            return losses

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

        if not True:# Calculate IoU of bboxes
            for data_sample, pred_instances in zip(batch_data_samples, results_list):
                gt_labels = data_sample.gt_instances.labels
                gt_bboxes = data_sample.gt_instances.bboxes
                pred_labels = pred_instances.labels
                pred_bboxes = pred_instances.bboxes
                indices = (pred_instances.scores > 0.3).nonzero()[:, 0]
                pred_labels = pred_labels[indices]
                pred_bboxes = pred_bboxes[indices]
                for i in range(len(gt_labels)):
                    label = gt_labels[i].item()
                    iou=0
                    if label in pred_labels:
                        idx = torch.where(pred_labels==label)[0][0].item()
                        iou = box_iou(gt_bboxes[i:i+1], pred_bboxes[idx:idx+1]).item()
                    if self.save_results_list==[]:
                        self.save_results_list.append(dict())
                        self.save_results_list[0][0]=[]
                    if label+1 not in self.save_results_list[0]:
                        self.save_results_list[0][label+1] = []
                    self.save_results_list[0][label+1].append(iou)
                    self.save_results_list[0][0].append(iou)



        if self.is_det:
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

        seg_list = self.seg_head.predict(
            **seg_input_dict,
            batch_data_samples=batch_data_samples,
            detection_results=results_list,
            image_input=batch_inputs[:, 0:1],  # rgb相同所以取一个
        )
        # the 'pred_panoptic_seg' has a key ``sem_seg``, which is a tensor of shape (1, h, w).

        colors = np.random.rand(36, 3)
        colors[0] = [0, 0, 0]

        for data_sample, pred_instances, entity, seg in zip(
            batch_data_samples, results_list, entities, seg_list
        ):
            """
            pred = ((seg.sem_seg[0]) + 1) % 256
            gt = ((data_sample.gt_sem_seg.sem_seg[0]) + 1) % 256

            tmp = dict()
            tmp["pred"] = pred.detach().cpu().numpy()
            tmp["gt"] = gt.detach().cpu().numpy()
            tmp["img_path"] = data_sample.img_path

            self.save_results_list.append(tmp)

            segmentationpred = np.zeros(
                (pred.shape[0], pred.shape[1], 3), dtype=np.uint8
            )
            segmentationgt = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for label in range(36):
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
            # """

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
