# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer, PipeTransformerLayer

import deepspeed
from deepspeed.pipe import PipelineModule

class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        # self.layers = nn.ModuleList(
        #     [
        #         TransformerLayer(
        #             self.embed_dim,
        #             4 * self.embed_dim,
        #             self.attention_heads,
        #             add_bias_kv=False,
        #             use_esm1b_layer_norm=True,
        #             use_rotary_embeddings=True,
        #         )
        #         for _ in range(self.num_layers)
        #     ]
        # )

        self.gpu_num = 4
        self.layers = [
            PipeTransformerLayer(
                layer_id,
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
            )
            for layer_id in range(self.num_layers)
        ]
        self.pipe_layers = PipelineModule(self.layers, num_stages=self.gpu_num)
        ds_config = {
            "train_batch_size": self.gpu_num,
            "train_micro_batch_size_per_gpu": 1, 
        }
        self.pipe_engine, _, _, _ = deepspeed.initialize(model = self.pipe_layers, config = ds_config)

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        # for layer_idx, layer in enumerate(self.layers):
        #     x, attn = layer(
        #         x,
        #         self_attn_padding_mask=padding_mask,
        #         need_head_weights=need_head_weights,
        #     )
        #     if (layer_idx + 1) in repr_layers:
        #         hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        #     if need_head_weights:
        #         # (H, B, T, T) => (B, H, T, T)
        #         attn_weights.append(attn.transpose(1, 0))

        def _get_data_iter():
            # micro_x_list = torch.chunk(x, self.gpu_num, dim=1)
            padding_mask_or_zero = padding_mask if padding_mask is not None else torch.tensor(0)
            repr_layers_tensor = torch.tensor(list(repr_layers))
            need_head_weights_tag = torch.tensor(1) if need_head_weights else torch.tensor(0)
            T, B, E = x.shape
            hidden_representations_tensor = torch.zeros(self.num_layers + 1, B, T, E)
            if 0 in repr_layers:
                hidden_representations_tensor[0] = x.transpose(0, 1)
            attn_weights_tensor = torch.zeros(self.num_layers, B, T, T)
            yield ((x, padding_mask_or_zero, repr_layers_tensor, need_head_weights_tag, hidden_representations_tensor, attn_weights_tensor), 0)

        pipe_output = self.pipe_engine.eval_batch(_get_data_iter(), compute_loss=False, reduce_output=None)
        if not self.is_last_stage():
            return 
        
        x, _, _, _, hidden_representations_tensor, attn_weights_tensor = pipe_output[0]
        hidden_representations = {idx: hidden_representations_tensor[idx] for idx in repr_layers}
        attn_weights = torch.chunk(attn_weights_tensor, self.num_layers, dim=0)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        # if (layer_idx + 1) in repr_layers:
        #     hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
    
    def is_last_stage(self):
        return self.pipe_engine.stage_id == self.gpu_num - 1
