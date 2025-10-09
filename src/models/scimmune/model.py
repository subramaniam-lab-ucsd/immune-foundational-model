# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Dict

# from transformers import PreTrainedModel
# from config import ScImmuneConfig
# # from flash_attn import flash_attention


# class ExprDecoder(nn.Module):

#     def __init__(self, d_model: int, explicit_zero_prob: bool = False):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(d_model, d_model),  # we don't use batch labels
#             nn.LeakyReLU(),
#             nn.Linear(d_model, d_model),
#             nn.LeakyReLU(),
#             nn.Linear(d_model, 1),
#         )
#         self.explicit_zero_prob = explicit_zero_prob
#         if explicit_zero_prob:
#             self.zero_logit = nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.LeakyReLU(),
#                 nn.Linear(d_model, d_model),
#                 nn.LeakyReLU(),
#                 nn.Linear(d_model, 1),
#             )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         pred_value = self.fc(x).squeeze(-1)
#         if not self.explicit_zero_prob:
#             return {"pred": pred_value}
#         zero_logits = self.zero_logit(x).squeeze(-1)
#         zero_probs = torch.sigmoid(zero_logits)
#         return {
#             "pred": pred_value,
#             "zero_probs": zero_probs
#         }  # TODO: what about inference / bernoulli?

# # class FlashTransformerEncoderLayer(nn.Module):

# #     def __init__(self,
# #                  d_model,
# #                  nhead,
# #                  dim_feedforward,
# #                  dropout,
# #                  norm_scheme="post"):
# #         super().__init__()
# #         from flash_attn.flash_attention import FlashMHA

# #         self.self_attn = FlashMHA(
# #             embed_dim=d_model,
# #             num_heads=nhead,
# #             dropout=dropout,
# #             attention_dropout=dropout,
# #         )
# #         self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward),
# #                                           nn.GELU(), nn.Dropout(dropout),
# #                                           nn.Linear(dim_feedforward, d_model))
# #         self.norm1 = nn.LayerNorm(d_model)
# #         self.norm2 = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #         self.norm_scheme = norm_scheme

# # Helper class to ensure we have the correct attention structure
# class MultiheadAttentionWithBias(nn.Module):

#     def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first

#         # Combined input projections for Q, K, V
#         self.in_proj_weight = nn.Parameter(
#             torch.empty((3 * embed_dim, embed_dim)))
#         self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

#         # Output projection
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Initialize parameters following PyTorch's MultiheadAttention initialization
#         nn.init.xavier_uniform_(self.in_proj_weight)
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         nn.init.constant_(self.in_proj_bias, 0.)
#         nn.init.constant_(self.out_proj.bias, 0.)

#     def forward(self, query, key, value, key_padding_mask=None):
#         return nn.functional.multi_head_attention_forward(
#             query,
#             key,
#             value,
#             self.embed_dim,
#             self.num_heads,
#             self.in_proj_weight,
#             self.in_proj_bias,
#             None,
#             None,
#             None,  # No bias_k, bias_v, or add_zero_attn
#             self.dropout,
#             self.out_proj.weight,
#             self.out_proj.bias,
#             training=self.training,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#             batch_first=self.batch_first)[0]

# class ScImmunePreTrainedModel(PreTrainedModel):
#     config_class = ScImmuneConfig
#     base_model_prefix = "scimmune"

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

# class ScImmuneModel(ScImmunePreTrainedModel):

#     def __init__(self, config):
#         super().__init__(config)

#         # Gene name embeddings remain the same
#         self.gene_encoder = nn.ModuleDict({
#             "embedding":
#                 nn.Embedding(config.vocab_size,
#                              config.embsize,
#                              padding_idx=config.pad_token_id),
#             "enc_norm":
#                 nn.LayerNorm(config.embsize)
#         })

#         # Value encoder remains the same
#         if config.input_emb_style == "continuous":
#             self.value_encoder = nn.ModuleDict({
#                 "linear1": nn.Linear(1, config.embsize),
#                 "linear2": nn.Linear(config.embsize, config.embsize),
#                 "norm": nn.LayerNorm(config.embsize),
#                 "dropout": nn.Dropout(config.dropout)
#             })
#         elif config.input_emb_style == "scaling":
#             self.value_encoder = nn.Identity()
#             raise Exception(
#                 "scaling input embedding style not supported because this model was trained on continuous style"
#             )
#         else:
#             raise Exception("unsupported embedding style")

#         # Modified transformer layers to use combined QKV projections
#         # self.transformer = nn.ModuleDict({
#         #     "layers": nn.ModuleList([
#         #         nn.ModuleDict({
#         #             "self_attn": MultiheadAttentionWithBias(
#         #                 config.embsize,
#         #                 config.nhead,
#         #                 dropout=config.dropout,
#         #                 batch_first=True
#         #             ),
#         #             "linear1": nn.Linear(config.embsize, config.d_hid),
#         #             "linear2": nn.Linear(config.d_hid, config.embsize),
#         #             "norm1": nn.LayerNorm(config.embsize),
#         #             "norm2": nn.LayerNorm(config.embsize),
#         #         }) for _ in range(config.nlayers)
#         #     ])
#         # })

#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.transformer = TransformerEncoder(
#             TransformerEncoderLayer(
#                 d_model=config.embsize,
#                 nhead=config.nhead,
#                 dim_feedforward=config.d_hid,
#                 dropout=config.dropout,
#                 batch_first=True,  # just for replication
#             ),
#             num_layers=config.nlayers)

#         # Decoder remains the same
#         self.expr_decoder = ExprDecoder(config.embsize,
#                                         config.explicit_zero_prob)

#         # we ignore cls_decoder because we do not pursue classification task
#         # we also ignore mvc and similarity because we ignore generative tasks

#         self.init_weights()

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         values: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         output_cell_emb: bool = True,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Args:
#             input_ids: Tensor of gene indices, shape [batch_size, seq_len]
#             values: Tensor of expression values, shape [batch_size, seq_len]
#             attention_mask: Optional mask tensor, shape [batch_size, seq_len]
#             output_cell_emb: Whether to output cell embeddings

#         Returns:
#             Dictionary containing:
#                 - 'pred': Predicted expression values
#                 - 'cell_emb': Cell embeddings (if output_cell_emb=True)
#                 - 'zero_probs': Zero probabilities (if config.explicit_zero_prob=True)
#         """
#         # Gene embeddings
#         gene_emb = self.gene_encoder["embedding"](input_ids)
#         gene_emb = self.gene_encoder["enc_norm"](gene_emb)

#         # Value encoding
#         if hasattr(self, 'value_encoder'):
#             values = values.unsqueeze(-1)  # Add feature dimension
#             value_emb = self.value_encoder["linear1"](values)
#             if "activation" in self.value_encoder:
#                 value_emb = self.value_encoder["activation"](value_emb)
#             value_emb = self.value_encoder["linear2"](value_emb)
#             value_emb = self.value_encoder["norm"](value_emb)
#             value_emb = self.value_encoder["dropout"](value_emb)

#             if self.config.input_emb_style == "continuous":
#                 hidden_states = gene_emb + value_emb
#             else:  # "scaling", currrently not supported
#                 hidden_states = gene_emb * value_emb
#         else:
#             hidden_states = gene_emb

#         # Convert attention_mask for transformer
#         # Flash attention expects mask of 0s for tokens to attend to and 1s for tokens to ignore
#         # if self.use_flash_attention and attention_mask is not None:
#         #     if attention_mask.dtype != torch.bool:
#         #         attention_mask = attention_mask.bool()
#         #     attention_mask = ~attention_mask # we assume user follows huggingface convention for the attention mask

#         # # Apply transformer layers
#         # if self.use_flash_attention:
#         #     for layer in self.transformer:
#         #         hidden_states = layer(
#         #             hidden_states,
#         #             src_key_padding_mask=attention_mask
#         #         )
#         # else:
#         hidden_states = self.transformer(hidden_states,
#                                          src_key_padding_mask=attention_mask)

#         # Get cell embeddings if requested
#         output_dict = {}
#         if output_cell_emb:
#             if self.config.cell_emb_style == "cls":
#                 cell_emb = hidden_states[:, 0]
#             elif self.config.cell_emb_style == "avg-pool":
#                 cell_emb = hidden_states.mean(dim=1)
#             else:  # w-pool
#                 # Weighted pooling using input values as weights
#                 weights = F.softmax(values, dim=1).unsqueeze(-1)
#                 cell_emb = (hidden_states * weights).sum(dim=1)
#             output_dict['cell_emb'] = cell_emb

#         # Decode expression values
#         decoder_output = self.expr_decoder(hidden_states)
#         output_dict.update(decoder_output)

#         return output_dict

#     def get_input_embeddings(self):
#         return self.gene_encoder["embedding"]

#     def set_input_embeddings(self, new_embeddings):
#         self.gene_encoder["embedding"] = new_embeddings

#     def _init_weights(self, module):
#         super()._init_weights(module)
#         if isinstance(module, nn.Linear):
#             # Additional initialization for linear layers
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)


import math
from dataclasses import dataclass
from typing import Dict, Optional, Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ----------------------------
# Config
# ----------------------------
class ScImmuneConfig(PretrainedConfig):
    model_type = "scimmune"

    def __init__(
        self,
        vocab_size: int,
        embsize: int = 512,
        nhead: int = 8,
        nlayers: int = 12,
        d_hid: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        # value encoding
        input_emb_style: str = "continuous",  # "continuous" or "category"
        n_bins: int = 51,                     # used if category
        pad_value: int = -2,                  # pad for expr
        mask_value: int = -1,                 # mask for expr
        explicit_zero_prob: bool = False,
        # routing
        use_generative_training: bool = True,
        cell_emb_style: str = "cls",          # "cls" | "avg-pool" | "w-pool"
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.nhead = nhead
        self.nlayers = nlayers
        self.d_hid = d_hid
        self.dropout = dropout

        self.input_emb_style = input_emb_style
        self.n_bins = n_bins
        self.pad_value = pad_value
        self.mask_value = mask_value
        self.explicit_zero_prob = explicit_zero_prob

        self.use_generative_training = use_generative_training
        self.cell_emb_style = cell_emb_style


# ----------------------------
# Small modules
# ----------------------------
class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: float = 512.0):
        super().__init__()
        self.lin1 = nn.Linear(1, d_model)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.max_value = max_value

    def forward(self, v: Tensor) -> Tensor:
        # v: [B, L]
        v = torch.clamp(v, max=self.max_value).unsqueeze(-1)  # [B,L,1]
        x = self.lin2(self.act(self.lin1(v)))
        x = self.norm(x)
        return self.drop(x)


class CategoryValueEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, v_bins: Tensor) -> Tensor:
        # v_bins: [B, L] long
        return self.norm(self.embedding(v_bins.long()))


class ExprDecoder(nn.Module):
    def __init__(self, d_model: int, explicit_zero_prob: bool = False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred = self.fc(x).squeeze(-1)  # [B,L]
        if not self.explicit_zero_prob:
            return {"pred": pred}
        zero_probs = torch.sigmoid(self.zero_head(x).squeeze(-1))
        return {"pred": pred, "zero_probs": zero_probs}


# ----------------------------
# Model
# ----------------------------
class ScImmunePreTrainedModel(PreTrainedModel):
    config_class = ScImmuneConfig
    base_model_prefix = "scimmune"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class ScImmuneModel(ScImmunePreTrainedModel):
    """
    Handles BOTH and PCPT modes.

    BOTH (generative_training=True):
      forward(**{
        'pcpt_gene','pcpt_expr','pcpt_key_padding_mask',
        'gen_gene','gen_key_padding_mask'
      }) -> {'pcpt_preds','gen_preds','cell_emb', ...}

    PCPT (generative_training=False):
      forward(input_ids=..., values=..., attention_mask=...)
      -> {'pred','cell_emb', ...}
    """

    def __init__(self, config: ScImmuneConfig):
        super().__init__(config)

        self.gene_encoder = GeneEncoder(config.vocab_size, config.embsize, padding_idx=config.pad_token_id)

        # Value encoder
        if config.input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(config.embsize, config.dropout)
            self._use_category = False
        elif config.input_emb_style == "category":
            self.value_encoder = CategoryValueEncoder(config.n_bins, config.embsize, padding_idx=config.pad_value)
            self._use_category = True
        else:
            raise ValueError("input_emb_style must be 'continuous' or 'category'.")

        # Standard Transformer (no FlashAttention dependency)
        enc_layer = TransformerEncoderLayer(
            d_model=config.embsize,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=config.nlayers)

        self.expr_decoder = ExprDecoder(config.embsize, explicit_zero_prob=config.explicit_zero_prob)
        self.config = config
        self.init_weights()

    # ---- Shared helpers ----
    def _make_total_emb(self, token_ids: Tensor, values: Tensor) -> Tensor:
        tok = self.gene_encoder(token_ids)          # [B,L,E]
        val = self.value_encoder(values)            # [B,L,E]
        return tok + val                            # scGPT-style sum fusion

    def _cell_emb(self, hidden: Tensor, values: Optional[Tensor] = None) -> Tensor:
        if self.config.cell_emb_style == "cls":
            return hidden[:, 0]
        elif self.config.cell_emb_style == "avg-pool":
            return hidden.mean(dim=1)
        else:  # w-pool (values required)
            assert values is not None, "values required for w-pool"
            w = F.softmax(values, dim=1).unsqueeze(-1)
            return (hidden * w).sum(dim=1)

    # ---- PCPT path ----
    def perceptual_forward(
        self,
        input_ids: Tensor,
        values: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        input_ids: [B,L] gene IDs
        values:    [B,L] expr (continuous or bins)
        attention_mask: [B,L] 1=keep, 0=pad (HF convention) or bool False/True;
                         we convert to PyTorch encoder mask where True=ignore
        """
        # Convert mask to src_key_padding_mask (True = ignore)
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                src_kpm = ~attention_mask.bool()
            else:
                src_kpm = attention_mask
        else:
            src_kpm = None

        total = self._make_total_emb(input_ids, values)         # [B,L,E]
        hidden = self.encoder(total, src_key_padding_mask=src_kpm)
        out = self.expr_decoder(hidden)                          # {'pred', ...}

        cell_emb = self._cell_emb(hidden, values if self.config.cell_emb_style == "w-pool" else None)
        out["cell_emb"] = cell_emb
        return out

    # ---- BOTH (PCPT + GEN) path ----
    def generative_forward(
        self,
        pcpt_gene: torch.Tensor,
        pcpt_expr: torch.Tensor,
        pcpt_key_padding_mask: torch.Tensor,
        gen_gene: torch.Tensor,
        gen_key_padding_mask: torch.Tensor,
        input_cell_emb: Optional[torch.Tensor] = None,   # <-- add this
    ) -> Dict[str, torch.Tensor]:

        # Perceptual stream
        pcpt_tok  = self.gene_encoder(pcpt_gene)      # (B,P,E)
        pcpt_val  = self.value_encoder(pcpt_expr)     # (B,P,E)
        pcpt_total = pcpt_tok + pcpt_val              # (B,P,E)

        # If provided, overwrite CLS (position 0) with the given cell embedding
        if input_cell_emb is not None:
            # expect input_cell_emb: (B, E)
            if input_cell_emb.dim() == 2 and input_cell_emb.size(1) == self.config.embsize:
                pcpt_total[:, 0, :] = input_cell_emb
            else:
                raise ValueError(
                    f"input_cell_emb shape {tuple(input_cell_emb.shape)} must be (B,{self.config.embsize})"
                )

        # Encode perceptual
        pcpt_hidden = self.encoder(pcpt_total, src_key_padding_mask=pcpt_key_padding_mask)

        # Generative tokens (IDs only)
        gen_tok = self.gene_encoder(gen_gene)         # (B,G,E)

        # Concatenate and mix
        full_inp  = torch.cat([pcpt_hidden, gen_tok], dim=1)   # (B,P+G,E)
        full_kpm  = torch.cat([pcpt_key_padding_mask, gen_key_padding_mask], dim=1)
        mixed     = self.encoder(full_inp, src_key_padding_mask=full_kpm)

        # Decode and split
        dec       = self.expr_decoder(mixed)
        full_pred = dec["pred"]
        P         = pcpt_gene.size(1)
        out = {
            "pcpt_preds": full_pred[:, :P],
            "gen_preds":  full_pred[:, P:],
            "cell_emb":   self._cell_emb(mixed),
        }
        if "zero_probs" in dec:
            out["zero_probs"] = dec["zero_probs"][:, P:]
        return out

    # ---- HF forward router ----
    def forward(self, *args, **kwargs) -> Dict[str, Tensor]:
        """
        Route by kwarg 'generative_training' (preferred) or config.use_generative_training.
        """
        if "generative_training" in kwargs:
            do_gen = bool(kwargs.pop("generative_training"))
        else:
            do_gen = bool(self.config.use_generative_training)

        if do_gen:
            return self.generative_forward(**kwargs)
        else:
            return self.perceptual_forward(**kwargs)

    # HF embedding helpers
    def get_input_embeddings(self):
        return self.gene_encoder.embedding

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.gene_encoder.embedding = new_embeddings
