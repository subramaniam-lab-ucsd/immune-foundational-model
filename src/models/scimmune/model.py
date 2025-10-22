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
