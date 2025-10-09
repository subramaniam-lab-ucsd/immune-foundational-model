from transformers import PretrainedConfig
from torch import cuda
from typing import Optional, Dict


class ScImmuneConfig(PretrainedConfig):
    model_type = "scimmune"

    def __init__(
            self,
            vocab_size=60697,
            embsize=512,
            d_hid=512,
            nlayers=12,
            nhead=8,
            max_seq_len=1536,
            dropout=0.0,
            pad_token_id=0,
            use_fast_transformer=True,
            input_emb_style="continuous",
            cell_emb_style="cls",  # output embedding vector with
            norm_scheme="post",
            explicit_zero_prob=False,
            use_flash_attention=True,
            **kwargs):
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.use_fast_transformer = use_fast_transformer
        if input_emb_style not in ["continuous"]:
            raise ValueError(
                f"Invalid input_emb_style: {input_emb_style}. Only continuous embeddings currently supported."
            )
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.norm_scheme = norm_scheme
        self.explicit_zero_prob = explicit_zero_prob
        self.use_flash_attention = self.use_fast_transformer and cuda.is_available(
        ) and use_flash_attention
        super().__init__(pad_token_id=pad_token_id, **kwargs)
