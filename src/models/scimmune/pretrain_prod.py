#!/usr/bin/env python3
import os
import argparse
from typing import Optional


import torch

from datasets import load_from_disk
from transformers import set_seed

# your local modules
from tokenizer import ScImmuneTokenizer
from config import ScImmuneConfig
from model import ScImmuneModel
from collator import ScImmuneDataCollator
from trainer import ScImmunePretrainingTrainer, ScImmuneTrainingArguments



def build_datasets(
    tokenized_data_path: str,
    cls_token_id: int,
    num_metadata_tokens: int,
    skip_postprocess: bool = False,
):
    ds = load_from_disk(tokenized_data_path)

    if not skip_postprocess:
        def ensure_prefix_and_zeros(example):
            # unify values -> expressions, prepend zeros for metadata tokens
            vals = example.get("values", None)
            if vals is None:
                vals = example["expressions"]
            if isinstance(vals, list):
                vals = torch.tensor(vals, dtype=torch.float32)

            zeros = torch.zeros(num_metadata_tokens, dtype=vals.dtype)
            example["expressions"] = torch.cat((zeros, vals))

            # move <cls> to front if present
            genes = example["genes"]
            if isinstance(genes, list):
                genes = torch.tensor(genes, dtype=torch.long)
            pos = (genes == cls_token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0 and pos[0].item() != 0:
                i = pos[0].item()
                genes = torch.cat([genes[i:i+1], genes[:i], genes[i+1:]])
            example["genes"] = genes
            return example

        ds = ds.map(ensure_prefix_and_zeros, desc="Post-processing: zeros + <cls>")

    # Only keep what Trainer/collator need
    ds = ds.with_format(type="torch", columns=["genes", "expressions"])
    splits = ds.train_test_split(test_size=0.02, shuffle=True, seed=42)
    return splits["train"], splits["test"]


def make_collator(
    pad_token_id: int,
    num_metadata_tokens: int,
    max_len: int,
    do_binning: bool,
):
    return ScImmuneDataCollator(
        do_padding=True,
        pad_token_id=pad_token_id,
        pad_value=-2,                 # sentinel for padded expr
        do_mlm=True,
        do_binning=do_binning,        # False → continuous, True → 51-bin category
        mlm_probability=0.15,
        mask_value=-1,                # sentinel for masked expr
        max_length=max_len,
        sampling=True,
        keep_first_n_tokens=1 + num_metadata_tokens,   # <cls> + metadata prefix
        data_style="both",            # perceptual + generative
        scale_factor=100.0
    )


def parse_args():
    p = argparse.ArgumentParser("Continual pretraining for ScImmune")
    p.add_argument("--tokenized_data_path", type=str,
                   help="HF dataset saved with save_to_disk (contains genes/expressions)", default="scimmune-model/tokenized_data/")
    p.add_argument("--vocab_file", type=str, default="vocab_with_metadata.json")
    p.add_argument("--base_model_dir", type=str,
                   help="Folder with config.json + model.safetensors (your starting checkpoint)", default="scImmune_metadata_model/")
    p.add_argument("--output_dir", type=str, default="runs/scimmune-ctpt")
    p.add_argument("--num_metadata_tokens", type=int, default=6)
    p.add_argument("--max_len", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=18)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_binning", action="store_true",
                   help="If set, collator bins expr (category style); else continuous.")
    p.add_argument("--skip_postprocess", action="store_true",
                   help="Skip zeros prepend + <cls> move (if already saved that way).")
    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    return p.parse_args()


def main():
    args = parse_args()

    # --- Optional env knobs (feel free to remove if you handle elsewhere) ---
    os.environ.setdefault("WANDB_DISABLED", "true" if args.report_to == "none" else "false")
    # Single-node reliability with NCCL; adjust iface to your NIC if needed (e.g., "eth0", "enp134s0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    # os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # uncomment if needed

    set_seed(args.seed)

    # 1) Tokenizer
    tokenizer = ScImmuneTokenizer(vocab_file=args.vocab_file)
    cls_token_id = tokenizer.convert_tokens_to_ids("<cls>")

    # 2) Data
    train_dataset, eval_dataset = build_datasets(
        tokenized_data_path=args.tokenized_data_path,
        cls_token_id=cls_token_id,
        num_metadata_tokens=args.num_metadata_tokens,
        skip_postprocess=args.skip_postprocess,
    )

    # 3) Collator
    collator = make_collator(
        pad_token_id=tokenizer.pad_token_id,
        num_metadata_tokens=args.num_metadata_tokens,
        max_len=args.max_len,
        do_binning=args.use_binning,
    )

    # 4) Config & Model
    cfg = ScImmuneConfig.from_pretrained(
        args.base_model_dir,
        input_emb_style="continuous" if not args.use_binning else "category",
        pad_value=-2,
        mask_value=-1,
        use_generative_training=True,     # because data_style="both"
    )
    cfg.vocab_size   = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id
    if hasattr(cfg, "max_seq_len"):
        cfg.max_seq_len = max(getattr(cfg, "max_seq_len", args.max_len), args.max_len)

    model = ScImmuneModel.from_pretrained(args.base_model_dir, config=cfg)
    # in case vocab grew (metadata tokens), resize embeddings
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass  # some custom models don’t expose this cleanly; safe to ignore if sizes already match

    # 5) Training args
    training_args = ScImmuneTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=3,
        fp16=True,                          # set bf16=True instead if you want
        dataloader_num_workers=8,           # start conservative; raise once stable
        dataloader_pin_memory=False,        # ditto
        seed=args.seed,
        remove_unused_columns=False,        # IMPORTANT with custom collator/model
        report_to=None if args.report_to == "none" else args.report_to,

        # scGPT-ish extras your Trainer/Collator read:
        mlm_probability=0.15,
        max_length=args.max_len,
        MVC=False,
        scale_factor=100.0,
        max_grad_norm=1.0,         # (nice to have) gradient clipping
        # ddp_find_unused_parameters=False, # often safe; leave default if unsure
        optim="adamw_torch_fused",     # fused AdamW (Torch 2.x)
    )

    # 6) Trainer
    trainer = ScImmunePretrainingTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 7) Train & save
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()