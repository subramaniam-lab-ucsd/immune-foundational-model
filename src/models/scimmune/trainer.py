import json
from typing import Optional
from torch import nn
from transformers import Trainer, TrainingArguments
import torch
from loss import masked_mse_loss
from dataclasses import dataclass


@dataclass
class ScImmuneTrainingArguments(TrainingArguments):
    mlm_probability: Optional[float] = 0.50
    max_length: Optional[int] = 1200
    warmup_ratio_or_step: Optional[int] = 0.1
    MVC: Optional[bool] = False
    evaluation_strategy: Optional[str] = "steps"
    scale_factor: Optional[float] = 100.0
    # TODO: add custom arguments here

    # training loss, mvc loss, etc.
    def from_json_file(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)


def compute_metrics(eval_pred):
    # logits, labels = eval_pred
    raise NotImplementedError


class ScImmunePretrainingTrainer(Trainer):

    def compute_loss(
        self, model, data_dict, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        data_dict = dict(data_dict)

        gen_expr_target = data_dict.pop("gen_expr_target", None)
        _ = data_dict.pop("masked_expr", None)
        if gen_expr_target is None:
            raise ValueError("gen_expr_target missing from batch.")

        # DO NOT inject CLS/MVC/etc. – your model doesn't accept them
        # data_dict["CLS"] = False
        # data_dict["MVC"] = self.args.MVC
        # data_dict["generative_training"] = True  # harmless, but also not needed

        # Only keep the kwargs your model actually accepts
        allowed_keys = {
            "pcpt_gene",
            "pcpt_expr",
            "pcpt_key_padding_mask",
            "gen_gene",
            "gen_key_padding_mask",
            "batch_labels",   # only if you really use it
            "input_cell_emb",  # include ONLY if your generative_forward supports it
        }
        model_inputs = {k: v for k, v in data_dict.items() if k in allowed_keys}

        outputs = model(**model_inputs)
        gen_expr_preds = outputs.get("gen_preds")
        gen_kpm = model_inputs["gen_key_padding_mask"].bool()
        positions_to_match = ~gen_kpm

        loss_mse = masked_mse_loss(gen_expr_preds, gen_expr_target, positions_to_match)
        loss = loss_mse

        # Second pass (embed -> expr) only if your model supports input_cell_emb:
        if "cell_emb" in outputs:
            model_inputs_2 = dict(model_inputs)
            model_inputs_2["input_cell_emb"] = outputs["cell_emb"].detach()
            preds2 = model(**model_inputs_2).get("gen_preds")
            loss_gen = masked_mse_loss(preds2, gen_expr_target, positions_to_match)
            loss = loss + loss_gen
        else:
            loss_gen = None
        
            # ---- NEW: log unscaled MSE (no grad) ----
        sf = getattr(self.args, "scale_factor", None)
        if sf is not None and sf != 1.0:
            with torch.no_grad():
                mse_unscaled = masked_mse_loss(
                    gen_expr_preds * sf, gen_expr_target * sf, positions_to_match
                )
            # this shows up in the Trainer logs
            # self.log({"loss_mse_unscaled": mse_unscaled.detach().item()})

        if return_outputs:
            return loss, {
                "loss": loss,
                "loss_mse": loss_mse,
                "loss_gen": loss_gen,
                "gen_preds": gen_expr_preds,
                "cell_emb": outputs.get("cell_emb"),
            }
        return loss