import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

#---------------------------------------------------------------------
# Metadata-related losses
#---------------------------------------------------------------------

def metadata_cosine_loss(
    meta_emb: torch.Tensor, ref_emb: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute cosine alignment loss between model metadata embeddings
    and reference Node2Vec embeddings.

    Args:
        meta_emb: Tensor of shape [B, D] – model's metadata embeddings.
        ref_emb:  Tensor of shape [B, D] – corresponding Node2Vec embeddings.
        mask: Optional tensor [B], 1 for valid entries, 0 to ignore.
    Returns:
        Scalar cosine alignment loss = 1 - mean(cosine_similarity).
    """
    if mask is not None:
        meta_emb = meta_emb[mask.bool()]
        ref_emb = ref_emb[mask.bool()]

    meta_norm = F.normalize(meta_emb, dim=-1)
    ref_norm = F.normalize(ref_emb, dim=-1)
    cosine_sim = F.cosine_similarity(meta_norm, ref_norm, dim=-1)
    return 1.0 - cosine_sim.mean()


def metadata_aux_loss(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Auxiliary classification loss for predicting metadata (e.g., cell type).
    Used for λ_aux conditioning experiments.

    Args:
        logits: Tensor [B, num_classes].
        labels: Tensor [B], integer class IDs.
        class_weights: Optional tensor [num_classes], for imbalance correction.
    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels, weight=class_weights)