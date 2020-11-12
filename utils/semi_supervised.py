import torch
import torch.nn.functional as F


def fix_match_loss(logits, target):
    supervised_idx = target >= 0

    unsupervised_logits = logits[target < 0]
    pass


    return ce_loss(logits[supervised_idx], target[supervised_idx])


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss


def consistency_loss(logits_w, logits_s, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    logits_w = logits_w.detach()

    pseudo_label = torch.softmax(logits_w, dim=-1)
    _, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w/T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
    return masked_loss.mean(), mask.mean()