# import structured_map_ranking_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from confidnet.utils import misc

import numpy as np


class SelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)


class SelfConfidTCPRLoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        maxprob = probs.max(dim=1)[0]
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1) / maxprob) ** 2
        return torch.mean(loss)


class SelfConfidBCELoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        confidence = input[1].squeeze(dim=1)
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(input[0].argmax(dim=1) != target)] *= self.weighting
        return nn.BCEWithLogitsLoss(weight=weights)(
            confidence, (input[0].argmax(dim=1) == target).float()
        )


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        super().__init__()
        self.alpha = config_args["training"]["loss"].get("alpha", 0.25)
        self.gamma = config_args["training"]["loss"].get("gamma", 5)

    def forward(self, input, target):
        confidence = input[1].squeeze(dim=1)
        BCE_loss = F.binary_cross_entropy_with_logits(
            confidence, (input[0].argmax(dim=1) == target).float(), reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(loss)


class StructuredMAPRankingLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, mask):
        loss, ranking_lai = structured_map_ranking_loss.forward(input, target, mask)
        ctx.save_for_backward(input, target, mask, ranking_lai)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target, mask, ranking_lai = ctx.saved_variables
        grad_input = structured_map_ranking_loss.backward(
            grad_output, input, target, mask, ranking_lai
        )
        return grad_input, None, None


class StructuredMAPRankingLoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        confidence = input[1]
        mask = torch.ones_like(target).unsqueeze(dim=1)
        return StructuredMAPRankingLossFunction.apply(
            confidence,
            (input[0].argmax(dim=1) == target).float().unsqueeze(dim=1),
            mask.to(dtype=torch.uint8),
        )


class OODConfidenceLoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.device = device
        self.half_random = config_args["training"]["loss"]["half_random"]
        self.beta = config_args["training"]["loss"]["beta"]
        self.lbda = config_args["training"]["loss"]["lbda"]
        self.lbda_control = config_args["training"]["loss"]["lbda_control"]
        self.loss_nll, self.loss_confid = None, None
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1])

        # Make sure we don't have any numerical instability
        eps = 1e-12
        probs = torch.clamp(probs, 0.0 + eps, 1.0 - eps)
        confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

        if self.half_random:
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
            conf = confidence * b + (1 - b)
        else:
            conf = confidence

        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        probs_interpol = torch.log(conf * probs + (1 - conf) * labels_hot)
        self.loss_nll = nn.NLLLoss()(probs_interpol, target)
        self.loss_confid = torch.mean(-(torch.log(confidence)))
        total_loss = self.loss_nll + self.lbda * self.loss_confid

        # Update lbda
        if self.lbda_control:
            if self.loss_confid >= self.beta:
                self.lbda /= 0.99
            else:
                self.lbda /= 1.01
        return total_loss

class SelfConfidOnlineLoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        # probs_pred = F.softmax(input[0], dim=1)
        probs_confid = F.softmax(input[2], dim=1).detach()
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs_confid.argmax(dim=1) != target)] *= self.weighting
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        # if self.task == "segmentation":
        #     labels_hot = labels_hot.permute(0, 3, 1, 2)
        loss_confid = torch.mean(weights * (confidence - (probs_confid * labels_hot).sum(dim=1)) ** 2)
        loss_class = F.cross_entropy(input[0], target)
        loss_class_confid = F.cross_entropy(input[2], target)
        return loss_confid, loss_class, loss_class_confid

# PYTORCH LOSSES LISTS
PYTORCH_LOSS = {"cross_entropy": nn.CrossEntropyLoss}

# CUSTOM LOSSES LISTS
CUSTOM_LOSS = {
    "selfconfid_mse": SelfConfidMSELoss,
    "selfconfid_tcpr": SelfConfidTCPRLoss,
    "selfconfid_bce": SelfConfidBCELoss,
    "focal": FocalLoss,
    "ranking": StructuredMAPRankingLoss,
    "ood_confidence": OODConfidenceLoss,
    "selfconfid_online": SelfConfidOnlineLoss,
}

def mixup_data(x, y, alpha=1.0, intra_class=False, inter_class=False, mixup_norm=False, get_index=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if intra_class:
        clss = torch.unique(y)
        index = torch.arange(batch_size, device=y.device)
        for cls in clss:
            cls_index_mask = y == cls
            cls_index = torch.arange(batch_size, device=y.device)[cls_index_mask]
            new_cls_index = cls_index[torch.randperm(cls_index.size(0))]
            index[cls_index] = new_cls_index
    elif inter_class:
        clss = torch.randperm(torch.unique(y).size(0))
        orig_index = torch.arange(batch_size, device=y.device)
        index = torch.arange(batch_size, device=y.device)
        perm = torch.randperm(batch_size, device=y.device)
        taken = torch.zeros(batch_size, device=y.device)
        for cls in clss:
            cls_index_mask = y == cls
            cls_index = orig_index[cls_index_mask]
            for ind in cls_index:
                for i,elt in enumerate(perm):
                    if taken[i] == 0 and y[elt] != cls:
                        index[ind] = elt
                        taken[i] = 1
                        break
    else:
        index = torch.randperm(batch_size, device=x.device)
    if mixup_norm:
        mixed_x = x + (1-lam) * (x[index, :] - x) / torch.linalg.norm(x[index, :] - x, dim=(-2,-1), keepdim=True)
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    if not get_index:
        return mixed_x, y_a, y_b, lam
    else:
        return mixed_x, y_a, y_b, lam, index


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def similarity_mixup_criterion(criterion, pred, y_a, y_b, lam, cos):
    old_reduction = criterion.reduction
    criterion.reduction = 'none'
    loss = torch.mean(criterion(pred, y_a) * (1 - (1 - lam) * (cos + 1) / 2) + criterion(pred, y_b) * (1 - lam) * (cos + 1) / 2)
    criterion.reduction = old_reduction
    return loss

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """Construct an adversarial perturbation for a given batch of images X using ground truth y and a given model with a L_inf-PGD attack.
    Args:
        model (nn.Module): model to attack
        X (Tensor): batch of images
        y (Tensor): ground truth
        epsilon (float, optional): perturbation size. Defaults to 0.1.
        alpha (float, optional): step size. Defaults to 0.01.
        num_iter (int, optional): number of iterations. Defaults to 20.
        randomize (bool, optional): random start for the perturbation. Defaults to False.
    Returns:
        Tensor: perturbation to apply to the batch of images
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True, device=X.device)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True, device=X.device)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def get_nll(logits, targets):
    logsoftmax = F.log_softmax(logits, dim=1)
    out = torch.tensor(targets, dtype=torch.float)
    for i in range(len(targets)):
            out[i] = logsoftmax[i][targets[i]]

    return -out.sum()/len(out)
