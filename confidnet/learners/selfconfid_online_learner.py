import os
from collections import OrderedDict 

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import torch.nn.functional as F
import math

from confidnet.learners.learner import AbstractLearner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics
from confidnet.utils.losses import mixup_data,mixup_criterion,pgd_linf

LOGGER = get_logger(__name__, level="DEBUG")

class OnlineConfidLearner(AbstractLearner):
    def __init__(self, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        super().__init__(config_args, train_loader, val_loader, test_loader, start_epoch, device)
        # self.freeze_layers()
        self.freeze_encoder(verbose=True)
        self.disable_bn_encoder(verbose=True)
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout_encoder(verbose=True)
        self.ema_rate = self.config_args["model"].get("ema_rate", 0.996)
        LOGGER.info(f"Using a starting ema rate of {self.ema_rate}")
        self.mom_sched = self.config_args["model"].get("momentum_scheduler", "constant")
        LOGGER.info(f"Using a {self.mom_sched} ema scheduling")
        self.warmup_epochs = self.config_args["model"].get("warmup_epochs", 0)
        LOGGER.info(f"Warming up for {self.warmup_epochs} epochs")
        if self.warmup_epochs > 0:
            self.freeze_uncertainty(verbose=True)

    def set_optimizer(self, optimizer_name):
        optimizer_params = {
            k: v for k, v in self.config_args["training"]["optimizer"].items() if k != "name"
        }
        LOGGER.info(f"Using optimizer {optimizer_name}")
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD([p for n,p in self.model.named_parameters() if 'pred_network' in n or 'uncertainty_network.uncertainty' in n], **optimizer_params)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam([p for n,p in self.model.named_parameters() if 'pred_network' in n or 'uncertainty_network.uncertainty' in n], **optimizer_params)
        elif optimizer_name == "adadelta":
            self.optimizer = optim.Adadelta([p for n,p in self.model.named_parameters() if 'pred_network' in n or 'uncertainty_network.uncertainty' in n], **optimizer_params)
        else:
            raise KeyError("Bad optimizer name or not implemented (sgd, adam, adadelta).")


    def train(self, epoch, eval=True):
        self.model.train()
        self.disable_bn_encoder()
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout_encoder()
        metrics_mcp_pred = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        metrics_mcp_conf = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        metrics_tcp = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, confid_loss, class_loss, class_confid_loss = 0, 0, 0, 0
        len_steps, len_data = 0, 0

        if epoch < self.warmup_epochs:
            curr_ema_rate = 0.
        elif epoch == self.warmup_epochs:
            self.unfreeze_uncertainty(verbose=True)
            curr_ema_rate = self.update_momentum_rate(epoch)
        else:
            curr_ema_rate = self.update_momentum_rate(epoch)
        # self.update_momentum_encoder(curr_ema_rate)

        # Training loop
        with tqdm(self.train_loader) as loop:
            for batch_id, (data, target) in enumerate(loop):
                data, target = data.to(self.device), target.to(self.device)
                if self.mixup_augm:
                    data, target_a, target_b, lam = mixup_data(data,target)
                elif self.adv_augm:
                    delta = pgd_linf(self.model, data, target, epsilon=self.adv_eps, num_iter=self.adv_iter, randomize=True)
                    data = data + delta
                self.optimizer.zero_grad()
                output = self.model(data)

                # Potential temperature scaling
                if self.temperature:
                    output = list(output)
                    output[0] = output[0] / self.temperature
                    output = tuple(output)

                if self.task == "classification":
                    if self.mixup_pred:
                        current_loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                    else:
                        if self.mixup_augm:
                            if lam > 0.5:
                                current_loss = self.criterion(output, target_a)
                            else:
                                current_loss = self.criterion(output, target_b) 
                        else:
                            current_loss_confid, current_loss_class, current_loss_class_confid = self.criterion(output, target)
                            current_loss = current_loss_confid + current_loss_class
                elif self.task == "segmentation":
                    current_loss = self.criterion(output, target.squeeze(dim=1))
                current_loss.backward()
                # loss += current_loss
                confid_loss += current_loss_confid
                class_loss += current_loss_class
                class_confid_loss += current_loss_class_confid
                self.optimizer.step()
                self.update_momentum_encoder(curr_ema_rate)

                if self.task == "classification":
                    len_steps += len(data)
                    len_data = len_steps
                elif self.task == "segmentation":
                    len_steps += len(data) * np.prod(data.shape[-2:])
                    len_data += len(data)

                # Update metrics mcp + tcp
                conf_mcp_pred, pred = F.softmax(output[0], dim=1).max(dim=1, keepdim=True)
                conf_tcp = torch.sigmoid(output[1])
                conf_mcp_conf, pred_conf = F.softmax(output[2], dim=1).max(dim=1, keepdim=True)
                # metrics.update(pred, target, confidence)
                metrics_mcp_pred.update(pred, target, conf_mcp_pred, output[0])
                metrics_tcp.update(pred_conf, target, conf_tcp, output[2])
                metrics_mcp_conf.update(pred_conf, target, conf_mcp_conf, output[2])

                # Update the average loss
                loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
                loop.set_postfix(
                    OrderedDict(
                        {
                            "loss_nll": f"{(class_loss / len_data):05.3e}",
                            "loss_confid": f"{(confid_loss / len_data):05.3e}",
                            "loss_nll_confid": f"{(class_confid_loss / len_data):05.3e}",
                            "acc_pred": f"{(metrics_mcp_pred.accuracy / len_steps):05.2%}",
                            "acc_conf": f"{(metrics_mcp_conf.accuracy / len_steps):05.2%}",
                        }
                    )
                )
                loop.update()

        
        # Eval on epoch end
        scores_mcp_pred = metrics_mcp_pred.get_scores(split="train/pred")
        scores_mcp_conf = metrics_mcp_conf.get_scores(split="train/pred_conf")
        scores_tcp = metrics_tcp.get_scores(split="train/conf")
        scores = {}
        scores.update(scores_mcp_pred)
        scores.update(scores_mcp_conf)
        scores.update(scores_tcp)
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "ema": {"value": curr_ema_rate, "string": f"{curr_ema_rate:05.4e}"},
                "train/loss_confid": {
                    "value": confid_loss / len_data,
                    "string": f"{(confid_loss / len_data):05.4e}",
                },
                "train/loss_nll": {
                    "value": class_loss / len_data,
                    "string": f"{(class_loss / len_data):05.4e}",
                },
                "train/loss_nll_confid": {
                    "value": class_confid_loss / len_data,
                    "string": f"{(class_confid_loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        if eval:
            # Val scores
            val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
            logs_dict["val/loss_confid"] = {
                "value": val_losses["loss_confid"].item() / self.nsamples_val,
                "string": f"{(val_losses['loss_confid'].item() / self.nsamples_val):05.4e}",
            }
            logs_dict["val/loss_nll"] = {
                "value": val_losses["loss_nll"].item() / self.nsamples_val,
                "string": f"{(val_losses['loss_nll'].item() / self.nsamples_val):05.4e}",
            }
            logs_dict["val/loss_nll_confid"] = {
                "value": val_losses["loss_nll_confid"].item() / self.nsamples_val,
                "string": f"{(val_losses['loss_nll_confid'].item() / self.nsamples_val):05.4e}",
            }
            for sv in scores_val:
                logs_dict[sv] = scores_val[sv]

            # Test scores
            test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
            logs_dict["test/loss_confid"] = {
                "value": test_losses["loss_confid"].item() / self.nsamples_test,
                "string": f"{(test_losses['loss_confid'].item() / self.nsamples_test):05.4e}",
            }
            logs_dict["test/loss_nll"] = {
                "value": test_losses["loss_nll"].item() / self.nsamples_test,
                "string": f"{(test_losses['loss_nll'].item() / self.nsamples_test):05.4e}",
            }
            logs_dict["test/loss_nll_confid"] = {
                "value": test_losses["loss_nll_confid"].item() / self.nsamples_test,
                "string": f"{(test_losses['loss_nll_confid'].item() / self.nsamples_test):05.4e}",
            }
            for st in scores_test:
                logs_dict[st] = scores_test[st]

            # Print metrics
            misc.print_dict(logs_dict)

            # CSV logging
            misc.csv_writter(path=self.output_folder / "logs.csv", dic=OrderedDict(logs_dict))

            # Tensorboard logging
            self.save_tb(logs_dict)

            # Save the model checkpoint
            self.save_checkpoint(epoch)

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

    # eval both tcp + mcp
    def evaluate(self, dloader, len_dataset, split="test", verbose=False, **args):
        self.model.eval()
        metrics_mcp_pred = Metrics(self.metrics, len_dataset, self.num_classes)
        metrics_mcp_conf = Metrics(self.metrics, len_dataset, self.num_classes)
        metrics_tcp = Metrics(self.metrics, len_dataset, self.num_classes)
        loss = 0
        confid_loss, class_loss, class_confid_loss = 0, 0, 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                if self.task == "classification":
                    curr_conf_loss, curr_class_loss, curr_class_conf_loss = self.criterion(output, target)
                    confid_loss += curr_conf_loss
                    class_loss += curr_class_loss
                    class_confid_loss += curr_class_conf_loss
                elif self.task == "segmentation":
                    loss += self.criterion(output, target.squeeze(dim=1))
                # Update metrics mcp + tcp
                conf_mcp_pred, pred = F.softmax(output[0], dim=1).max(dim=1, keepdim=True)
                conf_mcp_conf, pred_conf = F.softmax(output[2], dim=1).max(dim=1, keepdim=True)
                conf_tcp = torch.sigmoid(output[1])
                # metrics.update(pred, target, confidence)
                metrics_mcp_pred.update(pred, target, conf_mcp_pred, output[0])
                metrics_tcp.update(pred_conf, target, conf_tcp, output[2])
                metrics_mcp_conf.update(pred_conf, target, conf_mcp_conf, output[2])

        scores_mcp_pred = metrics_mcp_pred.get_scores(split=f"{split}/mcp_pred")
        scores_tcp = metrics_tcp.get_scores(split=f"{split}/tcp")
        scores_mcp_conf = metrics_mcp_conf.get_scores(split=f"{split}/mcp_conf")
        scores = {}
        scores.update(scores_mcp_pred)
        scores.update(scores_tcp)
        scores.update(scores_mcp_conf)
        losses = {"loss_confid": confid_loss, "loss_nll": class_loss, "loss_nll_confid": class_confid_loss}
        return losses, scores

    def load_checkpoint(self, state_dict, uncertainty_state_dict=None, strict=True):
        if not uncertainty_state_dict:
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            self.model.pred_network.load_state_dict(state_dict, strict=strict)

            # 1. filter out unnecessary keys
            if self.task == "classification":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k not in ["fc2.weight", "fc2.bias"]
                }
            if self.task == "segmentation":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k
                    not in [
                        "up1.conv2.cbr_unit.0.weight",
                        "up1.conv2.cbr_unit.0.bias",
                        "up1.conv2.cbr_unit.1.weight",
                        "up1.conv2.cbr_unit.1.bias",
                        "up1.conv2.cbr_unit.1.running_mean",
                        "up1.conv2.cbr_unit.1.running_var",
                    ]
                }
            # 2. overwrite entries in the existing state dict
            self.model.uncertainty_network.state_dict().update(state_dict)
            # 3. load the new state dict
            self.model.uncertainty_network.load_state_dict(state_dict, strict=False)

    def freeze_layers(self):
        # Eventual fine-tuning for self-confid
        LOGGER.info("Freezing every layer except uncertainty")
        for param in self.model.named_parameters():
            if "uncertainty" in param[0]:
                print(param[0], "kept to training")
                continue
            param[1].requires_grad = False

    def disable_bn(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Keeping original BN parameters")
        for layer in self.model.named_modules():
            if "bn" in layer[0] or "cbr_unit.1" in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                layer[1].momentum = 0
                layer[1].eval()

    def disable_dropout(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Disable dropout layers to reduce stochasticity")
        for layer in self.model.named_modules():
            if "dropout" in layer[0]:
                if verbose:
                    print(layer[0], "set to eval mode")
                layer[1].eval()

    def freeze_encoder(self, verbose=False):
        LOGGER.info("Freezing encoder in uncertainty network")
        for param in self.model.uncertainty_network.named_parameters():
            if "uncertainty" not in param[0]:
                param[1].requires_grad = False
                if verbose:
                    LOGGER.info(f"{param[0]} frozen")
            else:
                if verbose:
                    LOGGER.info(f"{param[0]} kept training")

    def freeze_uncertainty(self, verbose=False):
        if verbose:
            LOGGER.info("Freezing uncertainty network")
        for param in self.model.uncertainty_network.named_parameters():
            # if "uncertainty" not in param[0]:
            param[1].requires_grad = False
            if verbose:
                LOGGER.info(f"{param[0]} frozen")
            # else:
            #     if verbose:
            #         LOGGER.info(f"{param[0]} kept training")

    def unfreeze_uncertainty(self, verbose=False):
        if verbose:
            LOGGER.info("Unfreezing uncertainty network")
        for param in self.model.uncertainty_network.named_parameters():
            if "uncertainty" in param[0]:
                param[1].requires_grad = True
                if verbose:
                    LOGGER.info(f"{param[0]} unfrozen")

    def disable_dropout_encoder(self, verbose=False):
        if verbose:
            LOGGER.info("Disabling dropout for uncertainty encoder")
        for layer in self.model.uncertainty_network.named_modules():
            if "dropout" in layer[0]:
                if verbose:
                    LOGGER.info(f"{layer[0]} set to eval to remove dropout")
                layer[1].eval()

    def disable_bn_encoder(self, verbose=False):
        if verbose:
            LOGGER.info("Keeping original BN parameters")
        for layer in self.model.uncertainty_network.named_modules():
            if "bn" in layer[0] or "cbr_unit.1" in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                layer[1].momentum = 0
                layer[1].eval()


    def update_momentum_encoder(self, ema_rate=0.996):
        uncertainty_params = OrderedDict(self.model.uncertainty_network.named_parameters())
        pred_params = OrderedDict(self.model.pred_network.named_parameters())

        for n_param, data_param in pred_params.items():
            uncertainty_params[n_param].data = uncertainty_params[n_param].data * ema_rate + data_param.data * (1. - ema_rate)

    def update_momentum_rate(self, epoch):
        if self.mom_sched == 'constant':
            return self.ema_rate
        elif self.mom_sched == 'cosine':
            momentum = 1 - (1 - self.ema_rate) * (math.cos(math.pi * (epoch) / (self.nb_epochs - 1)) + 1) / 2
            return momentum
        
        else:
            raise NotImplementedError(f"{self.mom_sched} is not supported")
