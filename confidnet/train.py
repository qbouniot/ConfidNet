import argparse
import os
from shutil import copyfile, rmtree

import click
import torch

import numpy as np
import random

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.utils.logger import get_logger
from confidnet.utils.misc import load_yaml
# from confidnet.utils.tensorboard_logger import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

LOGGER = get_logger(__name__, level="DEBUG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--from_scratch",
        "-f",
        action="store_true",
        default=False,
        help="Force training from scratch",
    )
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    config_args = load_yaml(args.config_path)
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # fix the seed for reproducibility
    seed = args.seed 
    # for multi-gpu -> + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    LOGGER.info(f"Random seed used: {seed}")


    # Start from scatch or resume existing model and optim
    if (config_args["training"]["output_folder"] / "ckpts").exists():
        list_previous_ckpt = sorted(
            [f for f in os.listdir(config_args["training"]["output_folder"] / "ckpts") if "model_epoch" in f]
        )
        if args.from_scratch or not list_previous_ckpt:
            LOGGER.info("Starting from scratch")
            # if click.confirm(
            #     "Removing current training directory ? ({}).".format(
            #         config_args["training"]["output_folder"]
            #     ),
            #     default=False,
            # ):
            #     rmtree(config_args["training"]["output_folder"])
            # os.makedirs(config_args["training"]["output_folder"], exist_ok=True)
            start_epoch = 1
        else:
            last_ckpt = list_previous_ckpt[-1]
            checkpoint = torch.load(config_args["training"]["output_folder"] / "ckpts" / str(last_ckpt))
            start_epoch = checkpoint["epoch"] + 1
    else:
        LOGGER.info("Starting from scratch")
        os.makedirs(config_args["training"]["output_folder"] / "ckpts")
        start_epoch = 1

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args,
        dloader.train_loader,
        dloader.val_loader,
        dloader.test_loader,
        start_epoch,
        device,
    )

    # Resume existing model or from pretrained one
    if start_epoch > 1:
        LOGGER.warning(f"Resuming from last checkpoint: {last_ckpt}")
        learner.model.load_state_dict(checkpoint["model_state_dict"])
        learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif config_args["model"]["resume"]:
        LOGGER.info(f"Loading pretrained model from {config_args['model']['resume']}")
        if config_args["model"]["resume"] == "vgg16":
            learner.model.init_vgg16_params()
        else:
            pretrained_checkpoint = torch.load(config_args["model"]["resume"])
            uncertainty_checkpoint = config_args["model"].get("uncertainty", None)
            if uncertainty_checkpoint:
                LOGGER.warning("Cloning training phase")
                learner.load_checkpoint(
                    pretrained_checkpoint["model_state_dict"],
                    torch.load(uncertainty_checkpoint)["model_state_dict"],
                    strict=False,
                )
            else:
                learner.load_checkpoint(pretrained_checkpoint["model_state_dict"], strict=False)

    # Log files
    LOGGER.info(f"Using model {config_args['model']['name']}")
    learner.model.print_summary(
        input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape])
    )
    # learner.tb_logger = TensorboardLogger(config_args["training"]["output_folder"])
    learner.tb_logger = SummaryWriter(config_args["training"]["output_folder"])
    copyfile(
        args.config_path, config_args["training"]["output_folder"] / f"config_{start_epoch}.yaml"
    )
    LOGGER.info(
        "Sending batches as {}".format(
            tuple(
                [config_args["training"]["batch_size"]]
                + [shape_i for shape_i in learner.train_loader.dataset[0][0].shape]
            )
        )
    )
    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")

    # Parallelize model
    nb_gpus = torch.cuda.device_count()
    if nb_gpus > 1:
        LOGGER.info(f"Parallelizing data to {nb_gpus} GPUs")
        learner.model = torch.nn.DataParallel(learner.model, device_ids=range(nb_gpus))

    if args.compile:
        LOGGER.info(f"Compiling model ...")
        learner.model = torch.compile(learner.model)
    # Set scheduler
    learner.set_scheduler()

    # Start training
    for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
        if epoch % config_args["training"].get("eval_every", 1) == 0 or (epoch > config_args["training"]["nb_epochs"] - config_args["training"].get("eval_every", 1)):
            learner.train(epoch,eval=True)
        else:
            learner.train(epoch,eval=False)


if __name__ == "__main__":
    main()
