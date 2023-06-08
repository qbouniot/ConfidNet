import argparse
import os
from shutil import copyfile, rmtree

import torch
from pathlib import Path

import numpy as np
import random

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.utils.logger import get_logger
from confidnet.utils.misc import load_yaml, dump_yaml
from scipy.optimize import minimize


LOGGER = get_logger(__name__, level="DEBUG")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )

    #### args for modifying config file
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--kernel_tau_x", type=float, default=None)
    parser.add_argument("--kernel_tau_y", type=float, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument('--kernel_mixup', action="store_true", default=None)
    parser.add_argument("--kernel_regmixup", action="store_true", default=None)
    parser.add_argument('--kernel_sim_mixup', action="store_true", default=None)
    ####

    parser.add_argument('--seed', default=42, type=int)
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

    if args.exp_dir is not None:
        config_args["training"]["output_folder"] = Path(args.exp_dir)
    if args.kernel_tau_x is not None:
        config_args["training"]["kernel_tau_x"] = args.kernel_tau_x
    if args.kernel_tau_y is not None:
        config_args["training"]["kernel_tau_y"] = args.kernel_tau_y
    if args.mixup_alpha is not None:
        config_args["training"]["mixup_alpha"] = args.mixup_alpha
    if args.kernel_mixup is not None:
        config_args["training"]["kernel_mixup"] = args.kernel_mixup
    if args.kernel_regmixup is not None:
        config_args["training"]["kernel_regmixup"] = args.kernel_regmixup
    if args.kernel_sim_mixup is not None:
        config_args["training"]["kernel_sim_mixup"] = args.kernel_sim_mixup

    # Start from scatch
    os.makedirs(config_args["training"]["output_folder"] / "ckpts", exist_ok=True)
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

    # Log files
    LOGGER.info(f"Using model {config_args['model']['name']}")
    learner.model.print_summary(
        input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape])
    )
    dump_yaml(config_args, start_epoch)

    f = lambda x: learner.optim_taus(x[0], x[1])

    # variables: x[0] = tau_x, x[1] = tau_y
    initial_values = [0., 0.]
    bnds = ((0., 1.), (0., 1.)) # 0 < tau < 1
    # cons # TODO add constraint tau_x > tau_y ?

    res = minimize(f, initial_values, method='nelder-mead', bounds=bnds, options={'disp':True, 'xatol':1e-3, 'fatol':1e-6})

    print(res.message)
    print(res.x)

if __name__ == "__main__":
    main()