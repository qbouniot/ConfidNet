import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import pickle

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.models import get_model
from confidnet.utils import trust_scores
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics
from confidnet.utils.misc import load_yaml

LOGGER = get_logger(__name__, level="DEBUG")

MODE_TYPE = ["mcp", "tcp", "mc_dropout", "trust_score", "confidnet"]
MAX_NUMBER_TRUSTSCORE_SEG = 3000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument("--epoch", "-e", type=int, default=None, help="Epoch to analyse")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="mcp",
        choices=MODE_TYPE,
        help="Type of confidence testing",
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=50, help="Samples in case of MCDropout"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument(
        "--get-confidences", action="store_true", default=False
    )

    parser.add_argument(
        "--val_temp_scaling", action="store_true", default=False
    )

    parser.add_argument(
        "--opt_temp_scaling", action="store_true", default=True
    )

    args = parser.parse_args()

    LOGGER.info(f"Args used: {args}")

    config_args = load_yaml(args.config_path)

    # Overwrite for release
    config_args["training"]["output_folder"] = Path(args.config_path).parent

    config_args["training"]["metrics"] = [
        "accuracy",
        "fpr_at_95tpr",
        "ap_errors",
        "ap_success",
        "auroc",
        "aurc",
        "ece",
        "brier",
        "nll"
    ]
    if config_args["training"]["task"] == "segmentation":
        config_args["training"]["metrics"].append("mean_iou")

    # Special case of MC Dropout
    if args.mode == "mc_dropout":
        config_args["training"]["mc_dropout"] = True

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args, dloader.train_loader, dloader.val_loader, dloader.test_loader, -1, device
    )

    # Initialize and load model
    ckpt_path = config_args["training"]["output_folder"] / "ckpts" / f"model_epoch_{args.epoch:03d}.ckpt"
    checkpoint = torch.load(ckpt_path)
    learner.model.load_state_dict(checkpoint["model_state_dict"])

    # Get scores
    LOGGER.info(f"Inference mode: {args.mode}")

    if args.mode != "trust_score":
        results = learner.evaluate(
            learner.test_loader,
            learner.prod_test_len,
            split="test",
            mode=args.mode,
            samples=args.samples,
            verbose=True,
            return_confidences=args.get_confidences
        )

        scores_test = results[1]

        if args.get_confidences:
            conf = results[2]

    # Special case TrustScore
    else:
        # For segmentation, reduce number of samples, else it is too long to compute
        if config_args["training"]["task"] == "segmentation":
            learner.prod_test_len = MAX_NUMBER_TRUSTSCORE_SEG * np.ceil(
                learner.nsamples_test / config_args["training"]["batch_size"]
            )

        # Create feature extractor model
        config_args["model"]["name"] = config_args["model"]["name"] + "_extractor"
        features_extractor = get_model(config_args, device).to(device)
        features_extractor.load_state_dict(learner.model.state_dict(), strict=False)
        LOGGER.info(f"Using extractor {config_args['model']['name']}")
        features_extractor.print_summary(
            input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape])
        )

        # Get features for KDTree
        LOGGER.info("Get features for KDTree")
        features_extractor.eval()
        metrics = Metrics(
            learner.metrics, learner.prod_test_len, config_args["data"]["num_classes"]
        )
        train_features, train_target = [], []
        with torch.no_grad():
            loop = tqdm(learner.train_loader)
            for j, (data, target) in enumerate(loop):
                data, target = data.to(device), target.to(device)
                output = features_extractor(data)
                if config_args["training"]["task"] == "segmentation":
                    # Select only a fraction of outputs for segmentation trustscore
                    output = (
                        output.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(output.size(0) * output.size(2) * output.size(3), -1)
                    )
                    target = (
                        target.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(target.size(0) * target.size(2) * target.size(3), -1)
                    )
                    idx = torch.randperm(output.size(0))[:MAX_NUMBER_TRUSTSCORE_SEG]
                    output = output[idx, :]
                    target = target[idx, :]
                else:
                    output = output.view(output.size(0), -1)
                train_features.append(output)
                train_target.append(target)
        train_features = torch.cat(train_features).detach().cpu().numpy()
        train_target = torch.cat(train_target).detach().cpu().numpy()

        LOGGER.info("Create KDTree")
        trust_model = trust_scores.TrustScore(
            num_workers=max(config_args["data"]["num_classes"], 20)
        )
        trust_model.fit(train_features, train_target)

        LOGGER.info("Execute on test set")
        test_features, test_pred = [], []
        learner.model.eval()
        with torch.no_grad():
            loop = tqdm(learner.test_loader)
            for j, (data, target) in enumerate(loop):
                data, target = data.to(device), target.to(device)
                output = learner.model(data)
                confidence, pred = output.max(dim=1, keepdim=True)
                features = features_extractor(data)

                if config_args["training"]["task"] == "segmentation":
                    features = (
                        features.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(features.size(0) * features.size(2) * features.size(3), -1)
                    )
                    target = (
                        target.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(target.size(0) * target.size(2) * target.size(3), -1)
                    )
                    pred = (
                        pred.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(pred.size(0) * pred.size(2) * pred.size(3), -1)
                    )
                    confidence = (
                        confidence.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(confidence.size(0) * confidence.size(2) * confidence.size(3), -1)
                    )
                    idx = torch.randperm(features.size(0))[:MAX_NUMBER_TRUSTSCORE_SEG]
                    features = features[idx, :]
                    target = target[idx, :]
                    pred = pred[idx, :]
                    confidence = confidence[idx, :]
                else:
                    features = features.view(features.size(0), -1)

                test_features.append(features)
                test_pred.append(pred)
                metrics.update(pred, target, confidence)

        test_features = torch.cat(test_features).detach().to("cpu").numpy()
        test_pred = torch.cat(test_pred).squeeze().detach().to("cpu").numpy()
        proba_pred = trust_model.get_score(test_features, test_pred)
        metrics.proba_pred = proba_pred
        scores_test = metrics.get_scores(split="test")

    LOGGER.info("Results")
    print("----------------------------------------------------------------")
    for st in scores_test:
        print(st)
        print(scores_test[st])
        print("----------------------------------------------------------------")

    if args.get_confidences:
        LOGGER.info("Saving confidence scores")
        with open(config_args["training"]["output_folder"] / f"confidence_scores.pkl", "wb") as f:
            pickle.dump(conf, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.val_temp_scaling:
        val_temp = learner.find_temp(split='val')
        LOGGER.info(f"Best temperature found on val set : {val_temp}")

        if args.mode != "trust_score":
            results_val_ts = learner.evaluate(
                learner.test_loader,
                learner.prod_test_len,
                split="test",
                mode=args.mode,
                samples=args.samples,
                verbose=True,
                return_confidences=args.get_confidences,
                temp=val_temp
            )

            scores_test_val_ts = results_val_ts[1]

            if args.get_confidences:
                conf_val_ts = results_val_ts[2]

        LOGGER.info("Results with approximated temperature:")
        print("----------------------------------------------------------------")
        for st in scores_test_val_ts:
            print(st)
            print(scores_test_val_ts[st])
            print("----------------------------------------------------------------")

        if args.get_confidences:
            LOGGER.info("Saving confidence scores")
            with open(config_args["training"]["output_folder"] / f"confidence_scores_val_ts.pkl", "wb") as f:
                pickle.dump(conf_val_ts, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.opt_temp_scaling:
        test_temp = learner.find_temp(split='test')
        LOGGER.info(f"Best temperature found on test set : {test_temp}")

        if args.mode != "trust_score":
            results_temp_ts = learner.evaluate(
                learner.test_loader,
                learner.prod_test_len,
                split="test",
                mode=args.mode,
                samples=args.samples,
                verbose=True,
                return_confidences=args.get_confidences,
                temp=test_temp
            )

            scores_test_temp_ts = results_temp_ts[1]

            if args.get_confidences:
                conf_temp_ts = results_temp_ts[2]

        LOGGER.info("Results with optimal temperature")
        print("----------------------------------------------------------------")
        for st in scores_test_temp_ts:
            print(st)
            print(scores_test_temp_ts[st])
            print("----------------------------------------------------------------")

        if args.get_confidences:
            LOGGER.info("Saving confidence scores")
            with open(config_args["training"]["output_folder"] / f"confidence_scores_temp_ts.pkl", "wb") as f:
                pickle.dump(conf_temp_ts, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
