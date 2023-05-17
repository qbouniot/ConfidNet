from confidnet.learners.default_learner import DefaultLearner
from confidnet.learners.oodconfid_learner import OODConfidLearner
from confidnet.learners.selfconfid_learner import SelfConfidLearner
from confidnet.learners.crl_learner import CRLLearner
from confidnet.learners.selfconfid_online_learner import OnlineConfidLearner

def get_learner(config_args, train_loader, val_loader, test_loader, start_epoch, device):
    """
        Return a new instance of model
    """

    # Available models
    learners_factory = {
        "default": DefaultLearner,
        "selfconfid": SelfConfidLearner,
        "oodconfid": OODConfidLearner,
        "crl": CRLLearner,
        "selfconfid_online": OnlineConfidLearner,
    }

    return learners_factory[config_args["training"]["learner"]](
        config_args=config_args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        start_epoch=start_epoch,
        device=device,
    )
