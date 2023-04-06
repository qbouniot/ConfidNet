import torch.nn as nn
from torchsummary import summary

import torchvision.models as models

from confidnet.utils.logger import get_logger
from confidnet.models.model import AbstractModel

LOGGER = get_logger(__name__, level="DEBUG")

class VGG16_torch(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)

        self.model = models.vgg16(pretrained=False, num_classes=config_args['data']['num_classes'])

    def forward(self, x):
        return self.model(x)