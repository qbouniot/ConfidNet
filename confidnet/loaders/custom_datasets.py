import torch
import numpy as np
from torchvision import datasets

from PIL import Image
from typing import Any, Tuple, Optional, Callable


class MNIST_idx(datasets.MNIST):

    # def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, return_index: bool=False) -> None:
    #     super().__init__(root, train, transform, target_transform, download)

    #     self.return_index = return_index

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Custom MNIST dataset that returns the index of the data in the dataset. Used for CRL loss.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
class CIFAR10_idx(datasets.CIFAR10):

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Custom CIFAR10 dataset that returns the index of the data in the dataset. Used for CRL loss.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
        
class CIFAR100_idx(datasets.CIFAR100):
     
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Custom CIFAR100 dataset that returns the index of the data in the dataset. Used for CRL loss.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class SVHN_idx(datasets.SVHN):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index