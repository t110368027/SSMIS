import random
from typing import List

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class CustomRotationTransform:
    """
    Apply a random rotation to the image.

    Parameters:
    - angles (List[int]): List of angles from which one will be chosen
    randomly to rotate the image.
    """

    def __init__(self, angles: List[int]):
        self.angles = angles

    def __call__(self, x: Image.Image) -> Image.Image:
        """
        Apply the rotation transform.

        Parameters:
        - x (PIL.Image.Image): The input image.

        Returns:
        - PIL.Image.Image: The rotated image.
        """
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def strong_aug() -> T.Compose:
    """
    Strong augmentation pipeline.

    Returns:
    - torchvision.transforms.Compose: Composed torchvision transforms for
    strong augmentation.
    """
    return T.Compose(
        transforms=[
            T.RandomChoice(
                transforms=[
                    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    T.RandomErasing(),
                    T.RandomAutocontrast(),
                    # T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                ]
            ),
        ]
    )


def weak_aug() -> T.Compose:
    """
    Weak augmentation pipeline.

    Returns:
    - torchvision.transforms.Compose: Composed torchvision transforms for weak
    augmentation.
    """
    return T.Compose(
        transforms=[
            CustomRotationTransform(angles=[-90, 0, 90]),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    )


def no_aug(img):
    return img
