"""
The implementation is refactoring from:
https://github.com/ildoonet/pytorch-randaugment/
https://github.com/automl/trivialaugment/
"""
from typing import Callable, List, Tuple
import random
import numpy as np
from .aug_method_pil import Aug_Methods
from .aug_method_tf import strong_aug, weak_aug, no_aug

__all__ = [
    "strong_aug",
    "weak_aug",
    "no_aug",
]


def aug_list() -> List[Tuple[Callable, float, float]]:
    augs = [
        (Aug_Methods["AutoContrast"], 0, 1),
        (Aug_Methods["Brightness"], 0.3, 1.2),
        (Aug_Methods["Contrast"], 0.3, 1.2),
        (Aug_Methods["Equalize"], 0, 1),
        (Aug_Methods["Invert"], 0, 1),
        (Aug_Methods["Posterize"], 4, 8),
        (Aug_Methods["Sharpness"], 0.1, 2.0),
        (Aug_Methods["SolarizeAdd"], 0, 80),
        (Aug_Methods["Blur"], 0, 1),
        (Aug_Methods["Detail"], 0, 1),
        (Aug_Methods["Emboss"], 0, 1),
        (Aug_Methods["Contour"], 0, 1),
        (Aug_Methods["Find_edges"], 0, 1),
        (Aug_Methods["Edge_enhance"], 0, 1),
        (Aug_Methods["Edge_enhance_more"], 0, 1),
        (Aug_Methods["Smooth"], 0, 1),
        (Aug_Methods["Smooth_more"], 0, 1),
        (Aug_Methods["Sharpen"], 0, 1),
        (Aug_Methods["Identity"], 0, 1),
    ]
    return augs


def float_parameter(min_val: float, max_val: float, magnitude: int) -> float:
    """
    Generates a floating-point value between min_val and max_val, using the
    given magnitude.

    Args:
        min_val (float): The minimum possible value.
        max_val (float): The maximum possible value.
        magnitude (int): The magnitude used for random selection.

    Returns:
        float: The generated floating-point number.
    """
    factor = np.random.randint(0, magnitude)
    val = (float(max_val - min_val) * float(factor) / magnitude) + min_val
    val = round(val, 2)
    return val


class RandAugment:
    """
    RandAugment: A data augmentation algorithm that dynamically generates
    an augmentation policy from a set of predefined augmentations.

    Args:
        n (int): Number of augmentations to apply sequentially.
        m (int): Magnitude for augmentations.
        trnsforms (callable, optional): A callable function returning a list
        of transformations.
    Reference:
        Cubuk, Ekin D., et al.
        "Randaugment: Practical automated data augmentation with a reduced
        search space."
        Proceedings of the NeurIPS 2020.
    """

    def __init__(
        self,
        n: int,
        m: int,
        augment: Callable[[], List[Tuple[Callable, float, float]]] = aug_list,
    ):
        self.n = n
        self.m = m
        self.augment_list = augment

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = float_parameter(min_val, max_val, self.m)
            img = op(img, val)
        return img


class TrivialAugment:
    """
    TrivialAugment: A data augmentation strategy that applies a single randomly
    chosen augmentation to the given image.

    Args:
        trnsforms (callable, optional): A callable function returning a list
        of transformations.

    Reference:
        Müller, Samuel G., and Frank Hutter.
        "Trivialaugment: Tuning-free yet state-of-the-art data augmentation."
        Proceedings of the ICCV'2021.
    """

    def __init__(
        self,
        augment: Callable[[], List[Tuple[Callable, float, float]]] = aug_list,
    ):
        self.augment_list = augment

    def __call__(self, img):
        op, min_val, max_val = random.choices(self.augment_list, k=1)
        val = float_parameter(min_val, max_val, random.randint(0, 20))
        img = op(img, val)
        return img


class UniAugment:
    """
    UniAugment: A data augmentation strategy that randomly selects and applies
    two transformations from the given list. Each transformation has a 50%
    chance of being applied.

    Args:
        trnsforms (callable, optional): A callable function returning a list
        of transformations.

    Reference:
        LingChen, Tom Ching, et al.
        "Uniformaugment: A search-free probabilistic data augmentation
        approach."
        Proceedings of the arXiv preprint arXiv:2003.14348 (2020).
    """

    def __init__(
        self,
        augment: Callable[[], List[Tuple[Callable, float, float]]] = aug_list,
    ):
        self.augment_list = augment

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=2)
        for op, min_val, max_val in ops:
            if random.random() < 0.5:
                val = float_parameter(min_val, max_val, random.randint(0, 20))
                img = op(img, val)
        return img
