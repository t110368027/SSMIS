from dataclasses import dataclass
from typing import List, Any
import torch
import torch.nn as nn


def rotate90(img: torch.Tensor) -> torch.Tensor:
    """Rotate an image by 90 degrees."""
    return torch.rot90(img, 3, (2, 3))


def rotate180(img: torch.Tensor) -> torch.Tensor:
    """Rotate an image by 180 degrees."""
    return torch.rot90(img, 2, (2, 3))


def rotate270(img: torch.Tensor) -> torch.Tensor:
    """Rotate an image by 270 degrees."""
    return torch.rot90(img, 1, (2, 3))


def hflip(img: torch.Tensor) -> torch.Tensor:
    """Flip an image horizontally."""
    return img.flip(3)


def vflip(img: torch.Tensor) -> torch.Tensor:
    """Flip an image vertically."""
    return img.flip(2)


@dataclass
class HorizontalFlip:
    """Class to handle horizontal image flipping."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return hflip(img=image)

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return hflip(img=mask)

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return hflip(img=label)


@dataclass
class VerticalFlip:
    """Class to handle vertical image flipping."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return vflip(img=image)

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return vflip(img=mask)

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return vflip(img=label)


@dataclass
class Rotate_0:
    """Class for no rotation."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return image

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return label


@dataclass
class Rotate_90:
    """Class to handle 90-degree rotation."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return rotate90(img=image)

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return rotate270(img=mask)

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return rotate270(img=label)


@dataclass
class Rotate_180:
    """Class to handle 180-degree rotation."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return rotate180(img=image)

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return rotate180(img=mask)

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return rotate180(img=label)


@dataclass
class Rotate_270:
    """Class to handle 270-degree rotation."""

    @staticmethod
    def apply_aug_image(image: torch.Tensor) -> torch.Tensor:
        return rotate270(img=image)

    @staticmethod
    def apply_deaug_mask(mask: torch.Tensor) -> torch.Tensor:
        return rotate90(img=mask)

    @staticmethod
    def apply_deaug_label(label: torch.Tensor) -> torch.Tensor:
        return rotate90(img=label)


class Merger:
    """
    Merge tensor inputs based on a specified type.
    Supported types: mean, gmean, sum, max, min, tsharpen.
    """

    SUPPORTED_TYPES = {"mean", "gmean", "sum", "max", "min", "tsharpen"}

    def __init__(self, merge_type: str = "mean", num_inputs: int = 1) -> None:
        """Initialize the Merger with specified type and number of inputs."""

        if merge_type not in Merger.SUPPORTED_TYPES:
            raise ValueError(
                f'Unsupported merge type "{merge_type}". '
                f"Supported types: {Merger.SUPPORTED_TYPES}"
            )

        self._output: torch.Tensor = None
        self._merge_type = merge_type
        self._num_inputs = num_inputs

    def append(self, tensor: torch.Tensor) -> None:
        """Append a tensor for merging."""

        if self._merge_type == "tsharpen":
            tensor = tensor**0.5

        if self._output is None:
            self._output = tensor
        else:
            self._merge_tensors(tensor)

    def _merge_tensors(self, tensor: torch.Tensor) -> None:
        """Merge stored tensor with new tensor based on merge type."""

        if self._merge_type in ["mean", "sum", "tsharpen"]:
            self._output += tensor
        elif self._merge_type == "gmean":
            self._output *= tensor
        elif self._merge_type == "max":
            self._output = torch.max(self._output, tensor)
        elif self._merge_type == "min":
            self._output = torch.min(self._output, tensor)

    @property
    def result(self) -> torch.Tensor:
        """Compute and return the result of merging."""

        if self._merge_type in ["sum", "max", "min"]:
            return self._output
        elif self._merge_type in ["mean", "tsharpen"]:
            return self._output / self._num_inputs
        elif self._merge_type == "gmean":
            return self._output ** (1 / self._num_inputs)
        else:
            raise ValueError(f'Unsupported merge type "{self._merge_type}"')


class TTA(nn.Module):
    """
    Test Time Augmentation module for improving model inference.

    The module applies a list of transformations on the input image and
    aggregates the results after passing them through the model.
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: List[nn.Module] = None,
        merge_mode: str = "mean",
    ) -> None:
        """
        Initialize TTA module.

        Parameters:
        - model: Neural network model for inference.
        - transforms: List of transformations for test time augmentation.
        - merge_mode: Merging strategy for combining augmented results.
        """

        super().__init__()
        if transforms is None:
            transforms = [
                Rotate_0(),
                Rotate_90(),
                Rotate_180(),
                Rotate_270(),
                HorizontalFlip(),
                VerticalFlip(),
            ]
        self.model = model.eval()
        self.transforms = transforms
        self.merge_mode = merge_mode

    def forward(self, image: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Forward pass with test time augmentation.

        Parameters:
        - image: Batch of images to be inferred.
        - args: Additional arguments for the model's forward method.

        Returns:
        - torch.Tensor: Merged model output after applying TTA.
        """

        batch_size = image.size(0)
        device = image.get_device()
        result_list = torch.Tensor().to(device)

        for b in range(batch_size):
            merger = Merger(merge_type=self.merge_mode, num_inputs=len(self.transforms))
            img = image[b].unsqueeze(dim=0)
            for transformer in self.transforms:
                augmented_image = transformer.apply_aug_image(img)
                with torch.no_grad():
                    augmented_output = self.model(augmented_image, *args)
                deaugmented_output = transformer.apply_deaug_mask(augmented_output)
                merger.append(deaugmented_output)

            result_list = torch.cat((result_list, merger.result))

        return result_list


if __name__ == "__main__":
    pass
