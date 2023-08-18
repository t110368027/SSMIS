from typing import List, Tuple, Union, Callable
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import cv2


def load_txt(filename: str) -> List[str]:
    """
    Load a text file and return its content as a list of strings.

    Parameters:
    - filename: Path to the text file.

    Returns:
    - List of strings from the text file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def default_loader(path: str) -> Image.Image:
    """
    Default image loader using PIL.

    Parameters:
    - path: Path to the image file.

    Returns:
    - Loaded image.
    """
    return Image.open(path)


def check_data_shape(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Ensure the image shape matches the desired shape.

    Parameters:
    - img: Input image as a numpy array.
    - size: Desired size (height, width).

    Returns:
    - Processed image with shape (height, width, 1).
    """
    h, w = size
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, -1)
    assert img.shape == (h, w, 1)
    return img.astype(np.float32)


class ImageDataSetTrain(Dataset):
    def __init__(
        self,
        txt_path: str,
        img_dir: Union[str, Path],
        label_dir: Union[str, Path],
        size: Tuple[int, int],
        transform: "Callable",
        loader: "Callable" = default_loader,
        sort: bool = False,
    ) -> None:
        """
        Dataset for training images.

        Parameters:
        - txt_path: Path to the text file containing file names.
        - img_dir: Directory containing images.
        - label_dir: Directory containing labels.
        - size: Desired size for the images (height, width).
        - transform: Image transformations function or library.
        - loader: Function to load images.
        - sort: Whether to sort filenames.
        """
        file_names = load_txt(txt_path)
        if sort:
            file_names.sort()
        self.file_names = file_names
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = size
        self.loader = loader
        self.transform = transform

    def preprocess(
        self,
        img: Image.Image,
        size: Tuple[int, int],
        label: Image.Image = None,
        DA: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-process the image and label.

        Parameters:
        - img: Input image.
        - size: Desired size for the image.
        - label: Image label.
        - DA: Whether to apply data augmentation.

        Returns:
        - Processed image and label.
        """
        img = img.resize(size, Image.BICUBIC)
        label = label.resize(size, Image.BICUBIC)
        img = np.array(img)
        label = np.array(label)
        if self.transform and DA:
            sample = self.transform(image=img, mask=label)
            img, label = sample["image"], sample["mask"]
        img = check_data_shape(img, size)
        label = check_data_shape(label, size)
        if img.max() > 1.0:
            img /= 255.0
        if label.max() > 1.0:
            label //= 255.0
        return img.transpose((2, 0, 1)), label.transpose((2, 0, 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Parameters:
        - idx: Index of the desired item.

        Returns:
        - Image and its corresponding label.
        """
        file_name = self.file_names[idx]
        img = self.loader(str(self.img_dir / file_name))
        label = self.loader(str(self.label_dir / file_name))
        img, label = self.preprocess(img, self.size, label=label, DA=True)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self) -> int:
        """Return the total number of files in the dataset."""
        return len(self.file_names)


class ImageDataSetValid(Dataset):
    def __init__(
        self,
        txt_path: str,
        img_dir: Union[str, Path],
        label_dir: Union[str, Path],
        size: Tuple[int, int],
        loader: Callable = default_loader,
        sort: bool = False,
    ) -> None:
        """
        Dataset for validation images.

        Parameters:
        - txt_path: Path to the text file containing file names.
        - img_dir: Directory containing images.
        - label_dir: Directory containing labels.
        - size: Desired size for the images (height, width).
        - loader: Function to load images.
        - sort: Whether to sort filenames.
        """
        file_names = load_txt(txt_path)
        if sort:
            file_names.sort()
        self.file_names = file_names
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = size
        self.loader = loader

    def preprocess(
        self, img: Image.Image, size: Tuple[int, int], label: bool = False
    ) -> np.ndarray:
        """
        Pre-process the image.

        Parameters:
        - img: Input image.
        - size: Desired size for the image.
        - label: Flag to indicate if the input is a label.

        Returns:
        - Processed image.
        """
        img = img.resize(size, Image.BICUBIC)
        img = np.array(img)
        img = check_data_shape(img, size)

        if label:
            # normalize
            if img.max() > 1.0:
                img //= 255.0
        else:
            # normalize
            if img.max() > 1.0:
                img /= 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Parameters:
        - idx: Index of the desired item.

        Returns:
        - Image and its corresponding label.
        """
        file_name = self.file_names[idx]
        img = self.loader(str(self.img_dir / file_name))
        label = self.loader(str(self.label_dir / file_name))
        img = self.preprocess(img, self.size)
        label = self.preprocess(label, self.size, label=True)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self) -> int:
        """Return the total number of files in the dataset."""
        return len(self.file_names)


if __name__ == "__main__":

    def loadTxt(filename):
        f = open(filename, encoding="utf-8")
        context = list()
        for line in f:
            context.append(line.replace("\n", ""))
        return context

    def test_load_txt_functions():
        # Step 1: Write sample text to a file
        sample_content = [
            "This is the first line.",
            "This is the second line.",
            "This is the third line.",
            "",
        ]
        test_filename = "test_file.txt"
        with open(test_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(sample_content))

        # Step 2: Read the content using both functions
        content_loadTxt = loadTxt(test_filename)
        content_load_txt = load_txt(test_filename)

        # Step 3: Compare the results
        assert (
            content_loadTxt == content_load_txt
        ), f"loadTxt: {content_loadTxt}, load_txt: {content_load_txt}"
        print("Both functions returned the same content!")

    test_load_txt_functions()
