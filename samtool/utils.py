import os

import numpy as np
from PIL import Image


def label_exists(labeldir: str, image_filename: str, num_channels: int) -> bool:
    """Tests whether the label exists for a given image.

    Args:
        labeldir (str): directory of the labels on the disk
        image_filename (str): name of the image that corresponds to this label
        num_channels (int): number of channels that is expected

    Returns:
        bool:
    """
    for i in range(num_channels):
        label_filename = os.path.splitext(image_filename)[0] + f"_{i}.png"
        if not os.path.isfile(os.path.join(labeldir, label_filename)):
            return False
    return True


def delete_label(labeldir: str, image_filename: str) -> None:
    """Deletes the label for a specific image if it exists

    Args:
        labeldir (str): directory of the labels on the disk
        image_filename (str): name of the image that corresponds to this label

    Returns:
        None:
    """
    i = -1
    while True:
        i += 1
        label_filename = os.path.splitext(image_filename)[0] + f"_{i}.png"
        if not os.path.isfile(os.path.join(labeldir, label_filename)):
            return

        os.remove(os.path.join(labeldir, label_filename))


def save_label(labeldir: str, image_filename: str, label: np.ndarray) -> None:
    """Saves the label to a series of png images on the disk given a npy array.

    Args:
        labeldir (str): directory of the labels on the disk
        image_filename (str): name of the image that corresponds to this label
        label (np.ndarray): an array of [W, H, C]

    Returns:
        None:
    """
    assert len(label.shape) == 3
    for i, layer in enumerate(np.transpose(label, (2, 0, 1))):
        label_filename = os.path.splitext(image_filename)[0] + f"_{i}.png"
        im = Image.fromarray(layer)
        im.save(os.path.join(labeldir, label_filename))


def retrieve_label(labeldir: str, image_filename: str) -> np.ndarray:
    """retrieve_label.

    Args:
        labeldir (str): directory of the labels on the disk
        image_filename (str): name of the image that corresponds to this label

    Returns:
        np.ndarray: the label as an array of [W, H, C]
    """
    npy_list = []
    while True:
        label_filename = os.path.splitext(image_filename)[0] + f"_{len(npy_list)}.png"
        if not os.path.isfile(os.path.join(labeldir, label_filename)):
            assert len(npy_list) != 0
            return np.stack(npy_list, axis=-1)

        im = Image.open(os.path.join(labeldir, label_filename))
        npy_list.append(np.array(im))
