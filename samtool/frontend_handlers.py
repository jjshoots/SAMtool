import os

import cv2
import numpy as np
import yaml

from samtool.colors import colors
from samtool.sammer import sam_model


class FileSeeker:
    """FileSeeker.

    Handles searching for next and previous files.
    """

    def __init__(self, images_dir: str, labels_dir: str, annotations_dir: str):
        """__init__."""
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.all_images = os.listdir(images_dir)

        # check the validity of the labels
        labels = yaml.safe_load(open(annotations_dir))
        labels_check = list(labels.values())
        assert all(isinstance(i, int) for i in list(labels.values()))
        assert 0 in labels_check
        labels_check.sort()
        labels_check = np.array(labels_check[1:]) - np.array(labels_check[:-1])
        assert all(labels_check == 1)
        self.labels = labels

    def get_full_paths(self, filename: str) -> tuple[str, str]:
        """Produces the image and label full paths.

        Args:
            filename (str): filename

        Returns:
            tuple[str, str]:
        """
        image_path = os.path.join(self.images_dir, filename)
        label_path = os.path.join(self.labels_dir, f"{filename}.npy")

        return image_path, label_path

    # next file previous file
    def file_increment(self, ascend: bool, unlabelled_only: bool, filename: str) -> str:
        """file_increment.

        Args:
            ascend (bool): ascend
            unlabelled_only (bool): unlabelled_only
            filename (str): filename

        Returns:
            str:
        """
        try:
            index = self.all_images.index(filename)
        except ValueError:
            index = 0

        while True:
            index += 1 if ascend else -1

            # don't exceed index
            if index <= -1 or index >= len(self.all_images):
                index += 1 if not ascend else -1
                break

            # we don't care if labelled or unlabelled
            if not unlabelled_only:
                break

            # we only care if unlabelled
            maskfile = os.path.join(self.labels_dir, f"{self.all_images[index]}.npy")
            if not os.path.isfile(maskfile):
                break

        return self.all_images[index]

class SammedImage:
    """SammedImage."""

    def __init__(
        self, labels: dict[str, int], image_path: str, label_path: str
    ):
        """__init__."""
        self.labels = labels

        # store paths
        self.image_path = image_path
        self.label_path = label_path

        # store the base image and mask
        self.base_image: np.ndarray = np.array([])
        self.part_mask: np.ndarray = np.array(None)

        # storage for points and validities
        self.coords: list[np.ndarray] = list()
        self.validity: list[bool] = list()

        # update the base image, store the embeddings
        self.base_image = cv2.imread(self.image_path)
        self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2RGB)
        self.embedding = sam_model.compute_embeddings(self.base_image)

    def reset(self) -> np.ndarray:
        """Resets all part mask selections.

        Args:

        Returns:
            np.ndarray:
        """
        # reset the part mask
        self.part_mask = np.array(None)
        return self.base_image

    def get_comp_image(self) -> np.ndarray:
        """Returns the full image with the composite mask, this is for the right side image.

        Args:

        Returns:
            np.ndarray:
        """
        image = self.base_image

        # check if we have a complete mask
        if os.path.isfile(self.label_path):
            # draw the masks if we have it
            comp_mask = np.load(self.label_path)
            for i, mask in enumerate(np.moveaxis(comp_mask, -1, 0).copy()):
                image = self.show_mask(image, mask, i)

        return image

    def clear_coords_validity_part(self) -> np.ndarray:
        """Clears coordinates and validities to reset selection.

        Args:

        Returns:
            np.ndarray:
        """
        self.coords = list()
        self.validity = list()
        self.part_mask = np.array(None)

        return self.base_image

    def add_coords_validity(self, coord: np.ndarray, validity: bool):
        """Adds the coords and validity to the currently tracked list, coords must be (2, ) array and validity must be a bool

        Args:
            coord: np.ndarray
            validity: bool
        """
        self.coords.append(coord)
        self.validity.append(validity)

    def update_part_image(self, label) -> np.ndarray:
        """Updates the masks on the ax using the coords and validities. This is for the left image.

        Args:
            label (str): label
        """
        assert label in self.labels

        if len(self.coords) != 0:
            # set the embeddings, generate the new mask
            sam_model.set_embedding(self.embedding)
            masks, scores, logits = sam_model.predict(
                point_coords=np.array(self.coords),
                point_labels=np.array(self.validity),
                multimask_output=False,
            )
            self.part_mask = masks[0]

            # display the mask
            return self.show_mask(self.base_image, self.part_mask, self.labels[label])
        else:
            return self.base_image

    def part_to_comp_mask(self, key: str, add: bool = True) -> None:
        """Adds the part mask to the composite mask. This is basically pushing the mask on the left to the right.

        Args:
            key (str): key
            add (bool): add

        Returns:
            None:
        """
        assert key in self.labels

        # get the compound mask
        if os.path.isfile(self.label_path):
            comp_mask = np.load(self.label_path)
        else:
            comp_mask = np.zeros(
                (*self.base_image.shape[:2], len(self.labels)), dtype=bool
            )

        # save the mask
        if add:
            comp_mask[..., self.labels[key]] |= self.part_mask
        else:
            comp_mask[..., self.labels[key]] &= np.logical_not(self.part_mask)
        np.save(self.label_path, comp_mask)

        # reset the coords and validity
        self.clear_coords_validity_part()

    def clear_comp_mask(self, label: None | str = None) -> None:
        """Clears a label from the composite mask. If the label is not provided, the whole mask is cleared.

        Args:
            label (None | str): label

        Returns:
            None:
        """
        if not os.path.isfile(self.label_path):
            return

        # if full reset, delete the mask, otherwise, just override
        if label is None:
            os.remove(self.label_path)
        else:
            assert label in self.labels
            comp_mask = np.load(self.label_path)
            comp_mask[..., self.labels[label]] &= False
            np.save(self.label_path, comp_mask)

        # reset the coords and validity
        self.clear_coords_validity_part()

    @staticmethod
    def show_mask(image: np.ndarray, mask: np.ndarray, color_index=0):
        """Adds a particular mask to the image.

        Args:
            mask (np.ndarray): (H, W) array of booleans
            ax (plt.Axes): ax
            random_color:
        """

        # ignore if all zeros
        if not mask.any():
            return image

        color = colors[color_index]
        image = image.astype(np.float32)
        image[mask[...], :] += color * 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
