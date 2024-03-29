import os

import cv2
import numpy as np
import torch
import yaml
from segment_anything import SamPredictor, sam_model_registry

from samtool.colors import colors
from samtool.utils import (delete_label, label_exists, retrieve_label,
                           save_label)


class FileSeeker:
    """FileSeeker.

    Handles searching for next and previous files.
    """

    def __init__(self, images_path: str, labels_path: str, annotations_path: str):
        """__init__."""
        self.images_path = images_path
        self.labels_path = labels_path

        self.all_images = os.listdir(images_path)
        self.all_labels = yaml.safe_load(open(annotations_path))

    # next file previous file
    def file_increment(self, ascend: bool, unlabelled_only: bool, filename: str):
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

            # we don't care if labelled of unlabelled
            if not unlabelled_only:
                break

            # we only care if unlabelled
            if label_exists(
                labeldir=self.labels_path,
                image_filename=filename,
                num_channels=len(self.all_labels),
            ):
                break

        return self.all_images[index]


class Sammer:
    """Sammer."""

    def __init__(self, labels: dict[str, int], images_path: str, labels_path: str):
        """__init__."""
        # check the validity of the labels
        labels_check = list(labels.values())
        assert all(isinstance(i, int) for i in list(labels.values()))
        assert 0 in labels_check
        labels_check.sort()
        labels_check = np.array(labels_check[1:]) - np.array(labels_check[:-1])
        assert all(labels_check == 1)
        self.labels = labels

        # store paths
        self.images_path = images_path
        self.labels_path = labels_path

        # store the base image and mask
        self.base_image: np.ndarray = np.array([])
        self.part_mask: np.ndarray = np.array(None)

        # storage for points and validities
        self.coords = list()
        self.validity = list()

        this_file_path = os.path.dirname(os.path.abspath(__file__))

        # downlaod the model if it doesn't exist
        if not os.path.isfile(os.path.join(this_file_path, "sam_vit_l_0b3195.pth")):
            print(
                "Model weights not found, downloading them from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth`..."
            )
            import wget

            wget.download(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                out=this_file_path,
            )

        # load the model and move them to device
        sam = sam_model_registry["vit_l"](
            checkpoint=os.path.join(this_file_path, "sam_vit_l_0b3195.pth")
        )
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(sam)

    @property
    def num_labels(self):
        return len(self.labels)

    def reset(self, filename: str, compute_embeddings: bool = True):
        # update the base image
        imagefile = os.path.join(self.images_path, filename)
        self.base_image = cv2.imread(imagefile)
        self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2RGB)

        # reset the part mask
        self.part_mask = np.array(None)

        # compute the embeddings using the image
        if compute_embeddings:
            self.predictor.set_image(self.base_image)

        return self.base_image

    def get_comp_image(self, filename: str) -> np.ndarray:
        image = self.base_image

        # check if we have a complete mask
        if label_exists(
            labeldir=self.labels_path,
            image_filename=filename,
            num_channels=self.num_labels,
        ):
            # draw the masks if we have it
            comp_mask = retrieve_label(
                labeldir=self.labels_path,
                image_filename=filename,
            )
            for i, mask in enumerate(np.moveaxis(comp_mask, -1, 0).copy()):
                image = self.show_mask(image, mask, i)

        return image

    def clear_coords_validity_part(self) -> np.ndarray:
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
        """Updates the masks on the ax using the coords and validities.

        Args:
            label (str): label
        """
        assert label in self.labels

        if len(self.coords) != 0:
            # generate the new mask
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(self.coords),
                point_labels=np.array(self.validity),
                multimask_output=False,
            )
            self.part_mask = masks[0]

            # display the mask
            return self.show_mask(self.base_image, self.part_mask, self.labels[label])
        else:
            return self.base_image

    def part_to_comp_mask(self, filename: str, key: str, add: bool = True):
        assert key in self.labels

        # get the compound mask
        if label_exists(
            labeldir=self.labels_path,
            image_filename=filename,
            num_channels=self.num_labels,
        ):
            comp_mask = retrieve_label(
                labeldir=self.labels_path,
                image_filename=filename,
            )
        else:
            comp_mask = np.zeros(
                (*self.base_image.shape[:2], self.num_labels), dtype=bool
            )

        # save the mask
        if add:
            comp_mask[..., self.labels[key]] |= self.part_mask
        else:
            comp_mask[..., self.labels[key]] &= np.logical_not(self.part_mask)
        save_label(
            labeldir=self.labels_path,
            image_filename=filename,
            label=comp_mask,
        )

        # reset the coords and validity
        self.clear_coords_validity_part()

    def clear_comp_mask(self, filename: str, label: None | str = None):
        if not label_exists(
            labeldir=self.labels_path,
            image_filename=filename,
            num_channels=self.num_labels,
        ):
            return

        # if full reset, delete the mask, otherwise, just override
        if label is None:
            delete_label(
                labeldir=self.labels_path,
                image_filename=filename,
            )
        else:
            assert label in self.labels
            comp_mask = retrieve_label(
                labeldir=self.labels_path,
                image_filename=filename,
            )
            comp_mask[..., self.labels[label]] &= False
            save_label(
                labeldir=self.labels_path,
                image_filename=filename,
                label=comp_mask,
            )

        # reset the coords and validity
        self.clear_coords_validity_part()

    @staticmethod
    def show_mask(image: np.ndarray, mask: np.ndarray, color_index=0):
        """show_mask.

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
        image[mask[...], :] *= 0.5
        image[mask[...], :] += color * 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
