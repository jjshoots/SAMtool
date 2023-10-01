import os

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


class _SAMModel:
    """SAM Model.

    Convenience class for the sam model that also returns the embeddings.
    """

    def __init__(self):
        # downlaod the model if it doesn't exist
        this_file_path = os.path.dirname(os.path.abspath(__file__))
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
        self.in_use = False

    def set_embedding(self, embedding):
        self.predictor.features = embedding

    def compute_embeddings(self, image: np.ndarray) -> torch.Tensor:
        while self.in_use:
            continue

        self.in_use = True

        self.predictor.set_image(image)
        if self.predictor.features is None:
            raise AssertionError("SAM produced None embeddings. This cannot happen.")

        self.in_use = False

        return torch.clone(self.predictor.features.detach())

    def predict(self, *args, **kwargs):
        return self.predictor.predict(*args, **kwargs)

sam_model = _SAMModel()
