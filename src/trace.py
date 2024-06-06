import os
from typing import Any, Mapping, Optional

import rootutils

import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.models.networks.resnet1d import Resnet1d
from src.utils.reduce_activations import reduce_activations


class CropCQT(nn.Module):
    def __init__(self, min_steps: int, max_steps: int):
        super(CropCQT, self).__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps

        # lower bin
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        # WARNING: didn't check that it works, it may be dangerous
        return spectrograms[..., self.max_steps: self.min_steps]

        # old implementation
        batch_size, _, input_height = spectrograms.size()

        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0, \
            f"With input height {input_height:d} and output height {output_height:d}, impossible " \
            f"to have a range of {self.max_steps - self.min_steps:d} bins."

        return spectrograms[..., self.lower_bin: self.lower_bin + output_height]


class PESTO(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 crop_kwargs: Optional[Mapping[str, Any]] = None,
                 reduction: str = "alwa"):
        super(PESTO, self).__init__()
        self.encoder = encoder
        self.bins_per_semitone = 2  # WARNING: completely hardcoded

        # crop CQT
        if crop_kwargs is None:
            crop_kwargs = {}
        self.crop_cqt = CropCQT(**crop_kwargs)

        self.reduction = reduction

        # constant shift to get absolute pitch from predictions
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

    def forward(self,
                x: torch.Tensor,
                convert_to_freq: bool = True,
                return_activations: bool = False):
        r"""

        Args:
            x (torch.Tensor): mono audio waveform or batch of mono audio waveforms,
                shape (batch_size?, num_samples)
            convert_to_freq (bool): whether to convert the result to frequencies or return fractional semitones instead.
            return_activations (bool): whether to return activations or pitch predictions only

        Returns:
            preds (torch.Tensor): pitch predictions in SEMITONES, shape (batch_size?, num_timesteps)
                where `num_timesteps` ~= `num_samples` / (`self.hop_size` * `sr`)
            confidence (torch.Tensor): confidence of whether frame is voiced or unvoiced in [0, 1],
                shape (batch_size?, num_timesteps)
            activations (torch.Tensor): activations of the model, shape (batch_size?, num_timesteps, output_dim)
        """
        x = self.crop_cqt(x)  # the CQT has to be cropped beforehand

        # # for now, confidence is computed very naively just based on energy in the CQT
        # confidence = x.mean(dim=-2).max(dim=-1).values
        # conf_min, conf_max = confidence.min(dim=-1, keepdim=True).values, confidence.max(dim=-1, keepdim=True).values
        # confidence = (confidence - conf_min) / (conf_max + 1e-5 - conf_min)

        activations = self.encoder(x)

        activations = activations.roll(-round(self.shift.cpu().item() * self.bins_per_semitone), -1)

        preds = reduce_activations(activations, reduction=self.reduction)

        if convert_to_freq:
            preds = 440 * 2 ** ((preds - 69) / 12)

        if return_activations:
            return preds, confidence, activations

        return preds  # , confidence


def load_model(checkpoint: str):
    if os.path.exists(checkpoint):  # handle user-provided checkpoints
        model_path = checkpoint
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", checkpoint + ".ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {checkpoint}.")

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    hparams = checkpoint["hparams"]
    state_dict = checkpoint["state_dict"]

    # instantiate PESTO encoder
    encoder = Resnet1d(**hparams["encoder"])

    # instantiate main PESTO module and load its weights
    model = PESTO(encoder, crop_kwargs=hparams["pitch_shift"])
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


if __name__ == "__main__":
    import sys

    ckpt_path = sys.argv[1]

    model = load_model(ckpt_path)

    frame = torch.randn(1, 1, 168)

    traced_script_module = torch.jit.trace(model, frame)

    output = traced_script_module(frame)
    output_2 = model(frame)

    print(output, output)
    print(output_2)

    for o, o2 in zip(output, output_2):
        torch.testing.assert_close(o, o2, rtol=1e-05, atol=1e-08)

    traced_script_module.save("checkpoints/traced_scqt_best.pt")
