import torch
import torch.nn as nn
import numpy as np
import pdb

import src.data.rtcqt as scqt

class HarmonicCQT(nn.Module):
    def __init__(
            self,
            harmonics,
            sr: int = 22050,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: float | None = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True
    ):
        super(HarmonicCQT, self).__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.scqt_instance = scqt.SlidingCqt24()
        self.scqt_instance.init(sr, hop_length)

        print("RTCQT initialized")

    def forward(self, audio_waveforms: torch.Tensor):
        r"""
        Args:
            audio_waveforms (torch.Tensor): Input audio waveforms of shape (batch_size, num_channels, num_samples)

        Returns:
            torch.Tensor: Harmonic CQT, shape (batch_size, num_channels, num_freqs, num_timesteps)
        """
        print(f"Input shape: {audio_waveforms.shape}")

        if len(audio_waveforms.shape) == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0).unsqueeze(0)
        elif len(audio_waveforms.shape) == 2:
            audio_waveforms = audio_waveforms.unsqueeze(1)

        batch_size, num_channels, num_samples = audio_waveforms.shape
        hop_length = 512
        n_blocks = num_samples // hop_length
        bins_per_octave = 24
        n_octaves = 7
        n_bins = bins_per_octave * n_octaves

        print(f"Processed shape: {audio_waveforms.shape}")
        print(f"batch_size: {batch_size}, num_channels: {num_channels}, num_samples: {num_samples}")

        nd_waveforms = audio_waveforms.cpu().detach().numpy()

        cqt_magnitudes = np.zeros((batch_size, num_channels, n_blocks, n_bins), dtype=np.float32)

        for b in range(batch_size):
            for c in range(num_channels):
                for i_block in range(n_blocks):
                    block_data = nd_waveforms[b, c, i_block * hop_length: (i_block + 1) * hop_length]
                    self.scqt_instance.inputBlock(block_data, hop_length)
                    for i_octave in range(n_octaves):
                        octave_data = self.scqt_instance.getOctaveValues(i_octave)
                        cqt_magnitudes[b, c, i_block, i_octave * bins_per_octave : (i_octave + 1) * bins_per_octave] = np.flip(np.abs(octave_data))

        cqt_magnitudes = torch.from_numpy(cqt_magnitudes).to(audio_waveforms.device)

        print(f"Output shape before squeeze: {cqt_magnitudes.shape}")

        if cqt_magnitudes.shape[0] == 1:
            cqt_magnitudes = cqt_magnitudes.squeeze(0)
        if cqt_magnitudes.shape[0] == 1:
            cqt_magnitudes = cqt_magnitudes.squeeze(0)

        print(f"Outptut shape after squeeze: {cqt_magnitudes.shape}")

        return cqt_magnitudes