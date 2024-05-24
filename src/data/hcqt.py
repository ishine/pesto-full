import torch
import torch.nn as nn
import numpy as np

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
        self.hop_length = hop_length
        self.n_bins_per_octave = 12 * bins_per_semitone
        self.n_bins = n_bins
        self.n_octaves = n_bins // self.n_bins_per_octave

    def forward(self, audio_waveforms: torch.Tensor):
        r"""
        Args:
            audio_waveforms (torch.Tensor): Input audio waveforms of shape (batch_size, num_channels, num_samples)

        Returns:
            torch.Tensor: Harmonic CQT, shape (batch_size, num_channels, num_freqs, num_timesteps)
        """

        if len(audio_waveforms.shape) == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0).unsqueeze(0)
        elif len(audio_waveforms.shape) == 2:
            audio_waveforms = audio_waveforms.unsqueeze(1)

        batch_size, num_channels, num_samples = audio_waveforms.shape
        n_blocks = num_samples // self.hop_length + 1

        nd_waveforms = audio_waveforms.cpu().detach().numpy()
        nd_waveforms = np.concatenate((
            nd_waveforms,
            np.zeros((1, 1, n_blocks * self.hop_length - num_samples), dtype=nd_waveforms.dtype)),
            axis=-1
        )

        cqt_magnitudes = np.zeros((batch_size, num_channels, n_blocks, self.n_bins), dtype=np.complex64)

        for b in range(batch_size):
            for c in range(num_channels):
                for i_block in range(n_blocks):
                    block_data = nd_waveforms[b, c, i_block * self.hop_length: (i_block + 1) * self.hop_length]
                    self.scqt_instance.inputBlock(block_data, self.hop_length)
                    for i_octave in range(self.n_octaves):
                        octave_data = self.scqt_instance.getOctaveValues(i_octave)
                        start_idx = i_octave * self.n_bins_per_octave
                        end_idx = (i_octave + 1) * self.n_bins_per_octave
                        cqt_magnitudes[b, c, i_block, start_idx:end_idx] = np.flip(octave_data)

        cqt_magnitudes = torch.from_numpy(cqt_magnitudes)
        cqt_magnitudes = torch.view_as_real(cqt_magnitudes)

        # if cqt_magnitudes.shape[0] == 1:
        #     cqt_magnitudes = cqt_magnitudes.squeeze(0)
        # if cqt_magnitudes.shape[0] == 1:
        #     cqt_magnitudes = cqt_magnitudes.squeeze(0)


        return cqt_magnitudes