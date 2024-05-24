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
        self.n_octaves = 7
        self.n_bins_per_octave = 24
        self.n_bins = self.n_octaves * self.n_bins_per_octave

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
        n_blocks = num_samples // self.hop_length

        print(f"Processed shape: {audio_waveforms.shape}")
        print(f"batch_size: {batch_size}, num_channels: {num_channels}, num_samples: {num_samples}")

        nd_waveforms = audio_waveforms.cpu().detach().numpy()

        cqt_magnitudes = np.zeros((batch_size, num_channels, n_blocks, self.n_bins), dtype=np.complex64)
        print(f"Shape cqt magnitudes: {cqt_magnitudes.shape}")

        for b in range(batch_size):
            for c in range(num_channels):
                #code snippet alain padding
                for i_block in range(n_blocks):
                    block_data = nd_waveforms[b, c, i_block * self.hop_length: (i_block + 1) * self.hop_length]
                    self.scqt_instance.inputBlock(block_data, self.hop_length)
                    for i_octave in range(self.n_octaves):
                        octave_data = self.scqt_instance.getOctaveValues(i_octave)
                        print(f"octave len: {len(octave_data)}")
                        start_idx = i_octave * self.n_bins_per_octave
                        end_idx = (i_octave + 1) * self.n_bins_per_octave
                        cqt_magnitudes[b, c, i_block, start_idx:end_idx] = np.flip(octave_data)
                        print(f"cqt_magnitudes: {cqt_magnitudes[b]}")

        cqt_magnitudes = torch.from_numpy(cqt_magnitudes).to(audio_waveforms.device)
        cqt_magnitudes = cqt_magnitudes.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
        print(f"Output shape before squeeze: {cqt_magnitudes.shape}")

        # if cqt_magnitudes.shape[0] == 1:
        #     cqt_magnitudes = cqt_magnitudes.squeeze(0)
        # if cqt_magnitudes.shape[0] == 1:
        #     cqt_magnitudes = cqt_magnitudes.squeeze(0)

        print(f"Output shape after squeeze: {cqt_magnitudes.shape}")

        return cqt_magnitudes