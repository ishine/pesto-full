import logging
from typing import Mapping

import numpy as np
import torch

import lightning.pytorch as pl

import mir_eval.melody as mir_eval

from .visualize import VisualizationCallback, wandb, wandb_only


log = logging.getLogger(__name__)


class MIREvalCallback(VisualizationCallback):
    def __init__(self, cdf_resolution: int = 0, metric_name: str = "accuracy", voicing_threshold: float = 0.8):
        super(MIREvalCallback, self).__init__()
        self.cdf_resolution = cdf_resolution
        self.metric_name = metric_name
        self.voicing_threshold = voicing_threshold

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        predictions = torch.cat(pl_module.predictions).cpu().numpy()

        confidences = np.ones_like(predictions)
        labels = torch.cat(pl_module.labels).cpu().numpy()
        print(predictions.shape, confidences.shape, labels.shape)

        log_path = self.metric_name + "/{}"

        metrics = self.compute_metrics(predictions, labels, voicing=confidences > self.voicing_threshold)
        self.print_metrics(metrics)
        pl_module.log_dict({log_path.format(k): v for k, v in metrics.items()}, sync_dist=True)

        self.plot_pitch_error_cdf(predictions, labels, labels > 0)

        # compute optimal shift to see its influence
        voiced = labels > 0
        optimal_shift = self.compute_optimal_shift(predictions, labels, voiced)
        predictions[voiced] -= optimal_shift

        optimal_metrics = self.compute_metrics(predictions, labels)
        pl_module.log_dict({log_path.format('v' + k): v for k, v in optimal_metrics.items()}, sync_dist=True)

        # plot diff between optimal and actual shift
        pl_module.log("corrected shift", optimal_shift)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        predictions = torch.cat(pl_module.predictions).cpu().numpy()
        if pl_module.confidences is None:
            confidences = np.ones_like(predictions)
        else:
            confidences = torch.cat(pl_module.confidences).squeeze(-1).cpu().numpy()
        labels = torch.cat(pl_module.labels).cpu().numpy()
        print(predictions.shape, confidences.shape, labels.shape)

        log_path = "test/" + self.metric_name + "/{}"

        metrics = self.compute_metrics(predictions, labels, voicing=confidences > self.voicing_threshold)
        self.print_metrics(metrics)
        pl_module.log_dict({log_path.format(k): v for k, v in metrics.items()}, sync_dist=True)

        # Check if logger is wandb, if not, skip plotting
        if isinstance(pl_module.logger, pl.loggers.wandb.WandbLogger):
            self.plot_pitch_error_cdf(predictions, labels, labels > 0)

        # compute optimal shift to see its influence
        voiced = labels > 0
        optimal_shift = self.compute_optimal_shift(predictions, labels, voiced)
        predictions[voiced] -= optimal_shift

        optimal_metrics = self.compute_metrics(predictions, labels)
        pl_module.log_dict({log_path.format('v' + k): v for k, v in optimal_metrics.items()}, sync_dist=True)

        # plot diff between optimal and actual shift
        pl_module.log("corrected shift", optimal_shift)

    @staticmethod
    def compute_metrics(predictions: np.ndarray, labels: np.ndarray, voicing=None) -> Mapping[str, float]:
        # convert semitones to cents and infer voicing
        ref_cent, ref_voicing = mir_eval.freq_to_voicing(100 * labels)
        est_cent, est_voicing = mir_eval.freq_to_voicing(100 * predictions, voicing=voicing)

        # compute mir_eval metrics
        metrics = {}
        metrics["RPA"] = mir_eval.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
        metrics["RCA"] = mir_eval.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
        metrics["OA"] = mir_eval.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)

        return metrics

    @staticmethod
    def compute_optimal_shift(predictions: np.ndarray, labels: np.ndarray, voiced: np.ndarray) -> float:
        diff = predictions[voiced] - labels[voiced]
        shift = np.median(diff)
        std = diff.std()
        print("TODO: use calibrated model!!!", shift, std)
        return shift

    @staticmethod
    def print_metrics(metrics: Mapping[str, float]) -> None:
        print(metrics)

    @wandb_only
    def plot_pitch_error_cdf(self, predictions: np.ndarray, labels: np.ndarray, voiced: np.ndarray):
        sorted_errors = np.sort(np.abs(predictions[voiced] - labels[voiced]))
        total = len(sorted_errors)
        cumul_probs = np.arange(1, total + 1) / total

        cols = ["Pitch error (semitones)", "Cumulative Density Function"]
        fig = wandb.Table(data=list(zip(sorted_errors[::self.cdf_resolution], cumul_probs[::self.cdf_resolution])),
                          columns=cols)
        self.logger.experiment.log({f"pitch_error/{self.metric_name}": wandb.plot.line(fig, *cols)})