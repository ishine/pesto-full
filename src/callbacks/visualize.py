import logging

import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# W&B should always be an optional requirement.
# To avoid having to catch import errors everywhere, we only try to import it once and then check if wandb is None
try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


log = logging.getLogger(__name__)


def wandb_only(func):
    def wrapper(*args, **kwargs):
        if wandb is not None:
            return func(*args, **kwargs)
        log.warning(f"Method {func.__name__} can be used only with wandb.")
        return None
    return wrapper


class VisualizationCallback(pl.Callback):
    r"""Base class for visualization callbacks. Handles conditional Weights & Biases logging.
    """
    def __init__(self):
        super(VisualizationCallback, self).__init__()
        self.logger = None

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
                break
        if self.logger is None:
            global wandb
            wandb = None
            log.warning(f"As of now, `{self.__class__.__name__}` is only compatible with `WandbLogger`. "
                             f"Loggers: {pl_module.loggers}.")

    @wandb_only
    def _plot_wandb_line(self, title, **kwargs):
        assert len(kwargs) == 2, f"You should provide x and y, got {kwargs}"
        data = [tensor.cpu().numpy() for tensor in kwargs.values()]
        cols = list(kwargs.keys())
        table = wandb.Table(data=list(zip(*data)), columns=cols)
        self.logger.experiment.log({title: wandb.plot.line(table, *cols)})

    @wandb_only
    def _plot_wandb_histogram_old(self, title: str, values: torch.Tensor):
        self.logger.experiment.log({title: wandb.Histogram(values.cpu())})

    @wandb_only
    def _plot_wandb_histogram(self, title: str, values: torch.Tensor, xlabel: str = "predictions"):
        fig = wandb.Table(data=[[v] for v in values], columns=[xlabel])
        fig_name = title.lower().replace(" ", "_")
        self.logger.experiment.log({fig_name: wandb.plot.histogram(fig, xlabel, title=title)})

    @wandb_only
    def _plot_wandb_table(self, title: str, **kwargs):
        table = wandb.Table(data=list(zip(*kwargs.values())), columns=list(kwargs.keys()))
        self.logger.experiment.log({title: table})