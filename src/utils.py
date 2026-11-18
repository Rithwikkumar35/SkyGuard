"""
src/utils.py
Utility functions for SkyGuard: paths, logging, plotting style, and formatting.
"""

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the data directory path, ensuring it exists."""
    path = get_project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Return the models directory path, ensuring it exists."""
    path = get_project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dir() -> Path:
    """Return the results directory path, ensuring it exists."""
    path = get_project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_docs_dir() -> Path:
    """Return the docs directory path, ensuring it exists."""
    path = get_project_root() / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str = "SkyGuard", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_academic_plot_style() -> None:
    """Apply clean, publication-grade styling for matplotlib/seaborn plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 13,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
