import pickle
import warnings
from pathlib import Path

import hydra
import scipy
from omegaconf import DictConfig
from scipy import signal
from tqdm import tqdm

from utils.help_func import load_dataset


def preprocess(trial_data, cfg, bandpass=True, freq_range=(0.5, 4), zscore=True, normalize=True):
    """Preprocess Trial Data

    Parameters
    ----------
    trial_data : [sample, seq_len]
        Trial data
    bandpass : bool
        Whether to apply bandpass filter
    freq_range : tuple
        Frequency range for the bandpass filter.
        (-1,cutoff_freq) for low-pass, (cutoff_freq,-1) for high-pass, (low, high) for band-pass
    zscore : bool
        Whether to apply z-score normalization
    normalize : bool
        Whether to apply min-max normalization

    Output
    ------
    trial_data : [sample, seq_len]
        Preprocessed trial data
    """
    # Resample
    trial_data = signal.resample(trial_data, cfg.preprocess.resample_rate * cfg.preprocess.segment_len, axis=1)

    # Freq Filter
    if bandpass:
        nyq = 0.5 * cfg.preprocess.resample_rate
        low, high = freq_range[0] / nyq, freq_range[1] / nyq
        if freq_range[0] < 0:  # Low-pass
            b, a = signal.butter(4, high, btype="low")
        elif freq_range[1] < 0:  # High-pass
            b, a = signal.butter(4, low, btype="high")
        else:  # Band-pass
            b, a = signal.butter(4, [low, high], btype="band")
        trial_data = signal.filtfilt(b, a, trial_data)

    # Z Score
    if zscore:
        trial_data = scipy.stats.zscore(trial_data, axis=1)

    # Normalize (-1 ~ 1)
    if normalize:
        min_vals = trial_data.min(axis=1, keepdims=True)
        max_vals = trial_data.max(axis=1, keepdims=True)
        normalized = (trial_data - min_vals) / (max_vals - min_vals + 1e-8)
        trial_data = normalized * 2 - 1  # scale to [-1, 1]

    return trial_data


@hydra.main(version_base=None, config_path="../config/", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    dataset_name = cfg.preprocess.dataset
    proc_path = f"{cfg.preprocess.procdata_path}/{dataset_name}"
    Path(proc_path).mkdir(parents=True, exist_ok=True)
    dataset_cfg = getattr(cfg.preprocess, dataset_name)

    print(f"Preprocessing {dataset_name}...")
    for sub_idx in tqdm(range(dataset_cfg.subject_num)):
        x_data, y_data = load_dataset(dataset_name, sub_idx, cfg, dataset_cfg)
        x_data = preprocess(
            x_data,
            cfg,
            cfg.preprocess.ppg_bandpass,
            cfg.preprocess.ppg_freq_range,
            cfg.preprocess.ppg_zscore,
            cfg.preprocess.ppg_normalize,
        )
        y_data = preprocess(
            y_data,
            cfg,
            dataset_cfg.label_bandpass,
            dataset_cfg.label_freq_range,
            dataset_cfg.label_zscore,
            dataset_cfg.label_normalize,
        )
        data = {"x_data": x_data, "y_data": y_data}
        with open(f"{proc_path}/subject{sub_idx}.pkl", "wb") as f:
            pickle.dump(data, f)

    print(f"Subject {sub_idx} X_data shape : {x_data.shape}")
    print(f"Subject {sub_idx} y_data shape: {y_data.shape}")
    print("Done!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
