import pickle
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy.signal as signal
import torch
from biosppy.signals import ecg, tools
from scipy.fftpack import fft
from scipy.signal import hilbert
from thop import profile
from torch.utils.data import Dataset
from tqdm import tqdm

LOAD_DATASET_REGISTRY = {}
BASELINE_REGISTRY = {}


def register_dataset(name):
    def decorator(func):
        LOAD_DATASET_REGISTRY[name] = func
        return func

    return decorator


def register_baseline(name):
    def decorator(func):
        BASELINE_REGISTRY[name] = func
        return func

    return decorator


def load_dataset(dataset_name, sub_idx, cfg, dataset_cfg):
    """
    Returns
    -------
    x_data : np.ndarray of float32, shape (sample_num, seq_len)
        PPG signal samples.
    y_data : np.ndarray of float32, shape (sample_num, seq_len)
        Corresponding signals (e.g. ECG, Resp, ABP) for each sample.
    """
    if dataset_name not in LOAD_DATASET_REGISTRY:
        raise ValueError(f"{dataset_name} is not supported")
    return LOAD_DATASET_REGISTRY[dataset_name](sub_idx, cfg, dataset_cfg)


def initialize_model(cfg, device="cuda"):
    model_name = cfg.train.model
    model_cfg = getattr(cfg.models, model_name)

    model = BASELINE_REGISTRY[model_name](
        sample_rate=cfg.preprocess.resample_rate,
        segment_len=cfg.preprocess.segment_len,
        device=device,
        **model_cfg,
    )
    model = model.to(device)

    return model


class DualWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, message):
        for w in self.writers:
            w.write(message)
            w.flush()

    def flush(self):
        for w in self.writers:
            w.flush()


def fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def summarize(cfg, device):
    model = initialize_model(cfg, device)
    sample = torch.randn((1, 1, int(cfg.preprocess.resample_rate * cfg.preprocess.segment_len)))
    sample = sample.to(device)
    flops, params = profile(model, inputs=sample)
    print(f"Params: {params / 1e6:.2f}M")
    print(f"GFLOPs: {flops / 1e9:.2f}")
    print("==========================================================================================")


def get_Rpeaks_ECG(filtered, sampling_rate):
    # segment
    (rpeaks,) = ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    (rpeaks,) = ecg.correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)

    # extract templates
    _, rpeaks = ecg.extract_heartbeats(signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate, before=0.2, after=0.4)
    rr_intervals = np.diff(rpeaks)
    return rpeaks, rr_intervals


def heartbeats_ecg(filtered, sampling_rate):
    rpeaks, rr_intervals = get_Rpeaks_ECG(filtered, sampling_rate)
    if rr_intervals.size != 0:
        # compute heart rate
        _, hr = tools.get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
        if len(hr) == 0:
            hr = [-1]
    else:
        hr = [-1]
    return hr


def calc_ecg_hr(ecg_signal, sampling_rate=128, filter=False):
    final_bpm = []
    for k in ecg_signal:
        if filter:
            k = nk.ecg_clean(k, sampling_rate=sampling_rate, method="pantompkins1985")
        hr = heartbeats_ecg(k, sampling_rate)
        bpm = np.mean(hr)
        final_bpm.append(bpm)
    return np.array(final_bpm)


def fid_features_to_statistics(features):
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        "mu": mu,
        "sigma": sigma,
    }


def fid_statistics_to_metric(stat_1, stat_2):
    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.ndim == 2 and sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return fid


def compute_metrics(pred_signal, target_signal, task_metric, cfg):
    if task_metric == "SBPError":
        pred_sbp, target_sbp = torch.max(pred_signal, dim=-1)[0], torch.max(target_signal, dim=-1)[0]
        sbp_mae = torch.mean(torch.abs(pred_sbp - target_sbp))
        return sbp_mae.item()

    elif task_metric == "DBPError":
        pred_dbp, target_dbp = torch.min(pred_signal, dim=-1)[0], torch.min(target_signal, dim=-1)[0]
        dbp_mae = torch.mean(torch.abs(pred_dbp - target_dbp))
        return dbp_mae.item()

    elif task_metric == "HeartRateError":
        pred_signal, target_signal = pred_signal.reshape(1, -1), target_signal.reshape(1, -1)

        RR_sample_rate = 128
        RR_seqlen = RR_sample_rate * cfg.preprocess.segment_len
        pred_signal = signal.resample(pred_signal.detach().cpu().numpy(), RR_seqlen, axis=1)
        target_signal = signal.resample(target_signal.detach().cpu().numpy(), RR_seqlen, axis=1)
        pred_hr = calc_ecg_hr(pred_signal, RR_sample_rate, filter=True)
        target_hr = calc_ecg_hr(target_signal, RR_sample_rate, filter=False)

        # correction
        mask = (pred_hr != -1) & (target_hr != -1)
        hr_mae = np.mean(np.abs(pred_hr[mask] - target_hr[mask])) if np.any(mask) else 0.0

        return hr_mae

    elif task_metric == "RespRateError":
        nyq = 0.5 * cfg.preprocess.resample_rate
        b, a = signal.butter(8, 1 / nyq, btype="low")
        pred_signal = signal.filtfilt(b, a, pred_signal.detach().cpu().numpy())
        target_signal = target_signal.detach().cpu().numpy()
        T = pred_signal.shape[-1]

        pred_freq, target_freq = fft(pred_signal), fft(target_signal)
        freq_bin = np.fft.fftfreq(T, d=1 / cfg.preprocess.resample_rate)
        pred_mag, target_mag = np.abs(pred_freq), np.abs(target_freq)

        positive_freq_idx = np.where(freq_bin > 0)
        freq_bin = freq_bin[positive_freq_idx]
        pred_mag, target_mag = pred_mag[positive_freq_idx], target_mag[positive_freq_idx]

        pred_bpm = 60 * freq_bin[np.argmax(pred_mag)]
        target_bpm = 60 * freq_bin[np.argmax(target_mag)]

        resp_rate_mae = np.abs(pred_bpm - target_bpm)

        return resp_rate_mae

    elif task_metric == "FD":
        pred_stats = fid_features_to_statistics(pred_signal.detach().cpu())
        target_stats = fid_features_to_statistics(target_signal.detach().cpu())
        fd = fid_statistics_to_metric(pred_stats, target_stats)
        return fd

    else:
        raise ValueError(f"{task_metric} is not supported")


class PPGDataset(Dataset):
    def __init__(self, state, path_list, cfg, device="cuda"):
        self.device = device
        self.data = {"x_data": [], "y_data": []}
        for file in tqdm(path_list, desc=f"Loading {state} data"):
            with open(file, "rb") as f:
                segment_data = pickle.load(f)
            self.data["x_data"].append(segment_data["x_data"])
            self.data["y_data"].append(segment_data["y_data"])

        self.data["x_data"] = torch.tensor(np.concatenate(self.data["x_data"], axis=0), dtype=torch.float32)
        self.data["y_data"] = torch.tensor(np.concatenate(self.data["y_data"], axis=0), dtype=torch.float32)

        if cfg.train.model == "RDDM" and state == "Train":
            sample_num, seq_len = self.data["y_data"].shape
            peak_indices = np.zeros((sample_num, 1, seq_len))
            for i in tqdm(range(sample_num), desc="Calculating peak indices"):
                target_seg = self.data["y_data"][i].clone().cpu().numpy()
                _, info = nk.ecg_peaks(
                    target_seg,
                    sampling_rate=cfg.preprocess.resample_rate,
                    method="pantompkins1985",
                    correct_artifacts=True,
                    show=False,
                )
                for peak in info["ECG_R_Peaks"]:
                    roi_start = max(0, peak - cfg.models.RDDM.roi_size // 2)
                    roi_end = min(roi_start + cfg.models.RDDM.roi_size, seq_len)
                    peak_indices[i, 0, roi_start:roi_end] = 1
            self.data["peak_indices"] = torch.tensor(peak_indices, dtype=torch.float32)
        else:
            self.data["peak_indices"] = None

    def __len__(self):
        return len(self.data["x_data"])

    def __getitem__(self, idx):
        ppg_signal = self.data["x_data"][idx].to(self.device)
        target_signal = self.data["y_data"][idx].to(self.device)

        if self.data["peak_indices"] is not None:
            peak_roi = self.data["peak_indices"][idx].to(self.device)
        else:
            peak_roi = []

        return ppg_signal, target_signal, peak_roi


def load_checkpoint(filepath, device="cuda"):
    checkpoint = torch.load(filepath, weights_only=False)
    model = initialize_model(checkpoint["cfg"], device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")

    return model


def plot_signal(ppg_signal, pred_signal, target_signal, file_path):
    ppg_signal = ppg_signal.detach().cpu().numpy()[0]
    pred_signal = pred_signal.detach().cpu().numpy()[0]
    target_signal = target_signal.detach().cpu().numpy()[0]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(target_signal)
    axs[0].set_title("Target")

    axs[1].plot(pred_signal)
    axs[1].set_title("Pred")

    axs[2].plot(ppg_signal)
    axs[2].set_title("PPG")

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()