import glob
import pickle
import random

import h5py
import numpy as np
import pandas as pd
import scipy

from utils.help_func import register_dataset


def load_dataset_path(cfg):
    dataset_cfg = getattr(cfg.preprocess, cfg.train.dataset)
    file_list = glob.glob(f"{cfg.preprocess.procdata_path}/{cfg.train.dataset}/*")
    file_list = random.sample(file_list, len(file_list))
    val_size = dataset_cfg.subject_num // cfg.train.training.fold_num

    dataset_path = {}
    dataset_path["val_path"] = file_list[:val_size]
    dataset_path["test_path"] = file_list[val_size : 2 * val_size]
    dataset_path["train_path"] = [file for file in file_list if file not in (dataset_path["val_path"] + dataset_path["test_path"])]

    return dataset_path


@register_dataset("WildPPG")
def load_WildPPG(sub_idx, cfg, dataset_cfg):
    def load_wildppg_participant(path):
        """
        Loads the data of a WildPPG participant and cleans it to receive nested dictionaries
        """
        loaded_data = scipy.io.loadmat(path)
        loaded_data["id"] = loaded_data["id"][0]
        if len(loaded_data["notes"]) == 0:
            loaded_data["notes"] = ""
        else:
            loaded_data["notes"] = loaded_data["notes"][0]

        for bodyloc in ["sternum", "head", "wrist", "ankle"]:
            bodyloc_data = dict()  # data structure to feed cleaned data into
            sensors = loaded_data[bodyloc][0].dtype.names
            for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
                bodyloc_data[sensor_name] = dict()
                field_names = sensor_data[0][0].dtype.names
                for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                    bodyloc_data[sensor_name][sensor_field] = field_data[0]
                    if sensor_field == "fs":
                        bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            loaded_data[bodyloc] = bodyloc_data
        return loaded_data

    sub_path = sorted(glob.glob(f"{cfg.preprocess.rawdata_path}/WildPPG/*.mat"))
    sub_data = load_wildppg_participant(sub_path[sub_idx])

    ecg_data, ecg_fs = sub_data["sternum"]["ecg"]["v"], sub_data["sternum"]["ecg"]["fs"]
    ecg_data = np.lib.stride_tricks.sliding_window_view(
        ecg_data,
        ecg_fs * cfg.preprocess.segment_len,
    )[:: ecg_fs * cfg.preprocess.segment_len]

    ppg_data_list = []
    for loc in dataset_cfg.locations:
        for color in dataset_cfg.colors:
            ppg_data, ppg_fs = sub_data[loc][f"ppg_{color}"]["v"], sub_data[loc][f"ppg_{color}"]["fs"]
            ppg_data = np.lib.stride_tricks.sliding_window_view(
                ppg_data,
                ppg_fs * cfg.preprocess.segment_len,
            )[:: ppg_fs * cfg.preprocess.segment_len]
            ppg_data_list.append(ppg_data)

    ecg_data = np.tile(ecg_data, (len(ppg_data_list), 1))
    ppg_data = np.concatenate(ppg_data_list, axis=0)
    x_data = ppg_data
    y_data = ecg_data

    return x_data, y_data


@register_dataset("DaLiA")
def load_DaLiA(sub_idx, cfg, dataset_cfg):
    with open(f"{cfg.preprocess.rawdata_path}/DaLiA/PPG_FieldStudy/S{sub_idx + 1}/S{sub_idx + 1}.pkl", "rb") as f:
        sub_data = pickle.load(f, encoding="latin1")
    ecg_data = sub_data["signal"]["chest"]["ECG"].squeeze()
    ppg_data = sub_data["signal"]["wrist"]["BVP"].squeeze()
    ecg_data = np.lib.stride_tricks.sliding_window_view(
        ecg_data,
        dataset_cfg.label_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.label_fs * cfg.preprocess.segment_len]
    ppg_data = np.lib.stride_tricks.sliding_window_view(
        ppg_data,
        dataset_cfg.ppg_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.ppg_fs * cfg.preprocess.segment_len]
    x_data, y_data = ppg_data, ecg_data
    return x_data, y_data


@register_dataset("UCI-BP")
def load_UCI_BP(sub_idx, cfg, dataset_cfg):
    with h5py.File(f"{cfg.preprocess.rawdata_path}/UCI-BP/Part_{sub_idx // 4 + 1}.mat", "r") as f:
        part_data = f[f"Part_{sub_idx // 4 + 1}"]
        onset_idx, offset_idx = sub_idx % 2 * 1500, (sub_idx % 2 + 1) * 1500
        refs = part_data[onset_idx:offset_idx, 0]
        sub_data = [f[ref][:] for ref in refs]

    abp_data = np.concatenate(
        [
            np.lib.stride_tricks.sliding_window_view(
                sample[:, 1],
                dataset_cfg.label_fs * cfg.preprocess.segment_len,
            )[:: dataset_cfg.label_fs * cfg.preprocess.segment_len]
            for sample in sub_data
        ],
        axis=0,
    )
    ppg_data = np.concatenate(
        [
            np.lib.stride_tricks.sliding_window_view(
                sample[:, 0],
                dataset_cfg.ppg_fs * cfg.preprocess.segment_len,
            )[:: dataset_cfg.ppg_fs * cfg.preprocess.segment_len]
            for sample in sub_data
        ],
        axis=0,
    )
    x_data, y_data = ppg_data, abp_data
    return x_data, y_data


@register_dataset("MIMIC-BP")
def load_MIMIC_BP(sub_idx, cfg, dataset_cfg):
    ppg_path_list = sorted(glob.glob(f"{cfg.preprocess.rawdata_path}/MIMIC-BP/ppg/*.npy"))
    abp_path_list = sorted(glob.glob(f"{cfg.preprocess.rawdata_path}/MIMIC-BP/abp/*.npy"))
    sub_ppg = np.load(ppg_path_list[sub_idx])
    sub_abp = np.load(abp_path_list[sub_idx])
    sub_ppg = np.lib.stride_tricks.sliding_window_view(
        sub_ppg,
        (1, dataset_cfg.ppg_fs * cfg.preprocess.segment_len),
    )[:, :: dataset_cfg.ppg_fs * cfg.preprocess.segment_len].reshape(-1, dataset_cfg.ppg_fs * cfg.preprocess.segment_len)
    sub_abp = np.lib.stride_tricks.sliding_window_view(
        sub_abp,
        (1, dataset_cfg.label_fs * cfg.preprocess.segment_len),
    )[:, :: dataset_cfg.label_fs * cfg.preprocess.segment_len].reshape(-1, dataset_cfg.label_fs * cfg.preprocess.segment_len)
    x_data, y_data = sub_ppg, sub_abp

    return x_data, y_data


@register_dataset("WESAD")
def load_WESAD(sub_idx, cfg, dataset_cfg):
    sub_path_list = sorted(glob.glob(f"{cfg.preprocess.rawdata_path}/WESAD/S*/*.pkl"))
    with open(sub_path_list[sub_idx], "rb") as f:
        sub_data = pickle.load(f, encoding="latin1")
    ppg_data = sub_data["signal"]["wrist"]["BVP"].squeeze()
    resp_data = sub_data["signal"]["chest"]["Resp"].squeeze()
    ppg_data = np.lib.stride_tricks.sliding_window_view(
        ppg_data,
        dataset_cfg.ppg_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.ppg_fs * cfg.preprocess.segment_len]
    resp_data = np.lib.stride_tricks.sliding_window_view(
        resp_data,
        dataset_cfg.label_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.label_fs * cfg.preprocess.segment_len]
    x_data, y_data = ppg_data, resp_data
    return x_data, y_data


@register_dataset("BIDMC")
def load_BIDMC(sub_idx, cfg, dataset_cfg):
    sub_data = pd.read_csv(f"{cfg.preprocess.rawdata_path}/BIDMC/1.0.0/bidmc_csv/bidmc_{sub_idx + 1:02d}_Signals.csv")
    ppg_data = sub_data[" PLETH"].values
    resp_data = sub_data[" RESP"].values
    ppg_data = np.lib.stride_tricks.sliding_window_view(
        ppg_data,
        dataset_cfg.ppg_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.ppg_fs * cfg.preprocess.segment_len]
    resp_data = np.lib.stride_tricks.sliding_window_view(
        resp_data,
        dataset_cfg.label_fs * cfg.preprocess.segment_len,
    )[:: dataset_cfg.label_fs * cfg.preprocess.segment_len]
    x_data, y_data = ppg_data, resp_data
    return x_data, y_data
