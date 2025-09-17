import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.help_func import (
    DualWriter,
    PPGDataset,
    compute_metrics,
    fix_seed,
    initialize_model,
    load_checkpoint,
    plot_signal,
    summarize,
)
from utils.load_data import load_dataset_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(state, model, dataloader, cfg, epoch=None, optimizer=None, plot=False, plot_num=5):
    assert state in ["train", "val", "test"]
    epoch_num = cfg.train.training.epoch_num
    dataset_cfg = getattr(cfg.preprocess, cfg.train.dataset)
    task_spec_metrics = cfg.train.task_specific_metrics[dataset_cfg.label]
    metrics = {"loss": [], "time": [], "mae": []}
    trial_path = f"{cfg.train.logging.log_path}/{cfg.train.model}_{cfg.train.dataset}_{cfg.train.logging.description}"
    plot_path = f"{trial_path}/{cfg.train.logging.plot_folder}"

    if state == "test":
        for task_metric in task_spec_metrics:
            metrics[task_metric] = []
        window_size = cfg.train.task_specific_metrics.window_size[task_spec_metrics[0]]
        segment_len = cfg.preprocess.segment_len
        assert window_size % segment_len == 0
        metric_seq_num = window_size // segment_len
        metric_ppg_signal = torch.zeros(metric_seq_num, segment_len * cfg.preprocess.resample_rate)
        metric_pred_signal = torch.zeros(metric_seq_num, segment_len * cfg.preprocess.resample_rate)
        metric_target_signal = torch.zeros(metric_seq_num, segment_len * cfg.preprocess.resample_rate)
        metric_signal_idx = 0

    if state == "train":
        model.train()
    elif state == "val" or state == "test":
        model.eval()

    plot_count = 0
    for ppg, target_signal, peak_roi in tqdm(dataloader, total=len(dataloader)):
        start_time = time.perf_counter()
        torch.cuda.synchronize()
        if state == "train":
            target_signal_scaled = target_signal
            pred_signal = model(ppg, target_signal=target_signal_scaled, peak_roi=peak_roi)
        else:
            pred_signal = model(ppg)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        if state == "train":
            loss = model.optimize(pred_signal, target_signal, optimizer)

        # Calc Metrics
        mae = torch.mean(torch.abs(pred_signal - target_signal))
        if state == "train":
            metrics["loss"].append(loss.item())
        if state == "test":
            pred_signal, target_signal, ppg = pred_signal.detach().cpu(), target_signal.detach().cpu(), ppg.detach().cpu()
            for pred_sample, target_sample, ppg_sample in zip(pred_signal, target_signal, ppg):
                metric_ppg_signal[metric_signal_idx] = ppg_sample
                metric_pred_signal[metric_signal_idx] = pred_sample
                metric_target_signal[metric_signal_idx] = target_sample
                metric_signal_idx += 1
                if metric_signal_idx == metric_seq_num:
                    for task_metric in task_spec_metrics:
                        metrics[task_metric].append(
                            compute_metrics(
                                metric_pred_signal.reshape(-1),
                                metric_target_signal.reshape(-1),
                                task_metric,
                                cfg,
                            )
                        )
                    metric_signal_idx = 0
                    if plot:
                        if plot_count < plot_num:
                            plot_count += 1
                            file_path = f"{plot_path}/{state}_{plot_count}.png"
                            plot_signal(
                                metric_ppg_signal.reshape(1, -1),
                                metric_pred_signal.reshape(1, -1),
                                metric_target_signal.reshape(1, -1),
                                file_path,
                            )
        metrics["time"].append(end_time - start_time)
        metrics["mae"].append(mae.item())

    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    metric_str = f"Epoch [{epoch + 1}/{epoch_num}]" if state == "train" else "Val" if state == "val" else "Test"
    metric_str += f", Loss: {metrics['loss']:.4f}" if state == "train" else ""
    metric_str += f", MAE: {metrics['mae']:.4f}"
    print(metric_str)

    return model, metrics


def validation(cfg):
    model_cfg = getattr(cfg.models, cfg.train.model)
    dataset_cfg = getattr(cfg.preprocess, cfg.train.dataset)
    trial_path = f"{cfg.train.logging.log_path}/{cfg.train.model}_{cfg.train.dataset}_{cfg.train.logging.description}"
    task_spec_metrics = cfg.train.task_specific_metrics[dataset_cfg.label]
    qual_plot = cfg.train.logging.qual_plot
    qual_plot_num = cfg.train.logging.qual_plot_num

    # Initializing training components
    model = initialize_model(cfg, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=cfg.train.training.weight_decay)

    # Dataset construction
    file_path = load_dataset_path(cfg)
    train_dataset = PPGDataset("Train", file_path["train_path"], cfg, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.training.batch_size, shuffle=True)
    if cfg.train.training.monitor_val:
        val_dataset = PPGDataset("Val", file_path["val_path"], cfg, device=DEVICE)
        val_loader = DataLoader(val_dataset, batch_size=cfg.train.training.batch_size, shuffle=False)
    test_dataset = PPGDataset("Test", file_path["test_path"], cfg, device=DEVICE)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.training.batch_size, shuffle=False)

    # Training / Validation
    best_metric = 1e9
    no_improve_count = 0
    print("Training...")
    for epoch in range(cfg.train.training.epoch_num):
        model, train_metrics = train("train", model, train_loader, cfg, epoch=epoch, optimizer=optimizer)
        if cfg.train.logging.wandb:
            wandb.log(
                {
                    f"{cfg.train.dataset}_train_loss": train_metrics["loss"],
                    f"{cfg.train.dataset}_train_mae": train_metrics["mae"],
                },
                step=epoch,
            )

        if cfg.train.training.monitor_val:
            _, val_metrics = train("val", model, val_loader, cfg)
            if cfg.train.logging.wandb:
                wandb.log(
                    {
                        f"{cfg.train.dataset}_val_mae": val_metrics["mae"],
                    },
                    step=epoch,
                )

            comp_metric = val_metrics[cfg.train.training.earlystop_metric]
            if comp_metric < best_metric:
                best_metric = comp_metric
                best_val_metrics = val_metrics
                best_epoch = epoch
                no_improve_count = 0

                # save checkpoint
                print(f"Saving checkpoint at epoch {epoch}")
                Path(f"{trial_path}/{cfg.train.logging.ckpt_folder}").mkdir(parents=True, exist_ok=True)
                path = f"{trial_path}/{cfg.train.logging.ckpt_folder}/pretrain_ckpt.pth"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg}, path)
            else:
                no_improve_count += 1
                if no_improve_count >= cfg.train.training.earlystop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # save checkpoint
            Path(f"{trial_path}/{cfg.train.logging.ckpt_folder}").mkdir(parents=True, exist_ok=True)
            path = f"{trial_path}/{cfg.train.logging.ckpt_folder}/pretrain_ckpt.pth"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg}, path)

    if cfg.train.training.monitor_val:
        print(f"Best Validation Metrics at epoch {best_epoch + 1}:")
        print(f"MAE            : {best_val_metrics['mae']:.4f}")
        print(f"Inference Time : {best_val_metrics['time']:.4f}s")
    del train_loader, train_dataset

    # Test
    model = load_checkpoint(f"{trial_path}/{cfg.train.logging.ckpt_folder}/pretrain_ckpt.pth", DEVICE)
    _, test_metrics = train("test", model, test_loader, cfg, plot=qual_plot, plot_num=qual_plot_num)
    print(f"MAE  : {test_metrics['mae']:.4f}")
    for task_metric in task_spec_metrics:
        print(f"{task_metric} : {test_metrics[task_metric]:.4f}")

    print(f"Inference Time : {test_metrics['time']:.4f}s")


@hydra.main(version_base=None, config_path="../config/", config_name="config.yaml")
def main(cfg: DictConfig):
    trial_path = f"{cfg.train.logging.log_path}/{cfg.train.model}_{cfg.train.dataset}_{cfg.train.logging.description}"
    Path(f"{trial_path}/{cfg.train.logging.ckpt_folder}").mkdir(parents=True, exist_ok=True)
    if cfg.train.logging.qual_plot:
        Path(f"{trial_path}/{cfg.train.logging.plot_folder}").mkdir(parents=True, exist_ok=True)
    log_file = open(f"{trial_path}/output.log", "w")
    sys.stdout = DualWriter(sys.stdout, log_file)

    print("==========================================================================================")
    print("Training Config:")
    print(OmegaConf.to_yaml(cfg.train))
    print(f"{cfg.train.model} Config:")
    print(OmegaConf.to_yaml(getattr(cfg.models, cfg.train.model)))
    assert cfg.train.model in cfg.train.available_model, f"{cfg.train.model} is not available."

    # Set seed
    fix_seed(seed=cfg.seed)

    # Init WandB
    if cfg.train.logging.wandb:
        trail_name = f"{cfg.train.model}_{cfg.train.dataset}_{cfg.train.logging.description}"
        wandb.init(project="PPG_ICASSP", name=trail_name)
        wandb.config.model = cfg.train.model

    # Pretrain
    summarize(cfg, DEVICE)
    validation(cfg)


if __name__ == "__main__":
    main()
