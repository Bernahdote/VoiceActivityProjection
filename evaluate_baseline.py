import logging
from typing import Any

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _to_plain(cfg_obj: Any) -> dict[str, Any]:
    if cfg_obj is None:
        return {}
    return OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[return-value]


def _resolve_checkpoint(cfg: DictConfig) -> str:
    ckpt = cfg.get("checkpoint", None)
    if ckpt:
        return str(ckpt)

    ckpt = cfg.get("pretrained_checkpoint_path", None)
    if ckpt:
        return str(ckpt)

    raise ValueError(
        "No checkpoint path found. Set `checkpoint=...` (or `pretrained_checkpoint_path=...`)."
    )


def _attach_metrics_from_cfg(module: Any, cfg: DictConfig) -> None:
    module_cfg = cfg.get("module", None)
    if module_cfg is None:
        return

    val_metric_cfg = module_cfg.get("val_metric", None)
    test_metric_cfg = module_cfg.get("test_metric", None)

    # If one side is missing, reuse the other metric config.
    if val_metric_cfg is None and test_metric_cfg is not None:
        val_metric_cfg = test_metric_cfg
    if test_metric_cfg is None and val_metric_cfg is not None:
        test_metric_cfg = val_metric_cfg

    if hasattr(module, "val_metric") and val_metric_cfg is not None:
        module.val_metric = instantiate(val_metric_cfg)
    if hasattr(module, "test_metric") and test_metric_cfg is not None:
        module.test_metric = instantiate(test_metric_cfg)


def _build_eval_trainer(cfg: DictConfig) -> Trainer:
    # Optional dedicated eval trainer in config.
    if cfg.get("eval_trainer", None) is not None:
        return instantiate(cfg.eval_trainer)

    trainer_cfg = cfg.get("trainer", None)
    trainer_dict = _to_plain(trainer_cfg)

    allowed = [
        "accelerator",
        "devices",
        "strategy",
        "precision",
        "deterministic",
        "benchmark",
        "inference_mode",
    ]

    kwargs: dict[str, Any] = {
        k: trainer_dict[k] for k in allowed if k in trainer_dict and trainer_dict[k] is not None
    }

    # Keep evaluation side-effect free.
    kwargs.update(
        {
            "logger": False,
            "enable_checkpointing": False,
            "num_sanity_val_steps": 0,
        }
    )
    return Trainer(**kwargs)


def _first_present(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _extract_summary(
    val_metrics: dict[str, Any], test_metrics: dict[str, Any]
) -> dict[str, Any]:
    # val loss aliases across module variants
    val_loss_vap = _first_present(val_metrics, ["val_loss_vap", "loss_vap_val", "val_loss"])
    val_loss = _first_present(
        val_metrics,
        [
            "val_loss",
            "loss_val",
            "val_total_loss",
        ],
    )

    # If only vap-loss exists, expose both keys for downstream compatibility.
    if val_loss is None:
        val_loss = val_loss_vap

    summary = {
        "val_loss": val_loss,
        "val_loss_vap": val_loss_vap,
        "val_bacc_hs": _first_present(val_metrics, ["val_bacc_hs"]),
        "val_bacc_ls": _first_present(val_metrics, ["val_bacc_ls"]),
        "val_bacc_sp": _first_present(val_metrics, ["val_bacc_sp"]),
        "test_loss": _first_present(test_metrics, ["test_loss", "loss_test"]),
        "test_loss_vap": _first_present(test_metrics, ["test_loss_vap", "loss_vap_test", "test_loss"]),
        "test_bacc_hs": _first_present(test_metrics, ["test_bacc_hs"]),
        "test_bacc_ls": _first_present(test_metrics, ["test_bacc_ls"]),
        "test_bacc_sp": _first_present(test_metrics, ["test_bacc_sp"]),
    }
    return summary


def _require_eval_paths(cfg: DictConfig) -> None:
    dm_cfg = cfg.get("datamodule", None)
    if dm_cfg is None:
        raise ValueError("Missing `datamodule` config.")

    val_path = dm_cfg.get("val_path", None)
    test_path = dm_cfg.get("test_path", None)
    if not val_path:
        raise ValueError("Missing `datamodule.val_path` (required for validation).")
    if not test_path:
        raise ValueError("Missing `datamodule.test_path` (required for test evaluation).")


def _print_metrics_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    if not metrics:
        print("  (no metrics)")
        return
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")


@hydra.main(version_base=None, config_path="vap/conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed, workers=True)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    _require_eval_paths(cfg)
    ckpt_path = _resolve_checkpoint(cfg)

    module = instantiate(cfg.module)
    _attach_metrics_from_cfg(module, cfg)

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    trainer = _build_eval_trainer(cfg)

    val_list = trainer.validate(
        model=module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        verbose=False,
    )
    test_list = trainer.test(
        model=module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        verbose=False,
    )

    val_metrics = val_list[0] if len(val_list) > 0 else {}
    test_metrics = test_list[0] if len(test_list) > 0 else {}

    summary = _extract_summary(val_metrics, test_metrics)

    _print_metrics_block("Raw Validation Metrics", val_metrics)
    _print_metrics_block("Raw Test Metrics", test_metrics)
    _print_metrics_block("Baseline Evaluation Summary", summary)


if __name__ == "__main__":
    main()
