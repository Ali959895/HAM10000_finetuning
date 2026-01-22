import argparse
import os
import torch
import os
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))  # ensures "src" is on path
import lavis_ext  # IMPORTANT: registers clip_vitl_multilabel
import lavis.common.optims  # registers lr schedulers via decorators
from datetime import datetime
from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.common.dist_utils import init_distributed_mode, get_rank, is_main_process
from lavis.common.logger import setup_logger
from lavis.runners.runner_base import RunnerBase
from omegaconf import OmegaConf
from lavis.common.dist_utils import init_distributed_mode

# IMPORTANT: register our extensions
import lavis_ext  # noqa: F401

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg-path", required=True)
    parser.add_argument("--options", nargs="+", default=None)
    args = parser.parse_args()

    cfg = Config(args)

    os.makedirs(cfg.run_cfg.output_dir, exist_ok=True)
    setup_logger()
    # Only initialize distributed if torchrun/srun environment is present
    has_dist_env = (
        "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
        or "WORLD_SIZE" in os.environ
        or ("MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ)
    )
    if getattr(cfg.run_cfg, "distributed", False) and has_dist_env:
        init_distributed_mode(cfg.run_cfg)
    else:
        # force single-process mode
        cfg.run_cfg.distributed = False
        cfg.run_cfg.rank = 0
        cfg.run_cfg.world_size = 1
        cfg.run_cfg.gpu = 0

    # set output dir in registry (used by some LAVIS utilities)
    registry.register_path("output_dir", cfg.run_cfg.output_dir)
    #registry.register_path("result_dir", os.path.join(cfg.run_cfg.output_dir, "result"))

    # build task/datasets/model
    task = registry.get_task_class(cfg.run_cfg.task).setup_task(cfg)
    datasets = task.build_datasets(cfg)

    model = task.build_model(cfg)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    run_cfg = cfg.run_cfg
    job_id = getattr(run_cfg, "job_id", None)
    if job_id is None and hasattr(run_cfg, "get"):
        job_id = run_cfg.get("job_id", None)
    
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        
    if hasattr(registry, "mapping") and "paths" in registry.mapping:
        registry.mapping["paths"].pop("result_dir", None)
        registry.mapping["paths"].pop("output_dir", None)  # optional
    # allow adding missing keys
    # cfg.run_cfg is an OmegaConf DictConfig
    OmegaConf.set_struct(cfg.run_cfg, False)
    
    def set_default(k, v):
        if not hasattr(cfg.run_cfg, k):
            setattr(cfg.run_cfg, k, v)

    # Core RunnerBase fields
    set_default("evaluate", False)
    set_default("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_default("max_epoch", 20)
    
    # Dataloader / training loop fields (these are what you're missing now)
    set_default("batch_size_train", 16)
    set_default("batch_size_eval", 32)
    set_default("num_workers", 4)
    set_default("pin_memory", True)
    
    # Common extras to avoid more 'Missing key '' soon
    set_default("accum_grad_iters", 1)
    set_default("warmup_steps", 0)
    set_default("log_freq", 50)
    set_default("seed", 42)
    set_default("save_ckpt_freq", 1)
    set_default("resume_ckpt_path", None)
    
    # Optimizer / LR schedule defaults (RunnerBase expects these)
    set_default("lr_sched", "linear_warmup_cosine_lr")
    set_default("init_lr", 1e-5)          # start conservative for ViT-L fine-tuning
    set_default("min_lr", 1e-6)
    set_default("weight_decay", 0.05)
    set_default("warmup_lr", 1e-7)
    set_default("warmup_steps", 0)     
    
    # Minimal defaults RunnerBase expects
    if not hasattr(cfg.run_cfg, "device"):
        cfg.run_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not hasattr(cfg.run_cfg, "evaluate"):
        cfg.run_cfg.evaluate = False
    if not hasattr(cfg.run_cfg, "max_epoch"):
        cfg.run_cfg.max_epoch = 10   # choose what you want (e.g., 5/10/20)
    
    # Highly recommended to avoid the next common missing-key errors
    if not hasattr(cfg.run_cfg, "log_freq"):
        cfg.run_cfg.log_freq = 50
    if not hasattr(cfg.run_cfg, "seed"):
        cfg.run_cfg.seed = 42
    if not hasattr(cfg.run_cfg, "save_ckpt_freq"):
        cfg.run_cfg.save_ckpt_freq = 1
    if not hasattr(cfg.run_cfg, "resume_ckpt_path"):
        cfg.run_cfg.resume_ckpt_path = None

    runner = RunnerBase(
        cfg=cfg,          # or config=cfg depending on your code
        task=task,
        model=model,
        datasets=datasets,
        job_id=job_id,       # <-- IMPORTANT
    )

    runner.train()

if __name__ == "__main__":
    main()
