from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_arch: str = "llama"
    model_variant: str = "7b"
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/lustre/data"
    seq_length: int = 512
    prompt_length: int = 128
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange,lang=en/dataset=PG19"
    weights: str = "7700,500,550,28,17,22,25,8,100,500,175,250,100,25"
    logical_shards: int = 800
    file_type: str = "arrow"
    col_name: str = "tokens"
    tokenizer_path: str = "/fsx/tokenizer"
    vocab_size: int = 32000
    bos_token: Optional[int] = None
    eos_token: int = 128001
    bol_token: Optional[int] = None
    eol_token: Optional[int] = None
    strip_tokens: str = ""
    num_workers: int = 1

    # fsdp policies
    sharding_strategy: str = "hsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # continued training spec
    resuming_dataset: bool = False

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = False

    # speculator training
    model_path: str = "/lustre/llama_weights/8B-llama3-hf"
    n_speculator_heads: int = 3
    speculator_width: int = 4096
    stage2_start_step: int = 15000
    stage2_prompt_length: int = 64
    stage2_batch_size: int = 12
    stage2_seq_length: int = 256
