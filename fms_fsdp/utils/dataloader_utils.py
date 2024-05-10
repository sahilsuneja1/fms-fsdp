import torch

from fms_fsdp.utils.dataset_utils import (
    Buffer_Dataset,
    Checkpoint_Dataset,
    Preload_Buffer_Dataset,
    Preprocess_Dataset,
    Sampling_Dataset,
    Scalable_Shard_Dataset,
)


def get_dummy_loader(cfg, rank, world_size):
    """
    A simple dummy dataloader yielding incrementing vocab indices in an infinite loop
    """

    class SteadyCounter(torch.utils.data.IterableDataset):
        # Spit out incremental counts of constant length l, modulo vocab size v
        def __init__(self, l, v):
            self.i = 0
            self.l = l
            self.v = v

        def __iter__(self):
            while True:
                out = torch.IntTensor(
                    [x % self.v for x in range(self.i, self.i + self.l)]
                )
                yield out, out
                self.i += self.l

    data = SteadyCounter(cfg.seq_length, cfg.vocab_size)
    return torch.utils.data.DataLoader(data, batch_size=cfg.batch_size)


def get_data_loader(cfg, rank, world_size, postprocess=[causal_lm]):
    """
    Pytorch dataloader for stateful, distributed, and rescalable causal language model (CLM) training
    ...
    Args
    ----
    cfg : dataclass
        Training config containing seq len, dataset, dataset weight, datapath, etc. arguments
    rank : int
        Rank of current distributed worker. Used for handling dataset sharding logic.
    world_size : int
        Number of distributed workers. Used for handling dataset sharding logic.
    postprocess : List[Callable]
        Any task-specific postprocessing to apply before handing over data. Steps will apply in
        the order provided by the user. For CLM training, use postprocess=[causal_lm].        
    """

    datasets, weights = parse_data_args(cfg.datasets, cfg.weights)

    # Base streaming dataset. Returns doc chunks in sequence.
    # Implements dataset sampling and rescalability.
    data = Sampling_Dataset(
        cfg.data_path,
        Scalable_Shard_Dataset,
        rank,
        world_size,
        cfg.sep_token,
        min_length=3,
        datasets=datasets,
        weights=weights,
        seed=cfg.seed,
        verbose=(rank == 0),
        n_logical_shards=cfg.logical_shards,
    )
    # Wrap above dataset in packing logic to form constant-length lines.
    data = Buffer_Dataset(
        data,
        cfg.seq_length if causal_lm not in postprocess else cfg.seq_length + 1,
        pack_hard=True,
    )
    # Shuffle outputs in length 10k buffer. Consecutive lines appear 10k steps apart on average.
    data = Preload_Buffer_Dataset(data, 10000)
    # Apply desired postprocessing steps in sequence
    data = Preprocess_Dataset(data, torch.IntTensor)
    for p in postprocess:
        data = Preprocess_Dataset(data, p)

    # Enable auto-saving
    data = Checkpoint_Dataset(
        data,
        cfg.ckpt_load_path,
        cfg.checkpoint_interval,
        cfg.batch_size,
        cfg.ckpt_save_path,
    )
    return torch.utils.data.DataLoader(data, num_workers=1, batch_size=cfg.batch_size)


def parse_data_args(datas, weights):
    # Convert csv inputs into corresponding lists of values
    def splitstrip(x):
        if isinstance(x, str):
            return [item.strip() for item in x.split(",")]
        elif isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, (int, float, complex)):
            return [x]
        else:
            raise ValueError(f"arg input {x} cannot be parsed.")

    datas = splitstrip(datas)
    weights = [float(x) for x in splitstrip(weights)]
    return datas, weights


def causal_lm(data_seq, prompt_len=1):
    """
    Perform causal language modeling by right-shifting the input sequence.
    Sets first prompt_len tokens to be ignored by the loss.
    """
    data_seq = torch.IntTensor(data_seq)
    t = data_seq.clone()[1:]
    data_seq = data_seq[:-1]
    t[:prompt_len] = -100
    return data_seq, t
