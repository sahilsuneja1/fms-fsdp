import math
import time
import os
import re
from typing import Mapping

import fire  # type: ignore
import torch
import torch.optim as optim
from transformers import AutoTokenizer

from fms.models import get_model, register_model
from fms.models.llama import LLaMABlock, LLaMAConfig
from fms.models.llama import _hf_sd_to_fms_sd as _llama_hf_sd_to_fms_sd
from fms.utils import serialization, generation, tokenizers
from fms.utils.generation import generate
from fms_extras.models.speculator import MLPSpeculator, MLPSpeculatorLayer  # type: ignore
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.policies.ac_handler import apply_fsdp_checkpointing
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
)
from speculator.train_speculator_utils_tp import EmbedLLaMA, train_speculator

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

llama_3_8b_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=32,
    hidden_grow_factor=3.5,
    multiple_of=1024,
    max_expected_seq_len=8192,
    rope_theta=500000.0,
)

llama_3_70b_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=8192,
    norm_eps=1e-5,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=3.5,
    multiple_of=4096,
    max_expected_seq_len=8192,
    rope_theta=500000.0,
)

llama_3_405b_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=16384,
    norm_eps=1e-5,
    nheads=128,
    kvheads=16,
    nlayers=126,
    hidden_grow_factor=53248/16384,
    multiple_of=4096,
    max_expected_seq_len=16384,
    rope_theta=500000.0,
)


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"^lm_head.weight", "shared.head.weight"),
        (r"^model.embed_tokens.weight", "shared.emb.weight"),
        (r"^model.norm", "dec_norm"),
        (r"^model.layers", "layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}

    trans_required_pattern = re.compile("layers.[0-9]+.attn.(query|key).weight")
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # hf -> fms requires a transpose operation for the query and key
        if bool(trans_required_pattern.match(new_name)):
            temp = new_sd[new_name]
            # nheads is used in the transformation required for hf->fms
            # here we are using 128 as this value fits with all popular models
            #   7B, 13B, 70B to recover the number of heads
            nheads = int(temp.size(0) / 128)

            temp = (
                temp.view(nheads, 2, -1, temp.size(1))
                .transpose(1, 2)
                .reshape(*temp.size())
            )

            new_sd[new_name] = temp

    return new_sd

def _llama_factory_factory(config):
    def factory(**kwargs):
        return EmbedLLaMA(config, **kwargs)

    return factory


register_model("embedllama", "7b", _llama_factory_factory(LLaMAConfig()))
register_model("embedllama", "llama3_8b", _llama_factory_factory(llama_3_8b_config))
register_model("embedllama", "llama3_70b", _llama_factory_factory(llama_3_70b_config))
register_model("embedllama", "llama3_405b", _llama_factory_factory(llama_3_405b_config))
#serialization.register_adapter("embedllama", "hf", _hf_sd_to_fms_sd)
serialization.register_adapter("embedllama", "hf", _llama_hf_sd_to_fms_sd)




def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)
    cfg.seq_length = cfg.seq_length + cfg.n_speculator_heads + 1

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    torch.cuda.set_device(local_rank)

    if cfg.sharding_strategy != 'tp':
        setup()
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
        base_model_mesh = None
        speculator_mesh = None
    else:
        base_model_mesh = setup(dp=world_size//32, tp=32)
        speculator_mesh = dist.device_mesh.init_device_mesh('cuda', (world_size,))
        #base_model_mesh = setup(dp=2, tp=4) #simulated multi node in a single node
        #base_model_mesh = setup(dp=1, tp=8) #simulated multi node in a single node
        #speculator_mesh = dist.device_mesh.init_device_mesh('cuda', (8,))
        torch._C._distributed_c10d._register_process_group("default", base_model_mesh['tp'].get_group())
        #fms.distributed.tensorparallel.TP_MESH = base_model_mesh['tp']

    torch.cuda.empty_cache()
    setup_environ_flags()
    torch.set_default_dtype(torch.bfloat16)

    # get policy
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, LLaMABlock)

    # get base model
    model = get_model(
        #"embedllama",
        #"8b",
        #model_path=f"{cfg.model_path}/*.safetensors",
        cfg.model_arch,
        cfg.model_variant,
        model_path=cfg.model_path,
        device_type="cuda",
        source="hf",
        distributed_strategy=cfg.sharding_strategy,
        group=base_model_mesh['tp'].get_group() if cfg.sharding_strategy == 'tp' else None,
    )
    #model = model.bfloat16()
    if False:
        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy_policy,
            use_orig_params=cfg.use_torch_compile,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=cfg.low_cpu_fsdp,
            param_init_fn=lambda module: (
                module.to_empty(device=torch.device("cuda"), recurse=False)
                if cfg.low_cpu_fsdp
                else None
            ),
        )

    if cfg.sharding_strategy == 'tp':
        print(f"{local_rank}, {rank}, {world_size}, {base_model_mesh['tp'].get_group()}, {base_model_mesh['tp'].size()}, {base_model_mesh['tp'].get_local_rank()}")

    if rank == 0:
        print(f"{time.time()}")
        print(model.config)
        print(model)

    # get speculator
    if False:
        print("Testing model generation")
        model.eval()
        torch.set_grad_enabled(False)
        tokenizer = tokenizers.get_tokenizer(cfg.model_path)
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

        prompt = template.format(
            "Provide a list of instructions for preparing chicken soup."
        )
        tokens = tokenizer.tokenize(prompt)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        # include this line for embedllama
        ids = [tokenizer.bos_token_id] + ids
        ids = torch.tensor(ids, dtype=torch.long, device="cuda")
        result = generation.generate(
            model,
            ids,
            max_new_tokens=100,
            use_cache=True,
            do_sample=False,
            max_seq_len=8192,
        )
        result = generation.truncate_after_eos(result, tokenizer.eos_token_id)
        if rank == 0:
            print("quick test of base model")
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
        exit(1) 

    print("Loading speculator")
    speculator = MLPSpeculator(
        model.config.emb_dim,
        cfg.speculator_width,
        model.config.src_vocab_size,
        cfg.n_speculator_heads,
    )
    speculator.reset_parameters()

    if rank == 0:
        total_params = sum(
            p.numel() for p in speculator.parameters() if p.requires_grad
        )
        print(f"\n--> speculator has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        if cfg.sharding_strategy == 'tp':
            train_loader = get_data_loader(cfg, speculator_mesh.get_rank(), speculator_mesh.size(), postprocess=[])
        else:
            train_loader = get_data_loader(cfg, rank, world_size, postprocess=[])    
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)

        print("Datasets constructed!")

    # FSDP
    speculator = FSDP(
        speculator,
        auto_wrap_policy=None,
        mixed_precision=mixed_precision_policy,
        #sharding_strategy=ShardingStrategy.NO_SHARD,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: (
            module.to_empty(device=torch.device("cuda"), recurse=False)
            if cfg.low_cpu_fsdp
            else None
        ),
        device_mesh=speculator_mesh if cfg.sharding_strategy == 'tp' else None,
    )
    apply_fsdp_checkpointing(speculator, MLPSpeculatorLayer, 1)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print(f"--> enabling torch compile...")
            if cfg.fsdp_activation_checkpointing:
                raise ValueError(
                    "Compile does not yet work well with llama+ac, please"
                    "either use it without activation checkpointing, or disable"
                    "compile."
                )
        model = torch.compile(model)
        speculator = torch.compile(speculator)

    # Optimizer
    optimizer = optim.AdamW(
        speculator.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # optionally load from checkpoint (when continue pretraining)
    if cfg.sharding_strategy == 'tp':
        checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, "ddp", speculator_mesh.get_rank(), speculator_mesh.get_local_rank(), model_auto_placement=True)
    else:
        checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, "ddp", rank, local_rank)
    speculator, optimizer, train_loader, start_step, tokens_seen, _ = checkpointer.load(
        speculator,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
        is_compiled=cfg.use_torch_compile,
    )

    # LR schedule
    # These functions provide LR scaling factors in [0,1] based on step count.
    # Stage 1: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of max LR.
    warmup_interval1 = min(2000, cfg.stage2_start_step // 20)
    stage1_schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval1) / warmup_interval1) ** 2,
        0.1
        + 0.5
        * (1 - 0.1)
        * (
            1
            + math.cos(min(x, cfg.stage2_start_step) / cfg.stage2_start_step * math.pi)
        ),
    )
    # Stage 2: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of stage 1's final LR.
    warmup_interval2 = min(2000, (cfg.num_steps - cfg.stage2_start_step) // 20)
    stage2_schedule = lambda x: min(
        0.1 * (1 - (1 - min(x, warmup_interval2) / warmup_interval2) ** 2),
        0.01
        + 0.05
        * (1 - 0.1)
        * (
            1
            + math.cos(
                min(x, cfg.num_steps - cfg.stage2_start_step)
                / (cfg.num_steps - cfg.stage2_start_step)
                * math.pi
            )
        ),
    )
    # Assemble full scheduling function with correct step offsets.
    schedule = (
        lambda x: stage1_schedule(x)
        if x <= cfg.stage2_start_step
        else stage2_schedule(x - cfg.stage2_start_step)
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    torch.cuda.empty_cache()
    train_speculator(
        cfg,
        model,
        speculator,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        start_step,
        tokens_seen,
        profiler,
        base_model_mesh,
        speculator_mesh,        
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
