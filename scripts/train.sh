#!/bin/bash

# On AWS, the EFA and OFI paths enable NCCL to use optimized networking.
export LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

MODEL_ARGS_LLAMA3_8B="\
--model_path=/gpfs/llama3/hf/8b_instruction_tuned
--model_arch=embedllama
--model_variant=llama3_8b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama3-8b-specu2
--ckpt_save_path=/gpfs/suneja/checkpoints/llama3-8b-specu2
--logical_shards=768
--seq_length=8192
--batch_size=1
--report_interval=10
--checkpoint_interval=3000
--num_steps=21000
--stage2_start_step=15000
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=4096
--data_path=/gpfs/
--datasets='fineweb-edu'
--weights='1'
--low_cpu_fsdp=False
--use_torch_compile=False
"
#--data_path=/gpfs/suneja/datasets/llama3-dolma
#--datasets='dataset=stack'

MODEL_ARGS_LLAMA3_70B="\
--model_path=/gpfs/llama3/hf/70b_instruction_tuned
--model_arch=embedllama
--model_variant=llama3_70b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix
--ckpt_save_path=/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix
--sharding_strategy=tp
--logical_shards=768
--seq_length=8192
--batch_size=2
--report_interval=10
--checkpoint_interval=3000
--num_steps=14211
--stage2_start_step=11169
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=8192
--data_path=/gpfs/
--datasets='fineweb-edu'
--weights='1'
--low_cpu_fsdp=False
--use_torch_compile=False
"
#--data_path=/gpfs/llama3-common-crawl/rel0_7/lang=en
#--datasets='dataset=commoncrawl'
#--data_path=/gpfs/suneja/datasets/llama3-dolma
#--datasets='dataset=stack'

MODEL_ARGS_LLAMA2_7B="\
--model_path=/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/
--model_arch=embedllama
--model_variant=7b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama2-7b-tmp-1
--ckpt_save_path=/gpfs/suneja/checkpoints/llama2-7b-tmp-1
--logical_shards=768
--sharding_strategy=hsdp
--seq_length=4096
--batch_size=8
--report_interval=10
--checkpoint_interval=5000
--num_steps=15000
--stage2_start_step=10000
--stage2_batch_size=96
--n_speculator_heads=3
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--data_path=/gpfs1/users/suneja/datasets/bpv7_high_quality_rerun_fuzzy_deduped_incomplete/lang=en
--datasets="'dataset=commoncrawl'"
--seed=42
--weights="'1'"
"
#--data_path=/gpfs/suneja/datasets/bp7_llama2/lang=en
#--datasets="'dataset=arxiv'"


DO_BACKGROUND=1

if [ $DO_BACKGROUND -eq 1 ]
then
    FOUT=nohup-`date +%s`.out
    echo $FOUT

    nohup torchrun \
        --nproc_per_node=8 \
        speculator/train_speculator_tp.py \
        ${MODEL_ARGS_LLAMA3_70B}\
        >$FOUT &
else
    torchrun \
        --nproc_per_node=8 \
        speculator/train_speculator.py \
        ${MODEL_ARGS_LLAMA3_8B}
fi

