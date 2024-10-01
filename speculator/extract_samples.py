from fms_fsdp.utils.dataset_utils import Streaming_Doc_Dataset
import json

#data_path="/gpfs/suneja/datasets/bp7_llama2"
#subdata="lang=en/dataset=github_clean"
#data_path="/gpfs/bsc_data/"
#subdata="lang=es/dataset=wikipedia"
data_path="/gpfs/suneja/datasets/bp7_gpt2tokenizer/lang=en/"
subdata="dataset=commoncrawl"
seed = 211
prompt_len=64

local_rank=0
world_size=1

print("loading dataset", data_path)
dataset = Streaming_Doc_Dataset(
    data_path,
    local_rank, #for non fsdp model
    #0, #for fsdp model
    world_size,
    -1,
    datasets=[
        subdata,
    ],
    seed=seed,
    min_length=2148,
    max_chunksize=8192,
)
dataset = iter(dataset)
data = []
in_middle = False
print("pulling data to build reusable prompt set")
#while len(data) < 2:
while len(data) < 256:
    chunk = next(dataset)
    if not in_middle:
        data.append(chunk[: prompt_len])
    if chunk[-1] == -1:
        in_middle = False
    else:
        in_middle = True


with open('Granite8b-benchmark_speculator_prompts.json','w') as f:
    json.dump(data, f)
