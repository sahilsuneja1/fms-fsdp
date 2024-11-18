from fms_fsdp.utils.dataset_utils import Streaming_Doc_Dataset
import json

#data_path="/gpfs/suneja/datasets/bp7_llama2"
#subdata="lang=en/dataset=github_clean"
#data_path="/gpfs/bsc_data/"
#subdata="lang=es/dataset=wikipedia"
#data_path="/gpfs/suneja/datasets/bp7_gpt2tokenizer/lang=en/"
#subdata="dataset=commoncrawl"
#data_path="/gpfs/suneja/datasets/bluepile-granite/lang=en/"
#subdata="dataset=github_clean"
#data_path="/gpfs/bsc_data/new_version"
#subdata="lang=es/dataset=commoncrawl"
#subdata="lang=en/dataset=wikimedia"
#subdata="lang=en/dataset=textbook"
#subdata="lang=es/commoncrawl_test"
data_path="/gpfs/suneja/datasets/bluepile-granite/"
subdata="lang=en/dataset=github_clean"
seed = 211
prompt_len=120000

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
    min_length=120000+8192+256,
    max_chunksize=10_000_000,    
    #min_length=2148,
    #max_chunksize=8192,
)
dataset = iter(dataset)
data = []
in_middle = False
print("pulling data to build reusable prompt set")
#while len(data) < 2:
while len(data) < 256:
    chunk = next(dataset)
    if not in_middle:
        #data.append(chunk[: prompt_len])
        #data.append(chunk[2048: 2048+prompt_len])
        data.append(chunk[8192: 8192+prompt_len])
        #print(len(chunk))
    if chunk[-1] == -1:
        in_middle = False
    else:
        in_middle = True


with open('Granite-8b-code-speculator_prompts_mlen128448_cseek8192_plen120000.json','w') as f:
    json.dump(data, f)
