import os
import torch
from transformers import AutoTokenizer
from fms.models.hf.utils import to_hf_api
from fms.models import get_model, register_model
from fms_extras.models.speculator import _mlp_speculator_factory_factory, MLPSpeculator
from fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel, MLPSpeculatorConfig

torch.set_default_dtype(torch.half)

def convert_speculator(path, name, config, base_model_hf_repo):
    model = get_model("mlp_speculator",
                      name,
                      model_path=path,
                      device_type="cuda")

    print([i for i,j in model.named_parameters()])

    top_k_tokens_per_head = [4, 3, 2]
    extra_heads = config['n_predict'] - len(top_k_tokens_per_head)
    for i in range(0, extra_heads):
        top_k_tokens_per_head.append(2)
    print(top_k_tokens_per_head)    
        
    hf_model = to_hf_api(model,
                         top_k_tokens_per_head=top_k_tokens_per_head,
                         n_candidates=config['n_predict'],
                         scale_input=config['scale_input'],
                         tie_weights=config['tie_weights'])
    print([i for i,j in hf_model.named_parameters()])
    hf_model.config.torch_dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_model_hf_repo)
    path = os.path.join(os.path.dirname(path),
                        base_model_hf_repo.split('/')[-1],
                        'accelerator')
    #hf_model.save_pretrained(path)
    hf_model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)


def register_speculator(path, name, config) :
    weights = torch.load(path, map_location="cuda")["model_state"]
    path = os.path.dirname(path) + "/speculator.model_state.pth"
    torch.save(weights, path)
    register_model("mlp_speculator",
                    name,
                    _mlp_speculator_factory_factory(config))
    return path 


def get_speculator_info_granite3b():
    path = "/gpfs/suneja/checkpoints/granite-3b/checkpoints/step_21001_ckp.pth" 
    name = "llama.calico.3b.code.430m"
    config = {
        "emb_dim": 2560,
        "vocab_size": 49152,
        "n_predict": 5,
        "inner_dim": 4096,
        "tie_weights": True
    }
    base_model_hf_repo="ibm-granite/granite-3b-code-instruct"
    return path, name, config, base_model_hf_repo


def get_speculator_info_granite34b():
    path = "/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/step_21001_ckp.pth" 
    name = "gpt_bigcode.34b.code.680m"
    config = {
        "emb_dim": 6144,
        "vocab_size": 49152,
        "n_predict": 5,
        "inner_dim": 6144,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="ibm-granite/granite-34b-code-instruct"
    return path, name, config, base_model_hf_repo


def get_speculator_info_codellama34b():
    #path = "/gpfs/suneja/checkpoints/codellama-34b-st1/checkpoints/step_15001_ckp.pth" 
    path = "/gpfs/suneja/checkpoints/codellama-34b/checkpoints/step_21001_ckp.pth"
    name = "llama.34b.code.658m"
    config = {
        "emb_dim": 8192,
        "vocab_size": 32000,
        "n_predict": 5,
        "inner_dim": 8192,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="codellama/CodeLlama-34b-Instruct-hf"
    return path, name, config, base_model_hf_repo


def get_speculator_info_llama3_70b():
    path = "/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/step_11186_ckp.pth"
    name = "llama3.70b.961mm"
    config = {
        "emb_dim": 8192,
        "vocab_size": 128256,
        "n_predict": 4,
        "inner_dim": 3584,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="meta-llama/Meta-Llama-3-70B-Instruct"
    return path, name, config, base_model_hf_repo


def get_speculator_info_llama2_70b():
    path = "/gpfs/suneja/checkpoints/llama2-70b-tp-wtinitfix/checkpoints/step_18838_ckp.pth"
    name = "llama2.70b.658m"
    config = {
        "emb_dim": 8192,
        "vocab_size": 32000,
        "n_predict": 4,
        "inner_dim": 8192,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="meta-llama/Llama-2-70b-chat-hf"
    return path, name, config, base_model_hf_repo


def get_speculator_info_granite_13b():
    path = "/gpfs/suneja/checkpoints/granite-13b-chat-v2.1-cconly/checkpoints/step_17294_ckp.pth"
    name = "gpt_bigcode.13b.630m"
    config = {
        "emb_dim": 5632,
        "vocab_size": 50304,
        "n_predict": 4,
        "inner_dim": 5632,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
    return path, name, config, base_model_hf_repo


def get_speculator_info_llama3_70b_specu2_to_specu1():
    path = "/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix/checkpoints/step_14212_ckp_specu_v1.pth"
    name = "llama3.70b.2_2b"
    config = {
        "emb_dim": 8192,
        "vocab_size": 128256,
        "n_predict": 4,
        "inner_dim": 8192,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="/gpfs/llama3/hf/70b_instruction_tuned"
    return path, name, config, base_model_hf_repo


def get_speculator_info_bsc_8b_specu2_to_specu1():
    path = "/gpfs/prangan/ckpts/spanish/checkpoints/step_21001_ckp_specu_v1.pth"
    name = "bsc.8b.2b"
    config = {
        "emb_dim": 4096,
        "vocab_size": 256000,
        "n_predict": 4,
        "inner_dim": 4096,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="/gpfs/bsc_models/"
    return path, name, config, base_model_hf_repo


def get_speculator_info_llama3_405b_specu2_to_specu1():
    path = "/gpfs/suneja/checkpoints/llama3-405b-specu12k-1024/checkpoints/step_6001_ckp_specu_v1.pth" 
    name = "llama3.405b.3.5b"
    config = {
        "emb_dim": 16384,
        "vocab_size": 128256,
        "n_predict": 4,
        "inner_dim": 12288,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="/gpfs/llama3_405b_instruct"
    return path, name, config, base_model_hf_repo


def get_speculator_info_allam():
    path = "/gpfs/prangan/ckpts/arabic/checkpoints/step_21001_ckp.pth"
    name = "allama.13b.2.6b"
    config = {
        "emb_dim": 5120,
        "vocab_size": 61696,
        "n_predict": 4,
        "inner_dim": 5120,
        "scale_input": True,
        "tie_weights": False
    }
    base_model_hf_repo="/gpfs/suneja/models/dmf_models/allam-1-13b-instruct-20240607"
    return path, name, config, base_model_hf_repo


def get_speculator_info_granite_8b():
    path = "/gpfs/suneja/checkpoints/granite-8b-instruct/checkpoints/step_18782_ckp.pth"
    name = "granite.8b.436m"
    config = {
        "emb_dim": 4096,
        "vocab_size": 49155,
        "n_predict": 4,
        "inner_dim": 4096,
        "scale_input": True,
        "tie_weights": True
    }
    base_model_hf_repo="/gpfs/suneja/models/dmf_models/granite-8b-instruct-preview-4k-r240917a/"
    return path, name, config, base_model_hf_repo


path, name, config, base_model_hf_repo = get_speculator_info_granite_8b()
path = register_speculator(path, name, config)
convert_speculator(path, name, config, base_model_hf_repo)


