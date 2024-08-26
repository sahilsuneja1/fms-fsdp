import os
import torch
from collections import OrderedDict

#v2_ckpt_filepath = "/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix/checkpoints/step_14212_ckp.pth"
v2_ckpt_filepath = "/gpfs/prangan/ckpts/spanish/checkpoints/step_21001_ckp.pth"
v2 = torch.load(v2_ckpt_filepath)
d2 = v2['model_state']
nheads = 0
for k in v2['model_state'].keys():
    h = int(k.split('.')[1])
    if h > nheads:
        nheads = h
nheads += 1

new_d = OrderedDict()

def translate(new_d, name, do_bias=False):
    for i in range(nheads):
        i_ = str(i)
        pref = 'heads.'+i_+'.'
        new_d[name+'.'+i_+'.weight'] = d2[pref+name+'.weight']
        if do_bias:
            new_d[name+'.'+i_+'.bias'] = d2[pref+name+'.bias']
    return new_d

new_d = translate(new_d, 'emb')
new_d = translate(new_d, 'proj')
new_d = translate(new_d, 'head')
new_d = translate(new_d, 'ln', True)

new_v = {}
new_v['tokens_seen'] = v2['tokens_seen']
new_v['step'] = v2['step']
new_v['model_state'] = new_d

v2_ckpt_filename, suffix = v2_ckpt_filepath.split('/')[-1].split('.') 
v1_ckpt_filepath = os.path.dirname(v2_ckpt_filepath) + '/' + v2_ckpt_filename + '_specu_v1.' + suffix

torch.save(new_v, v1_ckpt_filepath)
