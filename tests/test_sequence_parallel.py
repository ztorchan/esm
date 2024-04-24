import os

import torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc

import esm

# run with: torchrun --nproc_per_node=4 sequence_parallel_fold.py

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

def get_sub_sequence(sequence):
    assert(len(sequence) % world_size)
    sub_seq_length = len(sequence) // world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank + 1) * sub_seq_length
    return sequence[sub_seq_start:sub_seq_end]

def run():
    # init torch dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # init gpc
    gpc._add_local_rank(ParallelMode.SEQUENCE, local_rank)
    gpc._add_world_size(ParallelMode.SEQUENCE, world_size)
    gpc._add_group(ParallelMode.SEQUENCE, dist.GroupMember.WORLD)
    gpc._add_ranks_in_group(ParallelMode.SEQUENCE, list(range(world_size)))
    gpc.add_global_rank(ParallelMode.GLOBAL, local_rank)

    # init model
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    # infer 
    sub_sequence = get_sub_sequence(sequence)
    with torch.no_grad():
        output = model.infer_pdb(sub_sequence)

    with open("result.pdb", "w") as f:
        f.write(output)

    import biotite.structure.io as bsio
    struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    print(struct.b_factor.mean())  # this will be the pLDDT

if __name__=="__main__":
    run()
