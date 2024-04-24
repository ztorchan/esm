import typing as T

import torch
import torch.nn as nn
from torch import nn
from torch.nn import LayerNorm

import esm
from esm.esmfold.v1.esmfold import ESMFold
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)

class DataParallelESMFold():
    def __init__(self, model: ESMFold, device_ids=None):
        self._device = model.device
        self._device_ids = device_ids
        self._model = torch.nn.DataParallel(model, device_ids).eval()

    def infer(
        self,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
    ):
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        output = self._model(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        # postprocess
        def split_and_concat(tensor, split_size):
            tensors = torch.split(tensor, tensor.shape[0] // split_size, dim=0)
            result = torch.cat(tensors, dim=1)
            return result
        need_split_and_concat = ["frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "states", "lddt_head"]
        for key in need_split_and_concat:
            if key in output:
                output[key] = split_and_concat(output[key], len(self._device_ids))

        return output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    @property
    def device(self):
        return self._device
