import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


class SalsaReference:
    def __init__(self, filepath, device):
        self.filepath = filepath
        self.data = torch.from_numpy(self.load_data()).to(device) 

    def load_data(self):
        data = np.load(self.filepath, allow_pickle=True)['full_ordered']
        data[:, 6:10] = self._make_orientation_relative(data[:, 6:10])
        return data

    def _make_orientation_relative(self, quats):
        # TODO: Make sure we're xyzw not wxyz
        rots = R.from_quat(quats)
        r0_inv = rots[0].inv()
        rots_inv = r0_inv * rots
        return rots_inv.as_quat()


class PointFootSFSalsaReferences:
    def __init__(self, files, device):
        self.references = []
        self.ref_lens = []
        max_len = 0
        for file in files:
            ref = SalsaReference(file, device)
            self.references.append(ref.data)
            l = ref.data.shape[0]
            self.ref_lens.append(l)
            if l > max_len:
                max_len = l
        self.references = torch.stack([
            torch.nn.functional.pad(ref, (0, 0, 0, max_len - l))
            for l, ref in zip(self.ref_lens, self.references)
        ], dim=0).to(device)
        self.ref_lens = torch.tensor(self.ref_lens, device=device, requires_grad=False)

    def get_reference_count(self):
        return len(self.references)

    def get_references(self, indices, phases):
        scaled = phases.clamp(0, 1) * self.ref_lens-1
        idx0 = torch.floor(scaled).long()
        idx1 = torch.ceil(scaled).long()
        w = (scaled - idx0).unsqueeze(-1)
        ref0 = self.references[indices, idx0, :]
        ref1 = self.references[indices, idx1, :]
        return (1-w) * ref0 + w * ref1
