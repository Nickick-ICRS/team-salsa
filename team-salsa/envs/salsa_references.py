import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


class SalsaReference:
    def __init__(self, filepath, device):
        self.filepath = filepath
        if 'foot' in filepath:
            self.type = 'foot'
        elif 'joint' in filepath:
            self.type = 'joint'
        self.data = torch.from_numpy(self.load_data_foot(self.type), dtype=torch.float32, device=device) 

    def load_data(self, type):
        try:
            data = np.load(self.filepath, allow_pickle=True).item()
            if type == 'foot':
                assert 'foot_pos_l' in data.keys() and 'foot_pos_r' in data.keys(), "Foot positions not found in data."
                assert 'chassis_orient' in data.keys(), "Chassis orientation not found in data."
                assert 'chassis_acc' in data.keys(), "Chassis acceleration not found in data."
                data['chassis_orient'][:] = self._make_orientation_relative(data['chassis_orient'])
                return data['foot_references']
            elif type == 'joint':
                assert 'joint_pos' in data.keys(), "Joint references not found in data."
                assert 'chassis_orient' in data.keys(), "Chassis orientation not found in data."
                assert 'chassis_acc' in data.keys(), "Chassis acceleration not found in data."
                data['chassis_orient'][:] = self._make_orientation_relative(data['chassis_orient'])
                return data['joint_references']
            else:
                raise ValueError("Unknown type. Must be 'foot' or 'joint'.")
        except Exception as e:
            raise ValueError(f"Error loading data from {self.filepath}: {e}")

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
