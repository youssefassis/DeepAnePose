import nibabel as ni
import numpy as np

import Data.IO as dio


def test_linear_normalization_rescales_to_unit_range(tmp_path):
    vol = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) + 10
    ni.save(ni.Nifti1Image(vol, np.eye(4)), tmp_path / "volume.nii.gz")

    data = dio.read_patient_data(str(tmp_path / "volume.nii.gz"), None, normalize="Linear")["data"]

    assert data.min() == 0 and data.max() == 1
