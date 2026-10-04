> **Note:** This repository is a copy of the original code hosted at [gitlab.inria.fr/yassis/DeepAnePose](https://gitlab.inria.fr/yassis/DeepAnePose), developed during my PhD at Inria/LORIA.

# DeepAnePose

Detection of intracranial aneurysms in 3D TOF-MRA together with their size and the orientation of their main axis,
for an immediate visualization in reformatted cut planes. This is the code and the annotations of the paper
[Aneurysm Pose Estimation with Deep Learning](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_51)
(MICCAI 2023).

<details>
<summary>Abstract</summary>

The diagnosis of unruptured intracranial aneurysms from time-of-flight Magnetic Resonance Angiography (TOF-MRA)
images is a challenging clinical problem that is extremely difficult to automate. We propose to go beyond the mere
detection of each aneurysm and also estimate its size and the orientation of its main axis for an immediate
visualization in appropriate reformatted cut planes. To address this issue, and inspired by the idea behind YOLO
architecture, a novel one-stage deep learning approach is described to simultaneously estimate the localization,
size and orientation of each aneurysm in 3D images. It combines fast and approximate annotation, data sampling and
generation to tackle the class imbalance problem, and a cosine similarity loss to optimize the orientation. We
evaluate our approach on two large datasets containing 416 patients with 317 aneurysms using a 5-fold
cross-validation scheme. Our method achieves a median localization error of 0.48 mm and a median 3D orientation
error of 12.27 degrees, demonstrating an accurate localization of aneurysms and an orientation estimation that
comply with clinical practice. Further evaluation is performed in a more classical detection setting to compare with
state-of-the-art nnDetecton[1] and nnUNet[2] methods. Competitive performance is reported with an average precision
of 76.60%, a sensitivity score of 82.93%, and 0.44 false positives per case.
</details>

Predicted (blue) and ground truth (green) markups on the public dataset [3]; each cut plane rotates around the
predicted aneurysm axis, as displayed by the [slicer-reformat-aneurysm](https://gitlab.inria.fr/yassis/slicer-reformat-aneurysm)
3D Slicer extension:

<img src="docs/examples.jpg" alt="Predicted and ground truth aneurysm markups in reformatted cut planes">

## Installation
The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10+:
```
git clone https://github.com/youssefassis/DeepAnePose.git && cd DeepAnePose
uv sync --extra cu126    # NVIDIA GPU (CUDA 12.6)
uv sync --extra cpu      # or CPU only
```
Commands run from the repository folder with `uv run`.

## Data
Each patient is a folder, either `P0001`-like or `sub-XXX/ses-YYYYMMDD` as in the public dataset:
```
Data/
  P0001/
    volume.nii.gz    TOF-MRA scan
    aneurysms.csv    aneurysms: two points (x, y, z columns, in mm) per aneurysm, see Annotation
    config.json      {"init volume": "volume.nii.gz", "pts aneurysm": "aneurysms.csv"}
    noskull.nii.gz   skull-stripped scan (written by remove_skull.py)
    points.csv       negative patch centers (written by extract_points.py)
```
Patients without aneurysm have no `aneurysms.csv`. A split file lists the training, validation and testing patient
folders, as in `reproducibility/fold1.json`.

## Usage
```
uv run python scripts/remove_skull.py Data/P*                                          # 1. skull stripping
uv run python scripts/extract_points.py Data/P*                                        # 2. negative patch centers
uv run python scripts/prepare.py Data Work reproducibility/fold1.json --name Train001   # 3. training folder
uv run python scripts/train.py Work/Train001                                           # 4. training (or resume)
uv run python scripts/predict.py Work/Train001                                         # 5. prediction
uv run python scripts/evaluate.py Work/Train001/Predictions/<n>_epochs/Validation_0.01/None_detections_per_patch
```
- `prepare.py` writes `Work/Train001/ndl_config.json`; the other training settings (model, loss, patch size,
  augmentation, optimizer...) are set in the script.
- Training saves `last_checkpoint.pytorch` and TensorBoard logs in the training folder.
- Prediction runs on the validation and testing patients of the split, and writes one JSON file per patient
  (`P0001.json`, `sub-013_ses-20101220.json`). Each detection has its neck (P1) and dome (P2) points, center and
  radius in mm, and a confidence between 0 and 1:
  ```
  {"0": {"P1": [x, y, z], "P2": [x, y, z], "center": [x, y, z], "radius": 2.86, "confidence": 0.996}}
  ```
- Evaluation matches the detections, from the most to the least confident, to the aneurysm they overlap most
  (IoU ≥ 0.1), and writes a CSV table of the detections and a summary figure of the pose estimation and detection
  metrics next to the predictions:

<img src="docs/evaluation_0.1.png" alt="Example of the evaluation figure">

## Annotation
Each aneurysm is approximated by a sphere defined by two points: the center of its neck (P1) and its dome (P2). The
[slicer-sphere-annotation](https://gitlab.inria.fr/yassis/slicer-sphere-annotation) 3D Slicer extension places them
in the volume rendering view, with a real-time display of the sphere in the canonical cut planes.

<div align="center">
    <img src="docs/annotation_draw.png" alt="Two points P1 and P2 define the sphere that approximates the aneurysm" width="25%">
</div>

## Reproducing the paper
- `reproducibility/annotations`: annotations of the public dataset [3] (to download from
  [OpenNeuro ds003949](https://openneuro.org/datasets/ds003949/)), one JSON file per subject. Write them as the
  `aneurysms.csv` of each patient folder with:
  ```
  uv run python scripts/convert_annotations.py reproducibility/annotations Data
  ```
- `reproducibility/fold1.json` to `fold5.json`: the 5-fold cross-validation splits. Their patient folders are
  relative to the data folder given to `prepare.py`.

## Citation
```
@inproceedings{assis2023aneurysm,
  title={Aneurysm Pose Estimation with Deep Learning},
  author={Assis, Youssef and Liao, Liang and Pierre, Fabien and Anxionnat, Ren{\'e} and Kerrien, Erwan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={543--553},
  year={2023},
  organization={Springer}
}

@inproceedings{assis2021efficient,
  title={An efficient data strategy for the detection of brain aneurysms from MRA with deep learning},
  author={Assis, Youssef and Liao, Liang and Pierre, Fabien and Anxionnat, Ren{\'e} and Kerrien, Erwan},
  booktitle={Deep Generative Models, and Data Augmentation, Labelling, and Imperfections: First Workshop, DGM4MICCAI 2021, and First Workshop, DALI 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, October 1, 2021, Proceedings 1},
  pages={226--234},
  year={2021},
  organization={Springer}
}
```

## References
[1] Baumgartner et al., nnDetection: A self-configuring method for medical object detection. MICCAI, pp. 530–539
(2021).

[2] Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature
Methods 18(2), pp. 203–211 (2021).

[3] Di Noto et al., Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical
knowledge. Neuroinformatics, pp. 1–14 (2022).

## Acknowledgements
This work was funded by Region Grand-Est and CHRU University Hospital of Nancy, France.
