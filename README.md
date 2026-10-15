# Lightweight YOLO Detectors for Crop Detection in Weed-Prone Fields

Benchmark of seven lightweight YOLO detectors (**YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, YOLO12n, YOLO26n**) for dense crop detection in field imagery, compared on accuracy, CPU/GPU latency, model size, convergence and robustness to imaging perturbations. This repository contains the experiment notebook, raw results, figures and the accompanying paper.

> **Status:** single-seed study. Differences below ~0.01 mAP should be treated as inconclusive (see [Limitations](#limitations)).

## Repository structure

```
.
├── complete_notebook.ipynb      # Full experiment pipeline (training, evaluation, latency, robustness, ranking)
├── results/                     # Tables and artefacts exported by the notebook
│   ├── table_main_results.csv / .tex
│   ├── table_ranking.csv
│   ├── table_robustness.csv, robustness_long.csv
│   ├── table_per_class_AP50.csv, table_per_class_AP50_95.csv
│   ├── accuracy_per_run.csv, latency_per_run.csv, per_class_per_run.csv
│   ├── master_mean.csv, master_std.csv
│   ├── dataset_summary.csv, dataset_class_distribution.csv, dataset_object_sizes.csv
│   ├── experiment_config.json   # hyper-parameters, hardware, library versions
│   ├── summary.txt
│   └── BEST_YOLOv9t.pt          # weights of the top-ranked model
├── figs/                        # 300-dpi figures (accuracy, Pareto, curves, robustness, qualitative, ...)
└── paper/                       # LNCS-style paper (paper.tex, paper.pdf, figures)
```

## Dataset

[AgriAISeg](https://www.kaggle.com/datasets/ismailismailtijjani/agriaiseg) (Kaggle), cabbage-field subset, 640×640 images, single object class (the crop plant). Weeds are **not** annotated, so the task is crop localisation; weeds can be inferred as non-crop vegetation in a downstream pipeline.

| Split | Images | Plants | Plants / image |
|---|---:|---:|---:|
| Train | 900 | 22,048 | 24.5 |
| Validation | 179 | 4,131 | 23.1 |
| Test | 119 | 2,719 | 22.8 |

The dataset is not redistributed here; the notebook downloads it with `kagglehub`. Three test images are ignored by Ultralytics because their label files mix box and polygon rows, so test metrics cover 116 images / 2,603 plants.

## Experimental protocol

All models are COCO-pretrained, fine-tuned with the Ultralytics pipeline (v8.4.177) under identical settings: 640 px, 100 epochs, batch 16, SGD (lr 0.01, momentum 0.937, weight decay 5e-4), 3 warm-up epochs, mosaic (off for last 10 epochs), HSV/flip/scale/translate augmentation, seed 0, no early stopping. Hardware: Tesla T4 (training, GPU latency); 4-thread CPU (CPU latency). Full details are in `results/experiment_config.json`.

Latency is end-to-end (pre-process + inference + post-process), batch size 1, 100 test images after 20 warm-up runs. Robustness is evaluated on the test set under five test-time perturbations (dark, bright, low contrast, blur σ=4, noise σ=25) without re-training.

## Main results (test set)

| Model | Params (M) | P | R | mAP@0.5 | mAP@0.5:0.95 | CPU ms | GPU ms | Robustness retained (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 2.64 | 0.918 | 0.899 | 0.944 | 0.825 | 75.6 | 8.1 | 91.5 |
| YOLOv8n | 3.14 | 0.912 | 0.914 | 0.949 | 0.831 | 78.9 | 7.4 | 90.9 |
| **YOLOv9t** | **2.11** | 0.926 | 0.902 | **0.953** | **0.839** | 112.2 | 17.2 | 89.9 |
| YOLOv10n | 2.74 | 0.928 | 0.887 | 0.947 | 0.829 | 77.3 | 8.6 | 91.6 |
| YOLO11n | 2.61 | 0.932 | 0.885 | 0.946 | 0.830 | 74.9 | 9.5 | 90.7 |
| YOLO12n | 2.58 | 0.909 | 0.900 | 0.950 | 0.832 | 97.2 | 13.5 | 91.7 |
| YOLO26n | 2.53 | 0.923 | 0.894 | 0.950 | 0.827 | 74.0 | 11.2 | 88.1 |

**Take-aways**
- Accuracy is tightly clustered (0.825–0.839 mAP@0.5:0.95); efficiency separates the models more.
- YOLOv9t is the most accurate and smallest but the slowest (≈52 % higher CPU latency than YOLO26n): parameter count does not predict latency.
- Over-exposure, blur and noise are the most damaging perturbations; YOLO26n is the least robust.
- Under a multi-criteria ranking (accuracy, CPU speed, parameters), YOLOv9t wins with accuracy-heavy weights and YOLO11n wins with edge-heavy weights.

## Reproducing

1. Open `complete_notebook.ipynb` in Kaggle or Colab with a GPU runtime.
2. Run top to bottom (installs `ultralytics`, downloads the dataset, trains, evaluates, benchmarks, exports tables/figures).
3. Set `SEEDS = [0, 1, 2]` in the configuration cell to obtain mean ± std over seeds (recommended).

```bash
pip install -U ultralytics kagglehub pandas matplotlib seaborn opencv-python-headless pyyaml tqdm
```

Note: later cells load trained checkpoints and training-curve CSVs from Kaggle input paths (`/kaggle/input/...`). Edit these paths to your own locations when running elsewhere.

To use the released weights:

```python
from ultralytics import YOLO
model = YOLO("results/BEST_YOLOv9t.pt")
model.predict("image.jpg", imgsz=640, conf=0.25)
```

## Paper

`paper/paper.pdf` (source: `paper/paper.tex`) uses the Springer LNCS format. `paper/llncs.cls` is a stand-in; replace it with the official class from Springer's LNCS author kit before submission. Build with `latexmk -pdf paper.tex`.

## Limitations

- **Single seed, small test set** (116 evaluated images): small gaps between models are not statistically supported.
- **Single crop, single class**; no weed annotations.
- **Class label:** the notebook's default class-name list names the class "Tomato", but the imagery shows cabbage. Box labels in some notebook figures therefore read "Tomato"; the underlying class is the crop plant.
- **GFLOPs** were not recorded correctly (exported as 0.0) and **training time** is NaN in the exported tables; neither is used in the paper. The notebook's ranking script includes a constant GFLOPs term, which does not change the order.
- Perturbations are simulated photometric transforms; latency was measured on a cloud CPU and a T4 GPU via the Python interface, not on embedded hardware.

## Citation

```bibtex
@misc{yolo_weed_benchmark,
  title  = {Benchmarking Lightweight YOLO Detectors for Crop Detection in Weed-Prone Fields: Accuracy, Edge Efficiency and Field Robustness},
  author = {Author Name},
  year   = {2026},
  note   = {Code and results: https://github.com/<user>/<repo>}
}
```

## License

Add a license of your choice (e.g. MIT for code). Ultralytics models and weights are distributed under AGPL-3.0; check its terms before redistributing trained weights or deploying commercially. The AgriAISeg dataset is subject to its own Kaggle license.

## Acknowledgements

[Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementations and the AgriAISeg dataset authors.
