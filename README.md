# ALGAVISION

A PyQt5 / Tkinter desktop suite for plankton image processing, automated identification, and active-learning-based model training. Developed at Xiamen University.

---

## Features

| Tool | Entry point | Description |
|---|---|---|
| **SORTA** | `run_sorta.py` | CNN-based plankton identification, morphometry, Excel export |
| **SEGMAN** | `run_segman.py` | Segment individual organisms from FlowCAM / microscope images |
| **TAGOR** | `run_tagor.py` | Manual image tagging and class-folder organization |
| **MODLER v2** | `run_modler.py` | Model training with active learning, unsupervised annotation, style transfer |
| **CutViewer** | `run_cutter.py` | Interactive parameter tuning for microscope image segmentation |

### MODLER v2 Highlights

- **14 SOTA backbones** via unified registry: EfficientNetV2, ConvNeXtV2, ViT, Swin, MaxViT, DINOv2
- **Unsupervised annotation** (P2): DINOv2 embeddings → HDBSCAN clustering → human cluster-naming GUI → labeled dataset
- **Active learning** (P3): entropy / margin / BALD / core-set sampling + energy-score OOD detection
- **Style transfer augmentation**: FDA / histogram matching / AdaIN to bridge domain gap between reference photos and FlowCAM/microscope field images
- **Mixed-precision training** (AMP) + gradient clipping
- **Workflow-driven 5-tab UI**: Data Setup → Auto-Annotation → Training → Batch Predict → Human Review

---

## Installation

1. Install PyTorch (GPU recommended):
   ```bash
   # Windows CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

```bash
# Plankton identification
python run_sorta.py

# Model training & active learning
python run_modler.py

# Image segmentation
python run_segman.py
```

---

## Project Structure

```
ALGAVISION/
├── src/                          # All source code
│   ├── sorta.py                  # Identification tool
│   ├── segman.py                 # Segmentation tool
│   ├── tagor.py                  # Tagging tool
│   ├── modler.py                 # Training tool v1
│   ├── modler_v2.py              # Training tool v2 (active learning)
│   ├── microscope_cut_viewer.py  # Segmentation parameter viewer
│   ├── child.py                  # Shared: image viewer, dataset split dialog
│   ├── imagecut.py               # Shared: image cutting algorithms
│   ├── morphometer.py            # Shared: morphological measurements
│   ├── Make_Datasets_Train_Val_Test.py
│   ├── models_registry.py        # SOTA model registry (14 architectures)
│   ├── unsupervised_annotator.py # Unsupervised clustering annotation
│   ├── cluster_review_gui.py     # Cluster naming & export GUI
│   ├── self_learning.py          # Active learning strategies + OOD detection
│   ├── style_aug.py              # Style transfer domain adaptation
│   ├── Image_Enhancement.py
│   └── Microscope_COI_Cut.py
├── resources/
│   ├── icons/                    # Application icons and logos
│   └── diagrams/                 # Active learning workflow diagram (.drawio)
├── Models/                       # Model configs (classes_names.txt, setting.log)
├── run_sorta.py                  # Launcher
├── run_segman.py
├── run_tagor.py
├── run_modler.py
├── run_cutter.py
├── requirements.txt
└── .gitignore
```

---

## Models

Pre-trained model weights are not included in this repository due to file size.  
Each model directory under `Models/` contains:
- `classes_names.txt` — taxon index mapping
- `setting.log` — model type, input size, and hardware settings

---

## Supported Model Architectures (MODLER v2)

| Key | Architecture | Params | Notes |
|---|---|---|---|
| B0–M | EfficientNetV2 (B0/B1/B2/B3/S/M) | 7M–54M | Production-ready |
| ConvNeXtV2-T/B | ConvNeXt V2 | 28M/89M | Strong long-tail performance |
| ViT-B/16 | Vision Transformer | 86M | Global attention |
| Swin-T/B | Swin Transformer | 28M/88M | Hierarchical attention |
| MaxViT-T | MaxViT | 31M | Local + global hybrid |
| DINOv2-S/B | DINOv2 (self-supervised) | 22M/86M | Best for unsupervised embedding |

---

## Citation

If you use ALGAVISION in your research, please cite:

```
@software{algavision,
  author  = {Brig Chen},
  title   = {ALGAVISION: Plankton Image Analysis Suite with Active Learning},
  year    = {2025},
  url     = {https://github.com/Brigchen/algavision}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

© Xiamen University
