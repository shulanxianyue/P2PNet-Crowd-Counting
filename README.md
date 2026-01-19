# P2PNet Crowd Counting

## Quickstart (end-to-end)
1) **Download the ShanghaiTech dataset (Part A + Part B)**
   - GitHub: https://github.com/desenzhou/ShanghaiTechDataset
   - Kaggle: https://www.kaggle.com/datasets/tthien/shanghaitech
   - Or multi-part archive:
     - Part 1: https://perso.esiee.fr/~chierchg/deep-learning/_downloads/8fa2b5234fa2aee0fa9aae9f3d8e62ab/ShanghaiTech.zip
     - Part 2: https://perso.esiee.fr/~chierchg/deep-learning/_downloads/86a7f671c171459c741eb6c826748ef6/ShanghaiTech.z01
     - Extract using a tool that supports multi-part archives (e.g., 7-Zip: https://www.7-zip.org/).

2) **Place the dataset in the expected structure**
   - Root folder: .data/ShanghaiTech/
   - Required layout:
     - part_A_final/train_data/images
     - part_A_final/train_data/ground_truth
     - part_A_final/test_data/images
     - part_A_final/test_data/ground_truth
     - part_B_final/train_data/images
     - part_B_final/train_data/ground_truth
     - part_B_final/test_data/images
     - part_B_final/test_data/ground_truth

3) **Export .npy annotations (recommended)**
   - This copies original images and exports filtered GT points to expd/ShanghaiTech
   - Run:
     - python export_data.py

4) **Train** (uses expd/ShanghaiTech by default)
   - Run:
     - python train.py
   - Checkpoints saved in weights/

5) **Test / Evaluate all checkpoints**
   - Run:
     - python run_test.py
   - Outputs metrics and plots in results/, and visualizations in vis_results/

6) **Demo (optional)**
   - Run with Streamlit:
     - streamlit run demo_app.py

## Repository Overview
- Backbone + P2PNet model: [backbone.py](backbone.py), [p2pnet.py](p2pnet.py)
- Dataset + online preprocessing: [dataset.py](dataset.py)
- Offline export to expd/: [export_data.py](export_data.py)
- Training (freeze backbone then fine-tune): [train.py](train.py)
- Evaluation + metrics & plots (all epochs): [run_test.py](run_test.py)
- Demo UI: [demo_app.py](demo_app.py)
- Checkpoints: weights/
- Visualization outputs: vis_results/
- Metrics & plots: results/ and results_train/

## Dataset
This repo expects the ShanghaiTech dataset organized as:

Dataset/ShanghaiTech/
	part_A_final/
		train_data/images
		train_data/ground_truth
		test_data/images
		test_data/ground_truth
	part_B_final/
		train_data/images
		train_data/ground_truth
		test_data/images
		test_data/ground_truth

## Preprocessing
Two options are available:

1) **Online preprocessing** (default in training/test):
	 - Resize shortest side to `crop_size`
	 - Random crop during training
	 - Keep original aspect ratio during test

2) **Offline export to expd/**:
	 - Run [export_data.py](export_data.py)
	 - Copies original images and saves `.npy` points (filtered to image bounds)
	 - Then set or keep `DATA_ROOT = "expd/ShanghaiTech"`

## Training
Freeze the backbone for the first `FREEZE_EPOCHS`, then unfreeze for full fine-tuning.

Key settings are in [train.py](train.py):
- `EPOCHS`, `BATCH_SIZE`, `LR`, `FREEZE_EPOCHS`

Checkpoints are saved every 10 epochs in weights/.

## Evaluation
Run evaluation across **all saved epochs** and generate plots:

python run_test.py

This produces:
- results/metrics.csv
- results/metrics.png (MAE/RMSE vs epoch)
- results/summary.txt

Optional visualizations of predictions are saved in vis_results/.

## Notes
- If a checkpoint file is 0 bytes (e.g., due to interrupted training), it will be skipped by the evaluation script.
