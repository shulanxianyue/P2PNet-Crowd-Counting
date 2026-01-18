# P2PNet Crowd Counting

Deep Learning project (E4 ESIEE Paris) implementing P2PNet for crowd counting on ShanghaiTech.

## Repository Overview
- Backbone + P2PNet model: [p2pnet.py](p2pnet.py)
- Dataset + preprocessing pipeline (online): [dataset.py](dataset.py)
- Offline preprocessing export: [export_data.py](export_data.py)
- Training (freeze backbone then fine-tune): [train.py](train.py)
- Evaluation + metrics & plots (all epochs): [run_test.py](run_test.py)
- Checkpoints: weights/
- Visualization outputs: vis_results/
- Metrics & plots: results/

## Dataset
This repo expects the ShanghaiTech dataset organized as:

Dataset/ShanghaiTech/
	part_A_final/
		train_data/images
		train_data/ground_truth
		test_data/images
		test_data/ground_truth
	part_B_final/...

## Preprocessing
Two options are available:

1) **Online preprocessing** (default in training/test):
	 - Resize shortest side to `crop_size`
	 - Random crop during training
	 - Keep original aspect ratio during test

2) **Offline preprocessing export**:
	 - Run [export_data.py](export_data.py)
	 - Outputs cropped images and `.npy` points into expd/

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

## Results
See outputs in results/ for metrics and plots.

## Notes
If a checkpoint file is 0 bytes (e.g., due to interrupted training), it will be skipped by the evaluation script.
