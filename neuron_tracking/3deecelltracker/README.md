# 3DeeCelltracker
## Commands Overview
The commands used are also specified within the `justfile` for convenience.

### Training
```bash
python inference.py --export_training # loads dataloader.py and exports into 3DeeCellTracker's format
CUDA_VISIBLE_DEVICES=<GPU_ID> python inference.py --train --fold <FOLD_NUMBER>
```

### Inference (Stardist segmentation backbone)
```bash
python inference.py --export_inference
python inference.py --inference
```

You can also spawn multiple processes on multiple GPUs for parallelism:
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python inference.py --parallel_inference
```

### Tracking (FFN linking)
```bash
python inference.py --tracking
```

Similar to inference, you can also run parallel tracking on multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python inference.py --parallel_tracking
```

### Evaluation
```bash
python inference.py --evaluate # exports performance metrics
python inference.py --metrics # pretty prints results
```

## Additional Options
- `--use_pretrained`: Use a pretrained model for inference instead of finetuned models.
- `--prob_thresh`: Specify the probability threshold for Stardist's detections (default is 0.4).
- `--radius`: Set the radius for neuron nuclei (default is 2.0).
