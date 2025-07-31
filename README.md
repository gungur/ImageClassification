# MiniPlaces Image Classification with LeNet

This repository contains a PyTorch implementation for image classification on the MiniPlaces dataset using a LeNet architecture.

## Overview

The project includes:
- A custom data loader for the MiniPlaces dataset
- LeNet model implementation
- Training and evaluation scripts
- Checkpoint saving functionality

## Dataset

The MiniPlaces dataset contains 100,000 images across 100 scene categories. The dataset is automatically downloaded and processed by the provided data loader.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/miniplaces-classification.git
   cd miniplaces-classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:
```bash
python train_miniplaces.py --epochs 10 --lr 0.001 --batch-size 32
```

Arguments:
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--resume`: Path to checkpoint to resume training (optional)

### Evaluation

To evaluate a trained model:
```bash
python eval_miniplaces.py --load path/to/checkpoint.pth.tar
```

### Model Implementation

The LeNet model is implemented in `student_code.py`. Key components include:
- `LeNet` class with customizable input shape and number of classes
- `train_model` function for training
- `test_model` function for evaluation

## File Structure

- `dataloader.py`: Handles dataset downloading, loading, and preprocessing
- `train_miniplaces.py`: Main training script
- `eval_miniplaces.py`: Model evaluation script
- `student_code.py`: Contains the LeNet model implementation and helper functions

## Results

The model achieves competitive accuracy on the MiniPlaces validation set. Performance can be improved by:
- Increasing model capacity
- Adjusting hyperparameters
- Using more advanced data augmentation

## Acknowledgments

- MiniPlaces dataset provided by MIT CSAIL
- PyTorch community for excellent documentation and examples
