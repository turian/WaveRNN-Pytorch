# WaveRNN-Pytorch
This repository started with Fatcord's [Alternative](https://github.com/fatchord/WaveRNN) WaveRNN (Faster training), which contains a fast-training, small GPU memory implementation of WaveRNN vocoder.

This repo refactors the code and converts code to a proper pytorch script.

# Highlights
* 10 bit quantized wav modeling for higher quality
* Weight pruning for reducing model complexity
* Fast, CPU only, C++ inference library running faster than real time on modern cpu.
* Compressed pruned weight format to make weight files small

# Planned
* Real time inference on modern ARM processors (e.g. inference on smartphone for high quality TTS) 
* Python bindings for the C++ library
* Combine with a Tacotron-2 implementation for TTS.

# Audio Samples
* coming soon

# Pretrained Checkpoints
* coming soon


# Requirements
* Python 3
* CUDA >=8.0
* PyTorch >= v1.0

# Installation
Ensure above requirements are met.

```
git clone https://github.com/geneing/WaveRNN-Pytorch.git
cd WaveRNN-Pytorch
pip install -r requirements.txt
```

# Usage
## 1. Adjusting Hyperparameters
Before running scripts, one can adjust hyperparameters in `hparams.py`.

Some hyperparameters that you might want to adjust:
* `input_type` (best performing ones are currently `bits` and `raw`, see `hparams.py` for more details)
* `batch_size`
* `save_every_step` (checkpoint saving frequency)
* `evaluate_every_step` (evaluation frequency)
* `seq_len_factor` (sequence length of training audio, the longer the more GPU it takes)
## 2. Preprocessing
This function processes raw wav files into corresponding mel-spectrogram and wav files according to the audio processing hyperparameters.

Example usage:
```
python preprocess.py /path/to/my/wav/files
```
This will process all the `.wav` files in the folder `/path/to/my/wav/files` and save them in the default local directory called `data_dir`.

Can include `--output_dir` to specify a specific directory to store the processed outputs.

## 3. Training
Start training process. checkpoints are by default stored in the local directory `checkpoints`.
The script will automatically save a checkpoint when terminated by `crtl + c`.


Example 1: starting a new model for training
```
python train.py data_dir
```
`data_dir` is the directory containing the processed files.

Example 2: Restoring training from checkpoint
```
python train.py data_dir --checkpoint=checkpoints/checkpoint0010000.pth
```
Evaluation `.wav` files and plots are saved in `checkpoints/eval`.










