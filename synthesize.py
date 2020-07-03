"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import librosa
import glob

from docopt import docopt
from model import *
from myhparams import hparams as hp
from utils import num_params_count
import pickle
import time
import numpy as np
import scipy as sp



if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")
    print("using device:{}".format(device))

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    mel_file_name = args['<mel_input.npy>']
    mel = np.load(mel_file_name)
    if mel.shape[0] > mel.shape[1]: #ugly hack for transposed mels
        mel = mel.T

    if checkpoint_path is None:
        flist = glob.glob(f'{checkpoint_dir}/checkpoint_*.pth')
        latest_checkpoint = max(flist, key=os.path.getctime)
    else:
        latest_checkpoint = checkpoint_path
    print('Loading: %s'%latest_checkpoint)
    # build model, create optimizer
    model = build_model().to(device)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    print("I: %.3f million"%(num_params_count(model.I)))
    print("Upsample: %.3f million"%(num_params_count(model.upsample)))
    print("rnn1: %.3f million"%(num_params_count(model.rnn1)))
    #print("rnn2: %.3f million"%(num_params_count(model.rnn2)))
    print("fc1: %.3f million"%(num_params_count(model.fc1)))
    #print("fc2: %.3f million"%(num_params_count(model.fc2)))
    print("fc3: %.3f million"%(num_params_count(model.fc3)))


    #onnx export
    model.train(False)
    #wav = np.load('WaveRNN-Pytorch/checkpoint/test_0_wav.npy')

    #doesn't work torch.onnx.export(model, (torch.tensor(wav),torch.tensor(mel)), checkpoint_dir+'/wavernn.onnx', verbose=True, input_names=['mel_input'], output_names=['wav_output'])


    #mel = np.pad(mel,(24000,0),'constant')
    # n_mels = mel.shape[1]
    # n_mels = hparams.batch_size_gen * (n_mels // hparams.batch_size_gen)
    # mel = mel[:, 0:n_mels]


    mel0 = mel.copy()
    mel0=np.hstack([np.ones([80,40])*(-4), mel0, np.ones([80,40])*(-4)])
    start = time.time()
    output0 = model.generate(mel0, batched=False, target=2000, overlap=64)
    total_time = time.time() - start
    frag_time = len(output0) / hparams.sample_rate
    print("Generation time: {}. Sound time: {}, ratio: {}".format(total_time, frag_time, frag_time/total_time))

    librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'_orig.wav'), output0, hparams.sample_rate)

    #mel = mel.reshape([mel.shape[0], hparams.batch_size_gen, -1]).swapaxes(0,1)
    #output, out1 = model.batch_generate(mel)
    #bootstrap_len = hp.hop_size * hp.resnet_pad
    #output=output[:,bootstrap_len:].reshape(-1)
    # librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'.wav'), output, hparams.sample_rate)
    with open(os.path.join(output_path, os.path.basename(mel_file_name)+'.pkl'), 'wb') as f:
        pickle.dump((output0,), f)
    print('done')
