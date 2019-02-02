"""Convert trained model for libwavernn

usage: convert_model.py [options] <checkpoint.pth>

options:
    --output-dir=<dir>           Output Directory [default: model_outputs]
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
from model import *

import struct
import numpy as np
import scipy as sp

elSize = 4 #change to 2 for fp16



def compress(W):
    N = W.shape[1]
    W_nz = W.copy()
    W_nz[W_nz!=0]=1
    L = W_nz.reshape([-1, N // hp.sparse_group, hp.sparse_group])
    S = L.max(axis=-1)
    #convert to compressed index
    #compressed representation has position in each row. "255" denotes row end.
    (row,col)=np.nonzero(S)
    idx=[]
    for i in range(S.shape[0]+1):
        idx += list(col[row==i])
        idx += [255]
    mask = np.repeat(S, hp.sparse_group, axis=1)
    idx = np.asarray(idx, dtype='uint8')
    return (W[mask!=0], idx)

def writeCompressed(f, W):
    weights, idx = compress(W)
    f.write(struct.pack('@i',weights.size))
    f.write(weights.tobytes(order='C'))
    f.write(struct.pack('@i',idx.size))
    f.write(idx.tobytes(order='C'))
    return


def linear_saver(f, layer):
    weight = layer.weight.cpu().detach().numpy()

    bias = layer.bias.cpu().detach().numpy()
    nrows, ncols = weight.shape
    v = struct.pack('@bii', elSize, nrows, ncols)
    f.write(v)
    writeCompressed(f, weight)
    f.write(bias.tobytes(order='C'))

def conv1d_saver(f, layer):
    pass

def conv2d_saver(f, layer):
    pass

def batchnorm1d_saver(f, layer):
    pass

def gru_saver(f, layer):
    weight_ih_l0 = layer.weight_ih_l0.detach().cpu().numpy()
    weight_hh_l0 = layer.weight_hh_l0.detach().cpu().numpy()
    bias_ih_l0 = layer.bias_ih_l0.detach().cpu().numpy()
    bias_hh_l0 = layer.bias_hh_l0.detach().cpu().numpy()

    W_ir,W_iz,W_in=np.vsplit(weight_ih_l0, 3)
    W_hr,W_hz,W_hn=np.vsplit(weight_hh_l0, 3)

    b_ir,b_iz,b_in=np.split(bias_ih_l0, 3)
    b_hr,b_hz,b_hn=np.split(bias_hh_l0, 3)

    hidden_size, input_size = W_ir.shape
    v = struct.pack('@bii', elSize, hidden_size, input_size)
    f.write(v)
    writeCompressed(f, W_ir)
    writeCompressed(f, W_iz)
    writeCompressed(f, W_in)
    writeCompressed(f, W_hr)
    writeCompressed(f, W_hz)
    writeCompressed(f, W_hn)
    f.write(b_ir.tobytes(order='C'))
    f.write(b_iz.tobytes(order='C'))
    f.write(b_in.tobytes(order='C'))
    f.write(b_hr.tobytes(order='C'))
    f.write(b_hz.tobytes(order='C'))
    f.write(b_hn.tobytes(order='C'))
    return

savers = { 'Conv1d':conv1d_saver, 'Conv2d':conv2d_saver, 'BatchNorm1d':batchnorm1d_saver, 'Linear':linear_saver, 'GRU':gru_saver }
layer_enum = { 'Conv1d':1, 'Conv2d':2, 'BatchNorm1d':3, 'Linear':4, 'GRU':5 }

def save_layer(f, layer):
    layer_type_name = layer._get_name()
    v = struct.pack('@b64s', layer_enum[layer_type_name], layer.__str__().encode() )
    f.write(v)
    savers[layer_type_name](f, layer)



if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    output_path = args["--output-dir"]

    device = torch.device("cpu")

    checkpoint_file_name = args['<checkpoint.pth>']

    # build model, create optimizer
    model = build_model().to(device)
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    x=1.+1./np.arange(1,113)
    W=model.I.weight.cpu().detach().numpy()
    print(np.matmul(W,x))

    with open(output_path+'/model.bin','wb') as f:
        save_layer(f, model.I)
        save_layer(f, model.rnn1)

    print()
