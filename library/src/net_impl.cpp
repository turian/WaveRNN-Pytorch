/* Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <vector>
#include "wavernn.h"
#include "net_impl.h"


void ResBlock::loadNext(FILE *fd)
{
    resblock.resize( RES_BLOCKS*4 );
    for(int i=0; i<RES_BLOCKS*4; ++i){
        resblock[i].loadNext(fd);
    }
}

Matrixf ResBlock::apply(const Matrixf &x)
{
    Matrixf y = x;

    for(int i=0; i<RES_BLOCKS; ++i){
        Matrixf residual = y;

        y = resblock[4*i](y);    //conv1
        y = resblock[4*i+1](y);  //batch_norm1
        y = relu(y);
        y = resblock[4*i+2](y);  //conv2
        y = resblock[4*i+3](y);  //batch_norm2

        y += residual;
    }
    return y;
}

void Resnet::loadNext(FILE *fd)
{
    conv_in.loadNext(fd);
    batch_norm.loadNext(fd);
    resblock.loadNext(fd); //load the full stack
    conv_out.loadNext(fd);
    stretch2d.loadNext(fd);
}

Matrixf Resnet::apply(const Matrixf &x)
{
    Matrixf y = x;
    y=conv_in(y);
    y=batch_norm(y);
    y=relu(y);
    y=resblock.apply(y);
    y=conv_out(y);
    y=stretch2d(y);
    return y;
}

void UpsampleNetwork::loadNext(FILE *fd)
{
    up_layers.resize( UPSAMPLE_LAYERS*2 );
    for(int i=0; i<up_layers.size(); ++i){
        up_layers[i].loadNext(fd);
    }
}

Matrixf UpsampleNetwork::apply(const Matrixf &x)
{
    Matrixf y = x;
    for(int i=0; i<up_layers.size(); ++i){
        y = up_layers[i].apply( y );
    }
    return y;
}

void Model::loadNext(FILE *fd)
{
    fread( &header, sizeof( Model::Header ), 1, fd);

    resnet.loadNext(fd);
    upsample.loadNext(fd);

    I.loadNext(fd);
    rnn1.loadNext(fd);
    fc1.loadNext(fd);
    fc2.loadNext(fd);
}


Matrixf pad( const Matrixf& x, int nPad )
{
    Matrixf y = Matrixf::Zero(x.rows(), x.cols()+2*nPad);
    y.block(0, nPad, x.rows(), x.cols() ) = x;
    return y;
}

Vectorf Model::apply(const Matrixf &x)
{
    Vectorf y;

    std::vector<int> rnn_shape = rnn1.shape();

    Vectorf h1(rnn_shape[0]);

    Matrixf mel_padded = pad(x, header.nPad);

    Matrixf mels = upsample.apply(mel_padded);
    Matrixf aux = resnet.apply(mel_padded);

    assert(mels.cols() == aux.cols());
    int seq_len = mels.rows();
    int n_aux = aux.rows();

    for(int i=0; i<seq_len; ++i){
        assert(0);
    }

    return y;
}


