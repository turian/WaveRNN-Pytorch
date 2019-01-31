/*
Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include "wavernn.h"


TorchLayer *TorchLayer::loadNext(FILE *fd)
{
    TorchLayer::Header header;
    fread(&header, sizeof(TorchLayer::Header), 1, fd);

    printf("Loading: %s\n", header.name);

    switch( header.layerType ){

    case TorchLayer::Header::LayerType::Linear:
    {
        LinearLayer* linear = reinterpret_cast<LinearLayer*>(this);
        linear->loadNext(fd);
        return linear;
    }
    break;

    case TorchLayer::Header::LayerType::GRU:
    {
        GRULayer* gru = reinterpret_cast<GRULayer*>(this);
        gru->loadNext(fd);
        return gru;
    }
    break;

    case TorchLayer::Header::LayerType::Conv1d:
    case TorchLayer::Header::LayerType::Conv2d:
    default:
        return nullptr;
    }
}

LinearLayer* LinearLayer::loadNext(FILE *fd)
{
    LinearLayer::Header header;
    fread( &header, sizeof(LinearLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);

    weight.resize(header.nRows, header.nCols);
    bias.resize(header.nRows);

    fread(weight.data(), header.elSize, header.nRows*header.nCols, fd);
    fread(bias.data(), header.elSize, header.nRows, fd);
}


GRULayer* GRULayer::loadNext(FILE *fd)
{
    GRULayer::Header header;
    fread( &header, sizeof(GRULayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);

    W_ir.resize(header.nRows, header.nCols);
    W_iz.resize(header.nRows, header.nCols);
    W_in.resize(header.nRows, header.nCols);

    W_hr.resize(header.nRows, header.nRows);
    W_hz.resize(header.nRows, header.nRows);
    W_hn.resize(header.nRows, header.nRows);;

    b_ir.resize(header.nRows);
    b_iz.resize(header.nRows);
    b_in.resize(header.nRows);

    b_hr.resize(header.nRows);
    b_hz.resize(header.nRows);
    b_hn.resize(header.nRows);


    fread(W_ir.data(), header.elSize, header.nRows*header.nCols, fd);
    fread(W_iz.data(), header.elSize, header.nRows*header.nCols, fd);
    fread(W_in.data(), header.elSize, header.nRows*header.nCols, fd);

    fread(W_hr.data(), header.elSize, header.nRows*header.nRows, fd);
    fread(W_hz.data(), header.elSize, header.nRows*header.nRows, fd);
    fread(W_hn.data(), header.elSize, header.nRows*header.nRows, fd);

    fread(b_ir.data(), header.elSize, header.nRows, fd);
    fread(b_iz.data(), header.elSize, header.nRows, fd);
    fread(b_in.data(), header.elSize, header.nRows, fd);

    fread(b_hr.data(), header.elSize, header.nRows, fd);
    fread(b_hz.data(), header.elSize, header.nRows, fd);
    fread(b_hn.data(), header.elSize, header.nRows, fd);
}


