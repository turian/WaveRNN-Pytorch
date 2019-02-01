#ifndef WAVERNN_H
#define WAVERNN_H

#include <stdio.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> Matrixf;
typedef Tensor<float, 3, RowMajor> Tensor3df;
typedef Tensor<float, 4, RowMajor> Tensor4df;
typedef VectorXf Vectorf;
typedef Matrix<int8_t, Dynamic, 1> Vectori8;


class CompMatrix{
    Vectorf weight;
    Vectori8 index;
public:

    void read(FILE* fd, int elSize){
        int nWeights, nIndex;

        fread(&nWeights, sizeof(int), 1, fd);

        weight.resize(nWeights);
        fread(weight.data(), elSize, nWeights, fd);

        fread(&nIndex, sizeof(int), 1, fd);
        index.resize(nIndex);
        fread(index.data(), sizeof(int8_t), nIndex, fd);
    }
};

class TorchLayer{
    struct alignas(1) Header{
        //int size; //size of data blob, not including this header
        enum class LayerType : char { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5 } layerType;
        char name[64]; //layer name for debugging
    };

public:
    TorchLayer* loadNext( FILE* fd );
    virtual Vectorf operator()( Vectorf& x ) = 0;
    virtual ~TorchLayer() = default;
};


class Conv1dLayer : TorchLayer{
    struct alignas(1) Header{
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        bool useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    Tensor3df weight;
    Vectorf bias;


    //call TorchLayer loadNext, not derived loadNext
    Conv1dLayer* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x );

};

class Conv2dLayer : TorchLayer{
    struct alignas(1) Header{
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        bool useBias;
        int inChannels;
        int outChannels;
        int kernelSize1;
        int kernelSize2;
    };

    Tensor4df weight;
    Vectorf bias;


    //call TorchLayer loadNext, not derived loadNext
    Conv2dLayer* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x );

};

class BatchNorm1dLayer : TorchLayer{
    struct alignas(1) Header{
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int inChannels;
    };

    Vectorf weight;
    Vectorf bias;
    Vectorf running_mean;
    Vectorf running_var;


public:
    //call TorchLayer loadNext, not derived loadNext
    BatchNorm1dLayer* loadNext( FILE* fd );

    virtual Vectorf operator()( Vectorf& x );

};


class LinearLayer : public TorchLayer{
    struct alignas(1) Header{
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nRows;
        int nCols;
    };

    CompMatrix mat;
    Vectorf bias;
    int nRows;
    int nCols;

public:
    LinearLayer() = delete;
    //call TorchLayer loadNext, not derived loadNext
    LinearLayer* loadNext( FILE* fd );
    virtual Vectorf operator()( Vectorf& x );
};


class GRULayer : public TorchLayer{
    struct alignas(1) Header{
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nRows;
        int nCols;
    };

    CompMatrix W_ir,W_iz,W_in;
    CompMatrix W_hr,W_hz,W_hn;
    Vectorf b_ir,b_iz,b_in;
    Vectorf b_hr,b_hz,b_hn;
    int nRows;
    int nCols;


public:
    GRULayer() = delete;

    //call TorchLayer loadNext, not derived loadNext
    GRULayer* loadNext( FILE* fd );
    virtual Vectorf operator()( Vectorf& x, Vectorf& hx );

};


#endif // WAVERNN_H
