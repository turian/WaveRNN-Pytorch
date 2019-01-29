#ifndef WAVERNN_H
#define WAVERNN_H

#include <stdio.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> Matrixf;
typedef Tensor<float, 3, RowMajor> Tensorf;
typedef VectorXf Vectorf;


class TorchLayer{
    struct Header{
        int size; //size of data blob, not including this header
        enum LayerType : char { Conv1d=1, BatchNorm1d=2, Linear=3, GRU=4 };
        char elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        char name[64]; //layer name for debugging
    };

public:
    TorchLayer* loadNext( FILE* fd );
    virtual Vectorf operator()( Vectorf& x ) = 0;
    virtual ~TorchLayer() = default;
};


class Conv1d : TorchLayer{
    struct Header{
        bool useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    Tensorf weight;
    Vectorf bias;


    //call TorchLayer loadNext, not derived loadNext
    Conv1d* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x );

};


class BatchNorm1d : TorchLayer{
    struct Header{
        int inChannels;
    };


    Vectorf weight;
    Vectorf bias;
    Vectorf running_mean;
    Vectorf running_var;

    //call TorchLayer loadNext, not derived loadNext
    BatchNorm1d* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x );

};


class Linear : TorchLayer{
    struct Header{
        int inChannels;
        int outChannels;
    };

    Matrixf weight;
    Vectorf bias;

    //call TorchLayer loadNext, not derived loadNext
    Linear* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x );

};


class GRU : TorchLayer{
    struct Header{
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    Matrixf W_ir,W_iz,W_in;
    Matrixf W_hr,W_hz,W_hn;
    Vectorf b_ir,b_iz,b_in;
    Vectorf b_hr,b_hz,b_hn;

    //call TorchLayer loadNext, not derived loadNext
    Linear* loadNext( FILE* fd );

public:

    virtual Vectorf operator()( Vectorf& x, Vectorf& hx );

};


#endif // WAVERNN_H
