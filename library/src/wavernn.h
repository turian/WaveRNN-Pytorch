#ifndef WAVERNN_H
#define WAVERNN_H

#include <stdio.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXf;
typedef Tensor<float, 3, RowMajor> TensorXf;


class TorchLayer{
    struct Header{
        int size; //size of data blob, not including this header
        enum LayerType : char { Conv1d=1, BatchNorm1d=2, Linear=3, GRU=4 };
    };

public:
    TorchLayer* loadNext( FILE* fd );
    virtual VectorXf operator()( VectorXf& x ) = 0;
    virtual ~TorchLayer() = default;
};


class Conv1d : TorchLayer{
    struct Header{
        bool useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };



    TensorXf weight;
    VectorXf bias;


    //call TorchLayer loadNext, not derived loadNext
    Conv1d* loadNext( FILE* fd );

public:

    virtual VectorXf operator()( VectorXf& x );

};


class BatchNorm1d : TorchLayer{
    struct Header{
        int inChannels;
    };


    VectorXf weight;
    VectorXf bias;
    VectorXf running_mean;
    VectorXf running_var;

    //call TorchLayer loadNext, not derived loadNext
    BatchNorm1d* loadNext( FILE* fd );

public:

    virtual VectorXf operator()( VectorXf& x );

};





#endif // WAVERNN_H
