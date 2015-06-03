#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "HiddenLayer.h"
#include "LogisticRegression.h"

class NeuralNetwork
{
public:
    NeuralNetwork(int n, int n_i, int n_o, int nhl, int*hls, int type);
    ~NeuralNetwork();

    void train(double** in_data, double** in_label, double lr, int epochs);
    void predict(double** in_data, int n);

private:
    int N; //样本数量
    int n_in; //输入维数
    int n_out; //输出维数
    int n_hidden_layer; //隐层数目
    int* hidden_layer_size; //中间隐层的大小 e.g. {3,4}表示有两个隐层，第一个有三个节点，第二个有4个节点

	int activityFunctionType;

    HiddenLayer **sigmoid_layers;
    LogisticRegression *log_layer;
};

void mlp();

#endif

