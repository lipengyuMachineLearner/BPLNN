#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "HiddenLayer.h"
#include "LogisticRegression.h"
#include "config.h"
#include <string>

class NeuralNetwork
{
public:
	NeuralNetwork(int n, int nt, int n_i, int n_o, int nhl, int*hls, int type, double *weight_decay_in);
    ~NeuralNetwork();

    void train(double** traindata, double** trainlabel, double **testData, double **testLabel, double lr, int epochs);
    double ** predict(double** in_data, int n);
	int *predictSoftMax(double** ppdata, int n);
	void save();
	void load();

private:
    int N; //样本数量
	int NT; //测试样本数量
    int n_in; //输入维数
    int n_out; //输出维数
    int n_hidden_layer; //隐层数目
    int* hidden_layer_size; //中间隐层的大小 e.g. {3,4}表示有两个隐层，第一个有三个节点，第二个有4个节点
	double *weight_decay;

	int activityFunctionType;

    HiddenLayer **sigmoid_layers;
    LogisticRegression *log_layer;

	double NeuralNetwork::getRMSE(double **pred, double **trainLabel, int N, int n_out);
	double NeuralNetwork::getAccuracy(int *pred, double **label, int N, int n_out);
};

#endif

