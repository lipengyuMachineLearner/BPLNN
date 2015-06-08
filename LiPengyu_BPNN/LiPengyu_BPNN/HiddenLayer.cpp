#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "HiddenLayer.h"
#include "util.h"

using namespace std;



HiddenLayer::HiddenLayer(int n_i, int n_o, int type, double weight_decay_in)
{
    n_in  = n_i;
    n_out = n_o;

	activityFunctionType = type;

	weight_decay = weight_decay_in;

    w = new double* [n_out];
    for(int i = 0; i < n_out; ++i)
    {
        w[i] = new double [n_in];
    }
    b = new double [n_out];

    double a = 1.0 / n_in;

    srand((unsigned)time(NULL));
    for(int i = 0; i < n_out; ++i)
    {
        for(int j = 0; j < n_in; ++j)
            w[i][j] = uniform(-a, a);
        b[i] = uniform(-a, a);
    }

    delta = new double [n_out];
    output_data = new double [n_out];
}

HiddenLayer::~HiddenLayer()
{
    for(int i=0; i<n_out; i++) 
		delete []w[i]; 
    delete[] w; 
    delete[] b;
    delete[] output_data;
    delete[] delta;
}
void HiddenLayer::forward_propagation(double* pdinputData)
{
    for(int i = 0; i < n_out; ++i)
    {
        output_data[i] = 0.0;
        for(int j = 0; j < n_in; ++j)
        {
            output_data[i] += w[i][j]*pdinputData[j];
        }
        output_data[i] += b[i];

        output_data[i] = activityFunction(activityFunctionType,output_data[i]);//sigmoid(output_data[i]);
    }
}

void HiddenLayer::back_propagation(double *pdinputData, double *pdnextLayerDelta,
        double** ppdnextLayerW, int iNextLayerOutNum, double dlr, int N)
{
    /*
    pdinputData          为输入数据
    *pdnextLayerDelta   为下一层的残差值delta,是一个大小为iNextLayerOutNum的数组
    **ppdnextLayerW      为此层到下一层的权值
    iNextLayerOutNum    实际上就是下一层的n_out
    dlr                  为学习率learning rate
    N                   为训练样本总数
    */

    double* sigma = new double[n_out];
    for(int i = 0; i < n_out; ++i)
        sigma[i] = 0.0;

    for(int i = 0; i < iNextLayerOutNum; ++i)
    {
        for(int j = 0; j < n_out; ++j)
        {
            sigma[j] += ppdnextLayerW[i][j] * pdnextLayerDelta[i];
        }
    }
    //计算得到本层的残差delta
    for(int i = 0; i < n_out; ++i)
    {
        delta[i] = sigma[i] * reciprocal(activityFunctionType,output_data[i]);//output_data[i] * (1 - output_data[i]);//
    }

    //调整本层的权值w
    for(int i = 0; i < n_out; ++i)
    {
        for(int j = 0; j < n_in; ++j)
        {
            w[i][j] += dlr * delta[i] * pdinputData[j] - weight_decay*w[i][j];
        }
        b[i] += dlr * delta[i];
    }
    delete[] sigma;
}

void HiddenLayer::load(string weight, string bias)
{
	;
}
