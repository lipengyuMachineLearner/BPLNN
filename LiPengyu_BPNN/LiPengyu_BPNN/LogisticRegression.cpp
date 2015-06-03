#include <cmath>
#include <ctime>
#include <iostream>
#include "LogisticRegression.h"
#include "util.h"
#include "config.h"
using namespace std;


LogisticRegression::LogisticRegression(int n_i, int n_o, int n_t)
{
    n_in = n_i;
    n_out = n_o;
    n_train = n_t;

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

LogisticRegression::~LogisticRegression()
{
    for(int i=0; i < n_out; i++) 
		delete []w[i]; 
    delete[] w; 
    delete[] b;
    delete[] output_data;
    delete[] delta;
}

void LogisticRegression::softmax(double* x)
{
    double _max = 0.0;
    double _sum = 0.0;

    for(int i = 0; i < n_out; ++i)
    {
        if(_max < x[i])
            _max = x[i];
    }
    for(int i = 0; i < n_out; ++i)
    {
        x[i] = exp(x[i]-_max);
        _sum += x[i];
    }

    for(int i = 0; i < n_out; ++i)
    {
        x[i] /= _sum;
    }
}

void LogisticRegression::forward_propagation(double* input_data)
{
    for(int i = 0; i < n_out; ++i)
    {
        output_data[i] = 0.0;
        for(int j = 0; j < n_in; ++j)
        {
            output_data[i] += w[i][j]*input_data[j];
        }
        output_data[i] += b[i];
    }

	if(target == "Classifier")
		softmax(output_data);
}

void LogisticRegression::back_propagation(double* input_data, double* label, double lr)
{
    for(int i = 0; i < n_out; ++i)
    {
        delta[i] = label[i] - output_data[i] > 0 ? 1 : -1;
        for(int j = 0; j < n_in; ++j)
        {
            w[i][j] += lr * delta[i] * input_data[j] / n_train;
        }
        b[i] += lr * delta[i] / n_train;
    }
}

double *LogisticRegression::predict(double *x)
{
	double *result = new double[n_out];

	forward_propagation(x);

	for(int i = 0 ; i < n_out ; i++)
		result[i] = output_data[i];

	return result;

}

int LogisticRegression::predictSoftMax(double *x)
{
	forward_propagation(x);

	int iresult = getMaxIndex(output_data, n_out);
	return iresult;

}

void LogisticRegression::train(double *x, double *y, double lr)
{
    forward_propagation(x);
	back_propagation(x, y, lr);
}



