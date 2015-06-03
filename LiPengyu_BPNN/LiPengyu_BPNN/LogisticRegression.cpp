#include <cmath>
#include <ctime>
#include <iostream>
#include "LogisticRegression.h"
#include "util.h"

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

void LogisticRegression::printwb()
{
    cout << "'****w****\n";
    for(int i = 0; i < n_out; ++i)
    {
        for(int j = 0; j < n_in; ++j)
            cout << w[i][j] << ' ';
        cout << endl;
            //w[i][j] = uniform(-a, a);
    }
    cout << "'****b****\n";
    for(int i = 0; i < n_out; ++i)
    {
        cout << b[i] << ' ';
    }
    cout << endl;
	cout << "'****output****\n";
    for(int i = 0; i < n_out; ++i)
    {
        cout << output_data[i] << ' ';
    }
    cout << endl;

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

    softmax(output_data);
}

void LogisticRegression::back_propagation(double* input_data, double* label, double lr)
{
    for(int i = 0; i < n_out; ++i)
    {
        delta[i] = label[i] - output_data[i] ;
        for(int j = 0; j < n_in; ++j)
        {
            w[i][j] += lr * delta[i] * input_data[j] / n_train;
        }
        b[i] += lr * delta[i] / n_train;
    }
}

int LogisticRegression::predict(double *x)
{
	forward_propagation(x);
	cout << "***result is ***" << endl;
	int iresult = getMaxIndex(output_data, n_out);
	cout << iresult << endl;
	//if (iresult == 1)
		printArr(output_data, n_out);
	return iresult;

}
void LogisticRegression::train(double *x, double *y, double lr)
{
    forward_propagation(x);
	back_propagation(x, y, lr);
}
double LogisticRegression::cal_error(double **ppdtest, double* pdlabel, int ibatch)
{
    double error = 0.0, dmax = 0;
	int imax = -1, ierrNum = 0;
	for (int i = 0; i < ibatch; ++i)
	{
		imax = predict(ppdtest[i]);
		if (imax != pdlabel[i])
			++ierrNum;
	}
	error = (double)ierrNum / ibatch;
	return error;
}
void LogisticRegression::makeLabels(int* pimax, double (*pplabels)[8])
{
	for (int i = 0; i < n_train; ++i)
	{
		for (int j = 0; j < n_out; ++j)
			pplabels[i][j] = 0;
		int k = pimax[i];
		pplabels[i][k] = 1.0;
	}
}


void test_lr() 
{
    srand(0);

    double learning_rate = 0.1;
    double n_epochs = 200;

    int test_N = 2;
    const int trainNum = 8, n_in = 3, n_out = 8;
	//int n_out = 2;
	double train_X[trainNum][n_in] = {
		{1, 1, 1},
		{1, 1, 0},
		{1, 0, 1},
		{1, 0, 0},
		{0, 1, 1},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 0}
	};
	//szimax存储的是最大值的下标
	int szimax[trainNum];
	for (int i = 0; i < trainNum; ++i)
		szimax[i] = trainNum - i - 1;
	double train_Y[trainNum][n_out];
	
	// construct LogisticRegression
	LogisticRegression classifier(n_in, n_out, trainNum);

	classifier.makeLabels(szimax, train_Y);
	// train online
	for(int epoch=0; epoch<n_epochs; epoch++) {
		for(int i=0; i<trainNum; i++) {
            //classifier.trainEfficient(train_X[i], train_Y[i], learning_rate);
            classifier.train(train_X[i], train_Y[i], learning_rate);
        }
    }


    // test data
    double test_X[2][n_out] = {
        {1, 0, 1},
        {0, 0, 1}
    };

    for(int i=0; i<trainNum; i++) {
		classifier.predict(train_X[i]);
        cout << endl;
    }
    cout << "*********\n";
    // test
    for(int i=0; i<test_N; i++) {
		classifier.predict(test_X[i]);
        cout << endl;
    }
}


