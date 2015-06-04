#include <iostream>
#include "NeuralNetwork.h"
#include "util.h"
#include <fstream>
#include <vector>
#include <algorithm>
//#include "HiddenLayer.h"
//#include "LogisticRegression.h"

using namespace std;
NeuralNetwork::NeuralNetwork(int n, int nt, int n_i, int n_o, int nhl, int *hls, int type)
{
	N = n;
	NT = nt;
	n_in = n_i;
	n_out = n_o;

	n_hidden_layer = nhl;
	hidden_layer_size = hls;

	activityFunctionType = type;
	//构造网络结构
	sigmoid_layers = new HiddenLayer* [n_hidden_layer];
	for(int i = 0; i < n_hidden_layer; ++i)
	{
		if(i == 0)
		{
			sigmoid_layers[i] = new HiddenLayer(n_in, hidden_layer_size[i], activityFunctionType);//第一个隐层
		}
		else
		{
			sigmoid_layers[i] = new HiddenLayer(hidden_layer_size[i-1], hidden_layer_size[i], activityFunctionType);//其他隐层
		}
	}

	log_layer = new LogisticRegression(hidden_layer_size[n_hidden_layer-1], n_out, N);//最后的softmax层

}

NeuralNetwork::~NeuralNetwork()
{
	//二维指针分配的对象不一定是二维数组
	for(int i = 0; i < n_hidden_layer; ++i)
		delete sigmoid_layers[i];  //删除的时候不能加[]
	delete[] sigmoid_layers;
	//log_layer只是一个普通的对象指针，不能作为数组delete
	delete log_layer;//删除的时候不能加[]
}

void NeuralNetwork::train(double** trainData, double** trainLabel, double **testData, double **testLabel, double dlr, int iepochs)
{
	ofstream outfile;
	outfile.open(outFile.c_str());
	vector<int> index;
	for(int i = 0 ; i < N ; i++)
		index.push_back(i);

	//反复迭代样本iepochs次训练
	for(int epoch = 0; epoch < iepochs; ++epoch)
	{
		random_shuffle(index.begin(), index.end());

		if(target ==  "Classifier")
		{
			int *pred_train = predictSoftMax(trainData, N);
			double Accuracy_train = getAccuracy(pred_train, trainLabel, N, n_out);

			int *pred_test = predictSoftMax(testData, NT);
			double Accuracy_test = getAccuracy(pred_test, testLabel, NT, n_out);

			delete []pred_train;
			delete []pred_test;

			std::cout << "epoch," << epoch << ",trainError," << Accuracy_train; 
			std::cout << ",testError," << Accuracy_test << endl;

			outfile << "epoch," << epoch << ",trainError," << Accuracy_train; 
			outfile << ",testError," << Accuracy_test << endl;
		}
		else if(target == "Regression")
		{
			double **pred_train = predict(trainData, N);
			double RMSE_train = getRMSE(pred_train, trainLabel, N, n_out);

			double **pred_test = predict(testData, NT);
			double RMSE_test = getRMSE(pred_test, testLabel, NT, n_out);

			for(int del_i = 0 ; del_i < N ; del_i ++)
				delete []pred_train[del_i];
			delete []pred_train;

			for(int del_i = 0 ; del_i < NT ; del_i ++)
				delete []pred_test[del_i];
			delete []pred_test;

			std::cout << "epoch," << epoch << ",trainError," << RMSE_train; 
			std::cout << ",testError" << RMSE_test << endl;

			outfile << "epoch," << epoch << ",trainError," << RMSE_train; 
			outfile << ",testError," << RMSE_test << endl;
		}
		else
			cout << "error in target chose" << endl;



		double e = 0.0;
		for(int index_i = 0; index_i < N; ++index_i)
		{
			int i = index[index_i];
			//前向传播阶段 
			for(int n = 0; n < n_hidden_layer; ++ n)
			{
				if(n == 0) //第一个隐层直接输入数据
				{
					sigmoid_layers[n]->forward_propagation(trainData[i]);
				}
				else //其他隐层用前一层的输出作为输入数据
				{
					sigmoid_layers[n]->forward_propagation(sigmoid_layers[n-1]->output_data);
				}
			}
			//softmax层使用最后一个隐层的输出作为输入数据
			log_layer->forward_propagation(sigmoid_layers[n_hidden_layer-1]->output_data);

			//e += log_layer->cal_error(trainLabel[i]);

			//反向传播阶段
			log_layer->back_propagation(sigmoid_layers[n_hidden_layer-1]->output_data, trainLabel[i], dlr);

			for(int n = n_hidden_layer-1; n >= 1; --n)
			{
				if(n == n_hidden_layer-1)
				{
					sigmoid_layers[n]->back_propagation(sigmoid_layers[n-1]->output_data, 
						log_layer->delta, log_layer->w, log_layer->n_out, dlr, N);
				}
				else
				{
					double *pdinputData;
					pdinputData = sigmoid_layers[n-1]->output_data;
					
					sigmoid_layers[n]->back_propagation(pdinputData, 
						sigmoid_layers[n+1]->delta, sigmoid_layers[n+1]->w, sigmoid_layers[n+1]->n_out, dlr, N);
				}
			}
			//这里该怎么写？
			if (n_hidden_layer > 1)
				sigmoid_layers[0]->back_propagation(trainData[i],
					sigmoid_layers[1]->delta, sigmoid_layers[1]->w, sigmoid_layers[1]->n_out, dlr, N);
			else
				sigmoid_layers[0]->back_propagation(trainData[i],
					log_layer->delta, log_layer->w, log_layer->n_out, dlr, N);
		}
		

		/*if(epoch%decay_lr_epoch == 0)
			dlr = decay_lr * dlr;*/
	}

}

int *NeuralNetwork::predictSoftMax(double** ppdata, int n)
{
	int *result = new int[n];

	for(int i = 0; i < n; ++i)
	{
		for(int n = 0; n < n_hidden_layer; ++ n)
		{
			if(n == 0) //第一个隐层直接输入数据
			{
				sigmoid_layers[n]->forward_propagation(ppdata[i]);
			}
			else //其他隐层用前一层的输出作为输入数据
			{
				sigmoid_layers[n]->forward_propagation(sigmoid_layers[n-1]->output_data);
			}
		}
		//softmax层使用最后一个隐层的输出作为输入数据
		result[i] = log_layer->predictSoftMax(sigmoid_layers[n_hidden_layer-1]->output_data);
		//log_layer->forward_propagation(sigmoid_layers[n_hidden_layer-1]->output_data);
	}

	return result;
}
double ** NeuralNetwork::predict(double** ppdata, int n)
{
	double **result = new double *[n];

	for(int i = 0; i < n; ++i)
	{
		for(int n = 0; n < n_hidden_layer; ++ n)
		{
			if(n == 0) //第一个隐层直接输入数据
			{
				sigmoid_layers[n]->forward_propagation(ppdata[i]);
			}
			else //其他隐层用前一层的输出作为输入数据
			{
				sigmoid_layers[n]->forward_propagation(sigmoid_layers[n-1]->output_data);
			}
		}
		//softmax层使用最后一个隐层的输出作为输入数据
		result[i] = log_layer->predict(sigmoid_layers[n_hidden_layer-1]->output_data);
		//log_layer->forward_propagation(sigmoid_layers[n_hidden_layer-1]->output_data);
	}

	return result;
}

double NeuralNetwork::getRMSE(double **pred, double **label, int N, int n_out)
{
	double result = 0;
	for(int i = 0 ; i < N ; i++)
	{
		for(int j = 0 ; j < n_out ; j++)
		{
			//result += (pred[i][j] - label[i][j]) * (pred[i][j] - label[i][j]);
			result += abs(pred[i][j] - label[i][j]);
		}
	}

	//result = sqrt(result/N);
	result = result/N;
	return result;
}

double NeuralNetwork::getAccuracy(int *pred, double **label, int N, int n_out)
{
	double result = 0;
	for(int i = 0 ; i < N ; i++)
	{
		int curLabel = 0;
		for(int j = 0 ; j < n_out ; j++)
		{
			if(abs(label[i][j]-1) < 0.00001)
				curLabel = j;
		}
		if(pred[i] == curLabel)
			result += 1;
	}

	result = result / N;
	return result;
}