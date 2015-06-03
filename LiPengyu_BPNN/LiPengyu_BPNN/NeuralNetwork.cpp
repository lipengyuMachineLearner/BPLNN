#include <iostream>
#include "NeuralNetwork.h"
#include "util.h"
//#include "HiddenLayer.h"
//#include "LogisticRegression.h"

using namespace std;

const int n_train = 8, innode = 3, outnode = 8;
NeuralNetwork::NeuralNetwork(int n, int n_i, int n_o, int nhl, int *hls, int type)
{
	N = n;
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

void NeuralNetwork::train(double** ppdinData, double** ppdinLabel, double dlr, int iepochs)
{
	printArrDouble(ppdinData, N, n_in);

	
	cout << "******label****" << endl;
	printArrDouble(ppdinLabel, N, n_out);

	//反复迭代样本iepochs次训练
	for(int epoch = 0; epoch < iepochs; ++epoch)
	{
		double e = 0.0;
		for(int i = 0; i < N; ++i)
		{
			//前向传播阶段 
			for(int n = 0; n < n_hidden_layer; ++ n)
			{
				if(n == 0) //第一个隐层直接输入数据
				{
					sigmoid_layers[n]->forward_propagation(ppdinData[i]);
				}
				else //其他隐层用前一层的输出作为输入数据
				{
					sigmoid_layers[n]->forward_propagation(sigmoid_layers[n-1]->output_data);
				}
			}
			//softmax层使用最后一个隐层的输出作为输入数据
			log_layer->forward_propagation(sigmoid_layers[n_hidden_layer-1]->output_data);

			//e += log_layer->cal_error(ppdinLabel[i]);

			//反向传播阶段
			log_layer->back_propagation(sigmoid_layers[n_hidden_layer-1]->output_data, ppdinLabel[i], dlr);

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
				sigmoid_layers[0]->back_propagation(ppdinData[i],
					sigmoid_layers[1]->delta, sigmoid_layers[1]->w, sigmoid_layers[1]->n_out, dlr, N);
			else
				sigmoid_layers[0]->back_propagation(ppdinData[i],
					log_layer->delta, log_layer->w, log_layer->n_out, dlr, N);
		}
		//if (epoch % 100 == 1)
			//cout << "iepochs number is " << epoch << "   cost function is " << e / (double)N << endl;
	}

}

void NeuralNetwork::predict(double** ppdata, int n)
{
	

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
		log_layer->predict(sigmoid_layers[n_hidden_layer-1]->output_data);
		//log_layer->forward_propagation(sigmoid_layers[n_hidden_layer-1]->output_data);
	}
}
//double **makeLabelSample(double **label_x)
double **makeLabelSample(double label_x[][outnode])
{
	double **pplabelSample;
	pplabelSample = new double*[n_train];
	for (int i = 0; i < n_train; ++i)
	{
		pplabelSample[i] = new double[outnode];
	}

	for (int i = 0; i < n_train; ++i)
	{
		for (int j = 0; j < outnode; ++j)
			pplabelSample[i][j] = label_x[i][j];
	}
	return pplabelSample;
}
double **maken_train(double train_x[][innode])
{
	double **ppn_train;
	ppn_train = new double*[n_train];
	for (int i = 0; i < n_train; ++i)
	{
		ppn_train[i] = new double[innode];
	}

	for (int i = 0; i < n_train; ++i)
	{
		for (int j = 0; j < innode; ++j)
			ppn_train[i][j] = train_x[i][j];
	}
	return ppn_train;
}
void disTrain(double **pptrain)
{
	for (int i = 0; i < n_train; ++i)
	{
		for (int j = 0; j < innode; ++j)
			cout << pptrain[i][j] << ' ';
		cout << endl;
	}
}

void mlp()
{
	//输入样本
	double X[n_train][innode]= {
		{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}
	};

	double Y[n_train][outnode]={
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 1},
	};
	const int ihiddenSize = 2;
	int phidden[ihiddenSize] = {5, 5};
	//printArr(phidden, 1);
	NeuralNetwork neural(n_train, innode, outnode, ihiddenSize, phidden, 0);
	double **train_x, **ppdlabel;
	train_x = maken_train(X);
	//printArrDouble(train_x, n_train, innode);
	ppdlabel = makeLabelSample(Y);
	neural.train(train_x, ppdlabel, 0.1, 3500);
	cout<<"trainning complete..."<<endl;
	neural.predict(train_x, n_train);

	for (int i = 0; i != n_train; ++i)
	{
		delete []train_x[i];
		delete []ppdlabel[i];
	}
	delete []train_x;
	delete []ppdlabel;
	cout<<endl;
}
