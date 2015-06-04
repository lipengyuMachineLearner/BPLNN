#include <iostream>
#include "NeuralNetwork.h"
#include "config.h"
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
/*main函数中调用的两个函数功能一样
*将3位二进制分类成十进制
*test_lr用的是单层的softmax回归
*mlp是含有多个隐藏层的神经网络
*/




double **getData(string dir, int num, int demsion)
{
	ifstream infile;
	infile.open(dir);
	
	double **data = new double *[num];
	for(int i = 0 ; i < num ; i++)
	{
		data[i] = new double [demsion];
		for(int j = 0 ; j < demsion ; j++)
			data[i][j] = 0;
	}

	string line("");
	for(int i = 0 ; i < num ; i++)
	{
		getline(infile, line);

		string::size_type pos1 = 0 , pos2 = 0;
		string character = "";
		if(target == "Regression")
		{
			for(int j = 0 ; j < col ; j++)
			{
				pos2 = line.find(",", pos1);
				pos1 = pos2+1;
			}
			for(int j = 0 ; j < demsion ; j++)
			{
				pos2 = line.find(",", pos1);
				character = line.substr(pos1, pos2-pos1);
				
				data[i][j] = atof(character.c_str());
				pos1 = pos2+1;
			}
		}
		else
		{
			string character = "";
			for(int j = 0 ; j < col ; j++)
			{
				pos2 = line.find(",", pos1);
				pos2 = pos1 + 1;
			}
			

			pos2 = line.find(",", pos1);
			character = line.substr(pos1, pos2-pos1);
			

			data[i][int(atof(character.c_str())+0.5)-1] = 1;
		}
	}

	return data;
}

double **splitData(double **data, int sta, int num, int demsion, vector<int> index)
{
	double **result = new double *[num];
	for(int i = sta ; i < num+sta ; i++)
	{
		result[i-sta] = new double[demsion];
		for(int j = 0 ; j < demsion ; j++)
			result[i-sta][j] = data[index[i]][j];
	}

	return result;
}

int main()
{
    //test_lr();
	cout << "****mlp****" << endl;
	
	//输入样本
	
	//printArr(phidden, 1);
	cout << "neural net intial..." << endl;
	NeuralNetwork neural(n_train, n_test, innode, outnode, ihiddenSize, phidden, 0);
	cout << "neural net complete" << endl;
	double **data, **label;
	cout <<"data loading..." << endl;
	data = getData(rootDir+DataFile, n_train+n_test, innode);
	label = getData(rootDir+LabelFile, n_train+n_test, outnode);
	cout <<"data load complete" << endl;

	vector<int> index;
	for(int i = 0 ; i < n_train+n_test ; i++)
		index.push_back(i);
	random_shuffle(index.begin() , index.end());

	for(int train_epoch =  0; train_epoch < 1; train_epoch++)
	{
		cout << "data spliting..." << endl;
		double **train_data, **train_label;
		double **test_data, **test_label;

		train_data = splitData(data, 0, n_train, innode, index);
		train_label = splitData(label, 0, n_train, outnode, index);
		
		test_data = splitData(data, n_train, n_test, innode, index);
		test_label = splitData(label, n_train, n_test, outnode, index);
		cout << "data spliting complete" << endl;


		cout << "training..." << endl;
		neural.train(train_data, train_label, test_data, test_label, lr, MaxEpoch);
		cout<<"trainning complete..."<<endl;
	}

	for (int i = 0; i != n_train+n_test; ++i)
	{
		delete []data[i];
		delete []label[i];
	}
	delete []data;
	delete []label;

    return 0;
}