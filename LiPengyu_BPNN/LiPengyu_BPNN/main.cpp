#include <iostream>
#include "NeuralNetwork.h"
#include "config.h"
#include <fstream>
using namespace std;
/*main�����е��õ�������������һ��
*��3λ�����Ʒ����ʮ����
*test_lr�õ��ǵ����softmax�ع�
*mlp�Ǻ��ж�����ز��������
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
		pos2 = line.find(" ", pos1);
		if(target == "Regression")
		{
			for(int j = 0 ; j < demsion ; j++)
			{
				string character = line.substr(pos1, pos2-pos1);
				while(character == "")
				{
					pos1 = pos2+1;
					pos2 = line.find(" ", pos1);
					character = line.substr(pos1, pos2-pos1);
				}

				data[i][j] = atof(character.c_str());
			}
		}
		else
		{
			string character = line.substr(pos1, pos2-pos1);
			while(character == "")
			{
				pos1 = pos2+1;
				pos2 = line.find(" ", pos1);
				character = line.substr(pos1, pos2-pos1);
			}

			data[i][int(atof(character.c_str())+0.5)-1] = 1;
		}
	}

	return data;
}

int main()
{
    //test_lr();
	cout << "****mlp****" << endl;
	
	//��������
	
	//printArr(phidden, 1);
	cout << "neural net intial..." << endl;
	NeuralNetwork neural(n_train, n_test, innode, outnode, ihiddenSize, phidden, 0);
	cout << "neural net complete" << endl;

	cout <<"train data loading..." << endl;
	double **train_data, **train_label;
	train_data = getData(rootDir+trainDataFile, n_train, innode);
	train_label = getData(rootDir+trainLabelFile, n_train , outnode);
	cout <<"train data load complete" << endl;

	cout <<"test data loading..." << endl;
	double **test_data, **test_label;
	test_data = getData(rootDir+testDataFile, n_test, innode);
	test_label = getData(rootDir+testLabelFile, n_test , outnode);
	cout <<"test data load complete" << endl;

	cout << "training..." << endl;
	neural.train(train_data, train_label, test_data, test_label, lr, MaxEpoch);
	cout<<"trainning complete..."<<endl;

	for (int i = 0; i != n_train; ++i)
	{
		delete []train_data[i];
		delete []train_label[i];
	}
	delete []train_data;
	delete []train_label;
	cout<<endl;

    return 0;
}