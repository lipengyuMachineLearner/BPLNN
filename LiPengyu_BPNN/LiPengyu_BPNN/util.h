#ifndef UTIL_H
#define UTIL_H
#include<iostream>
#include<string>

double uniform(double _min, double _max);
//void printArr(T *parr, int num);
void printArrDouble(double **pparr, int row, int col);
void initArr(double *parr, int num);
int getMaxIndex(double *pdarr, int num);

template <typename T>
void printArr(T *parr, int num)
{
	cout << "****printArr****" << endl;

	for (int i = 0; i < num; ++i)
		cout << parr[i] << ' ';
	cout << endl;
}

inline double sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
}

inline double reciprocal(int type, double x)
{
	switch (type)
	{
	case 0: return x*(1-x);
	default: return 0;
	}
	/*if(type=="sigmoid")
		return x*(1-x);
	else
		return 0;*/
}

inline double activityFunction(int type, double x)
{
	switch (type)
	{
	case 0: return sigmoid(x);
	default: return 0;
	}
	/*if(type=="sigmoid")
		return sigmoid(x);
	else
		return 0;*/
}

#endif