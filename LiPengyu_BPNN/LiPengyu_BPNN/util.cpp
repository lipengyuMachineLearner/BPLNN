#include "util.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
using namespace std;

int getMaxIndex(double *pdarr, int num)
{
	double dmax = -1;
	int imax = -1;
	for(int i = 0; i < num; ++i)
    {
		if (pdarr[i] > dmax)
		{
			dmax = pdarr[i];
			imax = i;
		}
    }
	return imax;
}


double uniform(double _min, double _max)
{
    return rand()/(RAND_MAX + 1.0) * (_max - _min) + _min;
}

void printArrDouble(double **pparr, int row, int col)
{
	cout << "****printArrDouble****" << endl;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			cout << pparr[i][j] << ' ';
		}
		cout << endl;
	}
}

void initArr(double *parr, int num)
{
	for (int i = 0; i < num; ++i)
		parr[i] = 0.0;
}

