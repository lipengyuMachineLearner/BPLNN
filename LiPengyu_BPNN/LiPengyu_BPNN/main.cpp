#include <iostream>
#include "NeuralNetwork.h"
using namespace std;
/*main函数中调用的两个函数功能一样
*将3位二进制分类成十进制
*test_lr用的是单层的softmax回归
*mlp是含有多个隐藏层的神经网络
*/
int main()
{
    //test_lr();
	cout << "****mlp****" << endl;
	mlp();
    return 0;
}
