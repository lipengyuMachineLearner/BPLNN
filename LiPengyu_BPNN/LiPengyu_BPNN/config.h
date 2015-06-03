#pragma once
std::string outFile("resultShow10_10.csv");//结果保存文件

std::string rootDir("F:\\心理学\\心理\\MatData\\tmp\\");//根目录
std::string trainDataFile("train_data.txt");//训练数据文件
std::string trainLabelFile("train_label.txt");//测试标签文件
std::string testDataFile("test_data.txt");//测试数据文件
std::string testLabelFile("test_label.txt");//测试标签文件

const int n_train = 30;//训练样本数目
const int n_test = 10;//测试样本数目
const int innode = 90;//输入唯度
const int outnode = 1;//输出维度

const int ihiddenSize = 2;//隐层数目
int phidden[ihiddenSize] = {70, 70};//每个隐层节点数

const double lr = 0.0001;//学习率
const double decay_lr = 0.99;//学习率衰减
const int decay_lr_epoch = 100;//学习率衰减频率

const int MaxEpoch = 35000;//最大训练次数
  
std::string target = "Regression";//"Regression";//"Classifier";//