#pragma once
std::string outFile("resultShow.csv");//结果保存文件

std::string rootDir("F:\\心理学\\心理\\MatData\\");//根目录
std::string DataFile("data_allProsocial.txt");//训练数据文件
std::string LabelFile("label_allProsocial.txt");//测试标签文件
const int col = 2;


const int n_train = 120;//训练样本数目
const int n_test = 40;//测试样本数目
const int innode = 90;//输入唯度
const int outnode = 1;//输出维度

const int ihiddenSize = 2;//隐层数目
int phidden[ihiddenSize] = {100, 70};//每个隐层节点数

const double lr = 0.0001;//学习率
const double decay_lr = 0.99;//学习率衰减
const int decay_lr_epoch = 100;//学习率衰减频率

const int MaxEpoch = 70000;//最大训练次数
  
std::string target = "Regression";//"Regression";//"Classifier";//