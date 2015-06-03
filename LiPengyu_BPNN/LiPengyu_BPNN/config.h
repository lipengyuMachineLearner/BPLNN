#pragma once
std::string outFile("resultShow10_10.csv");//��������ļ�

std::string rootDir("F:\\����ѧ\\����\\MatData\\tmp\\");//��Ŀ¼
std::string trainDataFile("train_data.txt");//ѵ�������ļ�
std::string trainLabelFile("train_label.txt");//���Ա�ǩ�ļ�
std::string testDataFile("test_data.txt");//���������ļ�
std::string testLabelFile("test_label.txt");//���Ա�ǩ�ļ�

const int n_train = 30;//ѵ��������Ŀ
const int n_test = 10;//����������Ŀ
const int innode = 90;//����Ψ��
const int outnode = 1;//���ά��

const int ihiddenSize = 2;//������Ŀ
int phidden[ihiddenSize] = {70, 70};//ÿ������ڵ���

const double lr = 0.0001;//ѧϰ��
const double decay_lr = 0.99;//ѧϰ��˥��
const int decay_lr_epoch = 100;//ѧϰ��˥��Ƶ��

const int MaxEpoch = 35000;//���ѵ������
  
std::string target = "Regression";//"Regression";//"Classifier";//