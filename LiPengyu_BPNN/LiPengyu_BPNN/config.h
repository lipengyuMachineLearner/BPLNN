#pragma once
std::string outFile("resultShow.csv");//��������ļ�

std::string rootDir("F:\\����ѧ\\����\\MatData\\");//��Ŀ¼
std::string DataFile("data_allProsocial.txt");//ѵ�������ļ�
std::string LabelFile("label_allProsocial.txt");//���Ա�ǩ�ļ�
const int col = 2;


const int n_train = 120;//ѵ��������Ŀ
const int n_test = 40;//����������Ŀ
const int innode = 90;//����Ψ��
const int outnode = 1;//���ά��

const int ihiddenSize = 2;//������Ŀ
int phidden[ihiddenSize] = {100, 70};//ÿ������ڵ���

const double lr = 0.0001;//ѧϰ��
const double decay_lr = 0.99;//ѧϰ��˥��
const int decay_lr_epoch = 100;//ѧϰ��˥��Ƶ��

const int MaxEpoch = 70000;//���ѵ������
  
std::string target = "Regression";//"Regression";//"Classifier";//