#pragma once
std::string outFile("resultShow.csv");//��������ļ�

std::string rootDir("D:\\����\\MatData\\");//��Ŀ¼
std::string DataFile("data_allProsocial.txt");//ѵ�������ļ�
std::string LabelFile("label_allProsocial.txt");//���Ա�ǩ�ļ�
const int col = 1;


const int n_train = 120;//ѵ��������Ŀ
const int n_test = 40;//����������Ŀ
const int innode = 90;//����Ψ��
const int outnode = 1;//���ά��

const int ihiddenSize = 2;//������Ŀ
int phidden[ihiddenSize] = {150, 100};//ÿ������ڵ���
double weidth_decay_[ihiddenSize + 1] = { 0.05, 0.05, 0.05 };

const double lr = 0.0001;//ѧϰ��
const double decay_lr = 0.99;//ѧϰ��˥��
const int decay_lr_epoch = 100;//ѧϰ��˥��Ƶ��

const int MaxEpoch = 170000;//���ѵ������
  
std::string target = "Regression";//"Regression";//"Classifier";//

bool signSave = true;