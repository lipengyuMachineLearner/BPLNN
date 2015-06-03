#pragma once
std::string outFile("resultShow10_10.csv");

std::string rootDir("F:\\心理学\\心理\\MatData\\tmp\\");
std::string trainDataFile("train_data.txt");
std::string trainLabelFile("train_label.txt");
std::string testDataFile("test_data.txt");
std::string testLabelFile("test_label.txt");

const int n_train = 30;
const int n_test = 10;
const int innode = 90;
const int outnode = 1;

const int ihiddenSize = 2;
int phidden[ihiddenSize] = {10, 10};

const double lr = 0.3;
const double decay_lr = 0.9;
const int decay_lr_epoch = 30;

const int MaxEpoch = 35000;
  