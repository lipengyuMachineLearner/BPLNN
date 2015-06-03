#ifndef LOGISTICREGRESSIONLAYER
#define LOGISTICREGRESSIONLAYER

class LogisticRegression
{
public:
	LogisticRegression(int n_i, int i_o, int);
	~LogisticRegression();

	void forward_propagation(double* input_data);
	void back_propagation(double* input_data, double* label, double lr);
	void softmax(double* x);
	void printwb();
	void train(double *x, double *y, double lr);

	double *predict(double *);
	double cal_error(double **ppdtest, double* pdlabel, int ibatch);
	//double cal_error(double* label);
	void makeLabels(int* pimax, double (*pplabels)[8]);

	//����ǰ�򴫲������ֵ��Ҳ�����յ�Ԥ��ֵ
	double* output_data;
	//���򴫲�ʱ����ֵ
	double* delta;

public:
	int n_in;
	int n_out;
	int n_train;
	double** w;
	double* b;

};

void test_lr();
#endif


