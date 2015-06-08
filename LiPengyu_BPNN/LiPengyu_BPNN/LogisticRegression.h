#ifndef LOGISTICREGRESSIONLAYER
#define LOGISTICREGRESSIONLAYER

class LogisticRegression
{
public:
	LogisticRegression(int n_i, int i_o, int n_t, double weight_decay_in);
	~LogisticRegression();

	void forward_propagation(double* input_data);
	void back_propagation(double* input_data, double* label, double lr);
	void softmax(double* x);
	void printwb();
	void train(double *x, double *y, double lr);

	double *predict(double *);
	int predictSoftMax(double *x);

	void makeLabels(int* pimax, double (*pplabels)[8]);
	void load(std::string weight, std::string bias);


	//本层前向传播的输出值，也是最终的预测值
	double* output_data;
	//反向传播时所需值
	double* delta;

public:
	int n_in;
	int n_out;
	int n_train;
	double** w;
	double* b;
private:
	double weight_decay;
};

void test_lr();
#endif


