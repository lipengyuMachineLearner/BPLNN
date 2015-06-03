#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

class HiddenLayer{
public:
	HiddenLayer(int n_i, int n_o, int type);
	~HiddenLayer();

	void forward_propagation(double* input_data);
	void back_propagation(double *input_data, double *next_layer_delta, double** next_layer_w, int next_layer_n_out, double lr, int N);

	//本层前向传播的输出值,作为下一层的输入值
	double* output_data;
	//反向传播时所需值
	double* delta;

	int activityFunctionType;

public:
	int n_in;
	int n_out;
	double** w;
	double*b;
};

#endif
