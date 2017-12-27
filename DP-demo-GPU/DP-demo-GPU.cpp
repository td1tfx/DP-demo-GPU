// DP-demo-GPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FullConnection.h"
#include <time.h>




int main()
{
	srand((unsigned)time(NULL));
	int in_num = 2;
	int out_num = 1;
	int hidden_num = 4;
	int batch_num = 4;
	double lr = 0.1;
	FullConnection fc(in_num, hidden_num, batch_num, lr);
	FullConnection fc1(hidden_num, hidden_num, batch_num, lr);
	FullConnection fc2(hidden_num, out_num, batch_num, lr);
	Matrix in_data(in_num,batch_num);
	Matrix bench_data(out_num, batch_num);
	auto in_data_t = new double[in_num*batch_num] {0,0,0,1,1,0,1,1};
	auto bench_data_t = new double[out_num*batch_num]{0,0,0,1};
	in_data.copyDatafrom(in_data_t);
	bench_data.copyDatafrom(bench_data_t);
	delete in_data_t;
	delete bench_data_t;

	std::cout << "input:" << std::endl;
	in_data.printf();

	std::cout << "benchput:" << std::endl;
	bench_data.printf();

	Matrix* out_data = NULL;
	double mse = 1;
	int run_num = 0;
	Matrix* loss_data;
	while (mse >= 0.001 && run_num < 10000) {
		out_data = fc.forward(&in_data);
		out_data = fc1.forward(out_data);
		out_data = fc2.forward(out_data);
		mse = fc2.squareLoss(&bench_data);
		std::cout << "mse = " << mse << std::endl;
		//std::cout << "LossMatrix:" << std::endl;
		//fc2.getLossMatrix()->printf();

		loss_data = fc2.getLossMatrix();
		loss_data = fc2.backward(loss_data);
		loss_data = fc1.backward(loss_data);
		loss_data = fc.backward(loss_data);
		run_num++;
	}
	std::cout << "output:" << std::endl;
	out_data->printf();
	system("pause");
	return 0;
}

