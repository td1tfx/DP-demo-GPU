#pragma once
#include "Matrix.h"
#include <cblas.h>

class FullConnection
{
private:
	int m_in_num;
	int m_out_num;
	int m_batch_num;
	double m_lr;

	Matrix* m_in_data;
	Matrix* m_out_data;
	Matrix* m_w;
	Matrix* m_b;
	Matrix* m_loss;
	Matrix* m_residual_z;
	Matrix* m_residual_x;
	Matrix* m_grad_w;
	Matrix* m_grad_b;

public:
	FullConnection();
	FullConnection(int in_num_t, int out_num_t, int batch_num_t, double lr);
	~FullConnection();

	Matrix* forward(Matrix* in_data);
	Matrix* backward(Matrix* loss);
	Matrix* getLossMatrix() { return m_loss; };
	double squareLoss(Matrix* bench_data);

};

