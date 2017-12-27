#include "stdafx.h"
#include "FullConnection.h"



FullConnection::FullConnection()
{
	FullConnection(3, 3, 3, 0.1);
}

FullConnection::FullConnection(int in_num_t, int out_num_t, int batch_num_t, double lr = 0.01)
{
	m_in_num = in_num_t;
	m_out_num = out_num_t;
	m_batch_num = batch_num_t;
	m_lr = lr;
	m_in_data = new Matrix(m_in_num,m_batch_num);
	m_out_data = new Matrix(m_out_num, m_batch_num);
	m_w = new Matrix(m_out_num, m_in_num);
	m_w->randomInit(1);
	m_b = new Matrix(m_out_num, 1);
	m_loss = new Matrix(m_out_num, m_batch_num);
	m_residual_z = new Matrix(m_out_num, m_batch_num);
	m_residual_x = new Matrix(m_in_num, m_batch_num);
	m_grad_w = new Matrix(m_out_num, m_in_num);
	m_grad_b = new Matrix(m_out_num, 1);
}


FullConnection::~FullConnection()
{
	if (m_in_data != NULL) {
		delete m_in_data;
		m_in_data = NULL;
	}
	if (m_out_data != NULL) {
		delete m_out_data;
		m_out_data = NULL;
	}
	if (m_w != NULL) {
		delete m_w;
		m_w = NULL;
	}
	if (m_b != NULL) {
		delete m_b;
		m_b = NULL;
	}
	if (m_loss != NULL) {
		delete m_loss;
		m_loss = NULL;
	}
	if (m_residual_x != NULL) {
		delete m_residual_x;
		m_residual_x = NULL;
	}
	if (m_residual_z != NULL) {
		delete m_residual_z;
		m_residual_z = NULL;
	}
	if (m_grad_w != NULL) {
		delete m_grad_w;
		m_grad_w = NULL;
	}
	if (m_grad_b != NULL) {
		delete m_grad_b;
		m_grad_b = NULL;
	}
}

Matrix* FullConnection::forward(Matrix* in_data) {
	in_data->copyto(m_in_data);
	in_data->dot(m_w, m_out_data);
	m_out_data->addb(m_b, m_out_data, m_batch_num);
	m_out_data->sigmoid();
	return m_out_data;
}

double FullConnection::squareLoss(Matrix* bench_data) {
	if (m_loss->size() != bench_data->size()) {
		std::cout << "SquareLoss error!! the size of output_data and bench_data is not euqal!" << std::endl;
	}
	for (int i = 0; i < m_loss->size(); i++) {
		m_loss->data()[i] = m_out_data->data()[i] - bench_data->data()[i];
	}
	return	m_loss->mulSum(m_loss)/m_loss->size()/2;
}

Matrix* FullConnection::backward(Matrix* loss) {
	loss->elementMul(m_out_data, m_residual_z);
// 	loss->printf();
// 	m_out_data->printf();
// 	m_residual_z->printf();
	Matrix loss_t(loss->width(), loss->hight(), 1);
	m_out_data->add(&loss_t,-1);
	m_residual_z->elementMul(&loss_t, m_residual_z);
	m_in_data->dot(m_residual_z, m_grad_w, 1, 0);
// 	m_in_data->printf();
// 	m_residual_z->printf();
// 	m_grad_w->printf();
	m_residual_z->sumRow(m_grad_b);
// 	m_residual_z->printf();
// 	m_grad_b->printf();
// 	m_grad_w->printf();
	m_grad_w->mulNum(m_lr);
	m_grad_w->add(m_w, -1);
	m_grad_b->mulNum(m_lr);
	m_grad_b->add(m_b, -1);
	m_residual_z->dot(m_w, m_residual_x,0,1);
// 	m_residual_z->printf();
// 	m_w->printf();
// 	m_residual_x->printf();
	return m_residual_x;
}