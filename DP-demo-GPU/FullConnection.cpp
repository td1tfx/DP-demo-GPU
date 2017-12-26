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
}

Matrix* FullConnection::forward(Matrix* in_data) {
	in_data->copyto(m_in_data);
	in_data->dot(m_w, m_out_data);
	m_out_data->addb(m_b, m_out_data,m_batch_num);
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
	return	m_loss->mul(m_loss)/m_loss->size()/2;
}

