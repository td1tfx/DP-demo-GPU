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
	m_in_data = in_data->copy();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m_batch_num, m_out_num, m_in_num, 1,
		m_in_data->data(), m_in_num, m_w->data(), m_out_num,
		0, m_out_data->data(), m_batch_num);

}


