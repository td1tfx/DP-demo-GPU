#include "stdafx.h"
#include "Matrix.h"



Matrix::Matrix()
{
	Matrix(3, 3);
}

Matrix::Matrix(int column_num, int row_num)
{
	m_row_num = row_num;
	m_column_num = column_num;
	m_size = m_row_num * m_column_num;
	m_data = new double[m_size];
	init(0);
}


Matrix::~Matrix()
{
	if (m_data != NULL) {
		delete[] m_data;
		m_data = NULL;
	}
}

double& Matrix::getData(int co_num, int r_num) {
	return m_data[m_column_num*r_num + co_num];
}

Matrix* Matrix::copy() {
	Matrix* matrix_new = new Matrix(m_column_num, m_row_num);
	cblas_dcopy(m_size, m_data, 1, matrix_new->data(), 1);
// 	for (int i = 0; i < m_size; i++) {
// 		matrix_new->data()[i] = m_data[i];
// 	}
	return matrix_new;
}

void Matrix::init(double num) {
	for (int i = 0; i < m_size; i++) {
		m_data[i] = num;
	}
}

void Matrix::randomInit(double max_num) {
	for (int i = 0; i < m_size; i++) {
		m_data[i] = (double)((rand() % (int)(max_num*1000))/1000.0000);
	}
}


void Matrix::add(Matrix* b) {
	cblas_daxpy(m_size, 1, m_data, 1, b->data(), 1);
}

void Matrix::addb(Matrix* b, Matrix* out, int batch_num) {
	

}

double Matrix::sum() {
	return cblas_dasum(m_size, m_data, 1);
}

