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
