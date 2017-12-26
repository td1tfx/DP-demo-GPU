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

void Matrix::copyto(Matrix* out) {
	cblas_dcopy(m_size, m_data, 1, out->data(), 1);

}

void Matrix::copyDatafrom(double* indata) {
	cblas_dcopy(m_size, indata, 1, m_data, 1);
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

void Matrix::printf() {
	for (int i = 0; i < m_row_num; i++) {
		for (int j = 0; j < m_column_num; j++) {
			std::cout << " " << m_data[i*m_column_num + j] ;
		}
		std::cout << std::endl;
	}
}

void Matrix::add(Matrix* b) {
	cblas_daxpy(m_size, 1, m_data, 1, b->data(), 1);
}

void Matrix::addb(Matrix* b, Matrix* out, int batch_num) {
	for (int i = 0; i < batch_num; i++) {
		cblas_daxpy(b->size(), 1, b->data(), 1, &out->data()[b->size()*i], 1);
	}
}

void Matrix::dot(Matrix* b, Matrix* out) {
	if (m_column_num != b->hight()) {
		std::cout << "Matrix dot error!! the hight of b is not march!" << std::endl;
	}
	if (m_row_num != out->hight()) {
		std::cout << "Matrix dot error!! the hight of out is not march!" << std::endl;
	}
	if (b->width() != out->width()) {
		std::cout << "Matrix dot error!! the width of b and out is not march!" << std::endl;
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m_row_num,b->width(),m_column_num,1,
		m_data, m_column_num,b->data(),b->width(),
		0,out->data(),out->width());
}

double Matrix::mul(Matrix* b) {
	if (b->size() != m_size) {
		std::cout << "Matrix multiply error!! the size of b is not march!" << std::endl;
	}
	return cblas_ddot(m_size, b->data(), 1, b->data(), 1);
}

double Matrix::sum() {
	return cblas_dasum(m_size, m_data, 1);
}

void Matrix::sigmoid() {
	for (int i = 0; i < m_size; i++) {
		m_data[i] = 1 / (1 + exp(-m_data[i]));
	}
}