#pragma once
#include <iostream>
#include <stdlib.h>
#include <cblas.h>

class Matrix
{
private:
	int m_row_num;
	int m_column_num;
	int m_size;
	double* m_data;

public:
	Matrix();
	Matrix( int column_num, int raw_num, double init_num = 0);	//row is the first order, default init is 0;
	~Matrix();
	
	double* data() { return m_data; }
	int size() { return m_size; }
	int hight() { return m_row_num; }
	int width() { return m_column_num; }

	void init(double num);
	void randomInit(double max_num);
	void printf();

	double& getData(int co_num, int r_num);
	Matrix* copy();

	//function
	void copyto(Matrix* out);
	void copyDatafrom(double* in_data);
	void add(Matrix* b, double Alpha = 0);
	void addb(Matrix* b, Matrix* out, int batch_num);	//matrix add b vector
	void dot(Matrix* b, Matrix* out, bool a_trans = 0, bool b_trans = 0);
	void sigmoid();
	void elementMul(Matrix* b, Matrix* out);
	void mulNum(double x);
	double mulSum(Matrix* b);
	double sum(int length);
	void sumColumn(Matrix* out);
	void sumRow(Matrix* out);

};

