#pragma once

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
	Matrix( int column_num, int raw_num);	//row is the first order, default init is 0;
	~Matrix();
	
	double* data() { return m_data; }
	int size() { return m_size; }
	int hight() { return m_row_num; }
	int width() { return m_column_num; }

	void init(double num);
	void randomInit(double max_num);

	double& getData(int co_num, int r_num);

	//function
	Matrix* copy();
	void add(Matrix* b);
	void addb(Matrix* b, Matrix* out, int batch_num);	//matrix add b vector
	double sum();

};

