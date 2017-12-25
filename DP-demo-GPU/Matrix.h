#pragma once
class Matrix
{
private:
	int m_row_num;
	int m_column_num;
	int m_size;
	double* m_data;

public:
	Matrix();
	Matrix( int column_num, int raw_num);	//row is the first order
	~Matrix();
	
	double* data() { return m_data; }
	int size() { return m_size; }
	int hight() { return m_row_num; }
	int width() { return m_column_num; }

	double& getData(int co_num, int r_num);

};

