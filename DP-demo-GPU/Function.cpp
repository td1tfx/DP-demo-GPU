#include "stdafx.h"
#include "Function.h"


Function::Function()
{
}


Function::~Function()
{
}

Matrix* Function::Sigmoid(Matrix* in_data) {
	for (int i = 0; i < in_data->size(); i++) {
		in_data->data()[i] = 1 / (1 + exp(-in_data->data()[i]));
	}
	return in_data;
}