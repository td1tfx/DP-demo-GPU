#pragma once
#include "Matrix.h"
#include <cblas.h>
#include <math.h>

class Function
{
public:
	Function();
	~Function();

	static Matrix* Sigmoid(Matrix* in_data);
};

