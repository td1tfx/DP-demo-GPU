// DP-demo-GPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cblas.h"
#include <stdlib.h>
#include <iostream>



int main()
{
	float a[1] = { 2 };
	float b[1] = { 3 };
	float c[1] = { 0 };
	cblas_sgemm(CblasRowMajor,
		CblasNoTrans, CblasNoTrans,
		1, 1, 1,
		1,
		a, 1,
		b, 1,
		0,
		c, 1);
	std::cout << *c << std::endl;
	system("pause");
    return 0;
}

