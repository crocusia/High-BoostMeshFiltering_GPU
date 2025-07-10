#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <vector>
#include <math.h>
using namespace std;

class HighboostDevice
{
public:
	double3* _vertices; //vertex À§Ä¡
	double3* _devVerticies;
	int* _faces;
	int* _devFaces;
	int* _isStop;
	
	int* _Vnbfaces; 
	int* _VnbStart; 
	int* _VnbEnd; 
	int* _dVnbfaces;
	int* _dVnbStart;
	int* _dVnbEnd;

	int* _Fnbfaces;
	int* _FnbStart;
	int* _FnbEnd;
	int* _dFnbfaces;
	int* _dFnbStart;
	int* _dFnbEnd;

	double3* _bNorm;
	double3* _devSmooth;
	double3* _devBNorm;
	double3* _devGradient;
	
public:
	void init(int numV, int numF, int numVnb, int numFnb);
	void initGPU(int numThreadV, int numV, int numF, int numVnb, int numFnb);
	void free();
	void copyHtoD(int numV, int numF, int numVnb, int numFnb);
	void copyDtoH(int numV, int numF, int numVnb, int numFnb);
};

