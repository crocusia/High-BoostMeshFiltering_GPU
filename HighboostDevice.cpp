#include "HighboostDevice.h"

void HighboostDevice::init(int numV, int numF, int numVnb, int numFnb) {
	_vertices = new double3[numV];
	_faces = new int[numF * 3];
	_Vnbfaces = new int[numVnb];
	_VnbStart = new int[numV];
	_VnbEnd = new int[numV];
	_Fnbfaces = new int[numFnb];
	_FnbStart = new int[numF];
	_FnbEnd = new int[numF];
	_bNorm = new double3[numF];
}

void HighboostDevice::initGPU(int numThreadV, int numV, int numF, int numVnb, int numFnb) {
	checkCudaErrors(cudaMalloc((void**)&_devVerticies, sizeof(double3) * numThreadV));
	checkCudaErrors(cudaMalloc((void**)&_devFaces, sizeof(int) * numF * 3));

	checkCudaErrors(cudaMalloc((void**)&_dVnbfaces, sizeof(int) * numVnb));
	checkCudaErrors(cudaMalloc((void**)&_dVnbStart, sizeof(int) * numV));
	checkCudaErrors(cudaMalloc((void**)&_dVnbEnd, sizeof(int) * numV));	
	
	checkCudaErrors(cudaMalloc((void**)&_dFnbfaces, sizeof(int) * numFnb));
	checkCudaErrors(cudaMalloc((void**)&_dFnbStart, sizeof(int) * numF));
	checkCudaErrors(cudaMalloc((void**)&_dFnbEnd, sizeof(int) * numF));

	checkCudaErrors(cudaMalloc((void**)&_devSmooth, sizeof(double3) * numF));
	checkCudaErrors(cudaMalloc((void**)&_devBNorm, sizeof(double3) * numF));
	checkCudaErrors(cudaMalloc((void**)&_devGradient, sizeof(double3) * numV));
	checkCudaErrors(cudaMalloc((void**)&_isStop, sizeof(int) * numV));
	checkCudaErrors(cudaMemset(_isStop, 0, sizeof(int) * numV));
}

void HighboostDevice::free() {
	delete[] _vertices;
	delete[] _faces;
	delete[] _bNorm;

	delete[] _Vnbfaces;
	delete[] _VnbStart;
	delete[] _VnbEnd;	
	
	delete[] _Fnbfaces;
	delete[] _FnbStart;
	delete[] _FnbEnd;

	checkCudaErrors(cudaFree(_devFaces));

	checkCudaErrors(cudaFree(_dVnbfaces));
	checkCudaErrors(cudaFree(_dVnbEnd));
	checkCudaErrors(cudaFree(_dVnbStart));	
	
	checkCudaErrors(cudaFree(_dFnbfaces));
	checkCudaErrors(cudaFree(_dFnbEnd));
	checkCudaErrors(cudaFree(_dFnbStart));

	checkCudaErrors(cudaFree(_devSmooth));
	checkCudaErrors(cudaFree(_devBNorm));
	checkCudaErrors(cudaFree(_devGradient));
	checkCudaErrors(cudaFree(_devVerticies));
	checkCudaErrors(cudaFree(_isStop));
}

void HighboostDevice::copyHtoD(int numV, int numF, int numVnb, int numFnb) {
	checkCudaErrors(cudaMemcpy(_devVerticies, _vertices, numV * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_devFaces, _faces, numF * sizeof(int) * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_devBNorm, _bNorm, sizeof(double3) * numF, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_dVnbfaces, _Vnbfaces, sizeof(int) * numVnb, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dVnbStart, _VnbStart, sizeof(int) * numV, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dVnbEnd, _VnbEnd, sizeof(int) * numV, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_dFnbfaces, _Fnbfaces, sizeof(int) * numFnb, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dFnbStart, _FnbStart, sizeof(int) * numF, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dFnbEnd, _FnbEnd, sizeof(int) * numF, cudaMemcpyHostToDevice));
}

void HighboostDevice::copyDtoH(int numV, int numF, int numVnb, int numFnb) {
	checkCudaErrors(cudaMemcpy(_vertices, _devVerticies, numV * sizeof(double3), cudaMemcpyDeviceToHost));
}