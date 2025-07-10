#include "HighboostKernel.h"

__device__ double GetNormD(double3 v) {
	double result = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	return result;
}

__device__ double DotD(double3 v1, double3 v2) {
	double result = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	return result;
}

__device__ double3 projectionD(double3 v, double3 b) {
	double norm = GetNormD(b);
	double dist = (v.x*b.x + v.y*b.y + v.z*b.z + norm) * -1;
	double k = dist / (norm * norm);
	double3 result;
	result.x = v.x + b.x * k;
	result.y = v.y + b.y * k;
	result.z = v.z + b.z * k;
	return result;
}

__device__ double3 sumVecD(double3 v1, double3 v2) {
	double3 result; 
	result.x = v1.x + v2.x;
	result.y = v1.y + v2.y;
	result.z = v1.z + v2.z;
	return result;
}

__device__ double3 subVecD(double3 v1, double3 v2) {
	double3 result;
	result.x = v1.x - v2.x;
	result.y = v1.y - v2.y;
	result.z = v1.z - v2.z;
	return result;
}

__device__ double calcAngleD(double3 vec1, double3 vec2) {
	double dot = DotD(vec1, vec2);
	double normVec1 = GetNormD(vec1);
	double normVec2 = GetNormD(vec2);
	double theta = dot / (normVec1 * normVec2);
	double result = acos(theta);
	return result;
}

__device__ int3 setPivot(int tid, int fid, int* _faces) {
	int index = fid * 3;
	int pivot = fid * 3;
	int3 result = make_int3(-1, -1, -1);
	for (int i = 0; i < 3; i++) {
		if (_faces[index + i] == tid) { 
			pivot = index + i; 
			break;
		}
	}
	
	int id0 = pivot % 3;
	int id1 = (pivot + 1) % 3;
	int id2 = (pivot + 2) % 3;

	result = make_int3(index + id0, index + id1, index + id2);
	return result;
}


__device__ double3 calcAreaD(double3 v1, double3 v2, double3 v3) {
	double3 vec1 = subVecD(v2, v1); //v2 - v1
	double3 vec2 = subVecD(v3, v1); //v3 - v1
	double3 vec3 = subVecD(v3, v2); //v3 - v2

	double3 m_vec1 = make_double3(vec1.x * -1, vec1.y * -1, vec1.z * -1);
	double3 m_vec2 = make_double3(vec2.x * -1, vec2.y * -1, vec2.z * -1);
	double3 m_vec3 = make_double3(vec3.x * -1, vec3.y * -1, vec3.z * -1);

	double cotA = 1 / tan(calcAngleD(m_vec2, m_vec3));
	double cotB = 1 / tan(calcAngleD(m_vec1, vec3));

	double3 result;
	result.x = (vec1.x * cotA + vec2.x * cotB) * -0.5;
	result.y = (vec1.y * cotA + vec2.y * cotB) * -0.5;
	result.z = (vec1.z * cotA + vec2.z * cotB) * -0.5;
	return result;
}

__device__ double3 calcdistD(int3 fid1, int3 fid2, double3* _vertex) {
	double3 result;

	double3 f1v1 = _vertex[fid1.x];
	double3 f1v2 = _vertex[fid1.y];
	double3 f1v3 = _vertex[fid1.z];

	double3 f2v1 = _vertex[fid2.x];
	double3 f2v2 = _vertex[fid2.y];
	double3 f2v3 = _vertex[fid2.z];

	double3 centerF1 = make_double3((f1v1.x + f1v2.x + f1v3.x) / 3, (f1v1.y + f1v2.y + f1v3.y) / 3, (f1v1.z + f1v2.z + f1v3.z) / 3);
	double3 centerF2 = make_double3((f2v1.x + f2v2.x + f2v3.x) / 3, (f2v1.y + f2v2.y + f2v3.y) / 3, (f2v1.z + f2v2.z + f2v3.z) / 3);

	result = subVecD(centerF1, centerF2);
	return result;
}
__device__ double gaussiandistD(double value, double sigma) {
	double result = 0.0;
	result = exp(-(value * value) / (2*sigma*sigma)) / (2*3.14*sigma*sigma);
	return result;
}
__device__ double3 gaussiandiffD(double3 value, double sigma) {
	double3 result = make_double3(0.0, 0.0, 0.0);
	result.x = exp(-(value.x * value.x) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
	result.y = exp(-(value.y * value.y) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
	result.z = exp(-(value.z * value.z) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
	return result;
}
__global__ void calcBilateralD(HighboostDevice _hbd, int sigmaDist, int sigmaValue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	double w = 0.0;
	double3 bilateral;

	int start = _hbd._dFnbStart[tid];
	int end = _hbd._dFnbEnd[tid];

	int3 _face = make_int3(_hbd._devFaces[tid * 3], _hbd._devFaces[tid * 3 + 1], _hbd._devFaces[tid * 3 + 2]);
	double3 _boost = _hbd._devBNorm[tid];

	for (int i = start; i < end + 1; i++) {
		int fid = _hbd._dFnbfaces[i];
		int3 _fnbf = make_int3(_hbd._devFaces[fid * 3], _hbd._devFaces[fid * 3 + 1], _hbd._devFaces[fid * 3 + 2]);
		double3 dist = calcdistD(_face, _fnbf, _hbd._devVerticies);
		double value = GetNormD(dist); 
		double g_dist = gaussiandistD(value, sigmaDist);

		double3 _nfboost = _hbd._devBNorm[fid];
		double3 diff = subVecD(_boost, _nfboost);
		diff.x = abs(diff.x);
		diff.y = abs(diff.y);
		diff.z = abs(diff.z);
		double3 g_diff = gaussiandiffD(diff, sigmaValue);

		double3 g = make_double3(g_diff.x * g_dist, g_diff.y * g_dist, g_diff.z * g_dist);
		double3 bg = make_double3(g.x * _nfboost.x, g.y * _nfboost.y, g.z * _nfboost.z);
		bilateral = sumVecD(bilateral, bg);

		w += GetNormD(g);
	}

	_hbd._devSmooth[tid] = make_double3(bilateral.x / w, bilateral.y / w, bilateral.z / w);
}

__global__ void updateBoostD(HighboostDevice _hbd) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	_hbd._devBNorm[tid] = _hbd._devSmooth[tid];
}

__global__ void calcGradientD(HighboostDevice _hbd) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;  //vertex index
	int start = _hbd._dVnbStart[tid];
	int end = _hbd._dVnbEnd[tid];
	int isStop = _hbd._isStop[tid];
	double3 result = make_double3(0.0, 0.0, 0.0);
	if (isStop == 0) {
		for (int i = start; i < end + 1; i++) {
			int fid = _hbd._dVnbfaces[i]; //nbface face index
			int3 pivot = setPivot(tid, fid, _hbd._devFaces);
			int3 vertex = make_int3(_hbd._devFaces[pivot.x], _hbd._devFaces[pivot.y], _hbd._devFaces[pivot.z]);
			double3 b = _hbd._devBNorm[fid];

			double3 v1 = _hbd._devVerticies[vertex.x];
			double3 v2 = _hbd._devVerticies[vertex.y];
			double3 v3 = _hbd._devVerticies[vertex.z];

			double3 p_v1 = projectionD(v1, b);
			double3 p_v2 = projectionD(v2, b);
			double3 p_v3 = projectionD(v3, b);

			double3 resultR = calcAreaD(v1, v2, v3);
			double3 resultS = calcAreaD(p_v1, p_v2, p_v3);
			double3 gradient;
			gradient = subVecD(resultR, resultS);
			result = sumVecD(result, gradient);
		}
		result.x = result.x * 2;
		result.y = result.y * 2;
		result.z = result.z * 2;

		_hbd._devGradient[tid] = result;
	}
}

__global__ void updatePosD(HighboostDevice _hbd, double learningrate) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double gNorm = GetNormD(_hbd._devGradient[tid]);
	double3 gradient;
	double3 pos;
	int isStop = _hbd._isStop[tid];
	
	//if (isStop == 0) {
		if (gNorm >= learningrate) {
			gradient = make_double3(_hbd._devGradient[tid].x * 0.01, _hbd._devGradient[tid].y * 0.01, _hbd._devGradient[tid].z * 0.01);
			pos = subVecD(_hbd._devVerticies[tid], gradient);
			_hbd._devVerticies[tid] = pos;
		}
		else {
			_hbd._isStop[tid] = 1;
		}
	//}
	_hbd._devGradient[tid] = make_double3(0.0, 0.0, 0.0);
}

void HighboostKernel::calcBoost(HighboostDevice _hbd, int numBlocks, int numThreads, int sigmaDist, int sigmaValue) {
	printf("Bilateral GPU\n");
	calcBilateralD << < numBlocks, numThreads >> > (_hbd, sigmaDist, sigmaValue);
	updateBoostD << < numBlocks, numThreads >> > (_hbd);
}

void HighboostKernel::calcGradient(HighboostDevice _hbd, int numBlocks, int numThreads, int iterate, double learningrate) {
	printf("Gradient GPU\n");
	for (int i = 0; i < iterate; i++) {
		calcGradientD <<< numBlocks, numThreads >>> (_hbd);
		updatePosD <<< numBlocks, numThreads >>> (_hbd, learningrate);
	}
}