#include "HighboostSystem.h"

HighboostSystem::HighboostSystem() {}
HighboostSystem::~HighboostSystem() {}

int setBlockSize(int n, int blocksize) {
	int size = blocksize; // i = 64
	int diff = blocksize;
	int result;

	if (n > size) {
		for (int i = size; i < 513; i += size) { // i = 64 i+= 64 , 64 128 192 256 320 384 448 512
			int numB = n / i + 1;
			int r = numB * i - n;
			if (r < diff) {
				result = i;
			}
		}
		return result;
	}
	return n;
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeGridSize(int n, int blockSize, int& numBlocks, int& numThreads)
{
	numThreads = setBlockSize(n, blockSize);
	numBlocks = iDivUp(n, numThreads);
	printf("numBlocks %d numThreads %d numData %d diff %d\n", numThreads, numBlocks, numThreads * numBlocks, numThreads * numBlocks - n);
}

void HighboostSystem::init() {
	_highboost.init(_numV, _numF, _numVnb, _numFnb);
	int numThreadV, numBlockV;
	computeGridSize(_numV, 32, numBlockV, numThreadV);
	_highboost.initGPU(numThreadV*numBlockV, _numV, _numF, _numVnb, _numFnb);
}

void HighboostSystem::setData(Mesh* _mesh) {
	for (int i = 0; i < _numV; i++) {
		_highboost._vertices[i] = make_double3(_mesh->_vertices[i]->_pos.x(), _mesh->_vertices[i]->_pos.y(), _mesh->_vertices[i]->_pos.z());
		_highboost._VnbStart[i] = _mesh->_VnbStart[i];
		_highboost._VnbEnd[i] = _mesh->_VnbEnd[i];
	}
	for (int j = 0; j < _numF; j++) {
		_highboost._faces[j * 3] = _mesh->_faces[j]->_vertices[0]->_index;
		_highboost._faces[j * 3 + 1] = _mesh->_faces[j]->_vertices[1]->_index;
		_highboost._faces[j * 3 + 2] = _mesh->_faces[j]->_vertices[2]->_index;
		_highboost._FnbStart[j] = _mesh->_FnbStart[j];
		_highboost._FnbEnd[j] = _mesh->_FnbEnd[j];
		_highboost._bNorm[j] = make_double3(_mesh->_faces[j]->_boost.x(), _mesh->_faces[j]->_boost.y(), _mesh->_faces[j]->_boost.z());
	}
	for (int m = 0; m < _numVnb; m++) {
		_highboost._Vnbfaces[m] = _mesh->_VnbFace[m];
	}

	for (int n = 0; n < _numFnb; n++) {
		_highboost._Fnbfaces[n] = _mesh->_FnbFace[n];
	}
}

void HighboostSystem::free() {
	_highboost.free(); 
}

void HighboostSystem::update() {
	_highboost.copyHtoD(_numV, _numF, _numVnb, _numFnb);				//CPU -> GPU
//#ifdef BILATERALGPU
	int numThreadF, numBlockF;
	computeGridSize(_numF, 32, numBlockF, numThreadF);
	_kernel.calcBoost(_highboost, numBlockF, numThreadF, 70, 70);		//BoostNormal에 양방향 필터 적용
//#endif

//#ifdef GRADIENTGPU
	int numThreadV, numBlockV;
	computeGridSize(_numV, 32, numBlockV, numThreadV);
	_kernel.calcGradient(_highboost, numBlockV, numThreadV,_iterate, _learningrate); //경사하강법으로 새로운 위치 계산
//#endif

	_highboost.copyDtoH(_numV, _numF, _numVnb, _numFnb);				//GPU -> CPU
}

void HighboostSystem::applyMesh(Mesh* _mesh) {
	for (int i = 0; i < _numV; i++) {
		double3 p = _highboost._vertices[i];
		_mesh->_vertices[i]->_pos.Set(p.x, p.y, p.z);
	}
	_mesh->computeNormal();
}

