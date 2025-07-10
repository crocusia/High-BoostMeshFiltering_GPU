#pragma once
#include "Mesh.h"
#include "HighboostKernel.h"
#include <stdio.h>

class HighboostSystem {
public:
	HighboostSystem();
	~HighboostSystem();
public:
	HighboostDevice _highboost;
	HighboostKernel _kernel;
public:
	int _numV;
	int _numF;
	int _numVnb;
	int _numFnb;

	int _iterate;
	double _learningrate;

public:
	void init();
	void setData(Mesh* _mesh); 
	void update();
	void applyMesh(Mesh* _mesh);
	void free();

};