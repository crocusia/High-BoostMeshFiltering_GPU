#pragma once
#include "HighboostDevice.h"
#include <helper_math.h>
class HighboostKernel {
public:
	void calcBoost(HighboostDevice _hbd, int numBlocks, int numThreads, int sigmaDist, int sigmaValue);
	void calcGradient(HighboostDevice _hbd, int numBlocks, int numThreads, int iterate, double learningrate);
};