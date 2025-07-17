#include "Mesh.h"

Mesh::Mesh()
{
}

Mesh::~Mesh()
{
}

void Mesh::buildAdjacency(void)
{
	for (auto v : _vertices) {
		v->_nbFaces.clear();
		v->_nbVertices.clear();
	}

	// v-f
	for (auto f : _faces) {
		for (int j = 0; j < 3; j++) {
			f->_vertices[j]->_nbFaces.push_back(f);
		}
	}
	
	// v-v
	for (auto v : _vertices) {
		for (auto nf : v->_nbFaces) {
			auto pivot = nf->getIndex(v); // 0 : 1,2, 1 : 2,0, 2: 0,1
			int other_id0 = (pivot + 1) % 3;
			int other_id1 = (pivot + 2) % 3;
			if (!v->hasNbVertex(nf->_vertices[other_id0])) {
				v->_nbVertices.push_back(nf->_vertices[other_id0]);
			}
			if (!v->hasNbVertex(nf->_vertices[other_id1])) {
				v->_nbVertices.push_back(nf->_vertices[other_id1]);
			}
		}
	}

	// f-f 자기 자신을 포함함
	for (auto f : _faces) {
		for (int j = 0; j < 3; j++) {
			for (auto nf : f->_vertices[j]->_nbFaces) {
				f->_fnbFaces.push_back(nf);
			}
		}
		sort(f->_fnbFaces.begin(), f->_fnbFaces.end());
		f->_fnbFaces.erase(unique(f->_fnbFaces.begin(), f->_fnbFaces.end()), f->_fnbFaces.end());
	}

	printf("build adjacency list\n");
}

void Mesh::computeNormal(void)
{
	// f-normal
	for (auto f : _faces) {
		auto a = f->_vertices[0]->_pos;
		auto b = f->_vertices[1]->_pos;
		auto c = f->_vertices[2]->_pos;
		auto normal = (a - b).Cross(a - c);
		normal.Normalize();
		f->_normal = normal;
		f->_smooth = normal; //init smooth normal
	}

	// v-normal
	for (auto v : _vertices) {
		v->_normal.Clear();
		for (auto nf : v->_nbFaces) {
			v->_normal += nf->_normal;
		}
		v->_normal /= v->_nbFaces.size();
	}
	printf("compute normal\n");
}

//GPU에서 사용하기 위해 Vertex와 Face의 이웃 정보를 동적 배열에 별도로 저장
void Mesh::forGPUdata() {
	int startV = 0;
	for (auto v : _vertices) {
		_VnbStart.push_back(startV);
		for (auto n : v->_nbFaces) {
			_VnbFace.push_back(n->_index);
		}
		startV += v->_nbFaces.size();
		_VnbEnd.push_back(startV - 1);
	}
	int startF = 0;
	for (auto f : _faces) {
		_FnbStart.push_back(startF);
		for (auto n : f->_fnbFaces) {
			_FnbFace.push_back(n->_index);
		}
		startF += f->_fnbFaces.size();
		_FnbEnd.push_back(startF - 1);
	}
}

void Mesh::smoothNormal() {
	vector<Vec3<double>> smooths;
	for (auto f1 : _faces) {
		auto result = f1->calcSmooth();
		result.Normalize();
		smooths.push_back(result);
	}
	for (auto f2 : _faces) {
		f2->_smooth = smooths[f2->_index];
	}	
	smooths.clear();
}

//GPU와 비교를 위한 그래디언트 계산 함수
void Mesh::calcGradient_case(int iterate, double learningrate) {
	for (int i = 0; i < iterate; i++) {
		for (auto v : _vertices) {
			v->_gradient.Clear();
			if (v->_isStop == false) {
				for (auto nf : v->_nbFaces) {
					auto resultR = nf->calcAreaRi(v, i);
					auto resultS = nf->calcAreaSi(v, i);
					v->_gradient += (resultR - resultS);
				}
				v->_gradient = v->_gradient * 2;
	
			}
		}

		for (auto v : _vertices) {

			if (v->_gradient.GetNorm() >= learningrate) {
				v->_pos = v->_pos - v->_gradient * 0.01;
			}
			else {
				v->_isStop = true;
			}
		}
	}
}

void Mesh::highBoostFilter_CPU(int smoothK, double threshold, int iterate, double learningrate) {
	printf("highBoostFilter start\n");
	clock_t start0, start1, start2, start3, start4, finish0, finish1, finish2, finish3, finish4;
	double duration0, duration1, duration2, duration3, duration4;

	//0.calcArea
	start0 = clock();
	
	for (auto f : _faces) {
		f->computeArea();
		f->calcCenter();
	}
	finish0 = clock();
	duration0 = (double)(finish0 - start0) / CLOCKS_PER_SEC;
	printf("calcArea : %f sec\n", duration0);

	//1.smooth Normal 계산
	start1 = clock();
	for (int i = 0; i < smoothK; i++) {
		smoothNormal();	  
	}
	finish1 = clock();
	duration1 = (double)(finish1 - start1) / CLOCKS_PER_SEC;
	printf("calcSmoothNormal : %f sec\n", duration1);
	
	//2.boost Normal 계산
	start2 = clock();
	for (auto f : _faces) {
		f->calcBoost(threshold);
	}
	finish2 = clock();
	duration2 = (double)(finish2 - start2) / CLOCKS_PER_SEC;
	printf("calcBoostNormal : %f sec\n", duration2);
/*
#ifndef BILATERALGPU
	//3.boost Normal 과장
	start3 = clock();
	vector<Vec3<double>> bi_results;
	for (auto f : _faces) {
		auto b_result = f->bilateralFilter(50, 50);
		bi_results.push_back(b_result);
	}
	for (auto f : _faces) {
		f->_boost = bi_results[f->_index];
	}

	finish3 = clock();
	duration3 = (double)(finish3 - start3) / CLOCKS_PER_SEC;
	printf("bilateralFilter : %f sec\n", duration3);
#endif // !BILATERALGPU

#ifndef GRADIENTGPU
	start4 = clock();

	calcGradient_case(iterate, learningrate);

	finish4 = clock();
	duration4 = (double)(finish4 - start4) / CLOCKS_PER_SEC;
	printf("calcGradient : %f sec\n", duration4);

#endif // !GRADIENTGPU
*/
	printf("highBoostFilter CPU end\n");
}

void Mesh::drawSolid(void)
{
	glEnable(GL_LIGHTING);
	for (auto f : _faces) {
		glBegin(GL_POLYGON);
		glNormal3f(f->_normal.x(), f->_normal.y(), f->_normal.z());
		for (int j = 0; j < 3; j++) {
			glVertex3f(f->_vertices[j]->x(), f->_vertices[j]->y(), f->_vertices[j]->z());
		}
		glEnd();
	}
	glEnable(GL_LIGHTING);
}
