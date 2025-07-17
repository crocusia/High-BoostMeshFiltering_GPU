#include <stdlib.h>
#include <io.h>
#include <string>
#include <iostream>
#include "HighboostSystem.h"

#define ITERATE 100
#define LEARNINGRATE 0.001
#define MINBLOCKSIZE 32

double _zoom = 15.0f; // 화면 확대,축소
double _rot_x = 0.0f; // x축 회전
double _rot_y = 0.001f; // y축 회전
double _trans_x = 0.0f; // x축 이동
double _trans_y = 0.0f; // y축 이동
int _last_x = 0; // 이전 마우스 클릭 x위치
int _last_y = 0; // 이전 마우스 클릭 y위치
unsigned char _buttons[3] = { 0 }; // 마우스 상태(왼쪽,오른쪽,휠 버튼)

Mesh* _mesh;
HighboostSystem* _hbSystem = new HighboostSystem();
vector<string> _file;

int _renderMode = 0;

void Init(void)
{
	glEnable(GL_DEPTH_TEST);
}

void highboostGPU() {
	
	_mesh->forGPUdata();												//1.GPU에서 사용할 Vertex와 Face의 이웃 정보 준비

	_hbSystem->_numV = _mesh->_vertices.size();
	_hbSystem->_numF = _mesh->_faces.size();
	_hbSystem->_numVnb = _mesh->_VnbFace.size();
	_hbSystem->_numFnb = _mesh->_FnbFace.size();
	_hbSystem->_iterate = ITERATE;
	_hbSystem->_learningrate = LEARNINGRATE;

	_mesh->highBoostFilter_CPU(ITERATE, 5.0, ITERATE, LEARNINGRATE);	//2.CPU에서 BoostNormal까지 계산

	_hbSystem->init();													//3.메모리 할당
	printf("1.init end\n");

	_hbSystem->setData(_mesh);											//4.CPU 데이터를 GPU로 복사하기 위한 작업
	printf("2.setData end\n");
	
	_hbSystem->update();												//5.BoostNormal에 양바양 필터 적용
	printf("3.update end\n");											//  & 경사하강법으로 새로운 Vertex 위치 계산
	
	_hbSystem->applyMesh(_mesh);										//6.GPU 연산 결과를 CPU Mesh에 반영
	printf("4.apply gpu result end\n");
	
	_hbSystem->free();
	printf("5.free end\n");
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '1':
		highboostGPU();
		break;
	case 'Q':
	case 'q':
		exit(0);
	case 's':
	case 'S':
		_renderMode = 0;
		break;
	case 'w':
	case 'W':
		_renderMode = 1;
		break;
	case 'p':
	case 'P':
		_renderMode = 2;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void main(int argc, char** argv)
{
	_mesh = new Mesh("OBJ\\armadillo.obj");
	//readfile();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("HighBoost@Renew");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse); 
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard); 
	Init(); 
	glutMainLoop();
}