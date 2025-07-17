#include <stdlib.h>
#include <io.h>
#include <string>
#include <iostream>
#include "HighboostSystem.h"

#define ITERATE 100
#define LEARNINGRATE 0.001
#define MINBLOCKSIZE 32

double _zoom = 15.0f; // ȭ�� Ȯ��,���
double _rot_x = 0.0f; // x�� ȸ��
double _rot_y = 0.001f; // y�� ȸ��
double _trans_x = 0.0f; // x�� �̵�
double _trans_y = 0.0f; // y�� �̵�
int _last_x = 0; // ���� ���콺 Ŭ�� x��ġ
int _last_y = 0; // ���� ���콺 Ŭ�� y��ġ
unsigned char _buttons[3] = { 0 }; // ���콺 ����(����,������,�� ��ư)

Mesh* _mesh;
HighboostSystem* _hbSystem = new HighboostSystem();
vector<string> _file;

int _renderMode = 0;

void Init(void)
{
	glEnable(GL_DEPTH_TEST);
}

void highboostGPU() {
	
	_mesh->forGPUdata();												//1.GPU���� ����� Vertex�� Face�� �̿� ���� �غ�

	_hbSystem->_numV = _mesh->_vertices.size();
	_hbSystem->_numF = _mesh->_faces.size();
	_hbSystem->_numVnb = _mesh->_VnbFace.size();
	_hbSystem->_numFnb = _mesh->_FnbFace.size();
	_hbSystem->_iterate = ITERATE;
	_hbSystem->_learningrate = LEARNINGRATE;

	_mesh->highBoostFilter_CPU(ITERATE, 5.0, ITERATE, LEARNINGRATE);	//2.CPU���� BoostNormal���� ���

	_hbSystem->init();													//3.�޸� �Ҵ�
	printf("1.init end\n");

	_hbSystem->setData(_mesh);											//4.CPU �����͸� GPU�� �����ϱ� ���� �۾�
	printf("2.setData end\n");
	
	_hbSystem->update();												//5.BoostNormal�� ��پ� ���� ����
	printf("3.update end\n");											//  & ����ϰ������� ���ο� Vertex ��ġ ���
	
	_hbSystem->applyMesh(_mesh);										//6.GPU ���� ����� CPU Mesh�� �ݿ�
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