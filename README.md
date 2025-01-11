# [요약]

본 프로젝트는 High-Boost Mesh Filtering에 Bilateral Filter를 적용하여 Mesh irreguralization을 해결함으로써 Mesh의 과장을 안정적으로 표현할 수 있도록 하였다.
또한 Mesh Saliency를 기반으로, Saliency 방향성을 기존 필터링에 통합하여 aliasing 발생을 방지 및 완화하였으며 이를 GPU 기반에서 설계하여 고속화하였다.
(GPU 기반 코드는 별도로 정리하였다.)

# [선행 연구]
[High-Boost Mesh Filtering이란?]
2D 이미지 처리에서 사용되는 샤프닝 알고리즘을 3D Mesh에 적용하도록 고안된 알고리즘이다.

![image](https://github.com/user-attachments/assets/05d91462-293f-40a4-a6bc-1d137d11596b)

Figure 1. Stanford Bunny model. (a) Original one, (b) Enhanced by the high-boost meshltering.
- Mesh의 Triangle Face가 뒤집히는 현상을 방지한다. 
- Mesh의 irregularization(불규칙성)과 특정 영역에서의 aliasing(계단현상)이 발생하는 문제가 있다.
  
![image](https://github.com/user-attachments/assets/52f73b46-8d07-43f4-a8a7-14b92126bc1f)

Figure 2. Aliasing of (b)

Hirokazu Yagou외 2인은 필터링이 적용된 결과물에 Laplacian Smoothing으로 후처리를 하여 문제를 해결했으나, 필터링 자체에서 이 문제 현상을 방지 또는 개선하는 기능이 필요하다.



### [개선사항(1) Bilateral Filter 적용 과정]
Mesh irreguralization이 발생하는 원인은 곡률을 강화한 Boosted Normal Vector을 계산할 때 Mesh의 잡음도 함께 강화하기 때문이라고 추론하였다.
잡음을 제거할 때 주로 사용되는  _Bilateral Filter(양방향 필터)_ 를 Boosted Normal Vector에 적용하여 잡음을 제거하고자 하였다. 
양방향 필터는 가우시안 필터를 픽셀의 거리 차와 두 픽셀의 색상값의 차에 따라 총 두 번 실행한다.
픽셀의 거리 차이를 Triangle Face의 중심 좌표 간의 거리 차이로, 픽셀의 색상값 차이를 Boosted Normal Vector에 대해 
