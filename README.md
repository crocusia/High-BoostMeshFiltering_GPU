⚠️ 본 저장소에는 지도 교수님께서 제공하신 3D Mesh 관련 코드가 비공개로 유지되어,<br> 
연구에서 사용된 전체 코드가 포함되어 있지 않습니다. <br>
이로 인해 본 저장소의 코드만으로는 프로젝트 전체를 실행할 수 없고,<br>
GPU 연산 및 관련 서브모듈만 포함되어 있음을 알려드립니다. <br>
<hr>

# [ Summary ]
High-Boost Mesh Filtering의 문제점인 `Aliasing(계단현상)`과 `Mesh irreguralization(불규칙화)`을 개선하기 위해<br>
Mesh 형상을 강화하는데 사용하는 **Boosted Normal에 Bilateral Filter를 적용**한다.<br>

# [연구 목표]
## [High-Boost Mesh Filtering이란?]
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
