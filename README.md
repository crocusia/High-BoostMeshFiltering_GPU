⚠️ 본 저장소에는 지도 교수님께서 제공하신 3D Mesh 관련 코드가 비공개로 유지되어,<br> 
연구에서 사용된 전체 코드가 포함되어 있지 않습니다. <br>
이로 인해 본 저장소의 코드만으로는 프로젝트 전체를 실행할 수 없고,<br>
GPU 연산 및 관련 서브 모듈만 포함되어 있음을 알려드립니다. <br>
<hr>

# 🗃️ Summary 
High-Boost Mesh Filtering의 문제점인 `Aliasing(계단현상)`과 `Mesh irreguralization(불규칙화)`을 개선하기 위해<br>
Mesh 형상을 강화하는 데 사용하는 **Boosted Normal에 Bilateral Filter를 적용**한다.<br>

<img width="1842" height="565" alt="image" src="https://github.com/user-attachments/assets/3f7d09cd-5809-48df-9ce1-ea68ef62a701" />

Figure 1. Horse model. (a) Original one, (b) Enhanced by the high-boost mesh filtering, (c) Enhanced by the high-boost mesh filtering With Bilateral Filter

- 기존 기술의 문제점 개선 및 안정적인 Mesh 강화
- GPU 기반 병렬화로 3배 이상 고속화


# 연구 목표
## High-Boost Mesh Filtering란?
2D 이미지 처리에서 사용되는 샤프닝 알고리즘을 3D Mesh에 적용하도록 고안된 알고리즘

![image](https://github.com/user-attachments/assets/05d91462-293f-40a4-a6bc-1d137d11596b)

Figure 2. Stanford Bunny model. (a) Original one, (b) Enhanced by the high-boost meshltering. <br><br>
`장점` : Mesh의 Triangle Face가 뒤집히는 현상 방지. <br>
`단점(1)` : Mesh의 irregularization(불규칙성)과 **Figure 2처럼 특정 영역에서의 aliasing(계단현상)** 발생<br>
`단점(2)` : **고해상도** Mesh일수록 CPU 기반 **연산 시간 증가**
  
![image](https://github.com/user-attachments/assets/52f73b46-8d07-43f4-a8a7-14b92126bc1f)

Figure 3. Aliasing of (b)

## 💡연구의 필요성
Hirokazu Yagou 외 2인은 필터링이 적용된 결과물에 Laplacian Smoothing으로 후처리를 하여 문제 개선 시도함<br>
따라서, High-Boost Mesh Filtering에서 **자체적으로 문제 현상을 방지 또는 개선하는 기능**이 필요

<hr>

# 제안 기법
`Aliasing 발생 원인` : Mesh 형태 강화 시, **잡음(noise)도 함께 과장**됨

## 양방향 필터 Bilateral Filter란?
- 2D 이미지의 잡음 제거에 사용되는 알고리즘
- 픽셀 간 거리, 픽셀 값 차이 두 가지를 고려하여 필터링
- Edge는 보존하고, Edge가 아닌 부분은 블러링

### 💡양방향 필터 적용 이유
- High-Boost Mesh Filtering이 2D 기반 알고리즘을 3D로 확장한 기법이므로,<br>
2D에서 사용하는 잡음 제거 방법을 적용하는 것이 효과적일 것이라 판단하였다.
- 잡음 제거 알고리즘 중 **Edge 보존 가능 여부**를 중점으로 비교 후, Bilateral Filter 선정

## ✅ 적용 방법
Mesh 강화를 위해 Normal Vector를 k번의 Smoothing하여 계산한 Boosted Normal Vector에 Bilateral Filter 적용
- 픽셀 간 거리 ➡️ Face의 중심 좌표 간 거리
- 픽셀 값 차이 ➡️ Face의 Boosted Normal 차이

<img width="2401" height="931" alt="image" src="https://github.com/user-attachments/assets/250f2d23-7f7f-4ae0-a94e-9b4e4f472457" />
Figure 4. Proposed Method Overview and Bilateral Filtering Integration Point

<hr>

# 결과


