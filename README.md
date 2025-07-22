⚠️ 본 저장소에는 지도 교수님께서 제공하신 3D Mesh 관련 코드가 비공개로 유지되어,<br> 
연구에서 사용된 전체 코드가 포함되어 있지 않습니다. <br>
이로 인해 본 저장소의 코드만으로는 프로젝트 전체를 실행할 수 없고,<br>
GPU 연산 및 관련 서브모듈만 포함되어 있음을 알려드립니다. <br>
<hr>

# 🗃️ Summary 
High-Boost Mesh Filtering의 문제점인 `Aliasing(계단현상)`과 `Mesh irreguralization(불규칙화)`을 개선하기 위해<br>
Mesh 형상을 강화하는데 사용하는 **Boosted Normal에 Bilateral Filter를 적용**한다.<br>

<img width="1842" height="565" alt="image" src="https://github.com/user-attachments/assets/3f7d09cd-5809-48df-9ce1-ea68ef62a701" />

- 기존 기술의 문제점 개선 및 안정적인 Mesh 강화
- GPU 기반 병렬화로 3배 이상 고속화


# 연구 목표
## High-Boost Mesh Filtering란?
2D 이미지 처리에서 사용되는 샤프닝 알고리즘을 3D Mesh에 적용하도록 고안된 알고리즘

![image](https://github.com/user-attachments/assets/05d91462-293f-40a4-a6bc-1d137d11596b)

Figure 1. Stanford Bunny model. (a) Original one, (b) Enhanced by the high-boost meshltering. <br><br>
`장점` : Mesh의 Triangle Face가 뒤집히는 현상 방지. <br>
`단점` : Mesh의 irregularization(불규칙성)과 특정 영역에서의 aliasing(계단현상) 발생 <br>
  
![image](https://github.com/user-attachments/assets/52f73b46-8d07-43f4-a8a7-14b92126bc1f)

Figure 2. Aliasing of (b)

## 💡연구의 필요성
Hirokazu Yagou외 2인은 필터링이 적용된 결과물에 Laplacian Smoothing으로 후처리를 하여 문제 개선 시도함<br>
따라서, High-Boost Mesh Filtering에서 **자체적으로 문제 현상을 방지 또는 개선하는 기능**이 필요

