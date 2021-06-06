# Maze-Solving

```bash
$python main.py --maze-path "./미로 1.txt" --size 32 --is-visible True
$python main.py --maze-path "./maze_16_16.txt" --size 16 --is-visible False
```


미로 찾기 문제를 풀기 위한 다양한 접근 방법 (알고리즘, ML/DL)

> 가운데 2 by 2가 goal이라고 가정, 이를 제외하고 랜덤하게 구조가 변경되는 것으로 생각

- 객체
    - 미로 생성
        1. `Room`
        2. `Maze`
        
    - 미로 학습
        1. `Agent`
- 학습용 코드
    1. 데이터셋 생성 (최적경로 탐색)
    2. 모델 생성 및 학습
    3. 테스트 코드

- 평가용 코드
    1. 미로 읽기: `read_maze()`
    2. 미로 탐색: `explore_maze()`
        - 실시간으로 보여줄 수 있어야함
        - 도착점 한곳에만 닿아도 
    3. 평가: `evaluate_solving()`
        - 총 걸린 시간 (회전, 이동 시간 고려)
        - 출발과 도착, 2phase에서 min 값으로 


# 1. 기능별 설명

## 1.1. 미로 학습 과정

### 미로 랜덤 생성 - `Maze`

### 미로 랜덤 생성 - `Agent`

### 최적 경로 탐색

### 실시간 업데이트

# 2. 미로 탐색 학습

## 2.1. 트리 서치

## 2.2. `ConvNet`

## 2.3. Q-Learning
