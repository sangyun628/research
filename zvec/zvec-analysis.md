# Zvec (Alibaba) 오픈소스 벡터 데이터베이스 심층 분석

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **리포지토리** | https://github.com/alibaba/zvec |
| **라이선스** | Apache 2.0 |
| **주요 언어** | C++ (81.3%), Python (7.8%), SWIG (8.4%) |
| **빌드 시스템** | CMake (C++) + scikit-build-core (Python) |
| **C++ 표준** | C++17 |
| **GitHub Stars** | 8.6k+ |

**Zvec**는 Alibaba의 프로덕션 벡터 검색 엔진 **Proxima**를 기반으로 구축된 경량, 고성능, 인-프로세스(in-process) 벡터 데이터베이스다. 별도의 서버 구성 없이 애플리케이션 내에 직접 임베딩되어 수십억 개의 벡터에 대해 밀리초 단위의 유사도 검색을 제공한다.

### 핵심 특징

- **초고속 검색**: 수십억 벡터에 대한 밀리초 단위 검색
- **Zero-Config**: 서버/설정 없이 설치 즉시 사용 가능
- **Dense + Sparse 벡터**: 밀집/희소 벡터 모두 지원 및 멀티벡터 쿼리
- **하이브리드 검색**: 시맨틱 유사도 + 구조화된 필터 결합
- **크로스 플랫폼**: Linux (x86_64, ARM64), macOS (ARM64)
- **다국어 바인딩**: Python (3.10-3.12), Node.js

---

## 2. 전체 아키텍처

Zvec는 명확하게 분리된 **3-Tier 모듈 아키텍처**를 채택한다.

```
┌──────────────────────────────────────────────────────────────┐
│                    사용자 애플리케이션                         │
│              (Python / Node.js / C++)                         │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│              Python Binding Layer (pybind11)                  │
│         _zvec 모듈: Collection, Doc, Schema, Params          │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                   DB Layer (src/db/)                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Collection│  │ Segment  │  │  WAL     │  │  SQL Engine  │  │
│  │ Manager  │  │ Manager  │  │  (Write  │  │  (ANTLR      │  │
│  │          │  │          │  │   Ahead  │  │   Parser +   │  │
│  │          │  │          │  │   Log)   │  │   Planner)   │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                Core Engine (src/core/)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │   FLAT   │  │   IVF    │  │   HNSW   │  │  Quantizer   │ │
│  │  (Brute  │  │(Inverted │  │(Graph    │  │ (Int4/Int8/  │ │
│  │  Force)  │  │  File)   │  │ Search)  │  │  FP16/PQ)    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Metric  │  │  Mixed   │  │ Storage  │                   │
│  │(L2/IP/   │  │ Reducer  │  │(MMAP/    │                   │
│  │ Cosine)  │  │          │  │ Buffer)  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│             AiLego Foundation Library (src/ailego/)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ SIMD     │  │ Thread   │  │Container │  │  Algorithm   │ │
│  │ Math     │  │  Pool    │  │(Bitmap,  │  │ (K-Means,    │ │
│  │ (SSE/    │  │ Parallel │  │ Bloom    │  │  Quantizer)  │ │
│  │  AVX)    │  │          │  │ Filter)  │  │              │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 디렉토리 구조

```
zvec/
├── src/                           # C++ 소스 코드 (507 파일)
│   ├── ailego/                    # 기반 유틸리티 라이브러리
│   │   ├── algorithm/             #   K-Means, 양자화 알고리즘
│   │   ├── container/             #   Bitmap, BloomFilter, Reservoir
│   │   ├── math/                  #   SIMD 거리 계산 (46+ 파일)
│   │   ├── math_batch/            #   배치 거리 연산
│   │   ├── parallel/              #   SpinMutex, ThreadPool, Semaphore
│   │   ├── internal/              #   CPU Feature Detection (SSE/AVX)
│   │   ├── io/                    #   File I/O
│   │   ├── pattern/               #   RAII ScopeGuard, Closure
│   │   └── utility/               #   Memory/File/Concurrency Helper
│   │
│   ├── core/                      # 벡터 검색 엔진 코어
│   │   ├── algorithm/             #   인덱스 알고리즘
│   │   │   ├── flat/              #     Flat (Brute Force) 검색
│   │   │   ├── flat_sparse/       #     Sparse Flat 검색
│   │   │   ├── hnsw/              #     HNSW 그래프 검색
│   │   │   ├── hnsw_sparse/       #     Sparse HNSW 검색
│   │   │   ├── ivf/               #     IVF (Inverted File) 검색
│   │   │   └── cluster/           #     클러스터링
│   │   ├── framework/             #   Builder/Searcher/Streamer 추상화
│   │   ├── interface/             #   공용 인터페이스 (Index, Query)
│   │   ├── metric/                #   거리 메트릭 구현체
│   │   ├── quantizer/             #   양자화 엔진
│   │   └── utility/               #   코어 유틸리티
│   │
│   ├── db/                        # 데이터베이스 엔진
│   │   ├── index/                 #   인덱스 구조
│   │   │   ├── column/            #     컬럼 기반 스토리지
│   │   │   │   ├── vector_column/ #       벡터 인덱스 컬럼
│   │   │   │   └── inverted_column/#      스칼라 역인덱스 (RocksDB)
│   │   │   ├── segment/           #     세그먼트 관리
│   │   │   ├── storage/           #     스토리지 레이어
│   │   │   │   └── wal/           #       Write-Ahead Log
│   │   │   └── common/            #     ID Map, Delete Store
│   │   ├── sqlengine/             #   SQL 쿼리 엔진
│   │   │   ├── antlr/             #     ANTLR 문법 정의
│   │   │   ├── parser/            #     쿼리 파서
│   │   │   ├── analyzer/          #     의미 분석기
│   │   │   └── planner/           #     쿼리 플래너 및 실행 노드
│   │   ├── proto/                 #   Protocol Buffers 스키마
│   │   └── common/                #   상수, 공통 유틸리티
│   │
│   └── binding/python/            # pybind11 바인딩 코드
│
├── python/                        # Python 패키지
│   ├── zvec/                      #   메인 패키지
│   │   ├── model/                 #     Collection, Doc, Schema
│   │   ├── typing/                #     타입 정의 (Enum)
│   │   ├── extension/             #     임베딩/리랭킹 함수
│   │   └── executor/              #     쿼리 실행기
│   └── tests/                     #   Python 테스트 (57 파일)
│
├── include/                       # 공용 C++ 헤더
├── tests/                         # C++ 테스트 (Google Test)
├── examples/c++/                  # C++ 예제 코드
├── tools/core/                    # 벤치마크/프로파일링 도구
├── thirdparty/                    # 서드파티 의존성 (12개 서브모듈)
├── cmake/                         # CMake 빌드 설정
├── CMakeLists.txt                 # 루트 CMake
└── pyproject.toml                 # Python 빌드 설정
```

---

## 4. 핵심 벡터 인덱스 알고리즘

Zvec의 코어 엔진은 3가지 주요 벡터 인덱스 타입을 지원하며, 각각 Dense와 Sparse 변형을 포함한다.

### 4.1 FLAT Index (Brute Force)

가장 단순하면서도 정확한 검색 방식으로, 모든 벡터를 순차적으로 스캔하여 정확한 결과를 반환한다.

```
┌─────────────────────────────────────────┐
│            FLAT Index                    │
│                                          │
│  [v₀] [v₁] [v₂] [v₃] ... [vₙ]        │
│    ↕    ↕    ↕    ↕        ↕            │
│  query와 모든 벡터 간 거리 계산          │
│  → Top-K 결과 반환                       │
└─────────────────────────────────────────┘
```

**특징**:
- 소규모 데이터셋이나 정확한 recall이 필요한 경우에 적합
- 데이터 구조 오버헤드 없음
- Row-Major 또는 Column-Major 스토리지 지원 (캐시 효율)
- `key → node_id` 매핑을 `unordered_map`으로 관리

**핵심 클래스**:
```cpp
FlatSearcher<BATCH_SIZE>    // 전체 스캔 검색
FlatBuilder<BATCH_SIZE>     // 벡터 레이아웃 구성
```

### 4.2 IVF Index (Inverted File)

2단계 계층적 검색을 통해 검색 범위를 대폭 줄이는 방식이다.

```
┌──────────────────────────────────────────────────────┐
│                    IVF Index                          │
│                                                      │
│  [Level 1: Coarse Quantization]                      │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐        ┌────┐         │
│  │ C₀ │ │ C₁ │ │ C₂ │ │ C₃ │  ...   │ Cₖ │        │
│  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘        └──┬─┘         │
│     │      │      │      │              │            │
│  [Level 2: Inverted Lists]                           │
│  ┌──┴──┐ ┌─┴──┐ ┌─┴──┐ ┌─┴──┐     ┌──┴──┐         │
│  │v₀,v₁│ │v₃  │ │v₅,v₆│ │v₇,v₈│    │vₙ   │        │
│  │v₂   │ │v₄  │ │     │ │v₉  │    │     │         │
│  └─────┘ └────┘ └─────┘ └────┘     └─────┘         │
│                                                      │
│  Query → nprobe개 센트로이드 선택 → 해당 리스트 검색  │
└──────────────────────────────────────────────────────┘
```

**검색 알고리즘 흐름**:
1. 쿼리 벡터를 변환 (L2 norm → Inner Product 변환)
2. `nprobe`개의 가장 가까운 센트로이드 탐색
3. 해당 Inverted List 내에서 벡터 검색
4. 결과 병합 및 리랭킹

**핵심 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `nlist` | 1024 | 센트로이드(클러스터) 수 |
| `niters` | 10 | K-Means 반복 횟수 |
| `nprobe` | - | 검색 시 탐색할 클러스터 수 |

**스토리지 포맷**:
```cpp
// Inverted List 헤더
struct InvertedIndexHeader {
    uint32_t total_vector_count;     // 전체 벡터 수
    uint32_t inverted_list_count;    // nlist (클러스터 수)
    uint32_t block_vector_count;     // 블록당 벡터 수
    uint32_t block_size;             // 블록 데이터 크기 (bytes)
};

// 각 Inverted List의 메타데이터
struct InvertedListMeta {
    uint64_t offset;                 // 세그먼트 내 데이터 오프셋
    uint32_t block_count;            // 블록 수
    uint32_t vector_count;           // 리스트 내 벡터 수
};
```

**빌드 워크플로**:
```
1. train()
   ├─ 벡터 샘플링
   ├─ K-Means 클러스터링 실행
   └─ 센트로이드 벡터 학습

2. build()
   ├─ 벡터 → 센트로이드 할당 (병렬 처리)
   ├─ 센트로이드별 벡터 그루핑
   ├─ 선택적 양자화 (Int8/Int4)
   ├─ 블록 단위 구성
   └─ 세그먼트 덤프

3. 생성되는 세그먼트:
   ├─ ivf.centroid           (nlist × dim 벡터)
   ├─ ivf.inverted_body      (압축된 벡터)
   ├─ ivf.inverted_meta      (InvertedListMeta 배열)
   ├─ ivf.keys               (uint64_t 기본 키)
   ├─ ivf.offsets             (오프셋 정보)
   ├─ ivf.int8_quantized_params (리스트별 스케일)
   └─ ivf.features           (원본 벡터, 선택적)
```

### 4.3 HNSW Index (Hierarchical Navigable Small World)

다중 레벨 근접 그래프를 활용한 근사 최근접 이웃(ANN) 검색 알고리즘이다.

```
┌──────────────────────────────────────────────────────┐
│                   HNSW Index                          │
│                                                       │
│  Level 3:  [Entry Point]                              │
│               │                                       │
│  Level 2:  [n₁]────[n₅]                              │
│             │ ╲      │                                │
│  Level 1:  [n₁]─[n₃]─[n₅]─[n₇]                     │
│             │╲  │╲   │╲  │╲                          │
│  Level 0:  [n₀][n₁][n₂][n₃][n₄][n₅][n₆][n₇][n₈]  │
│             ─────────────────────────────────────      │
│             (모든 노드, M×2 이웃)                      │
│                                                       │
│  Search: Top→Bottom 탐욕적 검색                       │
│  Level 3: entry → nearest                             │
│  Level 2: → nearest neighbors                         │
│  Level 1: → nearest neighbors                         │
│  Level 0: → ef 크기 beam search → Top-K              │
└──────────────────────────────────────────────────────┘
```

**검색 알고리즘**:
1. 최상위 레벨의 Entry Point에서 시작
2. 각 레벨에서 Greedy Search로 최근접 노드 탐색
3. 해당 노드의 하위 레벨 이웃으로 이동
4. Level 0까지 반복
5. Level 0에서 `ef` 크기의 Beam Search 수행
6. Top-K 결과 반환

**핵심 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `M` (Scaling Factor) | 50 | 레벨당 최대 이웃 수 |
| `ef_construction` | 500 | 빌드 시 Beam 폭 |
| `ef_search` | - | 검색 시 Beam 폭 |
| Level 0 이웃 수 | 100 (M×2) | Level 0 최대 이웃 |

**그래프 구조 (HNSWHeader)**:
```cpp
struct HNSWHeader {
    GraphHeader graph;       // Level-0 이웃 정보 (M×2)
    HnswHeader hnsw;         // 상위 레벨 이웃 정보 (M)
    uint32_t entry_point;    // 최상위 진입 노드
    uint32_t max_level;      // 현재 최대 레벨
    uint32_t scaling_factor; // M = 50 (설정 가능)
};
```

**빌드 워크플로**:
```
1. 순차 삽입으로 그래프 구축
   ├─ 지수 분포로 레벨 생성
   ├─ [0, level] 모든 레벨에 노드 삽입
   ├─ 최근접 이웃 탐색 및 연결
   └─ 최대 레벨 초과 시 Entry Point 갱신

2. 이웃 선택 전략:
   ├─ Entry Point로부터 Greedy 검색
   ├─ 거리 기준 M개 후보 랭킹
   └─ 레벨당 M개로 이웃 가지치기

3. 생성되는 세그먼트:
   ├─ graph.header       (GraphHeader + HnswHeader)
   ├─ graph.features     (벡터 데이터)
   ├─ graph.keys         (uint64_t 기본 키)
   ├─ graph.neighbors    (Level-0 이웃 리스트)
   ├─ graph.offsets      (이웃 오프셋 인덱스)
   ├─ hnsw.neighbors     (상위 레벨 이웃)
   └─ hnsw.offsets       (상위 레벨 오프셋)
```

### 4.4 인덱스 타입 비교

| 특성 | FLAT | IVF | HNSW |
|------|------|-----|------|
| **검색 방식** | 전수 조사 | 클러스터 기반 | 그래프 기반 |
| **정확도** | 100% (정확) | 높음 (nprobe 의존) | 높음 (ef 의존) |
| **검색 속도** | O(n) | O(n/nlist × nprobe) | O(log n) |
| **메모리** | 낮음 | 중간 (센트로이드) | 높음 (그래프) |
| **빌드 시간** | 빠름 | 중간 (K-Means) | 느림 (그래프 구축) |
| **적합 규모** | < 100K | 100K ~ 100M | 100K ~ 1B+ |
| **Sparse 지원** | O | X | O |

---

## 5. 거리 메트릭 및 양자화

### 5.1 지원 거리 메트릭

```cpp
enum MetricType { COSINE, IP, L2 };
```

| 메트릭 | 수식 | 용도 |
|--------|------|------|
| **L2 (Euclidean)** | `dist = Σ(aᵢ - bᵢ)²` | 기본 거리 메트릭 |
| **Inner Product** | `sim = Σ(aᵢ × bᵢ)` | 추천 시스템 |
| **Cosine** | `sim = (a·b) / (‖a‖×‖b‖)` | 텍스트 유사도 |
| **Hamming** | `dist = popcount(a ⊕ b)` | 바이너리 벡터 |
| **MIPS** | `max(a·b)` | Maximum Inner Product |

모든 메트릭은 **SIMD 최적화**(SSE4.2, AVX, AVX-512) 및 **배치 연산**을 지원하며, 데이터 타입별(FP32, FP16, Int8, Int4) 특화 구현을 갖추고 있다.

### 5.2 양자화 (Quantization)

벡터 데이터의 메모리 사용량을 줄이면서 검색 정확도를 유지하기 위한 압축 기법이다.

```
┌─────────────────────────────────────────────────────┐
│           Quantization Pipeline                      │
│                                                      │
│  빌드 시: Converter (원본 → 양자화)                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ FP32     │ →  │ Scale/   │ →  │ Int8/    │       │
│  │ 원본벡터 │    │ Bias 학습│    │ Int4/FP16│       │
│  └──────────┘    └──────────┘    └──────────┘       │
│                                                      │
│  쿼리 시: Reformer (쿼리 변환 + 스코어 역변환)       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ 쿼리벡터 │ →  │ 양자화   │ →  │ 결과     │       │
│  │          │    │ 공간변환 │    │ 역정규화 │       │
│  └──────────┘    └──────────┘    └──────────┘       │
└─────────────────────────────────────────────────────┘
```

| 양자화 타입 | 압축률 | 정확도 손실 | 설명 |
|------------|--------|------------|------|
| **FP16** | 2× | 최소 | IEEE 754 반정밀도 |
| **Int8** | 4× | 낮음 | scale=255/(max-min), bias=min |
| **Int4** | 8× | 중간 | 4비트 패킹 + 정렬 |
| **PQ** | 높음 | 중간~높음 | Product Quantization (서브양자화) |

---

## 6. 데이터베이스 엔진 (DB Layer)

### 6.1 세그먼트 기반 스토리지 아키텍처

Zvec는 데이터를 **세그먼트(Segment)** 단위로 분할하여 관리한다. 각 세그먼트는 독립적인 벡터 인덱스, 스칼라 인덱스, 포워드 데이터를 가진다.

```
┌──────────────────────────────────────────────────────┐
│                   Collection                          │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Segment 0   │  │  Segment 1   │  │ Segment N  │  │
│  │  (Active     │  │  (Sealed)    │  │ (Sealed)   │  │
│  │   Writing)   │  │              │  │            │  │
│  │              │  │              │  │            │  │
│  │ ┌──────────┐│  │ ┌──────────┐ │  │            │  │
│  │ │ Forward  ││  │ │ Forward  │ │  │            │  │
│  │ │ Data     ││  │ │ Data     │ │  │            │  │
│  │ │(Arrow IPC)│  │ │(Parquet) │ │  │            │  │
│  │ └──────────┘│  │ └──────────┘ │  │            │  │
│  │ ┌──────────┐│  │ ┌──────────┐ │  │            │  │
│  │ │ Vector   ││  │ │ Vector   │ │  │            │  │
│  │ │ Index    ││  │ │ Index    │ │  │            │  │
│  │ │(HNSW/IVF)│  │ │(HNSW/IVF)│ │  │            │  │
│  │ └──────────┘│  │ └──────────┘ │  │            │  │
│  │ ┌──────────┐│  │ ┌──────────┐ │  │            │  │
│  │ │ Inverted ││  │ │ Inverted │ │  │            │  │
│  │ │ Index    ││  │ │ Index    │ │  │            │  │
│  │ │(RocksDB) ││  │ │(RocksDB) │ │  │            │  │
│  │ └──────────┘│  │ └──────────┘ │  │            │  │
│  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐                   │
│  │   ID Map     │  │ Delete Store │                   │
│  │  (RocksDB)   │  │ (Roaring     │                   │
│  │ PK → Doc ID  │  │  Bitmap)     │                   │
│  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────┘
```

**핵심 상수**:
```cpp
MAX_DOC_COUNT_PER_SEGMENT = 10,000,000  // 세그먼트당 최대 문서 수
DEFAULT_MAX_BUFFER_SIZE = 64MB          // 인메모리 버퍼 크기
kMaxRecordBatchNumRows = 4,096          // Arrow RecordBatch 행 수
kMaxDenseDimSize = 20,000               // 최대 벡터 차원
kMaxVectorFieldSize = 5                 // 최대 벡터 필드 수
kMaxQueryTopk = 1,024                   // 최대 Top-K
```

### 6.2 Write-Ahead Log (WAL)

데이터 내구성을 보장하는 선행 기록 로그 시스템이다.

```
Write 요청 → WAL append → Memory Buffer → Segment Flush → Disk
```

- Append-only 로깅
- 설정 가능한 플러시 빈도 (`max_docs_wal_flush`)
- `append()`, `flush()`, `prepare_for_read()`, `next()` 연산 지원

### 6.3 컬럼 스토리지

포워드 데이터(스칼라 필드)는 **Apache Arrow** 포맷으로 저장한다.

| 스토리지 타입 | 설명 |
|-------------|------|
| `MemoryForwardStore` | 인메모리 버퍼 |
| `MmapForwardStore` | 메모리 매핑 파일 |
| `BufferPoolForwardStore` | 설정 가능한 버퍼 풀 |

### 6.4 Inverted Index (스칼라 필터링)

스칼라 필드에 대한 필터링을 위해 **RocksDB 기반 역인덱스**를 사용한다.

**인코딩 전략** (사전순 정렬 보존):
- INT32/INT64: 부호 비트 XOR → Big-endian
- FLOAT/DOUBLE: XOR 조작으로 순서 보존
- STRING: 원본 바이트

**압축 전략**:
- 문서 ID ≤ 3개: 리스트로 저장
- 문서 ID > 3개: **Roaring Bitmap**으로 압축

**지원 연산**: `EQ`, `NE`, `GT`, `GE`, `LT`, `LE`, `LIKE`, `CONTAIN_ALL`, `CONTAIN_ANY`, `IS_NULL`, `IS_NOT_NULL`, `HAS_PREFIX`, `HAS_SUFFIX`

### 6.5 Manifest (메타데이터 영속화)

컬렉션의 전체 상태를 Protocol Buffers로 직렬화한다.

```protobuf
message Manifest {
    uint32 version;
    CollectionSchema schema;
    bool enable_mmap;
    repeated SegmentMeta persisted_segment_metas;
    SegmentMeta writing_segment_meta;
    uint32 id_map_path_suffix;
    uint32 delete_snapshot_path_suffix;
    uint32 next_segment_id;
}
```

---

## 7. SQL 엔진

Zvec는 벡터 검색과 스칼라 필터링을 결합한 SQL-유사 쿼리 언어를 ANTLR 기반으로 지원한다.

### 7.1 쿼리 처리 파이프라인

```
사용자 쿼리
    │
    ▼
┌─────────────┐
│   Parser    │  ANTLR 기반 SQL 파싱
│  (ANTLR4)   │  → SQLInfo (AST)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analyzer   │  의미 분석 + 스키마 검증
│             │  → QueryInfo
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Planner    │  실행 계획 생성
│             │  → PlanInfo
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │  계획 실행
│             │  → Arrow RecordBatchReader
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Converter  │  결과 변환
│             │  → DocPtrList
└─────────────┘
```

### 7.2 지원 쿼리 문법

```sql
SELECT [fields | *]
FROM collection_name
WHERE [filter_conditions]
ORDER BY field [ASC|DESC], ...
LIMIT topk
```

**필터 조건**:
```sql
-- 관계 연산
field = value | field != value | field > value | field >= value

-- 문자열 연산
field LIKE pattern | field HAS_PREFIX prefix | field HAS_SUFFIX suffix

-- 배열 연산
field CONTAIN_ALL (v1, v2) | field CONTAIN_ANY (v1, v2)

-- NULL 체크
field IS NULL | field IS NOT NULL

-- 범위 연산
field IN (v1, v2, v3)

-- 논리 연산
(cond1) AND (cond2) | (cond1) OR (cond2)
```

### 7.3 필터 푸시다운 전략

3단계 필터링을 통해 성능을 최적화한다.

```
┌───────────────────────────────────────────┐
│  1단계: Inverted Index Filter (최저 비용)  │
│     RocksDB 역인덱스 → Roaring Bitmap     │
│     스칼라 필드 조건 즉시 평가             │
├───────────────────────────────────────────┤
│  2단계: Forward Filter (중간 비용)         │
│     Arrow Compute 표현식                   │
│     복잡한 스칼라 필터 평가                │
├───────────────────────────────────────────┤
│  3단계: Vector Filter (최고 비용)          │
│     벡터 검색 중 DocFilter 적용            │
│     Top-K 후보에 필터 결합                 │
└───────────────────────────────────────────┘
```

---

## 8. 메모리 관리 및 스토리지

### 8.1 메모리 블록 구조

```cpp
struct MemoryBlock {
    enum Type { MBT_MMAP, MBT_BUFFERPOOL };
    ailego::BufferHandle::Pointer buffer_handle_;  // Pinned 메모리
    void *data_;                                    // Raw 포인터
};
```

### 8.2 스토리지 세그먼트 인터페이스

```cpp
class IndexStorage::Segment {
    // 읽기 전용 접근
    size_t read(size_t offset, const void **data, size_t len);
    size_t read(size_t offset, MemoryBlock &block, size_t len);

    // 빌드 시 쓰기
    size_t write(size_t offset, const void *data, size_t len);
    size_t resize(size_t size);

    // 무결성 검사
    uint32_t data_crc();
    void update_data_crc(uint32_t crc);
};
```

### 8.3 메모리 최적화 기법

| 기법 | 설명 |
|------|------|
| **MMAP** | 메모리 매핑 파일 (Huge Pages 2MB 지원) |
| **Buffer Pool** | Pin/Unpin + 참조 카운팅 |
| **Column-Major Ordering** | 캐시 지역성 최적화 |
| **32-byte Alignment** | SIMD 연산 정렬 |
| **Container-Aware** | cgroup 메모리 제한 자동 감지 |

---

## 9. Python 바인딩 아키텍처

### 9.1 바인딩 기술 스택

```
Python 공개 API (zvec/)
    │
    ▼
Python Wrapper (model/, schema/, typing/)
    │
    ▼
pybind11 C++ 바인딩 (src/binding/python/)
    │
    ▼
C++ Core 클래스 (src/db/, src/core/)
```

### 9.2 pybind11 모듈 구조

```cpp
PYBIND11_MODULE(_zvec, m) {
    ZVecPyTyping::Initialize(m);      // DataType, IndexType, MetricType 등
    ZVecPyParams::Initialize(m);      // 인덱스/쿼리 파라미터
    ZVecPySchemas::Initialize(m);     // Collection/Field 스키마
    ZVecPyConfig::Initialize(m);      // 전역 설정
    ZVecPyDoc::Initialize(m);         // 문서 연산
    ZVecPyCollection::Initialize(m);  // 핵심 Collection 클래스
}
```

### 9.3 지원 데이터 타입 (24종)

| 카테고리 | 타입 |
|---------|------|
| **스칼라** | STRING, BOOL, INT32, INT64, FLOAT, DOUBLE, UINT32, UINT64 |
| **배열** | ARRAY_STRING, ARRAY_INT32, ARRAY_INT64, ARRAY_FLOAT, ARRAY_DOUBLE, ARRAY_UINT32, ARRAY_UINT64, ARRAY_BOOL |
| **밀집 벡터** | VECTOR_FP16, VECTOR_FP32, VECTOR_FP64, VECTOR_INT8 |
| **희소 벡터** | SPARSE_VECTOR_FP16, SPARSE_VECTOR_FP32 |

### 9.4 에러 처리 매핑

```cpp
NOT_FOUND       → py::key_error
INVALID_ARGUMENT → py::value_error
기타            → std::runtime_error
```

### 9.5 Python API 사용 예시

```python
import zvec

# 1. 초기화 (선택적)
zvec.init(
    log_type=zvec.LogType.CONSOLE,
    log_level=zvec.LogLevel.WARN,
    query_threads=4
)

# 2. 스키마 정의
schema = zvec.CollectionSchema(
    name="my_collection",
    fields=[
        zvec.FieldSchema("category", zvec.DataType.STRING,
                         index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("price", zvec.DataType.FLOAT, nullable=True),
    ],
    vectors=[
        zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32,
                          dimension=128,
                          index_param=zvec.HnswIndexParam(m=16, ef_construction=200)),
        zvec.VectorSchema("sparse_vec", zvec.DataType.SPARSE_VECTOR_FP32),
    ]
)

# 3. 컬렉션 생성/열기
collection = zvec.create_and_open(path="./db", schema=schema)

# 4. 문서 삽입
docs = [
    zvec.Doc(
        id="doc1",
        fields={"category": "electronics", "price": 99.9},
        vectors={
            "embedding": [0.1, 0.2, ...],           # 128차원 밀집 벡터
            "sparse_vec": {0: 1.0, 5: 0.5, 100: 0.3}  # 희소 벡터 (index→value)
        }
    )
]
collection.insert(docs)

# 5. 하이브리드 검색 (벡터 + 스칼라 필터)
results = collection.query(
    vectors=zvec.VectorQuery(
        field_name="embedding",
        vector=[0.15, 0.25, ...],
        param=zvec.HnswQueryParam(ef=300)
    ),
    topk=10,
    filter="category == 'electronics' AND price < 200.0",
    output_fields=["category", "price"]
)

# 6. ID로 직접 조회
doc_dict = collection.fetch(["doc1", "doc2"])
```

### 9.6 임베딩 확장 함수

Zvec는 외부 임베딩 모델과의 통합을 위한 확장 함수를 제공한다.

| 확장 | 모델 | 타입 |
|------|------|------|
| `OpenAIDenseEmbedding` | text-embedding-3-small/large | Dense |
| `QwenDenseEmbedding` | DashScope API | Dense |
| `JinaEmbeddingFunction` | Jina API | Dense |
| `SentenceTransformerEmbedding` | 로컬 모델 | Dense |
| `BM25EmbeddingFunction` | BM25 키워드 | Sparse |
| `RrfReRanker` | Reciprocal Rank Fusion | Rerank |
| `WeightedReRanker` | 가중치 기반 | Rerank |
| `QwenReRanker` | Qwen 리랭킹 | Rerank |

---

## 10. AiLego 기반 라이브러리

Zvec의 기반을 구성하는 고성능 유틸리티 라이브러리다.

### 10.1 SIMD 수학 연산

CPU 특성을 런타임에 감지하여 최적 SIMD 명령어를 선택한다.

**지원 명령어 세트**:
- SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2
- AVX/AVX2
- AVX-512 (F, DQ, BW, VL, VNNI 등 다수 변형)
- FMA, BMI1/BMI2

**거리 연산 구현** (46+ 파일):
- Euclidean (FP32, FP16, INT8, INT4)
- Inner Product (FP32, FP16, INT8)
- Cosine Distance
- Hamming Distance (uint32, uint64)
- MIPS (Quadratic/Spherical Injection)

### 10.2 병렬 처리

| 컴포넌트 | 설명 |
|---------|------|
| `SpinMutex` | Atomic 기반 스핀 락 |
| `SharedMutex` | Reader-Writer 락 |
| `Semaphore` | 바이너리/멀티 카운트 |
| `BinarySemaphores<N>` | 비트 연산 기반 세마포어 풀 |
| `ThreadPool` | CPU 어피니티 지원 스레드 풀 |
| `MultiThreadList` | 유한 Producer-Consumer 큐 |

### 10.3 컨테이너 자료구조

| 자료구조 | 설명 |
|---------|------|
| `Bitmap` | 버킷(65536비트) 기반 희소 비트맵 |
| `FixedBitset<N>` | 컴파일 타임 고정 크기 비트셋 |
| `BloomFilter<K>` | K개 해시 함수 블룸 필터 |
| `Reservoir` | 확률적 샘플링 |

---

## 11. 성능 최적화 요약

### 11.1 SIMD 가속

```
CPU Feature 감지 → 최적 SIMD 경로 선택
  ├─ AVX-512: 512비트 벡터 연산
  ├─ AVX2: 256비트 벡터 연산
  ├─ SSE4.2: 128비트 벡터 연산
  └─ Fallback: 스칼라 연산
```

### 11.2 메모리 레이아웃 최적화

- **Column-Major Ordering**: 배치 거리 계산 시 캐시 적중률 극대화
- **32-byte Aligned Allocation**: SIMD 연산 요구사항 충족
- **Huge Page (2MB)**: MMAP 성능 향상

### 11.3 검색 조기 종료

- HNSW: `ef` 크기 Beam Search 예산
- IVF: `nprobe` 클러스터 가지치기
- Distance 기반 후보 필터링

### 11.4 양자화 트레이드오프

| 양자화 | 메모리 절감 | 정확도 |
|--------|-----------|--------|
| FP16 | 2× | 최소 손실 |
| Int8 | 4× | 높은 정확도 |
| Int4 | 8× | 중간 정확도 |

---

## 12. 서드파티 의존성

| 라이브러리 | 버전 | 역할 |
|-----------|------|------|
| **RocksDB** | 8.1.1 | 역인덱스 스토리지 (KV Store) |
| **Apache Arrow** | 21.0.0 | 컬럼형 데이터 포맷 |
| **Protocol Buffers** | 3.21.12 | 메타데이터 직렬화 |
| **ANTLR4** | - | SQL 쿼리 파서 생성기 |
| **CRoaring** | 2.0.4 | Roaring Bitmap (비트맵 압축) |
| **LZ4** | 1.9.4 | 데이터 압축 |
| **Google Test** | 1.10.0 | C++ 유닛 테스트 |
| **gflags** | 2.2.2 | CLI 플래그 파싱 |
| **glog** | 0.5.0 | 로깅 |
| **yaml-cpp** | 0.6.3 | YAML 설정 파싱 |
| **sparsehash** | 2.0.4 | 희소 해시 테이블 |
| **magic_enum** | 0.9.7 | C++ enum 리플렉션 |

---

## 13. 연관 기술 비교

### 13.1 벡터 데이터베이스 생태계에서의 위치

| 특성 | Zvec | Faiss (Meta) | Milvus | Qdrant | ChromaDB |
|------|------|------|--------|--------|----------|
| **배포 방식** | In-process | Library | Client-Server | Client-Server | Client-Server/Embedded |
| **언어** | C++ | C++ | Go/C++ | Rust | Python |
| **HNSW** | O | O | O | O | O |
| **IVF** | O | O | O | X | X |
| **스칼라 필터링** | O (RocksDB) | X | O | O | O |
| **SQL 쿼리** | O (ANTLR) | X | X | X | X |
| **Sparse 벡터** | O | X | O | O | X |
| **WAL** | O | X | O | O | X |
| **양자화** | Int4/8/FP16/PQ | PQ/SQ/OPQ | SQ8/PQ | SQ/PQ | X |

### 13.2 핵심 연관 기술

**HNSW (Hierarchical Navigable Small World)**:
- 2016년 Malkov & Yashunin 논문에서 제안된 그래프 기반 ANN 알고리즘
- Skip List와 Small World Graph의 결합
- 대부분의 벡터 DB에서 기본 인덱스로 채택

**IVF (Inverted File Index)**:
- 클러스터링 기반 분할 검색 (K-Means + Inverted List)
- Faiss에서 대중화된 방식
- 대규모 데이터셋에서 메모리 효율적

**Product Quantization (PQ)**:
- 고차원 벡터를 서브벡터로 분할하여 코드북 기반 압축
- 메모리 사용량을 10~100× 감소 가능

**Apache Arrow**:
- 언어 독립적 컬럼형 메모리 포맷
- Zero-copy 데이터 교환
- Zvec에서 포워드 데이터 스토리지로 활용

**RocksDB**:
- Facebook이 개발한 고성능 임베디드 KV 스토어
- LSM-Tree 기반
- Zvec에서 역인덱스 + ID 매핑에 활용

**Roaring Bitmap**:
- 압축 비트맵 자료구조
- Union/Intersection 연산이 빠름
- Zvec에서 삭제 마킹 및 필터 결과 표현에 활용

---

## 14. 설계 패턴 요약

| 패턴 | 적용 위치 |
|------|----------|
| **Factory** | IndexFactory - 인덱스 인스턴스 생성 |
| **Strategy** | Pluggable 알고리즘 (Flat, IVF, HNSW) |
| **Iterator** | IndexHolder - 벡터 스트리밍 |
| **Template Method** | Builder/Searcher 상속 계층 |
| **Pimpl** | Entity 클래스 - 내부 상태 은닉 |
| **Two-Phase Init** | init() → load()/open() |
| **RAII / ScopeGuard** | 리소스 관리 (ailego pattern) |

---

## 15. 동시성 및 스레드 안전성

```
┌─────────────────────────────────────────────┐
│  Read (Search) 연산                          │
│  - IndexStorage MemoryBlock 동시 읽기        │
│  - 공유 메트릭 (락 불필요)                   │
│  - 스레드별 Searcher Context                 │
├─────────────────────────────────────────────┤
│  Write (Build) 연산                          │
│  - IVF Builder: Mutex 보호 라벨 할당         │
│  - HNSW Builder: SpinMutex 락 풀             │
│  - Flat Builder: 순차 삽입                   │
├─────────────────────────────────────────────┤
│  인덱스 상태 머신                            │
│  STATE_INIT → STATE_INITED → STATE_LOADED   │
└─────────────────────────────────────────────┘
```

---

## 16. 결론

Zvec는 Alibaba의 프로덕션 환경에서 검증된 Proxima 엔진을 기반으로 한 **고성능 인-프로세스 벡터 데이터베이스**다. 핵심 강점은 다음과 같다:

1. **완전한 인덱스 스택**: FLAT(정확), IVF(균형), HNSW(근사) 3종 알고리즘을 모두 지원하며, Dense와 Sparse 벡터를 동시에 처리한다.

2. **풍부한 양자화**: Int4/Int8/FP16/PQ를 통해 메모리-정확도 트레이드오프를 세밀하게 조절할 수 있다.

3. **하이브리드 검색**: ANTLR 기반 SQL 엔진과 3단계 필터 푸시다운(역인덱스 → Arrow Compute → 벡터 필터)으로 벡터 검색과 스칼라 필터링을 효율적으로 결합한다.

4. **프로덕션 수준 스토리지**: 세그먼트 기반 아키텍처, WAL, RocksDB 역인덱스, Apache Arrow 컬럼 스토리지로 데이터 내구성과 성능을 보장한다.

5. **하드웨어 최적화**: 런타임 CPU 특성 감지를 통한 SIMD 디스패치(SSE4.2~AVX-512), Huge Page 지원, 컨테이너 인식 메모리 관리 등 하드웨어 수준 최적화를 갖추고 있다.

6. **Zero-Config 임베딩**: 별도 서버 없이 애플리케이션에 직접 임베딩되어 운영 복잡성을 제거하면서도 프로덕션급 성능을 제공한다.

이러한 특성들은 Zvec를 **서버리스 환경, 엣지 디바이스, 개인 프로젝트부터 대규모 프로덕션 시스템까지** 폭넓게 활용할 수 있는 벡터 검색 솔루션으로 자리매김하게 한다.
