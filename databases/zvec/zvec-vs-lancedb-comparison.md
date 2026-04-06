# Zvec vs LanceDB: 인-프로세스 벡터 데이터베이스 심층 비교 분석

## 1. 개요

두 프로젝트 모두 **인-프로세스(in-process) 벡터 데이터베이스**로, 별도 서버 없이 애플리케이션에 직접 임베딩되어 동작한다. 그러나 설계 철학, 스토리지 아키텍처, 생태계 면에서 근본적으로 다른 접근 방식을 취한다.

| 항목 | Zvec (Alibaba) | LanceDB |
|------|---------------|---------|
| **리포지토리** | https://github.com/alibaba/zvec | https://github.com/lancedb/lancedb |
| **라이선스** | Apache 2.0 | Apache 2.0 |
| **핵심 언어** | C++ (81.3%) | Rust (코어) + Python/TypeScript SDK |
| **C/C++/Rust 표준** | C++17 | Rust 2021 Edition |
| **빌드 시스템** | CMake + scikit-build-core | Cargo + Maturin (Python) + NAPI (Node.js) |
| **바인딩 기술** | pybind11 | PyO3 (Python), NAPI-RS (Node.js) |
| **스토리지 포맷** | 자체 세그먼트 + RocksDB + Arrow IPC | Lance 포맷 (자체 컬럼형 포맷) |
| **쿼리 엔진** | 자체 ANTLR SQL 파서 | Apache DataFusion |
| **GitHub Stars** | 8.6k+ | 12k+ |
| **Python 지원** | 3.9+ | 3.10+ |

---

## 2. 설계 철학 비교

### Zvec: 전통적 데이터베이스 설계

Alibaba의 프로덕션 벡터 검색 엔진 **Proxima**를 기반으로 하며, 전통적인 DB 엔진 설계 패턴을 따른다. 여러 전문 스토리지 엔진(RocksDB, Arrow, Roaring Bitmap)을 조합하여 각 영역에서 최적의 성능을 추구한다.

```
┌──────────────────────────────────────────────────┐
│  Zvec 설계 핵심: "조합형 전문 엔진"               │
│                                                   │
│  벡터 인덱스: 자체 HNSW/IVF/FLAT 엔진             │
│  스칼라 인덱스: RocksDB (LSM-Tree 기반 KV Store)  │
│  포워드 데이터: Apache Arrow IPC/Parquet           │
│  삭제 마킹: Roaring Bitmap                        │
│  내구성: Write-Ahead Log                          │
│  메타데이터: Protocol Buffers                     │
│                                                   │
│  → 각 영역에 특화된 엔진을 조합하여 최적화        │
└──────────────────────────────────────────────────┘
```

### LanceDB: 데이터 레이크 설계

벡터와 스칼라 데이터를 **Lance라는 단일 컬럼형 포맷**으로 통합 저장하며, 불변(immutable) 파일과 MVCC를 통해 Git-like 버전 관리를 제공한다. Object Store 추상화로 로컬과 클라우드를 동일한 코드로 처리한다.

```
┌──────────────────────────────────────────────────┐
│  LanceDB 설계 핵심: "불변 파일 + 버전 관리"       │
│                                                   │
│  모든 데이터: Lance 포맷 (벡터+스칼라 통합)        │
│  모든 쓰기: 새로운 불변 파일 생성 (MVCC)          │
│  버전 관리: Manifest 파일로 스냅샷 추적            │
│  스토리지: Object Store 추상화 (로컬/S3/GCS/Azure)│
│  쿼리 엔진: Apache DataFusion (범용 SQL)          │
│  인덱스: Lance 포맷 내 통합                       │
│                                                   │
│  → 단일 포맷으로 통합하여 클라우드 친화적 설계     │
└──────────────────────────────────────────────────┘
```

---

## 3. 스토리지 아키텍처

두 시스템의 **가장 근본적인 차이**는 데이터 저장 방식에 있다.

### 3.1 Zvec: 세그먼트 기반 다중 엔진 스토리지

```
Collection (디렉토리)
├── manifest                      # Protobuf 메타데이터
├── segment_0/                    # 활성 세그먼트 (쓰기 가능)
│   ├── forward/                  #   스칼라 데이터 (Arrow IPC)
│   ├── vector_index/             #   벡터 인덱스 (HNSW/IVF 자체 바이너리)
│   │   ├── graph.header          #     HNSW 그래프 헤더
│   │   ├── graph.features        #     벡터 데이터
│   │   ├── graph.neighbors       #     이웃 리스트
│   │   └── hnsw.offsets          #     오프셋 인덱스
│   └── inverted/                 #   스칼라 역인덱스 (RocksDB)
├── segment_1/                    # 봉인된 세그먼트 (읽기 전용)
├── id_map/                       # PK → Doc ID 매핑 (RocksDB)
├── delete_store/                 # 삭제 마킹 (Roaring Bitmap)
└── wal/                          # Write-Ahead Log
```

**특징**:
- 세그먼트당 최대 1,000만 문서 (`MAX_DOC_COUNT_PER_SEGMENT`)
- 활성 세그먼트는 쓰기 가능, 임계치 초과 시 자동으로 봉인(seal) 후 새 세그먼트 생성
- RocksDB가 역인덱스와 ID 매핑 두 군데에서 독립적으로 사용됨
- WAL로 비정상 종료 시 데이터 복구 보장

### 3.2 LanceDB: Lance 포맷 불변 파일 스토리지

```
Table (디렉토리)
├── _latest.manifest              # 현재 버전 포인터
├── _versions/
│   ├── 1.manifest                # 버전 1 메타데이터 (불변)
│   ├── 2.manifest                # 버전 2 메타데이터 (불변)
│   └── 3.manifest                # 버전 3 메타데이터 (불변)
├── data/
│   ├── part-0001.lance           # 불변 데이터 파일 (벡터+스칼라 통합)
│   ├── part-0002.lance           # 새 쓰기 → 새 파일 생성
│   └── part-0003.lance
├── _indices/
│   └── ivf_pq_index/            # 벡터 인덱스 파일
└── _deletions/
    └── 2-3.arrow                 # 삭제 마킹 (버전 범위별)
```

**특징**:
- 모든 쓰기가 새로운 불변 데이터 파일 + 새 manifest 생성 (MVCC)
- 벡터와 스칼라가 동일 Lance 파일 내에 컬럼형으로 통합 저장
- 이전 manifest를 통해 과거 시점 데이터 접근 가능 (타임트래블)
- Object Store 추상화로 로컬 디스크와 S3/GCS/Azure 동일하게 접근

### 3.3 스토리지 접근 핵심 차이

| 관점 | Zvec | LanceDB |
|------|------|---------|
| **쓰기 방식** | In-place 수정 (세그먼트 내 블록 추가) | Append-only (새 불변 파일 생성) |
| **삭제 방식** | Roaring Bitmap 마킹 | Deletion 파일 + 새 manifest |
| **파일 수** | 세그먼트당 다수 (인덱스별 분리) | 데이터 파일 + manifest |
| **Compaction** | 세그먼트 단위 최적화 | 소파일 병합 (`compact_files()`) |
| **버전 관리** | 없음 | Git-like (태깅, 체크아웃) |
| **클라우드 스토리지** | 지원 안함 (로컬 전용) | S3/GCS/Azure/OSS 네이티브 |
| **RocksDB 의존** | 있음 (역인덱스 + ID맵) | 없음 |

---

## 4. 클라우드 스토리지 지원

**이것이 실무에서 가장 결정적인 차이**가 될 수 있다.

### Zvec: 로컬 전용

```python
# Zvec - 로컬 파일시스템 경로만 가능
collection = zvec.create_and_open(path="./local_db", schema=schema)

# S3에 저장하려면 애플리케이션이 별도로 동기화해야 함
# (예: 주기적으로 ./local_db 디렉토리를 S3에 업로드)
```

### LanceDB: Object Store 추상화

```python
# LanceDB - 로컬
db = lancedb.connect("./local_db")

# S3에서 직접 벡터 검색 (코드 변경 최소)
db = lancedb.connect("s3://my-bucket/vectors",
                     storage_options={"region": "us-west-2"})

# GCS
db = lancedb.connect("gs://my-bucket/vectors")

# Azure Blob
db = lancedb.connect("az://my-container/vectors")

# 이중 쓰기 (빠른 로컬 + 내구성 있는 S3)
# MirroringObjectStore로 primary(S3) + secondary(로컬) 구성 가능
```

LanceDB는 `object_store` 크레이트를 통해 **동일한 코드로 로컬↔클라우드를 전환**할 수 있으며, 자격 증명 자동 갱신(`StorageOptionsProvider`)도 지원한다.

---

## 5. 벡터 인덱스 알고리즘

### 5.1 지원 인덱스 타입 비교

| 인덱스 타입 | Zvec | LanceDB | 설명 |
|------------|------|---------|------|
| **FLAT** | O | O (스캔 모드) | 전수 검색 (Brute Force) |
| **IVF-Flat** | O | O | 클러스터 기반 + 원본 벡터 |
| **IVF-PQ** | △ (별도 PQ) | O | 클러스터 + Product Quantization |
| **IVF-SQ** | △ (별도 Int8) | O | 클러스터 + Scalar Quantization |
| **IVF-RQ** | X | O | 클러스터 + Residual Quantization |
| **HNSW (단독)** | **O** | X | 그래프 기반 ANN (독립 사용) |
| **IVF-HNSW-PQ** | X | **O** | IVF + HNSW + PQ 복합 |
| **IVF-HNSW-SQ** | X | **O** | IVF + HNSW + SQ 복합 |
| **Flat-Sparse** | **O** | X | 희소 벡터 전수 검색 |
| **HNSW-Sparse** | **O** | X | 희소 벡터 그래프 검색 |
| **Inverted (스칼라)** | **O** (RocksDB) | O (BTree/Bitmap) | 스칼라 필드 인덱스 |

### 5.2 인덱스 파라미터 비교

**HNSW 관련 파라미터**:

| 파라미터 | Zvec (독립 HNSW) | LanceDB (IVF-HNSW-PQ) |
|---------|-----------------|----------------------|
| `M` (이웃 수) | 50 (기본) | 20 (기본) |
| `ef_construction` | 500 (기본) | 300 (기본) |
| Level 0 이웃 수 | M×2 = 100 | M×2 = 40 |

**IVF 관련 파라미터**:

| 파라미터 | Zvec | LanceDB |
|---------|------|---------|
| `num_partitions` | 1024 (기본) | √n (자동 계산) |
| K-Means 반복 | 10 (기본) | 50 (기본) |
| `sample_rate` | N/A | 256 (학습 데이터 = rate × partitions) |
| `nprobe` | 쿼리 시 지정 | 20 (기본) |

### 5.3 핵심 차이점

**Zvec의 강점 — 독립 HNSW + Sparse 전용 인덱스**:
```
Zvec는 HNSW를 독립 인덱스로 사용 가능:
├─ 소규모~중규모 데이터에서 IVF 없이도 빠른 검색
├─ HNSW-Sparse로 희소 벡터를 그래프 기반으로 검색
└─ ef_construction=500, M=50으로 높은 기본 정확도
```

**LanceDB의 강점 — 복합 인덱스**:
```
LanceDB는 IVF + HNSW + 양자화를 결합:
├─ IVF-HNSW-PQ: 대규모 데이터에서 최적의 속도/정확도/메모리 균형
├─ IVF로 검색 범위를 좁힌 후 HNSW로 정밀 검색
└─ PQ/SQ 양자화로 메모리 16×~4× 절감
```

---

## 6. 거리 메트릭 및 양자화

### 6.1 거리 메트릭

| 메트릭 | Zvec | LanceDB |
|--------|------|---------|
| L2 (Euclidean) | O | O |
| Cosine | O | O |
| Inner Product (IP) | O | O (Dot) |
| Hamming | O | O |
| MIPS | O | X |

### 6.2 양자화

| 양자화 타입 | Zvec | LanceDB | 압축률 |
|------------|------|---------|--------|
| FP16 | O | O | 2× |
| Int8 (SQ) | O | O | 4× |
| Int4 | O | X | 8× |
| PQ (Product Quantization) | O | O | 16×+ |
| RQ (Residual Quantization) | X | O | - |

**Zvec**는 Int4(4비트) 양자화를 지원하여 극한의 메모리 절감이 가능하고, **LanceDB**는 RQ(Residual Quantization)로 PQ 대비 더 높은 정확도의 양자화를 제공한다.

### 6.3 SIMD 최적화

| 최적화 | Zvec | LanceDB |
|--------|------|---------|
| SSE4.2 | O | O (via Lance) |
| AVX/AVX2 | O | O (via Lance) |
| AVX-512 | **O** (다수 변형) | △ (제한적) |
| **런타임 CPU 감지** | **O** (CpuFeatures 클래스) | O (컴파일 타임) |
| ARM NEON | O (ARMv8.0~8.6) | O |

Zvec는 **런타임에 CPU 특성을 감지**하여 최적 SIMD 경로를 선택하는 반면, LanceDB는 주로 컴파일 타임에 결정한다. Zvec의 SIMD 지원이 더 세밀하다(AVX-512의 F, DQ, BW, VL, VNNI 등 세부 변형별 분기).

---

## 7. 쿼리 엔진

### 7.1 Zvec: 자체 ANTLR SQL 엔진

```
사용자 쿼리 문자열
    │
    ▼
┌──────────┐  ANTLR4 Lexer/Parser
│  Parser  │  → SQLInfo (AST)
└────┬─────┘
     ▼
┌──────────┐  스키마 검증 + 의미 분석
│ Analyzer │  → QueryInfo
└────┬─────┘
     ▼
┌──────────┐  3단계 필터 푸시다운
│ Planner  │  → PlanInfo
└────┬─────┘
     ▼
┌──────────┐
│ Executor │  → Arrow RecordBatchReader → DocPtrList
└──────────┘
```

**3단계 필터 푸시다운**:
1. **Inverted Index Filter** (최저 비용): RocksDB 역인덱스 → Roaring Bitmap
2. **Forward Filter** (중간 비용): Arrow Compute 표현식
3. **Vector Filter** (최고 비용): 벡터 검색 중 DocFilter 적용

### 7.2 LanceDB: Apache DataFusion 기반

```
Builder API / SQL 문자열
    │
    ▼
┌──────────────┐
│  DataFusion  │  SQL → Logical Plan → Optimized Physical Plan
│  (v52.1)     │
└──────┬───────┘
       ▼
┌──────────────┐
│  Lance       │  Projection Pushdown + Filter Pushdown
│  Scan Node   │  + Index Selection
└──────┬───────┘
       ▼
┌──────────────┐
│  Vector      │  벡터 인덱스 검색 + 필터 결합
│  Search Node │
└──────┬───────┘
       ▼
   Arrow RecordBatch Stream
```

**DataFusion 활용 이점**:
- SQL 함수, 집계, 조인 등 풍부한 SQL 표현력
- 자동 쿼리 최적화 (Projection Pushdown, Filter Pushdown)
- User-Defined Table Functions (UDTF) 확장 가능

### 7.3 쿼리 능력 비교

| 쿼리 기능 | Zvec | LanceDB |
|----------|------|---------|
| 벡터 유사도 검색 | O | O |
| 스칼라 필터 (=, >, <, LIKE) | O | O |
| AND/OR 논리 조합 | O | O |
| IS NULL / IS NOT NULL | O | O |
| CONTAIN_ALL / CONTAIN_ANY | O | O (`array_contains_all/any`) |
| HAS_PREFIX / HAS_SUFFIX | O | X (FTS로 대체) |
| IN 절 | O | O |
| GROUP BY 쿼리 | O | O (DataFusion) |
| 집계 함수 (COUNT, SUM) | X | **O** (DataFusion) |
| JOIN | X | **O** (DataFusion) |
| Window 함수 | X | **O** (DataFusion) |
| **풀텍스트 검색 (FTS)** | △ (키워드 수준) | **O** (Tantivy BM25) |

---

## 8. 데이터 버전 관리

이것은 **LanceDB만의 고유 기능**이다.

### Zvec: 버전 관리 없음

```python
# 삽입 후 이전 상태로 되돌릴 수 없음
collection.insert(docs)
collection.delete(["doc1"])  # doc1 영구 마킹, 복구 불가

# WAL은 비정상 종료 복구용이지 버전 관리가 아님
```

### LanceDB: Git-like MVCC 버전 관리

```python
# 모든 쓰기가 불변 버전 생성
table.add(data)              # → version 2 생성
table.add(more_data)         # → version 3 생성
table.delete("price > 1000") # → version 4 생성

# 타임트래블 — 과거 시점 데이터 쿼리
table.checkout_version(2)    # version 2 시점으로 이동
results = table.search(query).limit(10).to_list()

# 태깅 — 특정 버전에 이름 부여
table.tags.create("prod-release-v1", version=3)
table.checkout_tag("prod-release-v1")

# 버전 목록 확인
versions = table.list_versions()

# 오래된 버전 정리 (2주 이전 삭제)
table.cleanup_old_versions(older_than=timedelta(weeks=2))
```

**활용 시나리오**:
- ML 파이프라인에서 학습 데이터의 재현성(reproducibility) 보장
- A/B 테스트 시 동일 데이터셋 기준으로 비교
- 잘못된 데이터 삽입 시 이전 버전으로 롤백
- 프로덕션 배포 시 특정 버전 태깅 및 고정

---

## 9. 풀텍스트 검색 (FTS) 및 하이브리드 검색

### 9.1 Zvec: 역인덱스 기반 키워드 필터링

```python
# 스칼라 필드에 역인덱스 생성
schema = zvec.CollectionSchema(
    name="docs",
    fields=[
        zvec.FieldSchema("title", zvec.DataType.STRING,
                         index_param=zvec.InvertIndexParam()),
    ],
    vectors=[zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 128)]
)

# 필터링 수준의 텍스트 검색
results = collection.query(
    vectors=zvec.VectorQuery(field_name="embedding", vector=query_vec),
    filter="title LIKE '%machine learning%'",
    topk=10
)
```

- RocksDB 기반 역인덱스로 `LIKE`, `HAS_PREFIX`, `HAS_SUFFIX` 지원
- 토큰화, 스테밍, 불용어 제거 등 **전문 검색 기능은 없음**
- 패턴 매칭 수준의 필터링

### 9.2 LanceDB: Tantivy 기반 전문 검색

```python
# FTS 인덱스 생성 (Tantivy 기반)
table.create_index(
    "content",
    config=lancedb.index.FTS(
        language="English",
        stem=True,
        remove_stop_words=True,
        lower_case=True
    )
)

# 풀텍스트 검색
from lancedb.query import MatchQuery, PhraseQuery, BooleanQuery

# 단순 텍스트 매칭
results = table.search(MatchQuery("machine learning", fuzziness=1))

# 정확한 구문 검색
results = table.search(PhraseQuery("deep learning framework"))

# 복합 불리언 쿼리
query = BooleanQuery(
    must=[MatchQuery("vector database")],
    should=[MatchQuery("in-process"), MatchQuery("embedded")],
    must_not=[MatchQuery("deprecated")]
)
results = table.search(query)

# 하이브리드 검색 (벡터 + FTS + 리랭킹)
results = table.search_hybrid(
    query="lightweight vector database",
    vector_column_query=query_vector,
    text_columns=["title", "content"],
    reranker=lancedb.rerankers.RRFReranker(K=60)
)
```

**Tantivy 기능**: BM25 알고리즘, 27+ 언어 지원, 퍼지 매칭, 토큰화/스테밍/불용어 제거

---

## 10. 임베딩 생태계

### 10.1 지원 임베딩 함수 비교

| 카테고리 | 제공자 | Zvec | LanceDB |
|---------|--------|------|---------|
| **클라우드 API** | OpenAI | O | O |
| | Cohere | X | O |
| | Voyage AI | X | O |
| | Jina AI | O | O |
| | Google Gemini | X | O |
| | AWS Bedrock | X | O |
| | IBM WatsonX | X | O |
| | Qwen (DashScope) | O | X |
| **로컬 모델** | Sentence Transformers | O | O |
| | HuggingFace Transformers | X | O |
| | GTE (General Text Embeddings) | X | O |
| | Ollama | X | O |
| **Sparse** | BM25 | O | X (FTS로 대체) |
| **멀티모달** | OpenCLIP | X | O |
| | ImageBind | X | O |
| | SigLIP | X | O |
| | ColPali (문서 이미지) | X | O |

**Zvec**: 5개 임베딩 제공자 (Alibaba 생태계 중심)
**LanceDB**: 16+ 임베딩 제공자 (글로벌 생태계 광범위 지원)

### 10.2 리랭킹 비교

| 리랭커 | Zvec | LanceDB |
|--------|------|---------|
| RRF (Reciprocal Rank Fusion) | O | O |
| Weighted Reranker | O | O (Linear Combination) |
| Cohere Reranker | X | O |
| OpenAI Reranker | X | O |
| Voyage AI Reranker | X | O |
| Jina AI Reranker | X | O |
| CrossEncoder | X | O |
| ColBERT | X | O |
| Qwen Reranker | O | X |

---

## 11. 데이터 모델 및 스키마

### 11.1 데이터 타입 비교

| 카테고리 | Zvec (24종) | LanceDB (Arrow 기반) |
|---------|-----------|---------------------|
| **기본 스칼라** | STRING, BOOL, INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE | Arrow 전체 타입 지원 (수십 종) |
| **배열** | ARRAY_STRING, ARRAY_INT32 등 8종 | Arrow List 타입 (임의 중첩) |
| **밀집 벡터** | VECTOR_FP16, VECTOR_FP32, VECTOR_FP64, VECTOR_INT8 | FixedSizeList(Float16/32/64) |
| **희소 벡터** | SPARSE_VECTOR_FP16, SPARSE_VECTOR_FP32 | 별도 지원 없음 (일반 List로 표현) |
| **중첩 구조** | 지원 안함 | Arrow Struct/Map (임의 중첩) |

**핵심 차이**: Zvec는 **희소 벡터를 1등급 시민으로 취급**(전용 인덱스 포함)하는 반면, LanceDB는 **Arrow의 풍부한 타입 시스템**을 그대로 활용하여 중첩 구조 등 유연한 스키마를 지원한다.

### 11.2 스키마 진화

| 기능 | Zvec | LanceDB |
|------|------|---------|
| 컬럼 추가 | O (`add_column`) | O (`add_columns`) |
| 컬럼 삭제 | O (`drop_column`) | O (`drop_columns`) |
| 컬럼 이름 변경 | O (`alter_column`) | O (`alter_columns`) |
| 타입 변경 | X | O |
| 기존 데이터 재작성 | 필요 | **불필요** (새 파일만 추가) |

---

## 12. 동시성 및 일관성 모델

### Zvec

```
읽기: 다수 동시 읽기 가능 (Searcher Context per thread)
쓰기: 세그먼트 단위 락 (SpinMutex / SharedMutex)
일관성: 단일 프로세스 내 강한 일관성
다중 프로세스: 지원 안함 (단일 프로세스 전용)
```

### LanceDB

```
읽기: 다수 동시 읽기 가능 (불변 파일 기반)
쓰기: MVCC (새 파일 생성, Last-Write-Wins)
일관성: 3가지 모드 선택 가능
  ├─ Manual (Lazy): 캐시된 버전 사용, 수동 갱신
  ├─ Eventual: TTL 기반 백그라운드 갱신
  └─ Strong: 매 읽기마다 최신 버전 확인
다중 프로세스: 지원 (MVCC + Object Store)
```

---

## 13. 빌드 시스템 및 의존성

### 13.1 서드파티 의존성 비교

**Zvec (12개 Git 서브모듈)**:
| 의존성 | 버전 | 역할 |
|--------|------|------|
| RocksDB | 8.1.1 | 역인덱스 + ID 매핑 |
| Apache Arrow | 21.0.0 | 컬럼형 데이터 포맷 |
| Protocol Buffers | 3.21.12 | 메타데이터 직렬화 |
| ANTLR4 | - | SQL 파서 생성기 |
| CRoaring | 2.0.4 | 비트맵 압축 |
| LZ4 | 1.9.4 | 데이터 압축 |
| Google Test | 1.10.0 | 테스트 프레임워크 |

**LanceDB (Cargo 의존성)**:
| 의존성 | 버전 | 역할 |
|--------|------|------|
| Lance | 3.0.0-rc.2 | 코어 데이터 엔진 |
| Apache Arrow | 57.2 | 인메모리 데이터 포맷 |
| DataFusion | 52.1 | SQL 쿼리 엔진 |
| object_store | 0.12.0 | 클라우드 스토리지 추상화 |
| Tantivy | - | 전문 검색 엔진 |
| Tokio | - | 비동기 런타임 |
| Moka | - | LRU 캐시 |

### 13.2 빌드 복잡성

| 관점 | Zvec | LanceDB |
|------|------|---------|
| 빌드 시스템 | CMake (복잡) | Cargo (단순) |
| 서브모듈 | 12개 Git 서브모듈 | Cargo.toml 의존성 관리 |
| Python 빌드 | scikit-build-core + pybind11 | Maturin + PyO3 |
| 크로스 컴파일 | 복잡 (CMake 설정 필요) | Cargo의 target 시스템 |

---

## 14. 성능 특성 비교

### 14.1 강점 영역

```
Zvec가 유리한 영역:
├─ 단일 프로세스 로컬 검색 지연시간 (C++ 네이티브)
├─ SIMD 최적화 세밀도 (AVX-512 다수 변형별 분기)
├─ 희소 벡터 전용 인덱스 (HNSW-Sparse)
├─ RocksDB 기반 빠른 스칼라 필터링
├─ Int4 양자화로 극한의 메모리 절감
└─ 메모리 직접 제어 (Buffer Pool / MMAP / Huge Pages)

LanceDB가 유리한 영역:
├─ 클라우드 스토리지에서의 벡터 검색 (S3 직접 스캔)
├─ IVF-HNSW-PQ 복합 인덱스의 대규모 데이터 처리
├─ DataFusion 기반 복잡한 SQL 쿼리 (JOIN, 집계 등)
├─ 타임트래블로 데이터 재현성 보장
├─ Tantivy 기반 진정한 전문 검색 + 하이브리드 검색
├─ 다중 프로세스 동시 접근 (MVCC)
└─ 스키마 진화 (데이터 재작성 불필요)
```

### 14.2 인덱스 성능 특성

| 인덱스 | 검색 복잡도 | 빌드 시간 | 메모리 | 비고 |
|--------|-----------|----------|--------|------|
| **Zvec FLAT** | O(n) | O(n) | O(nd) | 정확한 검색 |
| **Zvec IVF** | O(n/nlist × nprobe) | O(n√n) | O(nd) + centroids | 균형 |
| **Zvec HNSW** | O(log n) | O(n log n) | O(nd) + graph | 빠른 검색 |
| **LanceDB IVF-Flat** | O(√n + k) | O(n√n) | O(nd) | 균형 |
| **LanceDB IVF-PQ** | O(√n + k) | O(n√n) | O(n/M) | 메모리 효율 |
| **LanceDB IVF-HNSW-PQ** | O(log n + k) | O(n log n) | O(n/M) + graph | 최적 균형 |

---

## 15. 관리형 서비스 및 생태계

| 관점 | Zvec | LanceDB |
|------|------|---------|
| **관리형 클라우드 서비스** | 없음 | **LanceDB Cloud** (서버리스) |
| **네임스페이스 (멀티테넌시)** | 없음 | O (계층적 네임스페이스) |
| **REST API** | 없음 | O (LanceDB Cloud) |
| **Python SDK** | O | O |
| **Node.js SDK** | 언급만 있음 (코드 없음) | **O** (네이티브 NAPI 바인딩) |
| **Rust SDK** | X (C++ 라이브러리) | **O** (네이티브) |
| **DuckDB 통합** | X | O |
| **Polars 통합** | X | O |
| **Pandas 통합** | X | O |

---

## 16. 사용 시나리오별 추천

| 시나리오 | 추천 | 이유 |
|---------|------|------|
| **로컬 고성능 벡터 검색 (최저 지연)** | **Zvec** | C++ 네이티브, SIMD 세밀 최적화, 직접 메모리 제어 |
| **희소 벡터 검색 (BM25 + 벡터)** | **Zvec** | HNSW-Sparse, Flat-Sparse 전용 인덱스 |
| **엣지/임베디드 디바이스** | **Zvec** | 최소 의존성, 로컬 전용 최적화 |
| **Alibaba 생태계 통합** | **Zvec** | Qwen 임베딩, DashScope API |
| **클라우드 스토리지 기반 검색** | **LanceDB** | S3/GCS/Azure 네이티브 Object Store |
| **ML 파이프라인 (데이터 버전 관리)** | **LanceDB** | MVCC + 타임트래블 + 태깅 |
| **풀텍스트 + 벡터 하이브리드 검색** | **LanceDB** | Tantivy FTS + BM25 + 다양한 리랭커 |
| **멀티모달 검색 (이미지+텍스트)** | **LanceDB** | OpenCLIP, ImageBind, ColPali, SigLIP |
| **복잡한 SQL 분석 쿼리** | **LanceDB** | DataFusion (JOIN, 집계, Window 함수) |
| **서버리스 클라우드 서비스** | **LanceDB** | LanceDB Cloud 관리형 서비스 |
| **다중 프로세스 동시 접근** | **LanceDB** | MVCC + 일관성 모드 선택 |
| **Node.js/TypeScript 환경** | **LanceDB** | 네이티브 NAPI 바인딩 |
| **기존 RocksDB 기반 시스템 통합** | **Zvec** | RocksDB 역인덱스 호환 |

---

## 17. 결론

**한 줄 요약**:
- **Zvec** = "로컬에서 극한의 성능을 짜내는 C++ 벡터 엔진" (전통적 DB 설계)
- **LanceDB** = "클라우드 네이티브 + 버전 관리가 가능한 현대적 벡터 DB" (데이터 레이크 설계)

두 프로젝트는 **근본적으로 다른 설계 패러다임**을 채택한다:

| 패러다임 | Zvec | LanceDB |
|---------|------|---------|
| **스토리지 모델** | Mutable 세그먼트 + WAL | Immutable 파일 + MVCC |
| **데이터 엔진** | 다중 엔진 조합 (RocksDB + Arrow + 자체) | 단일 Lance 포맷 통합 |
| **스케일링 방향** | Scale-Up (단일 머신 최적화) | Scale-Out (클라우드 Object Store) |
| **최적화 초점** | 지연시간 최소화 (SIMD, 메모리 제어) | 유연성 최대화 (버전, 클라우드, 생태계) |
| **유사 프로젝트** | SQLite + Faiss 결합 | DeltaLake + Faiss 결합 |

**선택 기준**:
- **"내 서버/디바이스에서 가장 빠르게 벡터를 검색해야 한다"** → Zvec
- **"클라우드에서 버전 관리하면서 다양한 검색을 통합해야 한다"** → LanceDB
