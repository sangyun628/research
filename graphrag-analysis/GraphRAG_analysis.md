# Microsoft GraphRAG 심층 분석

## 개요

**GraphRAG**는 Microsoft Research에서 개발한 그래프 기반 검색 증강 생성(Retrieval-Augmented Generation) 시스템이다. 기존 벡터 유사도 검색 기반의 RAG(Baseline RAG)의 한계를 극복하기 위해 **지식 그래프(Knowledge Graph)** 를 활용하여 LLM의 추론 능력을 대폭 향상시킨다.

- **GitHub**: https://github.com/microsoft/graphrag
- **논문**: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- **버전**: 2.7.0 (2025년 1월 기준)
- **라이선스**: MIT

---

## 왜 GraphRAG를 사용하는가?

### 기존 RAG(Baseline RAG)의 한계

1. **점과 점을 연결하지 못함 (Connect the Dots)**
   - 벡터 검색은 개별 텍스트 청크의 유사도만 계산
   - 서로 다른 문서에 흩어진 정보를 통합하여 새로운 인사이트 도출 불가
   - 예: "A 회사의 CEO가 B 회사와 어떤 관계인가?"와 같은 다중 홉(multi-hop) 질문에 취약

2. **전역적 질문 처리 불가 (Global Questions)**
   - "이 데이터셋의 주요 테마는 무엇인가?"와 같은 전체 데이터에 대한 질문
   - 벡터 검색은 이러한 질문에 대한 명시적인 검색 대상이 없음
   - Query-Focused Summarization(QFS) 작업에 본질적으로 부적합

3. **환각(Hallucination) 문제**
   - 맥락이 부족할 때 LLM이 사실이 아닌 정보를 생성
   - Baseline RAG는 더 짧고, 불완전하며, 환각이 많은 응답 생성

### GraphRAG의 해결책

```
┌─────────────────────────────────────────────────────────────────┐
│                    GraphRAG vs Baseline RAG                      │
├─────────────────────────────────────────────────────────────────┤
│  Baseline RAG          │  GraphRAG                              │
│  ─────────────         │  ────────                              │
│  벡터 유사도 검색       │  지식 그래프 + 커뮤니티 요약            │
│  개별 텍스트 청크       │  엔티티/관계 + 계층적 구조              │
│  지역적 질문만 가능     │  지역적 + 전역적 질문 모두 가능         │
│  단순 top-k 검색       │  구조적 그래프 탐색                     │
└─────────────────────────────────────────────────────────────────┘
```

**GraphRAG의 핵심 가치:**
- 지식 그래프를 통한 **관계적 추론** 가능
- 커뮤니티 계층 구조를 통한 **전역적 이해** 지원
- 프라이빗 데이터셋에 대한 **구조화된 분석** 제공

---

## 아키텍처 개요

GraphRAG는 크게 **인덱싱 파이프라인**과 **쿼리 엔진** 두 가지 핵심 컴포넌트로 구성된다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      GraphRAG 전체 아키텍처                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    인덱싱 파이프라인                          │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │    │
│  │  │Documents│→ │TextUnits│→ │ Graph   │→ │ Communities +   │ │    │
│  │  │         │  │(청킹)   │  │Extraction│  │ Reports         │ │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Knowledge Model                         │    │
│  │   Entities │ Relationships │ Communities │ Community Reports │    │
│  │   TextUnits│ Documents     │ Covariates  │ Embeddings        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      쿼리 엔진                                │    │
│  │   Local Search │ Global Search │ DRIFT Search │ Basic Search │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 인덱싱 파이프라인 상세

인덱싱 파이프라인은 7단계로 구성되며, 원시 텍스트를 구조화된 지식 모델로 변환한다.

### Phase 1: TextUnits 생성 (Compose TextUnits)

```
Documents → Chunk → TextUnits
```

- **TextUnit**: 분석 가능한 텍스트 청크 단위
- 기본 청크 크기: 300 토큰 (1200 토큰까지 조정 가능)
- 문서 경계 유지 (1:N 관계)
- 출처 추적(provenance)을 위한 참조 단위

```python
# 기본 엔티티 타입
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
```

### Phase 2: 그래프 추출 (Graph Extraction)

```
TextUnits → Entity & Relationship Extraction → Summarization → Graph Tables
```

#### 2.1 엔티티 및 관계 추출

LLM을 사용하여 각 TextUnit에서 엔티티와 관계를 추출한다.

**추출 프롬프트 구조:**
```
-Goal-
주어진 텍스트에서 지정된 엔티티 타입의 모든 엔티티와
그들 간의 관계를 식별

-Steps-
1. 엔티티 식별: entity_name, entity_type, entity_description
2. 관계 식별: source_entity, target_entity, relationship_description,
              relationship_strength
3. 튜플 형식으로 출력
```

**출력 예시:**
```
("entity"|CENTRAL INSTITUTION|ORGANIZATION|The Central Institution is...)
("relationship"|MARTIN SMITH|CENTRAL INSTITUTION|Martin Smith is the Chair...|9)
```

#### 2.2 엔티티/관계 요약

동일한 엔티티/관계가 여러 TextUnit에서 추출될 경우:
- 같은 title + type을 가진 엔티티 병합
- 같은 source + target을 가진 관계 병합
- 여러 설명을 LLM으로 하나의 간결한 요약으로 통합

```python
# 엔티티 병합 로직
def _merge_entities(entity_dfs) -> pd.DataFrame:
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    return (
        all_entities.groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            frequency=("source_id", "count"),
        )
        .reset_index()
    )
```

#### 2.3 Claims 추출 (선택사항)

- 시간 범위가 있는 사실적 진술 추출
- 평가 상태(evaluated status)를 포함
- Covariates 테이블로 내보냄
- 기본적으로 비활성화 (프롬프트 튜닝 필요)

### Phase 3: 그래프 증강 (Graph Augmentation)

```
Graph → Leiden Hierarchical Community Detection → Communities
```

#### 계층적 Leiden 알고리즘

```python
from graspologic.partition import hierarchical_leiden

def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """계층적 클러스터링 알고리즘 적용"""
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    # 레벨별 커뮤니티 구조 생성
    ...
```

**커뮤니티 계층 구조:**
```
Level 0 (최상위)     [전체 데이터셋 개요]
    ├── Level 1     [대분류 토픽]
    │   ├── Level 2 [중분류 토픽]
    │   │   ├── Level 3 [세부 토픽]
    │   │   └── ...
    │   └── ...
    └── ...
```

### Phase 4: 커뮤니티 요약 (Community Summarization)

```
Communities → Generate Reports → Summarize → Community Reports
```

**커뮤니티 리포트 구조:**
```json
{
    "title": "커뮤니티 제목",
    "summary": "구조와 주요 엔티티에 대한 요약",
    "rating": 5.0,  // 중요도 점수 (0-10)
    "rating_explanation": "중요도 평가 이유",
    "findings": [
        {
            "summary": "인사이트 1 요약",
            "explanation": "상세 설명 [Data: Entities (id); Relationships (id)]"
        },
        ...
    ]
}
```

### Phase 5: 문서 처리 (Document Processing)

- 원본 문서와 TextUnits 간의 링크 생성
- CSV 데이터의 경우 추가 필드 포함 가능
- Documents 테이블 내보내기

### Phase 6: 네트워크 시각화 (선택사항)

```
Graph → Node2Vec Embedding → UMAP Dimensionality Reduction → 2D Coordinates
```

- Node2Vec으로 그래프 임베딩 생성
- UMAP으로 2D 좌표 생성
- 시각화를 위한 x/y 좌표 제공

### Phase 7: 텍스트 임베딩 (Text Embedding)

벡터 검색을 위한 임베딩 생성:
- TextUnit 텍스트 임베딩
- 엔티티 설명 임베딩
- 커뮤니티 리포트 콘텐츠 임베딩

---

## Knowledge Model (지식 모델)

### 핵심 데이터 타입

| 타입 | 설명 | 주요 필드 |
|------|------|----------|
| **Document** | 입력 문서 | id, text, title |
| **TextUnit** | 분석 단위 청크 | id, text, document_ids, entity_ids |
| **Entity** | 추출된 엔티티 | id, title, type, description, community_ids, rank |
| **Relationship** | 엔티티 간 관계 | id, source, target, description, weight |
| **Community** | 엔티티 클러스터 | id, level, parent, children, entity_ids |
| **CommunityReport** | 커뮤니티 요약 | id, title, summary, rating, findings |
| **Covariate** | 추출된 클레임 | id, subject_id, type, description |

### Entity 데이터 모델

```python
@dataclass
class Entity(Named):
    type: str | None = None                    # 엔티티 유형
    description: str | None = None             # 설명
    description_embedding: list[float] | None  # 설명 임베딩
    community_ids: list[str] | None = None     # 소속 커뮤니티
    text_unit_ids: list[str] | None = None     # 출처 TextUnits
    rank: int | None = 1                       # 중요도 순위 (degree 기반)
    attributes: dict[str, Any] | None = None   # 추가 속성
```

### Community 데이터 모델

```python
@dataclass
class Community(Named):
    level: str              # 계층 레벨
    parent: str             # 부모 커뮤니티 ID
    children: list[str]     # 자식 커뮤니티 ID 목록
    entity_ids: list[str]   # 소속 엔티티 ID 목록
    relationship_ids: list[str]  # 관련 관계 ID 목록
    size: int | None        # 커뮤니티 크기
```

---

## 쿼리 엔진 상세

GraphRAG는 4가지 검색 모드를 제공한다.

### 1. Local Search (지역 검색)

**용도**: 특정 엔티티에 대한 상세 질문
- "카모마일의 치유 효능은 무엇인가?"
- 특정 개념이나 엔티티에 집중

**작동 원리:**
```
User Query → Entity Embedding Match → Fan-out to Related Data
     ↓
┌─────────────────────────────────────────────────────────────┐
│  1. 쿼리와 의미적으로 관련된 엔티티 식별                       │
│  2. 연결된 엔티티, 관계, 커뮤니티 리포트 수집                  │
│  3. 관련 TextUnits 추출                                     │
│  4. 우선순위 결정 및 필터링                                   │
│  5. 컨텍스트 윈도우 구성 → LLM 응답 생성                      │
└─────────────────────────────────────────────────────────────┘
```

**데이터 플로우:**
```
User Query + Conversation History
         ↓
   Entity Description Embedding
         ↓
   Extracted Entities
         ↓
   ┌─────────────────────────────────────────┐
   │ Entity-TextUnit Mapping → Text Units    │
   │ Entity-Report Mapping → Community Reports│
   │ Entity-Entity Relations → Entities       │
   │ Entity-Entity Relations → Relationships  │
   │ Entity-Covariate Mapping → Covariates   │
   └─────────────────────────────────────────┘
         ↓ (Ranking + Filtering)
   Prioritized Context → Response
```

### 2. Global Search (전역 검색)

**용도**: 전체 데이터셋에 대한 종합적 질문
- "데이터셋의 주요 5가지 테마는 무엇인가?"
- 전체 코퍼스에 대한 통찰 필요

**작동 원리 (Map-Reduce):**
```
┌─────────────────────────────────────────────────────────────┐
│  MAP 단계:                                                   │
│  - 커뮤니티 리포트를 배치로 분할                              │
│  - 각 배치에서 중요도가 평가된 중간 응답 생성                  │
│                                                             │
│  REDUCE 단계:                                                │
│  - 중요도 높은 포인트들을 집계                                │
│  - 필터링하여 최종 응답 생성                                  │
└─────────────────────────────────────────────────────────────┘
```

**데이터 플로우:**
```
User Query + Conversation History
         ↓
Shuffled Community Report Batch 1 ─┐
Shuffled Community Report Batch 2 ─┼→ Rated Intermediate Responses
Shuffled Community Report Batch N ─┘
         ↓ (Ranking + Filtering)
Aggregated Intermediate Responses
         ↓
Final Response
```

**주요 파라미터:**
- `max_data_tokens`: 컨텍스트 데이터 토큰 예산
- `map_llm_params`: MAP 단계 LLM 파라미터
- `reduce_llm_params`: REDUCE 단계 LLM 파라미터
- `concurrent_coroutines`: 병렬 처리 수준

### 3. DRIFT Search (동적 추론 검색)

**용도**: Local과 Global의 장점을 결합한 검색
- 커뮤니티 정보를 활용한 확장된 지역 검색
- 비용-품질 트레이드오프 조절 가능

**DRIFT = Dynamic Reasoning and Inference with Flexible Traversal**

**3단계 프로세스:**
```
┌─────────────────────────────────────────────────────────────┐
│  A. Primer (초기화)                                          │
│  - 쿼리와 가장 관련 있는 상위 K개 커뮤니티 리포트 비교         │
│  - 초기 답변 + 후속 질문 생성                                │
│                                                             │
│  B. Follow-Up (후속 탐색)                                    │
│  - Local Search로 쿼리 정제                                  │
│  - 추가 중간 답변 + 후속 질문 생성                           │
│  - 신뢰도 점수에 따라 탐색 확장/종료 결정                     │
│                                                             │
│  C. Output Hierarchy (계층적 출력)                           │
│  - 질문-답변의 계층적 구조                                   │
│  - 관련성에 따라 순위화                                      │
└─────────────────────────────────────────────────────────────┘
```

### 4. Basic Search (기본 검색)

**용도**: 벡터 RAG와 비교를 위한 기본 검색
- 표준 top-k 벡터 검색
- 간단한 질문에 적합

---

## 저장소 및 확장성

### 지원 벡터 스토어

| 벡터 스토어 | 설명 | 용도 |
|------------|------|------|
| **LanceDB** | 기본 벡터 DB | 로컬 개발, 소규모 |
| **Azure AI Search** | Azure 클라우드 검색 | 엔터프라이즈 |
| **CosmosDB** | Azure NoSQL DB | 대규모 분산 |

### Factory 패턴을 통한 확장

```python
class VectorStoreFactory:
    """커스텀 벡터 스토어 등록 및 생성"""

    @classmethod
    def register(cls, vector_store_type: str,
                 creator: Callable[..., BaseVectorStore]) -> None:
        """커스텀 구현 등록"""
        cls._registry[vector_store_type] = creator

# 기본 제공 벡터 스토어 등록
VectorStoreFactory.register(VectorStoreType.LanceDB.value, LanceDBVectorStore)
VectorStoreFactory.register(VectorStoreType.AzureAISearch.value, AzureAISearchVectorStore)
VectorStoreFactory.register(VectorStoreType.CosmosDB.value, CosmosDBVectorStore)
```

### 확장 가능한 서브시스템

| 서브시스템 | 설명 | 커스터마이징 |
|-----------|------|-------------|
| Language Model | LLM 제공자 | chat, embed 메서드 구현 |
| Cache | 캐시 저장소 | file, blob, CosmosDB 외 추가 |
| Logger | 로그 저장소 | 커스텀 로그 위치 |
| Storage | 데이터 저장소 | DB 등 추가 가능 |
| Vector Store | 벡터 저장소 | 커스텀 벡터 DB |
| Pipeline | 워크플로우 | 커스텀 워크플로우 스텝 |

---

## 의존성 및 기술 스택

### 핵심 의존성

```toml
[dependencies]
# LLM
fnllm[azure,openai]>=0.4.1
openai>=1.68.0
tiktoken>=0.11.0
litellm>=1.77.1

# 데이터 사이언스
numpy>=1.25.2
pandas>=2.2.3
networkx>=3.4.2
graspologic>=3.4.1   # Leiden 알고리즘
umap-learn>=0.5.6    # 차원 축소

# NLP
nltk==3.9.1
spacy>=3.8.4
textblob>=0.18.0

# 벡터 스토어
lancedb>=0.17.0
azure-search-documents>=11.5.2

# Azure
azure-cosmos>=4.9.0
azure-identity>=1.19.0
azure-storage-blob>=12.24.0
```

### 주요 알고리즘

| 알고리즘 | 용도 | 라이브러리 |
|---------|------|-----------|
| **Hierarchical Leiden** | 커뮤니티 감지 | graspologic |
| **Node2Vec** | 그래프 임베딩 | graspologic |
| **UMAP** | 차원 축소 | umap-learn |

---

## LazyGraphRAG 비교

Microsoft는 GraphRAG의 비용 효율적 대안으로 **LazyGraphRAG**를 제공한다.

### 비용 비교

| 항목 | GraphRAG | LazyGraphRAG |
|------|----------|--------------|
| 인덱싱 비용 | 높음 (LLM 사전 요약) | Vector RAG 수준 (0.1%) |
| 쿼리 비용 (Global) | 높음 | 700배 이상 저렴 |
| 사전 데이터 요약 | 필수 | 불필요 |

### 특성 비교

```
┌─────────────────────────────────────────────────────────────┐
│                GraphRAG vs LazyGraphRAG                      │
├─────────────────────────────────────────────────────────────┤
│  GraphRAG:                                                   │
│  - 고품질, 포괄적 결과 필요 시                               │
│  - 엔터프라이즈급 지식 관리                                  │
│  - 복잡한 데이터 분석                                        │
│  - 사전 인덱싱 비용 감당 가능                                │
│                                                             │
│  LazyGraphRAG:                                               │
│  - 비용 민감 시나리오                                        │
│  - 일회성 쿼리, 탐색적 분석                                  │
│  - 스트리밍 데이터                                           │
│  - 중소기업, 개인 개발자                                     │
└─────────────────────────────────────────────────────────────┘
```

### LazyGraphRAG 작동 원리

- 사전 요약 없이 그래프 기반 RAG 제공
- LLM 사용을 쿼리 시점으로 지연(lazy)
- 단일 파라미터(relevance test budget)로 비용-품질 조절
- 4% 쿼리 비용으로 모든 경쟁 방식 대비 우수한 성능

---

## 사용 방법

### 설치

```bash
pip install graphrag
```

### 초기화

```bash
graphrag init --root ./my_project
```

### 인덱싱 실행

```bash
graphrag index --root ./my_project
```

### 쿼리 실행

```bash
# Local Search
graphrag query --root ./my_project --method local --query "질문 내용"

# Global Search
graphrag query --root ./my_project --method global --query "질문 내용"

# DRIFT Search
graphrag query --root ./my_project --method drift --query "질문 내용"
```

### Python API

```python
from graphrag.api import index, query

# 인덱싱
await index(root_dir="./my_project")

# 쿼리
result = await query(
    root_dir="./my_project",
    method="local",
    query="질문 내용"
)
```

---

## 프롬프트 튜닝

기본 프롬프트로는 최적의 결과를 얻지 못할 수 있다. 데이터에 맞는 프롬프트 튜닝 권장.

```bash
graphrag prompt-tune --root ./my_project
```

**튜닝 대상:**
- 엔티티 추출 프롬프트
- 관계 추출 프롬프트
- 커뮤니티 리포트 생성 프롬프트
- 검색 프롬프트

---

## 적합한 사용 사례

### 적합한 경우

1. **복잡한 다중 홉 추론 질문**
   - 여러 문서에 걸친 정보 통합 필요
   - 엔티티 간 관계 파악 필요

2. **전역적 질문/주제 분석**
   - "이 문서들의 주요 테마는?"
   - 데이터셋 전체에 대한 통찰

3. **사실적 정확성 중요**
   - 환각 감소 필요
   - 출처 추적(provenance) 필요

4. **대규모 프라이빗 데이터셋**
   - 기업 내부 문서
   - 연구 자료

### 부적합한 경우

1. **단순한 사실 확인 질문**
   - 기본 RAG로 충분

2. **실시간 응답 필요**
   - 인덱싱 비용과 시간 고려

3. **빈번한 데이터 업데이트**
   - 재인덱싱 비용 발생

---

## 성능 벤치마크

Microsoft Research 발표 기준:

| 지표 | GraphRAG | Baseline RAG |
|------|----------|--------------|
| 포괄성(Comprehensiveness) | 72-83% | 낮음 |
| 다양성(Diversity) | 62-82% | 낮음 |
| 루트 레벨 요약 토큰 | 최대 97% 감소 | - |

---

## 참고 자료

- [Microsoft Research Blog - GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG Arxiv Paper](https://arxiv.org/abs/2404.16130)
- [LazyGraphRAG Blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [GitHub Repository](https://github.com/microsoft/graphrag)
- [Official Documentation](https://microsoft.github.io/graphrag/)
- [IBM - What is GraphRAG](https://www.ibm.com/think/topics/graphrag)

---

## 결론

GraphRAG는 기존 RAG의 한계를 극복하기 위해 **지식 그래프와 계층적 커뮤니티 구조**를 활용하는 혁신적인 접근 방식이다. 특히:

1. **관계적 추론**: 엔티티 간 관계를 명시적으로 모델링
2. **전역적 이해**: 커뮤니티 계층을 통한 데이터 구조 파악
3. **다양한 검색 모드**: 질문 유형에 따른 최적 검색 전략
4. **확장성**: Factory 패턴을 통한 유연한 커스터마이징

비용이 우려되는 경우 LazyGraphRAG를 통해 99.9% 저렴한 인덱싱으로 유사한 품질을 달성할 수 있다.
