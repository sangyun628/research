# Cognee 분석 보고서

> 저장소: https://github.com/topoteretes/cognee
> 분석일: 2026-01-21

---

## 1. 개요

Cognee는 원시 데이터를 벡터 검색과 그래프 데이터베이스를 결합하여 지속적이고 동적인 AI 메모리로 변환하는 오픈소스 메모리 및 지식 플랫폼입니다. 전통적인 RAG 시스템을 ECL(Extract, Cognify, Load) 파이프라인으로 대체합니다.

### 핵심 차별점

- **ECL 파이프라인**: 모듈식, 커스터마이징 가능한 데이터 처리 파이프라인 (단순 RAG가 아님)
- **그래프 + 벡터 하이브리드**: 벡터 유사도와 관계 기반 지식 그래프를 결합
- **온톨로지 통합**: 사전 정의된 어휘집에 대해 엔티티 검증
- **다중 검색 타입**: GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, CODE, CYPHER 등
- **트리플 임베딩**: 관계(S-P-O)를 임베딩하여 관계 인식 검색
- **30+ 데이터 소스**: 문서, 오디오, 이미지 등을 위한 네이티브 커넥터

### 연구 논문

- [지식 그래프와 LLM 간 인터페이스 최적화](https://arxiv.org/abs/2505.24478)

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                          Cognee                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    데이터 소스                            │   │
│  │   (문서, 오디오, 이미지, CSV, 웹, GitHub 등)              │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              cognee.add() - 데이터 수집                    │   │
│  │        (문서 처리, 청킹, 임베딩)                           │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           cognee.cognify() - 지식 그래프                   │   │
│  │     (엔티티 추출, 관계 탐지, 그래프 구축, 요약)             │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           cognee.memify() - 메모리 강화                    │   │
│  │     (규칙 연관, 그래프 확장, 코딩 규칙)                    │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────┬───────────────────┬──────────────────┐   │
│  │   벡터 저장소    │    그래프 저장소   │   관계형 DB      │   │
│  │  (임베딩)        │  (관계)           │   (메타데이터)   │   │
│  └──────────────────┴───────────────────┴──────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              cognee.search() - 검색                        │   │
│  │   (GRAPH_COMPLETION, RAG, CHUNKS, CODE, CYPHER 등)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 API

### 3.1 기본 사용법

```python
import cognee
import asyncio

async def main():
    # 1. cognee에 텍스트/문서 추가
    await cognee.add("여기에 문서나 텍스트")

    # 2. 지식 그래프 생성
    await cognee.cognify()

    # 3. 메모리 알고리즘 추가 (선택)
    await cognee.memify()

    # 4. 지식 검색
    results = await cognee.search(
        "이 문서가 말하는 것은?",
        query_type=SearchType.GRAPH_COMPLETION
    )

if __name__ == '__main__':
    asyncio.run(main())
```

### 3.2 CLI 인터페이스

```bash
# 데이터 추가
cognee-cli add "여기에 텍스트"

# 지식 그래프로 처리
cognee-cli cognify

# 검색
cognee-cli search "이것은 무엇에 관한 것인가?"

# 모두 삭제
cognee-cli delete --all

# 로컬 UI 시작
cognee-cli -ui
```

---

## 4. ECL 파이프라인: cognify()

`cognify()` 함수는 원시 데이터를 구조화된 지식 그래프로 변환하는 핵심 처리 단계입니다.

### 4.1 기본 파이프라인 태스크

```python
# /cognee/api/v1/cognify/cognify.py

async def get_default_tasks(user, graph_model, chunker, chunk_size, config, ...):
    default_tasks = [
        # 1. 문서 분류
        Task(classify_documents),

        # 2. 텍스트 청킹
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),

        # 3. 지식 그래프 추출
        Task(
            extract_graph_from_data,
            graph_model=graph_model,  # 기본: KnowledgeGraph
            config=config,
            custom_prompt=custom_prompt,
        ),

        # 4. 요약
        Task(summarize_text),

        # 5. 데이터 포인트 저장
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
        ),
    ]
    return default_tasks
```

### 4.2 그래프 추출

```python
# /cognee/tasks/graph/extract_graph_from_data.py

async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Config = None,
    custom_prompt: Optional[str] = None,
) -> List[DocumentChunk]:
    """문서 청크에서 지식 그래프를 추출합니다."""

    # LLM을 사용하여 각 청크에서 그래프 추출
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=custom_prompt)
            for chunk in data_chunks
        ]
    )

    # 엣지 검증 - 누락된 노드가 있는 엣지 제거
    for graph in chunk_graphs:
        valid_node_ids = {node.id for node in graph.nodes}
        graph.edges = [
            edge for edge in graph.edges
            if edge.source_node_id in valid_node_ids
            and edge.target_node_id in valid_node_ids
        ]

    # 온톨로지 검증과 통합
    return await integrate_chunk_graphs(
        data_chunks, chunk_graphs, graph_model, ontology_resolver
    )
```

### 4.3 시간 그래프 처리

```python
# 시간 인식 지식을 위한 temporal cognify

async def get_temporal_tasks(user, chunker, chunk_size, chunks_per_batch=10):
    temporal_tasks = [
        Task(classify_documents),
        Task(extract_chunks_from_documents, ...),
        Task(extract_events_and_timestamps),  # 시간 이벤트 추출
        Task(extract_knowledge_graph_from_events),  # 시간 KG 구축
        Task(add_data_points),
    ]
    return temporal_tasks
```

---

## 5. memify() - 메모리 강화

`memify()` 함수는 기존 지식 그래프에 메모리 알고리즘을 추가합니다.

```python
# /cognee/modules/memify/memify.py

async def memify(
    extraction_tasks: List[Task] = None,
    enrichment_tasks: List[Task] = None,
    data: Optional[Any] = None,
    dataset: str = "main_dataset",
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    ...
):
    """
    이미 구축된 그래프와 함께 작동하는 강화 파이프라인.
    커스텀 추출 및 강화 태스크를 추가할 수 있습니다.
    """

    # 지정되지 않으면 기본 태스크
    if not extraction_tasks:
        extraction_tasks = [Task(extract_subgraph_chunks)]
    if not enrichment_tasks:
        enrichment_tasks = [
            Task(
                add_rule_associations,
                rules_nodeset_name="coding_agent_rules",
            )
        ]

    # 데이터가 제공되지 않으면 기존 그래프를 입력으로 사용
    if not data:
        memory_fragment = await get_memory_fragment(
            node_type=node_type,
            node_name=node_name
        )
        data = [memory_fragment]

    # 강화 파이프라인 실행
    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=[*extraction_tasks, *enrichment_tasks],
        ...
    )
```

---

## 6. 검색 시스템

### 6.1 검색 타입

```python
# /cognee/modules/search/types.py

class SearchType(str, Enum):
    GRAPH_COMPLETION = "GRAPH_COMPLETION"  # LLM + 전체 그래프 컨텍스트
    RAG_COMPLETION = "RAG_COMPLETION"      # LLM + 문서 청크
    CHUNKS = "CHUNKS"                       # 원시 텍스트 세그먼트
    SUMMARIES = "SUMMARIES"                # 사전 생성된 요약
    CODE = "CODE"                          # 코드 특화 검색
    CYPHER = "CYPHER"                      # 직접 Cypher 쿼리
    FEELING_LUCKY = "FEELING_LUCKY"        # 최적 타입 자동 선택
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"      # 어휘 검색 (Jaccard)
```

### 6.2 검색 함수

```python
# /cognee/api/v1/search/search.py

async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: Optional[User] = None,
    datasets: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    verbose: bool = False,
) -> Union[List[SearchResult], CombinedSearchResult]:
    """
    검색 타입 및 사용 사례:

    GRAPH_COMPLETION (기본):
        - 전체 그래프 컨텍스트로 자연어 Q&A
        - 적합: 복잡한 질문, 분석, 통찰
        - 반환: 대화형 AI 응답

    RAG_COMPLETION:
        - 그래프 구조 없는 전통적 RAG
        - 적합: 직접 문서 검색
        - 반환: 텍스트 청크 기반 LLM 응답

    CHUNKS:
        - 쿼리와 일치하는 원시 텍스트 세그먼트
        - 적합: 특정 구절 찾기
        - 반환: 순위가 매겨진 텍스트 청크

    CODE:
        - 코드 특화 검색
        - 적합: 함수, 클래스 찾기
        - 반환: 구조화된 코드 정보

    CYPHER:
        - 직접 그래프 데이터베이스 쿼리
        - 적합: 고급 그래프 탐색
        - 반환: 원시 쿼리 결과
    """
```

### 6.3 검색 성능

| 검색 타입 | 속도 | 지능 | 사용 사례 |
|----------|------|------|----------|
| GRAPH_COMPLETION | 느림 | 높음 | 복잡한 추론 |
| RAG_COMPLETION | 중간 | 중간 | 문서 Q&A |
| CHUNKS | 빠름 | 낮음 | 구절 검색 |
| SUMMARIES | 빠름 | 낮음 | 빠른 개요 |
| CODE | 중간 | 높음 | 코드 이해 |
| FEELING_LUCKY | 가변 | 높음 | 일반 쿼리 |

---

## 7. 데이터 모델

### 7.1 KnowledgeGraph (기본 그래프 모델)

```python
# /cognee/shared/data_models.py

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class Node(BaseModel):
    id: str
    name: str
    type: str  # 엔티티 타입 (Person, Organization, Concept 등)
    properties: Dict[str, Any]

class Edge(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
```

### 7.2 Document Chunk

```python
# /cognee/modules/chunking/models/DocumentChunk.py

class DocumentChunk(BaseModel):
    id: UUID
    text: str
    document_id: UUID
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    contains: Optional[Any]  # 연결된 그래프/엔티티
```

### 7.3 Triplet (관계)

```python
# /cognee/modules/engine/models/Triplet.py

class Triplet(BaseModel):
    subject: Entity
    predicate: str
    object: Entity
    source_chunk_id: UUID
    confidence: float
```

---

## 8. 온톨로지 통합

Cognee는 온톨로지 기반 엔티티 검증 및 분류를 지원합니다.

### 8.1 온톨로지 리졸버

```python
# /cognee/modules/ontology/base_ontology_resolver.py

class BaseOntologyResolver:
    def get_subgraph(self, entity_type: str) -> Graph:
        """엔티티 타입에 대한 온톨로지 서브그래프를 가져옵니다."""
        pass

    def validate_entity(self, entity: Entity) -> bool:
        """온톨로지에 대해 엔티티를 검증합니다."""
        pass

    def resolve_entity_type(self, entity_name: str) -> str:
        """엔티티를 온톨로지 클래스에 매핑합니다."""
        pass
```

### 8.2 설정

```python
# 온톨로지를 위한 환경 변수

ONTOLOGY_FILE_PATH=/path/to/ontology.owl
ONTOLOGY_RESOLVER=default  # 또는 커스텀 리졸버 클래스
MATCHING_STRATEGY=fuzzy  # exact, fuzzy, semantic
```

---

## 9. 지원 통합

### 9.1 LLM 제공자

- OpenAI
- Anthropic
- Ollama (로컬)
- Azure OpenAI
- Google Gemini
- AWS Bedrock

### 9.2 벡터 데이터베이스

- Qdrant
- Weaviate
- Pinecone
- Milvus
- PostgreSQL (pgvector)
- 인메모리

### 9.3 그래프 데이터베이스

- Neo4j
- FalkorDB
- NetworkX (인메모리)

### 9.4 데이터 소스

- 텍스트 파일 (txt, md)
- 문서 (PDF, DOCX)
- 데이터 파일 (CSV, JSON)
- 이미지 (OCR 포함)
- 오디오 (전사 포함)
- 코드 저장소
- 웹 페이지
- Notion
- GitHub
- Google Drive
- 20+ 추가 커넥터

---

## 10. 트리플 임베딩

Cognee는 관계 인식 검색을 위한 관계(트리플) 임베딩을 지원합니다.

### 10.1 트리플 임베딩 설정

```python
# /cognee/modules/cognify/config.py

cognify_config = get_cognify_config()
embed_triplets = cognify_config.triplet_embedding  # True/False
```

### 10.2 트리플 임베딩 작동 방식

```
전통적: 노드만 임베딩
┌────────┐         ┌────────┐
│ Node A │───────→│ Node B │
│ [vec]  │ "knows" │ [vec]  │
└────────┘         └────────┘

트리플 임베딩 사용:
┌────────┐         ┌────────┐
│ Node A │───────→│ Node B │
│ [vec]  │ "knows" │ [vec]  │
└────────┘ [vec]   └────────┘
              ↑
    관계도 임베딩됨!

쿼리: "A가 아는 사람은?"
→ 노드 유사도뿐만 아니라 관계 유사도로도 검색 가능
```

---

## 11. 커스텀 파이프라인

Cognee는 커스텀 태스크로 커스텀 ECL 파이프라인을 구축할 수 있습니다.

### 11.1 커스텀 태스크 정의

```python
from cognee.modules.pipelines.tasks.task import Task

# 커스텀 태스크 정의
async def my_custom_extraction(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """커스텀 추출 로직."""
    for chunk in data_chunks:
        # 커스텀 처리
        chunk.metadata["custom_field"] = extract_custom_info(chunk.text)
    return data_chunks

# cognify에서 사용
await cognee.cognify(
    datasets=["my_data"],
    custom_tasks=[
        Task(classify_documents),
        Task(extract_chunks_from_documents),
        Task(my_custom_extraction),  # 커스텀 태스크
        Task(extract_graph_from_data),
        Task(add_data_points),
    ]
)
```

### 11.2 커스텀 그래프 모델

```python
from pydantic import BaseModel
from cognee.shared.DataPoint import DataPoint

class ScientificPaper(DataPoint):
    title: str
    authors: List[str]
    abstract: str
    methodology: str
    findings: List[str]
    citations: List[str]

await cognee.cognify(
    datasets=["research_papers"],
    graph_model=ScientificPaper,
)
```

---

## 12. 다른 메모리 시스템과의 비교

| 기능 | Cognee | Mem0 | MemU | OpenMemory | Second-Me |
|------|--------|------|------|------------|-----------|
| **아키텍처** | 그래프 + 벡터 | 벡터 + 그래프 | 파일 + 벡터 | HSG + 벡터 | 파인튜닝 LLM |
| **메모리 모델** | 지식 그래프 | 멀티레벨 | 5타입 + 카테고리 | 5섹터 | 모델 가중치 |
| **온톨로지 지원** | 예 | 아니오 | 아니오 | 아니오 | 아니오 |
| **트리플 임베딩** | 예 | 아니오 | 아니오 | 아니오 | N/A |
| **검색 타입** | 7+ 타입 | 1 타입 | 1 타입 | 1 타입 | N/A |
| **시간 지원** | 예 | 기본 | 기본 | 예 | 아니오 |
| **커스텀 파이프라인** | 예 | 아니오 | 아니오 | 아니오 | 아니오 |
| **데이터 소스** | 30+ | 10+ | 제한적 | 제한적 | 제한적 |

---

## 13. 강점과 약점

### 강점

1. **하이브리드 아키텍처**: 벡터 검색과 그래프 관계 결합
2. **모듈식 파이프라인**: 완전히 커스터마이징 가능한 ECL 파이프라인
3. **다중 검색 타입**: 다양한 사용 사례에 최적화
4. **온톨로지 통합**: 사전 정의된 어휘집에 대해 검증
5. **트리플 임베딩**: 관계 인식 검색
6. **시간 지원**: 시간 인식 지식 그래프
7. **광범위한 통합**: 30+ 데이터 소스, 다중 DB
8. **연구 기반**: KG-LLM 최적화 연구 기반
9. **프로덕션 준비**: 클라우드 플랫폼 이용 가능

### 약점

1. **복잡성**: 학습할 개념이 많음 (cognify, memify, 검색 타입)
2. **LLM 의존성**: 그래프 추출에 LLM 필요
3. **Decay 메커니즘 없음**: 자동 메모리 감쇠/통합 없음
4. **제한된 실시간**: 배치 처리, 실시간 메모리 아님
5. **아이덴티티 모델링 없음**: 사용자 프로필/아이덴티티 기능 없음

---

## 14. 사용 사례

### 14.1 문서 Q&A

```python
# 문서 추가
await cognee.add("path/to/documents/")

# 지식 그래프로 처리
await cognee.cognify()

# 쿼리
results = await cognee.search(
    "핵심 발견사항은 무엇인가?",
    query_type=SearchType.GRAPH_COMPLETION
)
```

### 14.2 코드 이해

```python
# 코드베이스 추가
await cognee.add("path/to/codebase/")

# 처리
await cognee.cognify()

# 코드 검색
results = await cognee.search(
    "인증은 어떻게 작동하나?",
    query_type=SearchType.CODE
)
```

### 14.3 지속적 에이전트 메모리

```python
# 에이전트 상호작용 저장
await cognee.add(conversation_history)
await cognee.cognify()
await cognee.memify()

# 관련 컨텍스트 검색
context = await cognee.search(
    current_query,
    query_type=SearchType.GRAPH_COMPLETION,
    session_id="agent_session"
)
```

---

## 15. 결론

Cognee는 벡터 검색을 그래프 기반 지식 표현과 결합하여 전통적인 RAG를 넘어서는 포괄적인 지식 관리 플랫폼을 제공합니다. 모듈식 ECL 파이프라인과 다중 검색 타입은 다양한 사용 사례에 높은 적응력을 제공합니다.

**적합한 사용 사례:**
- 지식 집약적 애플리케이션
- 문서 분석 및 Q&A
- 코드 이해 및 분석
- 관계 인식 검색이 필요한 애플리케이션
- 커스텀 지식 그래프 구축

**부적합한 사용 사례:**
- 실시간 메모리 업데이트
- 아이덴티티/페르소나 모델링
- 자동 메모리 통합
- 그래프 요구사항 없는 단순 RAG

---

*분석 완료일: 2026-01-21*
