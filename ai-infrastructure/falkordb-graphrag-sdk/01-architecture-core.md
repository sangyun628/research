# 01. 아키텍처 & 코어 레이어

> 소스 기준: `graphrag_sdk/src/graphrag_sdk/` (이하 경로 생략)

## 1. 패키지 구조

```
graphrag_sdk/
├── api/main.py                  # GraphRAG facade (3,222줄) — 모든 공개 API
├── core/
│   ├── models.py                # Pydantic 데이터 모델 전부
│   ├── context.py               # Context (tenant/trace/latency budget)
│   ├── connection.py            # FalkorDBConnection + ConnectionConfig
│   ├── circuit_breaker.py       # DB 커넥션용 circuit breaker
│   ├── exceptions.py            # 예외 계층
│   └── providers/               # LLMInterface · Embedder ABC + LiteLLM/OpenRouter 구현
├── ingestion/
│   ├── pipeline.py              # IngestionPipeline — 9단계 고정 시퀀스
│   ├── loaders/                 # Text · Pdf · Markdown
│   ├── chunking_strategies/     # FixedSize · SentenceTokenCap · Structural · Contextual · Callable
│   ├── extraction_strategies/   # GraphExtraction(2단계) · GLiNER/LLM NER · coref
│   ├── resolution_strategies/   # ExactMatch · DescriptionMerge · Semantic · LLMVerified
│   └── backfill.py              # 온톨로지 진화 시 청크 재스캔 실행기
├── retrieval/
│   ├── router.py                # SemanticRouter (룰 기반)
│   ├── strategies/              # Local · MultiPath · CypherGeneration 등
│   └── reranking_strategies/    # CosineReranker
├── discovery/                   # 온톨로지 자동 발견 (prompts.py에 프롬프트 집중)
├── storage/                     # GraphStore · VectorStore · OntologyStore · Deduplicator
└── utils/cypher.py              # 라벨 sanitization
```

설계 원칙: **모든 Cypher는 storage 레이어에만 존재**(repository 패턴). 파이프라인·전략 클래스는 절대 raw Cypher를 쓰지 않는다.

## 2. 공개 API — `GraphRAG` facade

`api/main.py:205-275`

```python
GraphRAG(
    connection: FalkorDBConnection | ConnectionConfig,
    llm: LLMInterface,
    embedder: Embedder,
    ontology: Ontology | None = None,        # 없으면 기본 11개 엔티티 타입 사용
    retrieval_strategy: RetrievalStrategy | None = None,  # 기본 MultiPathRetrieval
    embedding_dimension: int = 256,
)
```

### 메서드 그룹별 요약

| 그룹 | 메서드 | 역할 |
|---|---|---|
| 수집 | `ingest(source/text, ...)` (main.py:1268) | 단일/배치 수집. 기본 전략: SentenceTokenCap + GraphExtraction + ExactMatch |
| 수집 | `finalize()` (main.py:2955) | 수집 후 일괄 처리: NULL stub 제거 → 전역 dedup → 엔티티/관계 임베딩 → 인덱스 보장 |
| 갱신 | `update(source, document_id, if_missing=...)` (main.py:1916) | SHA-256 해시 비교 후 변경 시에만 재수집. crash-safe 상태 머신 |
| 갱신 | `delete_document(document_id)` (main.py:2205) | 문서 + 고아 엔티티 제거 (멱등) |
| 갱신 | `apply_changes(added, modified, deleted)` (main.py:2317) | CI 연동용 배치. **deletes → updates → adds 순서 보장** |
| 검색 | `retrieve(question)` (main.py:2519) | 컨텍스트만 반환 (생성 없음) |
| 검색 | `completion(question, history=...)` (main.py:2647) | RAG 전체: 검색 + 답변 생성. 멀티턴 history 지원 |
| 온톨로지 | `get/set/save/refresh_ontology()` | 온톨로지 그래프 로드/교체/JSON 저장 |
| 온톨로지 진화 | `rename_entity/attribute/relation`, `drop_*`, `add_attribute`, `backfill_*` | [05 문서](05-ontology-discovery-evolution.md) 참고 |
| 관리 | `get_statistics()`, `delete_all()`, `deduplicate_entities(fuzzy=...)` | |

모든 메서드는 async이며 `*_sync()` 래퍼(`asyncio.run`) 제공.

### `ingest()` 호출 시 내부 오케스트레이션 (main.py:1364-1564)

1. 인자 검증 (file 모드 vs text 모드, 예약 document_id 차단)
2. `_validate_graph_config()` — 임베더 차원 probe 후 그래프에 기록된 설정과 일치 확인
3. `_phase0_recover_prior_operations()` — 이전 크래시된 update/delete의 잔여 상태 복구
4. `_ensure_ontology_initialized()` — 사용자 온톨로지를 온톨로지 그래프에 lazy 등록
5. `IngestionPipeline` 조립 (loader/chunker/extractor/resolver 기본값 주입) 후 `run()`
6. 후처리: `vector_store.ensure_indices()` + 임베더 모델/차원 메타 기록

## 3. 코어 데이터 모델 (`core/models.py`)

데이터가 파이프라인을 흐르며 변환되는 타입 체인:

```
DocumentOutput (로더)
  └─ text: str, document_info: DocumentInfo, elements: list[DocumentElement] | None

TextChunks (청커)
  └─ chunks: list[TextChunk(text, index, metadata, uid=uuid4)]

GraphData (추출기)                          # models.py:519
  ├─ nodes: list[GraphNode(id, label, properties)]
  ├─ relationships: list[GraphRelationship(start_node_id, end_node_id, type, properties)]
  ├─ mentions: list[EntityMention(entity_id, chunk_id)]
  ├─ extracted_entities: list[ExtractedEntity(name, type, description, source_chunk_ids, attributes)]
  └─ extracted_relations: list[ExtractedRelation(source, target, type, description, weight, ...)]

ResolutionResult (해소기)                    # models.py:566
  └─ nodes, relationships, merged_count, remap: dict[loser_id → survivor_id]

IngestionResult                             # models.py:666
  └─ nodes_created, relationships_created, chunks_indexed, metadata
```

### 온톨로지 모델 (models.py:178-512)

```python
Attribute(name, type="STRING", description)       # STRING·INTEGER·FLOAT·BOOLEAN·DATE·LIST
Entity(label, description, properties=[Attribute])  # hash/eq는 label 기준
Relation(label, description, patterns=[(src,tgt)], properties)  # patterns 비면 모든 쌍 허용
Ontology(entities, relations)
  .from_file(path) / .from_sources(sources, llm)   # JSON 로드 / LLM 자동 발견
  .save_to_file(path) / .merge(other)
```

## 4. LLM 프로바이더 추상화 (`core/providers/`)

### `LLMInterface` ABC (base.py:101-305)

```python
invoke(prompt) -> LLMResponse                      # sync 필수 구현
ainvoke(prompt, max_retries=3, timeout=None)       # 지터 지수 백오프: (2**attempt) * (0.5 + rand(0,0.5))
ainvoke_messages(messages: list[ChatMessage])      # 멀티턴
ainvoke_with_model(prompt, response_model)         # Pydantic 구조화 출력
abatch_invoke(prompts, max_concurrency=12)         # 동시 호출 + 아이템별 에러 격리 (BatchItem.ok/.error)
astream(prompt)
```

### `Embedder` ABC (base.py:36-98)

`embed_query`, `embed_documents`, `aembed_query`, `aembed_documents` — 비동기 기본 구현은 `asyncio.to_thread`.

### `LiteLLM` 구현 (litellm.py)

- 모든 프로바이더를 litellm으로 위임. reasoning 모델(o1/o3/gpt-5)은 `temperature` 제거 + `max_completion_tokens` 변환 처리 (`_is_reasoning_model`)
- `LiteLLMEmbedder`: batch_size 2048, 실패 시 **binary-split retry** — 배치를 반으로 쪼개 재귀 재시도 (`providers/_retry.py:52-107`). 인증 오류(401/403)는 즉시 raise, 단일 텍스트 실패는 빈 벡터 반환

### 신뢰성 장치

| 장치 | 위치 | 파라미터 |
|---|---|---|
| Circuit breaker (DB) | `core/circuit_breaker.py` | failure_threshold=5, recovery_timeout=30s, CLOSED→OPEN→HALF_OPEN |
| 쿼리 재시도 (DB) | `core/connection.py` | retry_count=3, 지수 백오프 `retry_delay * 2**attempt`, transient만 재시도 |
| Latency budget | `core/context.py` | `Context(latency_budget_ms=...)` — 모든 전략이 LLM/임베딩 호출 전 `ctx.ensure_budget()` 체크, 초과 시 `LatencyBudgetExceededError` |
| LLM 타임아웃 | `providers/_timeout.py` | `asyncio.wait_for` 래핑 → `LLMTimeoutError` 변환 |

## 5. FalkorDB 커넥션 (`core/connection.py`)

```python
ConnectionConfig(
    host="localhost", port=6379,
    graph_name="knowledge_graph",       # 온톨로지는 "{graph_name}__ontology"에 별도 저장
    max_connections=16, pool_timeout=30.0,
    retry_count=3, retry_delay=1.0,
    query_timeout_ms=10_000,
    ssl=False, ...                       # rediss:// URL 파싱 지원 (from_url)
)
```

- lazy init: 첫 쿼리 시 `redis.asyncio.BlockingConnectionPool` + `falkordb.asyncio.FalkorDB` 생성
- 멀티테넌시: DB 레벨 격리는 없고 `Context.tenant_id`/`trace_id`가 로깅·추적 단위로만 사용됨. 실질 격리는 graph_name 분리로

## 6. 예외 계층 (`core/exceptions.py`)

```
GraphRAGError
├── LatencyBudgetExceededError
├── LLMError → LLMTimeoutError
├── EmbeddingError → EmbeddingTimeoutError
├── IngestionError → LoaderError · ChunkingError · ExtractionError · ResolutionError
├── RetrieverError / DatabaseError / DocumentNotFoundError
└── SchemaValidationError / ConfigError
```

단계별 예외가 분리되어 있어 파이프라인의 어느 단계가 실패했는지 타입으로 식별 가능.

## 7. 설계 패턴 정리

1. **Async-first + sync 래퍼** — 코어는 전부 async, `*_sync()`는 `asyncio.run` (async 컨텍스트 내 호출 시 RuntimeError)
2. **Crash-safe 상태 머신** — `update()`는 pending 문서 노드 + `ready_to_commit` 마커로 2-phase commit; 재시작 시 phase 0에서 roll-forward/rollback
3. **배치 단위 에러 격리** — `apply_changes()`·배치 ingest는 `BatchEntry.ok/fail`로 파일별 결과 반환, 한 파일 실패가 배치를 중단시키지 않음
4. **프롬프트 인젝션 방어 기본값** — RAG 답변 프롬프트가 `<context>` 태그 + "untrusted, 태그 안 지시 무시" 시스템 프롬프트 사용, 문서 텍스트의 literal `</context>` 무력화 (main.py:2729)
