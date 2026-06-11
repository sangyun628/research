# 02. 아키텍처 & 코어 레이어

> 소스: `lightrag/lightrag.py`(4,270줄), `base.py`(1,064줄), `kg/`, `constants.py`, `utils.py`(4,109줄)

## 1. 패키지 구조

```
lightrag/
├── lightrag.py        # LightRAG dataclass — 설정 + 스토리지 조립 + 공개 API
├── pipeline.py        # 수집 파이프라인 (enqueue → parse → analyze → process)
├── operate.py         # 핵심 알고리즘 전부 (추출·병합·질의) — 5,995줄
├── base.py            # 스토리지 ABC 4종 + QueryParam + DocStatus
├── prompt.py          # PROMPTS dict (프롬프트 전부)
├── constants.py       # 모든 기본값 (단일 출처)
├── chunker/           # F·R·V·P 4종 청킹 전략
├── parser/            # 문서 파서 (native docx · legacy · mineru · docling) + IR
├── kg/                # 스토리지 구현 12종 + shared_storage(락) + factory
├── llm/               # 프로바이더 바인딩 (openai · ollama · gemini · bedrock ...)
├── rerank.py          # 리랭커 (Cohere · Jina · Aliyun · generic API)
├── api/               # FastAPI 서버 + WebUI
└── utils.py           # 토크나이저 · 해시 ID · LLM 캐시 · 우선순위 큐 · 절단
```

GraphRAG-SDK가 "단계마다 Strategy 클래스"라면 LightRAG은 **"함수 중심 + 설정 주입"** — 알고리즘이 operate.py의 큰 async 함수들이고, dataclass 필드와 constants.py 환경변수로 동작을 조정한다.

## 2. LightRAG 클래스 — 주요 설정값

`lightrag.py:160-1154`. 전 필드가 dataclass + 환경변수 오버라이드. 핵심만 발췌:

| 그룹 | 필드 | 기본값 | 의미 |
|---|---|---|---|
| 스토리지 | `kv_storage` / `vector_storage` / `graph_storage` / `doc_status_storage` | JsonKV / NanoVectorDB / NetworkX / JsonDocStatus | 백엔드 선택 (문자열 → factory) |
| | `workspace` | `""` | 멀티테넌트 격리 키 |
| 청킹 | `chunk_token_size` / `chunk_overlap_token_size` | 1200 / 100 | 논문 설정 그대로 |
| | `tokenizer` | TiktokenTokenizer("gpt-4o-mini") | 플러그블 (encode/decode 구현체면 됨) |
| 추출 | `entity_extract_max_gleaning` | 1 | 재추출 횟수 |
| | `entity_extract_max_records` / `max_entities` | 100 / 40 | 응답당 추출 상한 |
| | `entity_extraction_use_json` | False | JSON 구조화 출력 모드 |
| | `force_llm_summary_on_merge` | 8 | description 조각 N개↑면 LLM 요약 |
| | `summary_max_tokens` / `summary_length_recommended` | 1200 / 600 | 요약 트리거/목표 길이 |
| 질의 | `top_k` / `chunk_top_k` | 40 / 20 | 엔티티·관계 / 청크 검색 수 |
| | `max_entity_tokens` / `max_relation_tokens` / `max_total_tokens` | 6000 / 8000 / 30000 | **통합 토큰 예산** |
| | `cosine_threshold` | 0.2 | 벡터 검색 유사도 하한 |
| | `related_chunk_number` | 5 | 엔티티/관계당 연결 청크 수 |
| | `kg_chunk_pick_method` | "VECTOR" | 청크 선택: WEIGHT(빈도) vs VECTOR(질문 유사도) |
| 동시성 | `llm_model_max_async` | 4 | LLM 동시 호출 |
| | `max_parallel_insert` | 3 | 문서 병렬 처리 |
| | `embedding_batch_num` / `embedding_func_max_async` | 10 / 8 | 임베딩 배치/동시성 |
| 상한 | `max_source_ids_per_entity` / `per_relation` | 200 / 200 | 엔티티당 청크 참조 상한 (KEEP=오래된 것 유지 / FIFO) |
| | `max_file_paths` | 75 | 메타데이터 파일경로 상한 |
| 캐시 | `enable_llm_cache` / `enable_llm_cache_for_entity_extract` | True / True | LLM 응답 캐시 |
| 리랭크 | `rerank_model_func` / `min_rerank_score` | None / 0.0 | 옵션 |
| 역할 | `role_llm_configs` | None | **역할별 LLM 분리** (extract/keyword/query/vlm 각각 다른 모델·동시성 가능) |

스토리지는 `__post_init__`에서 factory(`get_storage_class`)로 클래스만 바인딩하고, `await rag.initialize_storages()`에서 실제 초기화 (lazy).

## 3. 스토리지 추상화 — 4종 ABC × 네임스페이스

LightRAG의 가장 이식 가치 높은 설계. **"무엇을 저장하나(네임스페이스)"와 "어디에 저장하나(백엔드)"를 직교 분리**.

### 3.1 ABC 4종 (`base.py`)

| ABC | 핵심 메서드 | 비고 |
|---|---|---|
| `BaseKVStorage` (379-439) | get_by_id(s) · filter_keys(존재하지 않는 키 반환 — dedup용) · upsert · delete | 쓰기는 버퍼링, `index_done_callback()`이 커밋 포인트 |
| `BaseVectorStorage` (220-376) | query(text, top_k, query_embedding?) · upsert · get_vectors_by_ids | `cosine_better_than_threshold=0.2`, `meta_fields`로 임베딩 대상 필드 지정 |
| `BaseGraphStorage` (442-787) | upsert_node/edge · get_node/edge · node_degree · get_node_edges · batch 변형들 · get_knowledge_graph(BFS 서브그래프) | **엣지는 무방향**. 노드 키 = entity_name |
| `DocStatusStorage` (864-980) | get_docs_by_status · get_doc_by_file_basename / by_content_hash (dedup용) · 페이지네이션 | KV 확장 |

공통 부모 `StorageNameSpace`: `initialize / finalize / index_done_callback(커밋) / drop_pending_index_ops(배치 중단 시 버퍼 폐기) / drop`.

### 3.2 논리 네임스페이스 (`namespace.py`)

| 네임스페이스 | 타입 | 내용 | 임베딩되는 텍스트 |
|---|---|---|---|
| `full_docs` | KV | 원문 | — |
| `text_chunks` | KV | 청크 (+ `llm_cache_list` — 삭제 시 rebuild 재료) | — |
| `llm_response_cache` | KV | LLM 캐시, 키 = `{mode}:{cache_type}:{hash}` | — |
| `full_entities` / `full_relations` | KV | 문서 → 엔티티/관계 목록 (삭제 추적용) | — |
| `entity_chunks` / `relation_chunks` | KV | 엔티티/관계 → 청크 ID 전체 목록 | — |
| **`entities`** | Vector | 엔티티 KV의 K | `"{entity_name}\n{description}"` |
| **`relationships`** | Vector | 관계 KV의 K | `"{keywords}\t{src}\n{tgt}\n{description}"` |
| **`chunks`** | Vector | 청크 | content |
| `chunk_entity_relation` | Graph | 지식 그래프 본체 | — |
| `doc_status` | DocStatus | 상태 머신 | — |

### 3.3 구현체 매트릭스 (`kg/__init__.py`)

- KV: Json / Redis / PG / Mongo / OpenSearch
- Vector: NanoVectorDB / Milvus / PGVector / Faiss / Qdrant / Mongo / OpenSearch
- Graph: NetworkX / Neo4j / PG(AGE) / Mongo / Memgraph / OpenSearch
- DocStatus: Json / Redis / PG / Mongo / OpenSearch

기본 4종(Json/NanoVectorDB/NetworkX/JsonDocStatus)은 **파일 기반 + 단일 프로세스 쓰기** 모델:
- NanoVectorDB: `vdb_<ns>.json` 단일 파일, 벡터는 `float16 + zlib + base64` 압축 저장, **지연 임베딩** (upsert는 버퍼만, flush 때 배치 임베딩 — 같은 id 반복 upsert 시 1회만 임베딩)
- NetworkX: `graph_<ns>.graphml` (GraphML XML), 원자적 쓰기(`atomic_write`)
- 크로스 프로세스 동기화는 "파일 + update flag" — 다른 프로세스가 커밋하면 다음 읽기 때 전체 리로드

프로덕션 참조 (postgres_impl ~3,800줄): pgvector HNSW/IVFFLAT 인덱스, 배치 상한(16MiB payload / 200 레코드), tenacity 재시도(3회 지수 백오프). Neo4j: workspace를 백틱 라벨로, fulltext Lucene 인덱스.

## 4. 동시성 모델

### 4.1 락 체계 (`kg/shared_storage.py`, 1,743줄)

| 락 | 범위 | 용도 |
|---|---|---|
| `pipeline_status_lock` | workspace | 파이프라인 단일 실행 보장 (`busy` 플래그). 실행 중 enqueue는 `request_pending=True`만 세팅 |
| `enqueue_serialize_lock` | workspace | dedup 체크 + upsert 임계 구역 직렬화 |
| `KeyedUnifiedLock` | entity/edge 키 단위 | **병합 레이스 방지** — `get_storage_keyed_lock(sorted(names))`. 미사용 락은 300초 후 lazy 정리 |
| `UnifiedLock` | — | asyncio.Lock(단일 프로세스)과 Manager().Lock(멀티프로세스, gunicorn 멀티워커)을 동일 인터페이스로 래핑 |

### 4.2 LLM 우선순위 큐 (`utils.py` `priority_limit_async_func_call`)

모든 LLM 콜이 `asyncio.PriorityQueue`(max 1000)를 통과:

```
DEFAULT_QUERY_PRIORITY = 5        # 대화형 질의 — 최우선
DEFAULT_SUMMARY_PRIORITY = 8      # 병합 요약
DEFAULT_PROCESSING_PRIORITY = 10  # 수집 추출 — 최하위
```

→ **수집이 돌아가는 중에도 사용자 질의가 끼어들 수 있는** 구조. 역할별 LLM 분리(`role_llm_configs`)와 결합하면 "추출은 저렴한 모델 + 높은 동시성, 질의는 좋은 모델"이 설정만으로 가능.

### 4.3 ID 규약 — 콘텐츠 어드레서블

`compute_mdhash_id(content, prefix)` = `prefix + MD5(content)`:

| 접두사 | 해시 대상 | 효과 |
|---|---|---|
| `doc-` | 문서 내용 (또는 정규화된 파일 경로) | 문서 dedup |
| `chunk-` | 청크 내용 | **문서 횡단 청크 dedup** — 같은 텍스트 조각은 재처리 안 함 |
| `ent-` | entity_name | VDB ID |
| `rel-` | src+tgt | VDB ID (무방향 — 정렬 후 해시) |

> GraphRAG-SDK가 청크에 UUID를 쓰는 것과 대조적. **자체 구현 시 콘텐츠 해시 방식 권장** — 증분 수집에서 무료 dedup을 얻는다.

## 5. 토크나이저 · 유틸

- `Tokenizer` 프로토콜: `encode(str)→list[int]` / `decode(list[int])→str` 구현체면 무엇이든. 기본 tiktoken (모델명으로 인코딩 자동 선택)
- `truncate_list_by_token_size(list, key, max_token_size, tokenizer)` — 토큰 예산 절단의 단일 구현 (질의 컨텍스트 구축 전역에서 사용)
- `EmbeddingFunc`: 차원·max_token 메타를 가진 래퍼, 배치/동시성 제어 내장

## 6. 자체 구현에 가져갈 것

1. **네임스페이스 × ABC 직교 분리** — 전처리기의 저장 계약을 이 4 ABC로 정의하면 백엔드 교체가 자유로워짐
2. **콘텐츠 해시 ID 체계** (`doc-`/`chunk-` MD5) — 증분 수집의 기반
3. **LLM 우선순위 큐 + 역할별 모델 분리** — 수집·질의가 같은 LLM 풀을 쓰는 운영 환경에서 필수적 패턴
4. **`index_done_callback` 커밋 포인트 + `drop_pending_index_ops`** — 배치 실패 시 부분 쓰기를 버리는 단순한 원자성 모델
5. constants.py처럼 **기본값 단일 출처** + 환경변수 오버라이드
