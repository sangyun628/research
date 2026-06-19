# 02. 아키텍처 & 데이터 모델

> 소스: `graphiti_core/graphiti.py`(1,793줄), `nodes.py`(1,122줄), `edges.py`(1,046줄), `driver/`

## 1. 패키지 구조

```
graphiti_core/
├── graphiti.py          # Graphiti 클래스 — 공개 API (add_episode·search·build_communities)
├── nodes.py             # EntityNode · EpisodicNode · CommunityNode · SagaNode
├── edges.py             # EntityEdge(bi-temporal) · EpisodicEdge · CommunityEdge
├── prompts/             # LLM 프롬프트 (추출·dedup·모순·요약)
├── utils/maintenance/   # node_operations · edge_operations · community_operations · temporal
├── utils/
│   ├── content_chunking.py  # episode 청킹 (밀도 기반)
│   └── bulk_utils.py        # add_episode_bulk
├── search/              # 검색·리랭킹·필터 (04 문서)
├── llm_client/          # OpenAI · Anthropic · Gemini 클라이언트
├── embedder/            # 임베딩 클라이언트
├── cross_encoder/       # 리랭킹용 cross-encoder
└── driver/              # 그래프 DB 추상화
    ├── neo4j_driver.py · falkordb_driver.py · kuzu_driver.py · neptune_driver.py
    └── {backend}/operations/  # 백엔드별 쿼리 구현
```

별도로 `mcp_server/`(MCP 서버), `server/`(FastAPI REST)가 코어를 감싼다.

## 2. Graphiti 클래스 — 공개 API

`graphiti.py:138-248` 생성자:

```python
Graphiti(
    uri, user, password,           # 또는 graph_driver 직접 주입
    llm_client: LLMClient,         # OpenAI/Anthropic/Gemini
    embedder: EmbedderClient,
    cross_encoder: CrossEncoderClient,   # 리랭킹용
    store_raw_episode_content: bool = True,
    max_coroutines: int | None = None,   # 동시성 상한 (SEMAPHORE_LIMIT 오버라이드)
    tracer, trace_span_prefix='graphiti',
)
```

| 메서드 | 위치 | 역할 |
|---|---|---|
| `add_episode(...)` | 980-1228 | **핵심 수집** — episode 1개 처리 (추출·해소·무효화·임베딩) → `AddEpisodeResults` |
| `add_episode_bulk(...)` | 1230-1487 | 배치 수집 (공유 dedup 컨텍스트) |
| `build_communities(...)` | 1490-1524 | 커뮤니티 탐지 + 요약 |
| `search(query, center_node_uuid?, ...)` | 1527 | **간단 검색** — 엣지만 반환 (`list[EntityEdge]`) |
| `search_(query, config, ...)` | 1603 | **설정형 검색** — node/edge/episode/community 전부 (`SearchResults`) |
| `retrieve_episodes(reference_time, last_n)` | 927 | 최근 N개 episode 조회 |
| `add_triplet(source, edge, target)` | 1645 | 수동 fact 1개 추가 (dedup·무효화 포함) |
| `summarize_saga(...)` | 438 | saga 증분 요약 (watermark 기반) |

**Episode란?** 입력 1단위 — `name`, `content`(원본), `source`(message/json/text/fact_triple), `source_description`, `valid_at`(event time), `created_at`(transaction time). `reference_time` 파라미터로 이벤트 시점을 명시한다.

## 3. 데이터 모델 — 전체 필드

### EntityNode (`nodes.py:499-685`)
```python
uuid: str                          # 엔티티 식별자 (이름과 분리 — LightRAG와 차이)
name: str
group_id: str                      # 파티션 키 (멀티테넌시)
labels: list[str]                  # 엔티티 타입들
created_at: datetime
name_embedding: list[float] | None # 이름 임베딩
summary: str                       # 주변 엣지 요약
attributes: dict[str, Any]         # 커스텀 속성 (Pydantic 스키마)
```

### EpisodicNode (`nodes.py:318-497`)
```python
uuid, name, group_id, labels, created_at  # (Node 상속)
source: EpisodeType                # message | json | text | fact_triple
source_description: str            # "Chat API", "DART 공시" 등
content: str                       # 원본 (store_raw_episode_content=False면 비움)
valid_at: datetime                 # event time
entity_edges: list[str]            # 이 episode에서 추출된 EntityEdge uuid들
episode_metadata: dict | None      # 커스텀 필터용
```

### EntityEdge (`edges.py:263-286`) — bi-temporal
```python
uuid, group_id, source_node_uuid, target_node_uuid, created_at
name: str                          # 관계 타입 (예: WORKS_AT)
fact: str                          # 자연어 사실 문장
fact_embedding: list[float] | None # fact 임베딩 (검색 대상)
valid_at, invalid_at: datetime|None    # event time (참/거짓 구간)
expired_at: datetime | None            # transaction time (무효화 기록 시점)
reference_time: datetime | None        # 생성 episode 타임스탬프
episodes: list[str]                # 이 fact를 언급한 episode uuid들 (provenance)
attributes: dict[str, Any]         # 커스텀 엣지 속성
```

### 그 외 엣지 (구조 전용, 시간축 없음)
- `EpisodicEdge`: Episode →(MENTIONS)→ Entity
- `CommunityEdge`: Community →(HAS_MEMBER)→ Entity
- `HasEpisodeEdge`: Saga →(HAS_EPISODE)→ Episode, `NextEpisodeEdge`: Episode →(NEXT_EPISODE)→ Episode

> 핵심 설계: **EntityEdge만 bi-temporal**이고 나머지는 구조 엣지. fact(의미)에만 시간을 부여하고 provenance·구조는 시간 무관으로 분리한 것.

## 4. group_id — 멀티테넌시

모든 노드·엣지가 `group_id`(문자열 파티션 키)를 가진다. `add_episode(group_id=...)`로 지정, 기본값은 드라이버별(`''` Neo4j / `'_'` FalkorDB). 검색은 `group_ids`(리스트)로 스코프. **단일 Graphiti 인스턴스가 여러 격리된 그래프**를 서빙 — 사용자별·테넌트별 메모리 분리에 사용. (group_id가 드라이버 DB명과 다르면 드라이버를 복제해 별도 DB로 라우팅.)

## 5. 임베딩 저장

- `EntityNode.name_embedding` — 이름 임베딩 (노드 속성으로 저장)
- `EntityEdge.fact_embedding` — fact 문장 임베딩 (엣지 속성으로 저장, 검색의 핵심)
- `CommunityNode.name_embedding`
- `create_entity_edge_embeddings()` / `create_entity_node_embeddings()`로 배치 병렬 생성

> LightRAG이 엔티티·관계·청크를 별도 벡터 인덱스에 두는 것과 달리, Graphiti는 **그래프 노드/엣지 속성에 임베딩을 직접 저장**한다 (그래프 DB의 벡터 인덱스 기능 사용).

## 6. 드라이버 추상화 — 다중 백엔드

`driver/`가 그래프 DB를 추상화: **Neo4j · FalkorDB · Kuzu · Neptune**. `GraphProvider` enum으로 분기하고, 백엔드별 `operations/`에 쿼리(search_ops 등)를 구현. 쿼리 차이(Cypher 방언, 벡터 함수)를 드라이버가 흡수. → LightRAG의 12종 스토리지 ABC와 유사한 사상이나, Graphiti는 **그래프 DB만** 지원(KV/벡터를 그래프 DB 안에서 해결).

## 7. 동시성

```python
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))   # 기본 20
```
`semaphore_gather()`가 asyncio.gather를 세마포어로 감싸 LLM·검색·임베딩 호출을 제한. 적용처: 노드 dedup 병렬 검색, 엣지 검색(관련+무효화 후보), 속성 추출, 커뮤니티 빌드, 임베딩 생성. `max_coroutines`로 인스턴스별 오버라이드.

## 8. 자체 구현에 가져갈 것

1. **uuid 기반 엔티티 ID** (이름과 분리) — LightRAG의 "이름=ID" 한계(표기 분열·rename 불가)를 피하는 기본. resolution이 uuid를 canonical로 묶음
2. **fact에만 bi-temporal, 구조 엣지는 시간 무관** — 시간축을 의미 fact에 한정하는 분리
3. **임베딩을 그래프 속성에 저장** — 그래프 DB가 벡터 인덱스를 지원하면 별도 벡터 스토어 불필요 (단 우리는 OpenSearch라 LightRAG식 분리가 맞음)
4. **group_id 멀티테넌시** — 단일 인스턴스 다중 그래프. 회사별·도메인별 격리에 활용 가능
