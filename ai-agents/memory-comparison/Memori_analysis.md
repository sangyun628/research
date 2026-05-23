# Memori 에이전트 메모리 시스템 분석 보고서

## 1. 개요

**Memori**는 엔터프라이즈 AI를 위한 **메모리 패브릭(Memory Fabric)**입니다. LLM, 데이터스토어, 프레임워크에 구애받지 않으며, 기존 아키텍처에 한 줄의 코드로 통합할 수 있습니다.

- **GitHub**: https://github.com/MemoriLabs/Memori
- **라이선스**: Apache 2.0
- **Python 버전**: 3.8+
- **현재 버전**: v3

### 1.1 핵심 특징

| 기능 | 설명 |
|------|------|
| **제로 레이턴시** | Advanced Augmentation이 백그라운드에서 비동기 실행 |
| **LLM 애그노스틱** | OpenAI, Anthropic, Gemini, Grok, Bedrock 지원 |
| **데이터스토어 애그노스틱** | PostgreSQL, MySQL, SQLite, MongoDB, Oracle 등 |
| **Knowledge Graph** | 시맨틱 트리플(Subject-Predicate-Object) 자동 구축 |
| **자동 스키마 마이그레이션** | 데이터베이스 스키마 자동 생성/업데이트 |

### 1.2 MemU와의 차이점

| 측면 | MemU | Memori |
|------|------|--------|
| **메모리 추출** | 자체 LLM 파이프라인 | 외부 API (Advanced Augmentation) |
| **메모리 타입** | profile, event, knowledge, behavior, skill | facts, attributes, semantic triples |
| **계층 구조** | Resource → Item → Category | Entity → Fact / Process → Attribute |
| **통합 방식** | 명시적 memorize/retrieve 호출 | LLM 클라이언트 래핑 (자동) |
| **검색** | RAG + LLM 랭킹 | FAISS 기반 시맨틱 검색 |

---

## 2. 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         Memori                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   LLM Client Wrapper                     │    │
│  │   (OpenAI, Anthropic, Gemini, Grok, Bedrock 지원)        │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                 Advanced Augmentation                  │      │
│  │   ┌─────────────────────────────────────────────┐     │      │
│  │   │  • Fact 추출                                 │     │      │
│  │   │  • Semantic Triple (NER) 추출               │     │      │
│  │   │  • Conversation Summary 생성                │     │      │
│  │   │  • Process Attribute 추출                   │     │      │
│  │   └─────────────────────────────────────────────┘     │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                   Storage Driver                       │      │
│  │   ┌──────────┬──────────┬──────────┬──────────┐       │      │
│  │   │ SQLite   │ Postgres │  MySQL   │ MongoDB  │       │      │
│  │   └──────────┴──────────┴──────────┴──────────┘       │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                  Semantic Search                       │      │
│  │   ┌─────────────────────────────────────────────┐     │      │
│  │   │  FAISS + Sentence Transformers (768 dim)    │     │      │
│  │   └─────────────────────────────────────────────┘     │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Attribution 모델

Memori는 메모리를 **두 가지 핵심 엔티티**에 귀속시킵니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Attribution Model                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Entity (사용자/엔티티)              Process (에이전트/프로세스)  │
│  ├── Facts (사실)                   ├── Attributes (속성)        │
│  │   └── 임베딩 벡터                │   └── 프로세스가 다루는 주제 │
│  └── Semantic Triples               └── 사용자와의 상호작용 패턴  │
│      └── Knowledge Graph                                         │
│                                                                  │
│                    ┌───────────────┐                             │
│                    │    Session    │                             │
│                    │  (대화 세션)   │                             │
│                    └───────┬───────┘                             │
│                            │                                     │
│                    ┌───────▼───────┐                             │
│                    │  Conversation │                             │
│                    │   (대화 내역)  │                             │
│                    │   + Summary   │                             │
│                    └───────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 데이터베이스 스키마 (ERD)

```
┌──────────────────┐     ┌──────────────────┐
│  memori_entity   │     │  memori_process  │
├──────────────────┤     ├──────────────────┤
│ id (PK)          │     │ id (PK)          │
│ uuid             │     │ uuid             │
│ external_id (UK) │     │ external_id (UK) │
│ date_created     │     │ date_created     │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         │    ┌───────────────────┴───────────────────┐
         │    │           memori_session              │
         │    ├───────────────────────────────────────┤
         │    │ id (PK)                               │
         │    │ uuid                                  │
         │    │ entity_id (FK)                        │
         │    │ process_id (FK)                       │
         │    └───────────────────┬───────────────────┘
         │                        │
         │              ┌─────────▼─────────┐
         │              │ memori_conversation│
         │              ├───────────────────┤
         │              │ id (PK)           │
         │              │ session_id (FK)   │
         │              │ summary           │
         │              └─────────┬─────────┘
         │                        │
         │              ┌─────────▼─────────────────┐
         │              │ memori_conversation_message│
         │              ├───────────────────────────┤
         │              │ id (PK)                   │
         │              │ conversation_id (FK)      │
         │              │ role (user/assistant)     │
         │              │ content                   │
         │              └───────────────────────────┘
         │
┌────────▼─────────────┐
│  memori_entity_fact  │
├──────────────────────┤
│ id (PK)              │
│ entity_id (FK)       │
│ content              │
│ content_embedding    │  ← 768 dim 벡터
│ num_times            │  ← 언급 횟수
│ date_last_time       │  ← 마지막 언급 시간
│ uniq                 │  ← 중복 방지 해시
└──────────────────────┘

┌──────────────────────────┐
│ memori_process_attribute │
├──────────────────────────┤
│ id (PK)                  │
│ process_id (FK)          │
│ content                  │
│ uniq                     │
└──────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Tables                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  memori_subject      memori_predicate      memori_object        │
│  ├── id (PK)         ├── id (PK)           ├── id (PK)          │
│  ├── name            ├── content           ├── name             │
│  ├── type            └── uniq              ├── type             │
│  └── uniq                                  └── uniq             │
│                                                                  │
│                  memori_knowledge_graph                          │
│                  ├── id (PK)                                     │
│                  ├── entity_id (FK)                              │
│                  ├── subject_id (FK)                             │
│                  ├── predicate_id (FK)                           │
│                  └── object_id (FK)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 메모리 그룹화 및 계층화

### 3.1 메모리 유형 (Tulving 이론 매핑)

| Memori 유형 | Tulving 분류 | 설명 | 저장 위치 |
|------------|-------------|------|----------|
| **Facts** | Semantic | 사용자에 대한 사실적 정보 | memori_entity_fact |
| **Attributes** | Semantic | 프로세스가 다루는 주제/특성 | memori_process_attribute |
| **Semantic Triples** | Semantic | 지식 그래프 (S-P-O) | memori_knowledge_graph |
| **Conversation** | Episodic | 대화 내역 및 요약 | memori_conversation |
| **Session** | Episodic | 세션별 상호작용 그룹화 | memori_session |

### 3.2 2계층 귀속 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entity (사용자) 레벨                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Facts (사실)                                                    │
│  ├── "John likes hiking"                                        │
│  ├── "John works as a software engineer"                        │
│  └── "John's favorite color is blue"                            │
│                                                                  │
│  Knowledge Graph (Semantic Triples)                             │
│  ├── (John, likes, hiking)                                      │
│  ├── (John, works_as, software engineer)                        │
│  └── (John, favorite_color, blue)                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Process (에이전트) 레벨                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Attributes (속성)                                               │
│  ├── "handles travel recommendations"                           │
│  ├── "discusses outdoor activities"                             │
│  └── "provides career advice"                                   │
│                                                                  │
│  → Entity의 Facts와 Process의 Attributes를 매칭하여             │
│    가장 관련성 높은 컨텍스트 제공                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Fact 추출 및 중복 처리

```python
# Fact 저장 로직 (memori/storage/drivers/sqlite/_driver.py)

def create(self, entity_id: int, facts: list, fact_embeddings: list):
    for i, fact in enumerate(facts):
        embedding = fact_embeddings[i]
        uniq = generate_uniq([fact])  # 해시 생성

        # UPSERT: 중복 시 카운트 증가
        """
        INSERT INTO memori_entity_fact(...)
        VALUES (...)
        ON CONFLICT(entity_id, uniq) DO UPDATE SET
            num_times = num_times + 1,         ← 언급 횟수 증가
            date_last_time = datetime('now')   ← 최근 언급 시간 갱신
        """
```

**중복 처리 전략:**
- `uniq` 필드: 내용 기반 해시로 중복 판별
- `num_times`: 동일 정보 반복 언급 시 카운트 증가
- `date_last_time`: 최근성 기반 검색 우선순위

---

## 4. Advanced Augmentation 상세

### 4.1 처리 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                 Advanced Augmentation Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣ LLM 호출 감지                                               │
│     └── Memori가 등록된 LLM 클라이언트의 호출을 인터셉트          │
│                                                                  │
│  2️⃣ 대화 수집                                                   │
│     └── 메시지 히스토리 + 현재 응답 수집                          │
│                                                                  │
│  3️⃣ API 호출 (비동기)                                           │
│     └── Advanced Augmentation 서비스로 대화 전송                  │
│                                                                  │
│  4️⃣ 결과 처리                                                   │
│     ├── Facts 추출 + 임베딩 생성                                 │
│     ├── Semantic Triples (NER) 추출                             │
│     ├── Process Attributes 추출                                 │
│     └── Conversation Summary 생성                               │
│                                                                  │
│  5️⃣ 데이터 저장                                                 │
│     └── 각 테이블에 비동기로 저장                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 데이터 구조 (Memories 클래스)

```python
# memori/memory/_struct.py

class Memories:
    conversation: Conversation  # 대화 요약
    entity: Entity              # 사용자 관련 메모리
    process: Process            # 프로세스 관련 메모리

class Entity:
    facts: list[str]                    # 추출된 사실들
    fact_embeddings: list[list[float]]  # 768차원 임베딩
    semantic_triples: list[SemanticTriple]  # 지식 그래프 트리플

class SemanticTriple:
    subject_name: str   # 주어 이름
    subject_type: str   # 주어 타입 (person, place, thing 등)
    predicate: str      # 술어 (관계)
    object_name: str    # 목적어 이름
    object_type: str    # 목적어 타입

class Process:
    attributes: list[str]  # 프로세스가 다루는 주제들

class Conversation:
    summary: str | None  # 대화 요약
```

### 4.3 Fact → Semantic Triple 변환

```python
# API 응답에서 Semantic Triple → Fact 자동 생성
if not facts and triples:
    facts = [
        f"{t['subject']['name']} {t['predicate']} {t['object']['name']}"
        for t in triples
    ]

# 예시:
# Triple: (John, likes, hiking)
# → Fact: "John likes hiking"
```

---

## 5. 검색 시스템

### 5.1 시맨틱 검색 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Recall Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣ 쿼리 임베딩 생성                                            │
│     └── Sentence Transformers (768 dim)                          │
│                                                                  │
│  2️⃣ Entity Facts 조회                                           │
│     └── entity_id로 해당 사용자의 모든 Facts 조회                 │
│                                                                  │
│  3️⃣ FAISS 유사도 검색                                           │
│     └── 쿼리 임베딩 vs Fact 임베딩                               │
│                                                                  │
│  4️⃣ 하이브리드 랭킹 (Dense + Lexical)                           │
│     ├── Cosine Similarity (Dense)                               │
│     └── BM25-style Lexical Score                                │
│                                                                  │
│  5️⃣ Top-K Facts를 시스템 프롬프트에 주입                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 검색 코드 분석

```python
# memori/search/_core.py

def search_entity_facts_core(
    entity_fact_driver,
    entity_id: int,
    query_embedding: list[float],
    limit: int,
    embeddings_limit: int,
    *,
    query_text: str | None,
    find_similar_embeddings,    # FAISS 검색 함수
    lexical_scores_for_ids,     # Lexical 점수 계산
    dense_lexical_weights,      # 가중치 조정
):
    # 1. Entity의 임베딩 조회
    results = _get_embeddings_rows(entity_fact_driver, entity_id=entity_id)

    # 2. FAISS로 유사한 임베딩 검색
    similar = find_similar_embeddings(embeddings, query_embedding, cand_limit)

    # 3. 하이브리드 랭킹
    if query_text:
        lex_scores = lexical_scores_for_ids(query_text, candidate_ids, content_map)
        w_cos, w_lex = dense_lexical_weights(query_text)

        rank_score = w_cos * cosine_similarity + w_lex * lexical_score
```

### 5.3 임베딩 시스템

```python
# memori/embeddings/_sentence_transformers.py

class SentenceTransformersEmbedder:
    """
    Sentence Transformers 기반 임베딩 생성기
    - 768 차원 벡터
    - 정규화된 임베딩 (normalize_embeddings=True)
    - 긴 텍스트는 청크 분할 후 평균 풀링
    """

    def _encode_batch(self, encoder, inputs: list[str]) -> list[list[float]]:
        embeddings = encoder.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 정규화
        )
        return embeddings.tolist()

    def _mean_pool_and_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """긴 텍스트 청크들의 평균 풀링"""
        mean_vec = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec = mean_vec / norm
        return mean_vec
```

---

## 6. LLM 통합

### 6.1 지원 LLM 및 프레임워크

**LLM 프로바이더:**
- OpenAI (Chat Completions & Responses API)
- Anthropic
- Google Gemini
- xAI Grok
- AWS Bedrock

**프레임워크:**
- LangChain
- Agno

**지원 모드:**
- Synchronous / Asynchronous
- Streaming / Non-streaming

### 6.2 LLM 클라이언트 래핑

```python
# 사용 예시
from openai import OpenAI
from memori import Memori

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Memori가 클라이언트를 래핑
memori = Memori(conn=get_sqlite_connection).llm.register(client)
memori.attribution(entity_id="123456", process_id="test-ai-agent")

# 이후 모든 LLM 호출은 자동으로 Memori가 인터셉트
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}]
)
# → Memori가 자동으로:
#   1. 대화 저장
#   2. Fact 추출 ("user's favorite color is blue")
#   3. Semantic Triple 생성 ((user, favorite_color, blue))
#   4. 임베딩 생성 및 저장
```

### 6.3 래핑 메커니즘

```python
# memori/llm/_base.py

class BaseClient:
    def _wrap_method(self, obj, method_name, ...):
        """LLM 메서드를 Invoke 래퍼로 감싸기"""

        # 비동기 여부 자동 감지
        is_async = inspect.iscoroutinefunction(original)

        if is_async:
            wrapper_class = InvokeAsyncStream if stream else InvokeAsync
        else:
            wrapper_class = InvokeStream if stream else Invoke

        # 원본 메서드를 래퍼로 교체
        setattr(obj, method_name, wrapper_class(config, original).invoke)
```

---

## 7. 데이터스토어 통합

### 7.1 지원 데이터베이스

| 데이터베이스 | 드라이버 | 벡터 저장 방식 |
|------------|---------|--------------|
| SQLite | sqlite3 | BLOB (JSON 직렬화) |
| PostgreSQL | psycopg / SQLAlchemy | pgvector 또는 JSONB |
| MySQL/MariaDB | pymysql / MySQLdb | JSON 타입 |
| MongoDB | pymongo | Array 필드 |
| Oracle | cx_Oracle / oracledb | BLOB |
| CockroachDB | psycopg | JSONB |

### 7.2 드라이버 아키텍처

```python
# 추상 기본 클래스
class BaseStorageAdapter:
    conversation: BaseConversation
    entity: BaseEntity
    entity_fact: BaseEntityFact
    knowledge_graph: BaseKnowledgeGraph
    process: BaseProcess
    process_attribute: BaseProcessAttribute
    session: BaseSession

# 구체 구현 (SQLite 예시)
@Registry.register("sqlite")
class SqliteDriver(BaseStorageAdapter):
    def __init__(self, conn):
        self.conversation = Conversation(conn)
        self.entity = Entity(conn)
        self.entity_fact = EntityFact(conn)
        # ...
```

### 7.3 자동 스키마 마이그레이션

```python
# 마이그레이션 실행
Memori(conn=db_connection).config.storage.build()

# 내부적으로:
# 1. memori_schema_version 테이블 확인
# 2. 현재 버전과 최신 버전 비교
# 3. 필요한 마이그레이션 순차 실행
```

---

## 8. 성능 최적화

### 8.1 제로 레이턴시 설계

```
┌─────────────────────────────────────────────────────────────────┐
│                    Zero Latency Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Request ──► LLM Call ──► Response ──► User                │
│                       │                                          │
│                       └──► Background Thread ──► Augmentation    │
│                                                   ├── Fact 추출  │
│                                                   ├── 임베딩 생성│
│                                                   └── DB 저장    │
│                                                                  │
│  → LLM 응답 지연 없음 (메모리 처리는 백그라운드)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 검색 최적화

```python
# 후보 제한 전략
def _candidate_limit(*, limit: int, total_embeddings: int, query_text: str):
    if query_text:
        # 텍스트 쿼리 시 더 많은 후보 검색 후 재랭킹
        return max(limit, min(total_embeddings, max(limit * 10, 50)))
    return limit
```

### 8.3 임베딩 캐싱

```python
# 모델 지연 로딩 + 스레드 안전
class SentenceTransformersEmbedder:
    def _get_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._model is None:
                self._model = SentenceTransformer(self._model_name)
            return self._model
```

---

## 9. MemU와의 비교 분석

### 9.1 메모리 추출 접근법

| 측면 | MemU | Memori |
|------|------|--------|
| **추출 위치** | 자체 LLM 파이프라인 | 외부 API 서비스 |
| **메모리 타입** | 5종 (profile, event, knowledge, behavior, skill) | 3종 (facts, attributes, triples) |
| **커스터마이징** | 프롬프트 블록별 커스텀 가능 | API 의존 (제한적) |
| **비용** | 자체 LLM 토큰 소비 | API 무료 (레이트 리밋) |

### 9.2 데이터 모델

| 측면 | MemU | Memori |
|------|------|--------|
| **계층 구조** | 3계층 (Resource→Item→Category) | 2계층 (Entity/Process→Fact/Attribute) |
| **카테고리** | 10개 사전 정의 카테고리 | 없음 (Flat) |
| **요약** | 카테고리별 Markdown 요약 | 대화별 요약만 |
| **Knowledge Graph** | 없음 | Semantic Triple 지원 |

### 9.3 검색 방식

| 측면 | MemU | Memori |
|------|------|--------|
| **검색 방법** | RAG + LLM 랭킹 | FAISS + 하이브리드 랭킹 |
| **계층적 검색** | Category → Item → Resource | Flat (Facts만) |
| **충분성 검사** | LLM 기반 판단 | 없음 |
| **쿼리 재작성** | 지원 | 없음 |

### 9.4 통합 방식

| 측면 | MemU | Memori |
|------|------|--------|
| **통합 코드** | 명시적 memorize/retrieve 호출 | LLM 클라이언트 래핑 (자동) |
| **사용 복잡도** | 중간 | 낮음 (한 줄) |
| **제어 수준** | 높음 (파이프라인 커스텀) | 낮음 |

---

## 10. 사용 시나리오

### 10.1 Memori가 적합한 경우

- **빠른 통합**이 필요한 경우 (한 줄 코드)
- **제로 레이턴시**가 중요한 경우
- **Knowledge Graph**가 필요한 경우
- **다양한 데이터스토어**를 사용하는 경우

### 10.2 MemU가 적합한 경우

- **세밀한 메모리 타입 분류**가 필요한 경우
- **커스텀 프롬프트**로 도메인 특화가 필요한 경우
- **계층적 검색**이 필요한 경우
- **카테고리별 요약**이 필요한 경우

---

## 11. 주요 의존성

```toml
[dependencies]
faiss-cpu = "^1.7"           # 벡터 검색
sentence-transformers = "*"   # 임베딩 생성
numpy = "*"                   # 수치 연산
pydantic = "*"               # 데이터 검증

[optional]
openai = "*"
anthropic = "*"
google-generativeai = "*"
langchain = "*"
```

---

## 12. 결론

Memori는 **간편한 통합**과 **제로 레이턴시**에 초점을 맞춘 에이전트 메모리 솔루션입니다:

- **한 줄 통합**: LLM 클라이언트 래핑으로 자동 메모리 처리
- **Knowledge Graph**: Semantic Triple 기반 지식 그래프 자동 구축
- **유연한 스토리지**: 다양한 데이터베이스 지원
- **백그라운드 처리**: LLM 응답 지연 없음

MemU가 **정교한 메모리 분류와 검색**에 강점이 있다면, Memori는 **통합 용이성과 지식 그래프**에 강점이 있습니다. 프로젝트 요구사항에 따라 적절한 선택이 필요합니다.
