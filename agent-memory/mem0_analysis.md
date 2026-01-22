# Mem0 에이전트 메모리 시스템 분석 보고서

## 1. 개요

**Mem0** (발음: "mem-zero")는 AI 어시스턴트와 에이전트에 **지능형 메모리 레이어**를 제공하는 오픈소스 프레임워크입니다. 사용자 선호도를 기억하고, 개별 요구에 적응하며, 시간이 지남에 따라 지속적으로 학습합니다.

- **GitHub**: https://github.com/mem0ai/mem0
- **라이선스**: Apache 2.0
- **Python 버전**: 3.8+
- **현재 버전**: v1.0.0+
- **Y Combinator**: S24 배치

### 1.1 핵심 성능 지표 (LOCOMO 벤치마크)

| 지표 | 결과 |
|------|------|
| **정확도** | OpenAI Memory 대비 +26% |
| **응답 속도** | Full-context 대비 91% 빠름 |
| **토큰 사용량** | Full-context 대비 90% 절감 |

### 1.2 핵심 특징

| 기능 | 설명 |
|------|------|
| **Multi-Level Memory** | User, Session, Agent 상태를 적응형으로 유지 |
| **Graph Memory** | Neo4j 기반 Knowledge Graph 지원 |
| **Hybrid Search** | Vector + BM25 하이브리드 검색 |
| **Procedural Memory** | 에이전트 실행 히스토리 저장 |

---

## 2. 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                           Memory                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    add() / search()                      │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │              Parallel Processing (ThreadPool)          │      │
│  │   ┌─────────────────────┬─────────────────────┐       │      │
│  │   │   Vector Store      │    Graph Store      │       │      │
│  │   │   (Facts)           │    (Relations)      │       │      │
│  │   └─────────────────────┴─────────────────────┘       │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                     LLM Layer                          │      │
│  │   ┌────────────────────────────────────────────┐      │      │
│  │   │  Fact Extraction → Update Memory Decision  │      │      │
│  │   └────────────────────────────────────────────┘      │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                   Storage Layer                        │      │
│  │   ┌─────────────────────┬─────────────────────┐       │      │
│  │   │ Vector Stores       │  Graph Stores       │       │      │
│  │   │ (20+ 지원)          │  (Neo4j, Kuzu 등)   │       │      │
│  │   └─────────────────────┴─────────────────────┘       │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Multi-Level Attribution 모델

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attribution Hierarchy                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  user_id (사용자)                                                │
│  ├── 사용자별 개인 메모리                                        │
│  ├── 선호도, 습관, 개인 정보 저장                                │
│  └── 여러 에이전트/세션에 걸쳐 공유                              │
│                                                                  │
│  agent_id (에이전트)                                             │
│  ├── 에이전트별 특화 메모리                                      │
│  ├── 에이전트 성격, 스킬, 접근 방식                              │
│  └── Procedural Memory 저장 (agent_id 필수)                      │
│                                                                  │
│  run_id (실행)                                                   │
│  ├── 단일 실행 컨텍스트                                          │
│  ├── 세션 격리 메모리                                            │
│  └── 일시적 상태 저장                                            │
│                                                                  │
│  ⚠️ 최소 하나의 ID 필수 (user_id, agent_id, run_id 중 택일)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 데이터 모델

```python
class MemoryItem(BaseModel):
    id: str                           # UUID
    memory: str                       # 메모리 내용
    hash: Optional[str]               # 중복 검사용 해시
    metadata: Optional[Dict[str, Any]]  # user_id, agent_id, run_id 등
    score: Optional[float]            # 검색 점수
    created_at: Optional[str]         # 생성 시간
    updated_at: Optional[str]         # 수정 시간
```

---

## 3. 메모리 그룹화 및 계층화

### 3.1 메모리 유형 (Tulving 이론 매핑)

| Mem0 개념 | Tulving 분류 | 설명 | 추출 방식 |
|----------|-------------|------|----------|
| **User Memory** | Semantic | 사용자 Facts (선호도, 개인정보) | USER_MEMORY_EXTRACTION_PROMPT |
| **Agent Memory** | Semantic | 에이전트 Facts (성격, 스킬) | AGENT_MEMORY_EXTRACTION_PROMPT |
| **Procedural Memory** | Procedural | 에이전트 실행 히스토리 | PROCEDURAL_MEMORY_SYSTEM_PROMPT |
| **Graph Relations** | Semantic | 엔티티 간 관계 (Knowledge Graph) | Entity + Relation 추출 |

### 3.2 Fact 추출 카테고리

```
┌─────────────────────────────────────────────────────────────────┐
│                   7가지 정보 유형                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Personal Preferences (개인 선호도)                           │
│     • 음식, 제품, 활동, 엔터테인먼트 선호                         │
│                                                                  │
│  2. Important Personal Details (중요 개인 정보)                  │
│     • 이름, 관계, 중요 날짜                                      │
│                                                                  │
│  3. Plans and Intentions (계획/의도)                             │
│     • 예정된 이벤트, 여행, 목표                                  │
│                                                                  │
│  4. Activity/Service Preferences (활동/서비스 선호)              │
│     • 식사, 여행, 취미, 서비스 선호                               │
│                                                                  │
│  5. Health and Wellness (건강/웰니스)                            │
│     • 식이 제한, 피트니스 루틴, 웰니스 정보                       │
│                                                                  │
│  6. Professional Details (전문/직업 정보)                        │
│     • 직함, 업무 습관, 커리어 목표                                │
│                                                                  │
│  7. Miscellaneous (기타)                                         │
│     • 좋아하는 책, 영화, 브랜드 등                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 메모리 업데이트 결정 로직

```
┌─────────────────────────────────────────────────────────────────┐
│                 Memory Update Operations                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣ ADD - 새로운 정보 추가                                      │
│     • 기존 메모리에 없는 새로운 fact                             │
│     • 새 ID 생성                                                 │
│                                                                  │
│  2️⃣ UPDATE - 기존 정보 업데이트                                 │
│     • 기존 메모리와 의미적으로 유사하지만 더 구체적인 정보        │
│     • 동일 ID 유지, 내용만 변경                                  │
│     • old_memory 필드에 이전 내용 기록                           │
│                                                                  │
│  3️⃣ DELETE - 모순 정보 삭제                                     │
│     • 새 fact가 기존 메모리와 모순                               │
│     • 예: "좋아함" → "싫어함"                                    │
│                                                                  │
│  4️⃣ NONE - 변경 없음                                            │
│     • 이미 존재하는 동일 정보                                    │
│     • 불필요한 중복 방지                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Graph Memory (Knowledge Graph)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neo4j Graph Structure                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Entity Extraction:                                              │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │    Entity    │     │  Entity Type │                          │
│  │    "John"    │ ──► │   "person"   │                          │
│  └──────────────┘     └──────────────┘                          │
│                                                                  │
│  Relation Extraction:                                            │
│  ┌──────────────┐            ┌──────────────┐                   │
│  │    Source    │   WORKS_AT │   Target     │                   │
│  │    "John"    │ ──────────►│   "Google"   │                   │
│  └──────────────┘            └──────────────┘                   │
│                                                                  │
│  Cypher Query Example:                                           │
│  MATCH (n:__Entity__ {user_id: $user_id})-[r]->(m:__Entity__)   │
│  RETURN n.name AS source, type(r) AS relationship, m.name AS target │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 주요 프롬프트 상세

### 4.1 User Memory 추출 프롬프트

**목적**: 사용자 메시지에서만 facts 추출 (assistant 메시지 제외)

```
# Task Objective
You are a Personal Information Organizer, specialized in accurately storing
facts, user memories, and preferences.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES.
# DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.
# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM
# ASSISTANT OR SYSTEM MESSAGES.

# Types of Information to Remember:
1. Store Personal Preferences (음식, 제품, 활동, 엔터테인먼트)
2. Maintain Important Personal Details (이름, 관계, 중요 날짜)
3. Track Plans and Intentions (이벤트, 여행, 목표)
4. Remember Activity and Service Preferences (식사, 여행, 취미)
5. Monitor Health and Wellness Preferences (식이 제한, 피트니스)
6. Store Professional Details (직함, 업무 습관, 커리어 목표)
7. Miscellaneous Information (책, 영화, 브랜드 등)

# Output Format
{"facts": ["Fact 1", "Fact 2", ...]}
```

**Few-shot 예시:**
```
User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John! My name is Alex.
Output: {"facts": ["Name is John", "Is a Software engineer"]}

User: Me favourite movies are Inception and Interstellar. What are yours?
Assistant: Great choices! Mine are The Dark Knight and The Shawshank Redemption.
Output: {"facts": ["Favourite movies are Inception and Interstellar"]}
```

### 4.2 Agent Memory 추출 프롬프트

**목적**: Assistant 메시지에서만 에이전트 특성 추출

```
# Task Objective
You are an Assistant Information Organizer, specialized in accurately
storing facts, preferences, and characteristics about the AI assistant.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES.
# DO NOT INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.

# Types of Information to Remember:
1. Assistant's Preferences (활동, 관심 주제, 가상 시나리오)
2. Assistant's Capabilities (스킬, 지식 영역, 수행 가능 작업)
3. Assistant's Hypothetical Plans (가상 활동/계획)
4. Assistant's Personality Traits (성격 특성)
5. Assistant's Approach to Tasks (작업 접근 방식)
6. Assistant's Knowledge Areas (전문 분야)
7. Miscellaneous Information (기타 고유 정보)

# Output Format
{"facts": ["Fact 1", "Fact 2", ...]}
```

**Few-shot 예시:**
```
User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John! My name is Alex and I admire software engineering.
Output: {"facts": ["Admires software engineering", "Name is Alex"]}
```

### 4.3 Memory Update 프롬프트

**목적**: 새 facts와 기존 메모리를 비교하여 ADD/UPDATE/DELETE/NONE 결정

```
# Task Objective
You are a smart memory manager which controls the memory of a system.
You can perform four operations:
(1) add into the memory
(2) update the memory
(3) delete from the memory
(4) no change

# Guidelines:

## ADD
- New information not present in memory
- Generate new ID

## UPDATE
- Information already present but with different/more details
- Keep same ID, update content
- Include "old_memory" field

## DELETE
- Information that contradicts existing memory
- Example: "Likes X" → "Dislikes X"

## NONE
- Information already present
- No changes needed

# Output Format
{
    "memory": [
        {
            "id": "<memory_id>",
            "text": "<content>",
            "event": "ADD|UPDATE|DELETE|NONE",
            "old_memory": "<previous_content>"  // UPDATE only
        }
    ]
}
```

### 4.4 Procedural Memory 프롬프트

**목적**: 에이전트 실행 히스토리를 구조화된 요약으로 저장

```
# Task Objective
You are a memory summarization system that records and preserves
the complete interaction history between a human and an AI agent.

# Overall Structure:

## Overview (Global Metadata):
- Task Objective: 전체 목표
- Progress Status: 완료 비율 및 마일스톤

## Sequential Agent Actions (Numbered Steps):
Each step must include:

1. **Agent Action**: 정확한 행동 설명 (파라미터, 대상 요소 포함)

2. **Action Result (Mandatory, Unmodified)**:
   정확한 원본 출력 (HTML, JSON, 에러 등)

3. **Embedded Metadata**:
   - Key Findings: 발견된 중요 정보
   - Navigation History: 방문한 페이지 (URL 포함)
   - Errors & Challenges: 에러 메시지, 복구 시도
   - Current Context: 현재 상태, 다음 계획

# Guidelines:
1. Preserve Every Output - 출력 원본 유지, 요약 금지
2. Chronological Order - 시간순 번호 매김
3. Detail and Precision - URL, 인덱스, 에러 메시지 등 정확한 데이터
```

### 4.5 Entity Extraction 프롬프트 (Graph Memory)

```
# System Prompt
You are a smart assistant who understands entities and their types
in a given text.

If user message contains self reference such as 'I', 'me', 'my' etc.
then use {user_id} as the source entity.

Extract all the entities from the text.
***DO NOT*** answer the question itself if the given text is a question.

# Tool: extract_entities
{
    "entities": [
        {"entity": "John", "entity_type": "person"},
        {"entity": "Google", "entity_type": "organization"}
    ]
}
```

### 4.6 Relation Extraction 프롬프트 (Graph Memory)

```
# System Prompt
You are a top-tier algorithm for extracting information
from text to build a knowledge graph.

Extract entities and their relationships from the given text.

# Output Format
{
    "entities": [
        {
            "source": "John",
            "relationship": "WORKS_AT",
            "target": "Google"
        }
    ]
}
```

---

## 5. 검색 시스템

### 5.1 Hybrid Search 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Search Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣ Vector Search                                               │
│     └── Embedding similarity (Cosine)                           │
│                                                                  │
│  2️⃣ Graph Search (Optional)                                     │
│     ├── Entity extraction from query                            │
│     ├── Similar node retrieval (embedding similarity)           │
│     └── Relationship traversal                                  │
│                                                                  │
│  3️⃣ BM25 Reranking                                              │
│     └── Graph 결과에 대한 lexical reranking                      │
│                                                                  │
│  4️⃣ Reranker (Optional)                                         │
│     └── 최종 결과 재정렬                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 검색 코드 분석

```python
# mem0/memory/main.py - search 메서드

def search(self, query, *, user_id=None, agent_id=None, run_id=None, limit=100):
    # 1. Vector Store 검색
    embeddings = self.embedding_model.embed(query)
    memories = self.vector_store.search(
        query=embeddings,
        limit=limit,
        filters=filters
    )

    # 2. Graph Store 검색 (활성화 시)
    if self.enable_graph:
        graph_results = self.graph.search(query, filters=filters)
        # BM25로 reranking
        bm25 = BM25Okapi(search_outputs_sequence)
        reranked = bm25.get_top_n(tokenized_query, results, n=5)

    # 3. Reranker 적용 (설정 시)
    if self.reranker:
        memories = self.reranker.rerank(query, memories)

    return {"results": memories, "relations": graph_results}
```

---

## 6. 지원 통합

### 6.1 Vector Stores (20+)

| 카테고리 | 지원 스토어 |
|----------|------------|
| **Cloud** | Pinecone, Qdrant Cloud, Weaviate Cloud, Supabase |
| **Self-hosted** | Qdrant, Weaviate, Milvus, Chroma, FAISS |
| **Database** | PostgreSQL (pgvector), MongoDB, Redis, Elasticsearch |
| **Enterprise** | Azure AI Search, Vertex AI Vector Search, Databricks |
| **기타** | OpenSearch, Cassandra, Valkey |

### 6.2 LLM Providers

| Provider | 지원 기능 |
|----------|----------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| **Anthropic** | Claude 3.5, Claude 3 |
| **Google** | Gemini Pro, Gemini Flash |
| **AWS Bedrock** | Claude, Titan |
| **Azure OpenAI** | GPT-4, GPT-3.5 |
| **Ollama** | Llama, Mistral 등 로컬 모델 |
| **Together AI** | 다양한 오픈소스 모델 |
| **xAI** | Grok |

### 6.3 Graph Stores

| Store | 설명 |
|-------|------|
| **Neo4j** | 기본 그래프 스토어, Cypher 쿼리 |
| **Kuzu** | 임베디드 그래프 DB |
| **Memgraph** | 실시간 그래프 분석 |

### 6.4 프레임워크 통합

- **LangGraph**: 에이전트 워크플로우
- **CrewAI**: 멀티 에이전트 시스템
- **LangChain**: 체인 구성

---

## 7. 성능 최적화

### 7.1 병렬 처리

```python
# Vector Store와 Graph Store 병렬 처리
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
    future2 = executor.submit(self._add_to_graph, messages, filters)

    concurrent.futures.wait([future1, future2])

    vector_result = future1.result()
    graph_result = future2.result()
```

### 7.2 User/Agent Memory 자동 분기

```python
def _should_use_agent_memory_extraction(self, messages, metadata):
    """
    - agent_id 존재 + assistant 메시지 있음 → Agent Memory
    - 그 외 → User Memory
    """
    has_agent_id = metadata.get("agent_id") is not None
    has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

    return has_agent_id and has_assistant_messages
```

### 7.3 Hash 기반 중복 방지

```python
# 메모리 내용 해시로 중복 검사
memory_hash = hashlib.md5(memory_content.encode()).hexdigest()
```

---

## 8. MemU, Memori와 비교

### 8.1 아키텍처 비교

| 측면 | Mem0 | MemU | Memori |
|------|------|------|--------|
| **메모리 추출** | 자체 LLM 파이프라인 | 자체 LLM 파이프라인 | 외부 API |
| **Graph 지원** | Neo4j, Kuzu, Memgraph | 없음 | Semantic Triples |
| **Vector Store** | 20+ 지원 | 3종 (InMemory, Postgres, pgvector) | 5종 |
| **Reranker** | 지원 | 없음 | 하이브리드 내장 |

### 8.2 메모리 유형 비교

| 측면 | Mem0 | MemU | Memori |
|------|------|------|--------|
| **분류 기준** | User/Agent/Run ID | Memory Type + Category | Entity/Process |
| **타입 수** | 3 (User, Agent, Procedural) | 5 (profile, event, knowledge, behavior, skill) | 3 (facts, attributes, triples) |
| **계층 구조** | Flat (ID 기반 필터링) | 3계층 (Resource→Item→Category) | 2계층 (Entity→Fact) |
| **요약 생성** | 없음 | 카테고리별 자동 요약 | 대화 요약만 |

### 8.3 검색 방식 비교

| 측면 | Mem0 | MemU | Memori |
|------|------|------|--------|
| **기본 검색** | Vector (Cosine) | RAG + LLM 랭킹 | FAISS |
| **Graph 검색** | Neo4j + BM25 | 없음 | 없음 |
| **Reranking** | 외부 Reranker 지원 | LLM 기반 | BM25 하이브리드 |
| **충분성 검사** | 없음 | LLM 기반 | 없음 |

### 8.4 강점 비교

| 시스템 | 주요 강점 |
|--------|----------|
| **Mem0** | Graph Memory, 다양한 Vector Store, Procedural Memory |
| **MemU** | 세밀한 메모리 분류, 계층적 검색, 멀티모달 |
| **Memori** | 간편한 통합, Knowledge Graph, 제로 레이턴시 |

---

## 9. 주요 의존성

```toml
[dependencies]
pydantic = "^2.0"            # 데이터 검증
pytz = "*"                   # 시간대 처리

[optional]
langchain-neo4j = "*"        # Neo4j 통합
rank-bm25 = "*"              # BM25 재랭킹
openai = "*"                 # OpenAI LLM
qdrant-client = "*"          # Qdrant Vector Store
pinecone-client = "*"        # Pinecone Vector Store
chromadb = "*"               # Chroma Vector Store
faiss-cpu = "*"              # FAISS Vector Store
```

---

## 10. 사용 예시

### 10.1 기본 사용

```python
from mem0 import Memory
from openai import OpenAI

memory = Memory()
openai_client = OpenAI()

# 메모리 저장
memory.add(
    messages=[
        {"role": "user", "content": "My name is John. I love hiking."}
    ],
    user_id="john_123"
)

# 메모리 검색
results = memory.search(
    query="What does John like?",
    user_id="john_123"
)

# 결과: [{"memory": "Loves hiking", "score": 0.92}]
```

### 10.2 Graph Memory 사용

```python
from mem0 import Memory

config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    }
}

memory = Memory.from_config(config)

# 관계 추출 포함 저장
result = memory.add(
    messages="John works at Google as a software engineer.",
    user_id="john_123"
)

# 결과:
# {
#     "results": [{"memory": "Works at Google as a software engineer"}],
#     "relations": [{"source": "John", "relationship": "WORKS_AT", "target": "Google"}]
# }
```

### 10.3 Procedural Memory

```python
# agent_id 필수
memory.add(
    messages=agent_execution_history,
    agent_id="web_scraper_agent",
    memory_type="procedural_memory"
)
```

---

## 11. 결론

Mem0는 **프로덕션 레디 AI 에이전트**를 위한 확장 가능한 메모리 솔루션입니다:

- **Multi-Level Memory**: User/Agent/Run 레벨의 유연한 메모리 관리
- **Graph Memory**: Neo4j 기반 Knowledge Graph로 관계 추론
- **Procedural Memory**: 에이전트 실행 히스토리의 구조화된 저장
- **광범위한 통합**: 20+ Vector Stores, 다양한 LLM 지원
- **검증된 성능**: LOCOMO 벤치마크에서 OpenAI Memory 대비 +26% 정확도

### MemU, Memori와의 선택 가이드

| 요구사항 | 추천 시스템 |
|----------|------------|
| **Knowledge Graph 필요** | Mem0 (Neo4j 통합) |
| **세밀한 메모리 분류 필요** | MemU (5타입 + 10카테고리) |
| **빠른 통합 필요** | Memori (한 줄 통합) |
| **다양한 Vector Store 필요** | Mem0 (20+ 지원) |
| **멀티모달 지원 필요** | MemU (이미지, 비디오, 오디오) |
| **Procedural Memory 필요** | Mem0 |
