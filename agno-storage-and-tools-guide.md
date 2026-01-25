# Agno Storage 시스템 및 도구 활용 가이드

> 이 문서는 Agno 프레임워크의 Storage 시스템, 도구 정의 방식, 압축 기능 등에 대한 심층 분석입니다.

---

## 목차

1. [Storage 시스템 개요](#1-storage-시스템-개요)
2. [테이블별 기능 및 활성화 방법](#2-테이블별-기능-및-활성화-방법)
3. [도구 압축 (compress_tool_results)](#3-도구-압축-compress_tool_results)
4. [도구 정의: description vs instructions](#4-도구-정의-description-vs-instructions)

---

## 1. Storage 시스템 개요

### 1.1 DB가 저장하는 테이블 (13개)

| 테이블 | 저장 내용 | 용도 |
|--------|----------|------|
| `agno_sessions` | 세션, 채팅 히스토리, 상태 | 대화 연속성 |
| `agno_memories` | 사용자 메모리 | 장기 기억 |
| `agno_learnings` | 학습 내용 (user_profile, session_context 등) | 에이전트 학습 |
| `agno_knowledge` | 지식 콘텐츠 | RAG 지식베이스 |
| `agno_culture` | 문화적 지식 | 조직/도메인 지식 |
| `agno_traces` | 실행 트레이스 | 디버깅/모니터링 |
| `agno_spans` | 실행 스팬 | 상세 트레이싱 |
| `agno_metrics` | 사용량 메트릭 | 분석/과금 |
| `agno_eval_runs` | 평가 실행 기록 | 품질 평가 |
| `agno_components` | 에이전트/팀/워크플로우 정의 | 버전 관리 |
| `agno_component_configs` | 컴포넌트 설정 | 버전별 설정 |
| `agno_component_links` | 컴포넌트 연결 | 팀/워크플로우 구성 |
| `agno_schema_versions` | 스키마 버전 | 마이그레이션 |

### 1.2 지원 데이터베이스 백엔드

- PostgreSQL (권장)
- MySQL
- SQLite
- MongoDB
- Redis
- DynamoDB
- Firestore
- SingleStore
- SurrealDB
- GCS JSON
- In-Memory
- JSON File

### 1.3 왜 Storage가 필요한가?

| 상황 | DB 필요 여부 |
|------|-------------|
| 단일 세션, 서버 재시작 없음 | ❌ 불필요 |
| 서버 재시작 후 대화 이어가기 | ✅ 필요 |
| 여러 세션에서 사용자 기억 | ✅ 필요 |
| 멀티 사용자 서비스 | ✅ 필요 |
| session_state 영속화 | ✅ 필요 |
| 메모리/학습 기능 사용 | ✅ 필요 |
| 사용량 메트릭 수집 | ✅ 필요 |

---

## 2. 테이블별 기능 및 활성화 방법

### 2.1 세션 & 채팅 히스토리 (`agno_sessions`)

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb

db = PostgresDb(db_url="postgresql://...")

agent = Agent(
    model=model,
    db=db,  # ← DB 연결만 하면 자동으로 세션 저장

    # 히스토리를 컨텍스트에 포함 (선택)
    add_history_to_context=True,
    num_history_runs=5,  # 최근 5개 실행만
)

# 같은 session_id면 대화 이어감
agent.run("안녕", session_id="chat-001")
agent.run("방금 뭐라고 했어?", session_id="chat-001")
```

### 2.2 사용자 메모리 (`agno_memories`)

#### 방법 A: 자동 메모리 생성

```python
from agno.memory import MemoryManager

agent = Agent(
    model=model,
    db=db,

    # 메모리 매니저 설정
    memory_manager=MemoryManager(db=db),

    # 실행 후 자동으로 메모리 추출/저장
    update_memory_on_run=True,

    # 메모리를 시스템 메시지에 포함
    add_memories_to_context=True,
)

# 대화하면 LLM이 중요 정보를 자동 추출하여 저장
agent.run("나는 서울에 사는 개발자 김철수야", user_id="user-123")
# → DB에 저장: "User's name is 김철수, lives in Seoul, works as developer"

# 다른 세션에서도 기억
agent.run("내 이름이 뭐야?", session_id="new-session", user_id="user-123")
# → "김철수님이시네요"
```

#### 방법 B: 에이전트가 직접 관리

```python
agent = Agent(
    model=model,
    db=db,
    memory_manager=MemoryManager(db=db),

    # 에이전트에게 메모리 도구 제공
    enable_agentic_memory=True,
)

# 에이전트가 필요하다고 판단하면 직접 메모리 저장/수정/삭제
```

#### 메모리 직접 조회/관리

```python
# 특정 사용자의 메모리 조회
memories = db.get_user_memories(user_id="user-123")

# 메모리 삭제
db.delete_user_memory(memory_id="mem-xxx")

# 전체 메모리 초기화
db.clear_memories()
```

### 2.3 학습 시스템 (`agno_learnings`)

```python
from agno.learning import LearningMachine

agent = Agent(
    model=model,
    db=db,

    # 학습 활성화 (간단)
    learning=True,

    # 또는 상세 설정
    learning=LearningMachine(
        db=db,
        user_profile=True,      # 사용자 프로필 학습
        session_context=True,   # 세션 컨텍스트 학습
    ),

    # 학습 내용을 컨텍스트에 포함
    add_learnings_to_context=True,
)

# 대화하면 자동으로 학습
agent.run("나는 간결한 답변을 좋아해", user_id="user-123")
# → learnings 테이블에 사용자 선호도 저장

# 이후 대화에서 학습 내용 반영
agent.run("Python에 대해 설명해줘", user_id="user-123")
# → 간결하게 답변
```

#### 학습 내용 조회

```python
# 특정 사용자의 학습 내용 조회
learning = db.get_learning(
    learning_type="user_profile",
    user_id="user-123"
)
print(learning["content"])
# → {"preferences": {"response_style": "concise"}, ...}
```

### 2.4 문화적 지식 (`agno_culture`) - 실험적

```python
from agno.culture import CultureManager

agent = Agent(
    model=model,
    db=db,

    # 문화 관리자 설정
    culture_manager=CultureManager(db=db),

    # 자동 업데이트
    update_cultural_knowledge=True,

    # 컨텍스트에 포함
    add_culture_to_context=True,
)

# 조직/도메인 특화 지식 학습
# 예: 회사 용어, 프로세스, 규칙 등
```

### 2.5 지식 베이스 (`agno_knowledge`)

```python
from agno.knowledge import AgentKnowledge
from agno.vectordb.pgvector import PgVector

# 벡터 DB 설정
vector_db = PgVector(
    db_url="postgresql://...",
    table_name="my_embeddings",
)

# 지식 베이스 생성
knowledge = AgentKnowledge(
    db=db,  # 메타데이터 저장
    vector_db=vector_db,  # 임베딩 저장
)

# 문서 추가
knowledge.load_documents([
    Document(content="회사 정책 내용..."),
    Document(content="제품 매뉴얼..."),
])

# 에이전트에 연결
agent = Agent(
    model=model,
    db=db,
    knowledge=knowledge,
    search_knowledge=True,  # 지식 검색 도구 제공
)
```

### 2.6 세션 상태 (`session_state`)

```python
agent = Agent(
    model=model,
    db=db,

    # 초기 상태
    session_state={
        "user_tier": "premium",
        "cart": [],
        "step": 1,
    },

    # 에이전트가 상태 수정 가능
    enable_agentic_state=True,

    # 상태를 컨텍스트에 포함
    add_session_state_to_context=True,
)

# 상태 변경은 자동으로 DB에 저장됨
agent.run("상품 A 추가해줘", session_id="order-1")
# → session_state["cart"] = ["A"]

# 서버 재시작 후에도 상태 유지
agent.run("장바구니 보여줘", session_id="order-1")
# → "상품 A가 있습니다"
```

### 2.7 세션 요약 (`agno_sessions.summary`)

```python
agent = Agent(
    model=model,
    db=db,

    # 세션 요약 자동 생성
    enable_session_summaries=True,

    # 요약을 컨텍스트에 포함
    add_session_summary_to_context=True,
)

# 긴 대화 후 자동으로 요약 생성
# → 다음 대화에서 전체 히스토리 대신 요약 사용 (토큰 절약)
```

### 2.8 메트릭 & 트레이싱 (`agno_metrics`, `agno_traces`, `agno_spans`)

```python
agent = Agent(
    model=model,
    db=db,

    # 텔레메트리 활성화 (기본값 True)
    telemetry=True,
)

# 메트릭 조회
metrics, total = db.get_metrics(
    starting_date=date(2024, 1, 1),
    ending_date=date(2024, 12, 31),
)

# 트레이스 조회
traces, total = db.get_traces(
    agent_id="agent-123",
    limit=100,
)
```

### 2.9 전체 기능 활성화 예시

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager
from agno.learning import LearningMachine

db = PostgresDb(db_url="postgresql://...")

agent = Agent(
    model=model,
    db=db,

    # === 히스토리 ===
    add_history_to_context=True,
    num_history_runs=5,

    # === 메모리 ===
    memory_manager=MemoryManager(db=db),
    update_memory_on_run=True,
    add_memories_to_context=True,

    # === 학습 ===
    learning=True,
    add_learnings_to_context=True,

    # === 세션 상태 ===
    session_state={"initialized": True},
    enable_agentic_state=True,
    add_session_state_to_context=True,

    # === 세션 요약 ===
    enable_session_summaries=True,
    add_session_summary_to_context=True,

    # === 캐싱 ===
    cache_session=True,
)
```

### 2.10 기능 활성화 요약표

| 기능 | 활성화 옵션 | 테이블 |
|------|------------|--------|
| 채팅 히스토리 | `db=...` + `add_history_to_context=True` | `agno_sessions` |
| 사용자 메모리 | `memory_manager=...` + `update_memory_on_run=True` | `agno_memories` |
| 학습 | `learning=True` | `agno_learnings` |
| 문화 지식 | `culture_manager=...` + `update_cultural_knowledge=True` | `agno_culture` |
| 지식 베이스 | `knowledge=...` | `agno_knowledge` |
| 세션 상태 | `session_state=...` | `agno_sessions` |
| 세션 요약 | `enable_session_summaries=True` | `agno_sessions` |
| 트레이싱 | `telemetry=True` (기본값) | `agno_traces`, `agno_spans` |

---

## 3. 도구 압축 (compress_tool_results)

### 3.1 기본 정보

- **기본값**: `False` (명시적 활성화 필요)
- **방식**: LLM을 사용한 요약 (compaction/pruning 아님)
- **목적**: 도구 결과가 많을 때 토큰 절약

### 3.2 활성화 방법

```python
agent = Agent(
    model=model,
    compress_tool_results=True,  # 수동으로 활성화 필요
)
```

### 3.3 압축 트리거 조건

```python
# 두 가지 조건 중 하나 충족 시 압축 시작

# 조건 1: 도구 호출 횟수 기반 (기본값: 3회)
compress_tool_results_limit=3  # 미압축 도구 결과가 3개 이상이면 압축

# 조건 2: 토큰 수 기반
compress_token_limit=4000     # 전체 토큰이 4000 이상이면 압축
```

### 3.4 내부 동작 원리

```python
# CompressionManager._compress_tool_result()
def _compress_tool_result(self, tool_result: Message) -> Optional[str]:
    # 도구 결과를 LLM에게 요약 요청
    response = self.model.response(
        messages=[
            Message(role="system", content=DEFAULT_COMPRESSION_PROMPT),
            Message(role="user", content="Tool Results to Compress: " + tool_content),
        ]
    )
    return response.content
```

### 3.5 압축 프롬프트 규칙

```
ALWAYS PRESERVE (반드시 유지):
• 숫자, 통계, 가격, 수량, 메트릭
• 날짜, 시간 (짧은 형식: "Oct 21 2025")
• 사람, 회사, 제품, 위치
• URL, ID, 코드, 버전

COMPRESS TO ESSENTIALS (핵심만):
• 설명 → 핵심 속성만
• 리스트 → 가장 관련 있는 항목만

REMOVE ENTIRELY (완전 제거):
• 서론, 결론
• 헷지 언어 ("might", "possibly")
• 메타 코멘터리 ("According to", "The results show")
• 마크다운, HTML, JSON 구조
• 홍보성 문구, 필러 워드
```

### 3.6 압축 예시

**Before** (긴 도구 결과):
```
"According to recent market analysis and industry reports, OpenAI has made
several significant announcements in the technology sector. The company
revealed ChatGPT Atlas on October 21, 2025, which represents a new AI-powered
browser application that has been specifically designed for macOS users..."
```

**After** (압축된 결과):
```
"OpenAI - Oct 21 2025: ChatGPT Atlas (AI browser, macOS, search competitor);
Oct 6 2025: Apps in ChatGPT + SDK; Partners: Spotify, Zillow, Canva"
```

### 3.7 비용 효율적 사용

```python
# 저렴한 모델을 압축 전용으로 지정
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    compress_tool_results=True,
    compression_manager=CompressionManager(
        model=OpenAIChat(id="gpt-4o-mini"),  # 저렴한 모델로 압축
        compress_tool_results_limit=3,
    ),
)
```

---

## 4. 도구 정의: description vs instructions

### 4.1 두 방식의 차이점

| 속성 | 전달 위치 | 용도 |
|------|----------|------|
| **description** | LLM API의 `tools[].function.description` | 도구 선택 시 참고 |
| **instructions** | 시스템 메시지 본문 | 도구 사용 방법 상세 가이드 |

### 4.2 docstring 방식

```python
@tool
def search_web(query: str, max_results: int = 10) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드
        max_results: 최대 결과 수 (기본값: 10)

    Returns:
        검색 결과 문자열
    """
    return results
```

docstring은 자동으로 `description`으로 변환됩니다.

### 4.3 @tool 데코레이터 방식

```python
@tool(
    description="DeepSearch를 통해 기업 스냅샷을 조회합니다.",
    instructions="""회사명을 입력하면 기업 개요, 주주 정보 등을 반환합니다.

Args:
    company_name: 회사명 (예: "삼성전자")

Returns:
    기업 스냅샷 JSON

Examples:
    deepsearch_company_snapshot("삼성전자")
""",
)
def deepsearch_company_snapshot(company_name: str) -> str:
    pass
```

### 4.4 내부 처리 과정

#### description 처리

```python
# decorator.py:221-225
tool_config = {
    "name": kwargs.get("name", func.__name__),
    "description": kwargs.get("description", get_entrypoint_docstring(wrapper)),
    #                         ↑ 명시적 지정      ↑ 없으면 docstring에서 추출
    "instructions": kwargs.get("instructions"),
    ...
}
```

**description은 LLM API 호출 시 도구 스키마에 포함:**
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "웹에서 정보를 검색합니다.",
      "parameters": {...}
    }
  }]
}
```

#### instructions 처리

```python
# agent.py:6882-6883
# 도구 처리 시 instructions 수집
if tool.add_instructions and tool.instructions is not None:
    self._tool_instructions.append(tool.instructions)

# agent.py:8890-8892
# 시스템 메시지 구성 시 추가
if self._tool_instructions is not None:
    for _ti in self._tool_instructions:
        system_message_content += f"{_ti}\n"
```

**instructions는 시스템 메시지에 포함:**
```
[System Message]
You are a helpful assistant...

<additional_information>
- Current time: 2024-01-15
</additional_information>

회사명을 입력하면 기업 개요, 주주 정보 등을 반환합니다.
Args:
    company_name: 회사명 (예: "삼성전자")
...
```

### 4.5 토큰 소모 비교

**둘 다 LLM에 전달되어 토큰을 소모합니다.**

| 방식 | 전달 위치 | 토큰 소모 |
|------|----------|----------|
| docstring → `description` | `tools[]` 스키마 | O |
| `@tool(description=...)` | `tools[]` 스키마 | O |
| `@tool(instructions=...)` | 시스템 메시지 | O |

내용이 같으면 토큰 소모량도 거의 동일합니다. 분리하는 이유는 **토큰 절약이 아니라 역할 분리**입니다.

### 4.6 언제 뭘 써야 할까?

| 상황 | 권장 방식 |
|------|----------|
| 간단한 도구 | docstring만 (자동으로 description 추출) |
| 복잡한 사용법 | `description` (짧은 요약) + `instructions` (상세 가이드) |
| 예시가 필요한 도구 | `instructions`에 Examples 포함 |
| 제약조건이 많은 도구 | `instructions`에 상세 규칙 기술 |

### 4.7 권장 패턴

```python
@tool(
    # 짧고 명확한 한 줄 설명 (LLM이 도구 선택 시 참고)
    description="DeepSearch를 통해 기업 스냅샷을 조회합니다.",

    # 상세한 사용 가이드 (시스템 메시지에 포함)
    instructions="""회사명을 입력하면 기업 개요, 주주 정보 등을 반환합니다.

Args:
    company_name: 회사명 (예: "삼성전자")

Returns:
    기업 스냅샷 JSON

Examples:
    deepsearch_company_snapshot("삼성전자")
""",
    add_instructions=True,  # 기본값 True
)
def deepsearch_company_snapshot(company_name: str) -> str:
    pass
```

---

## 5. 도구가 LLM에 전달되는 전체 흐름

### 5.1 변환 과정

```
Python 함수
    ↓
Function.from_callable() 또는 @tool 데코레이터
    ↓
1. 함수 이름 추출: c.__name__
2. docstring 파싱: docstring_parser.parse()
3. description 추출: short_description + long_description
4. 파라미터 설명 추출: Args 섹션
5. 타입 힌트로 JSON Schema 생성
    ↓
Function 객체 생성
    ↓
Function.to_dict() → JSON 스키마
    ↓
LLM API 호출
```

### 5.2 최종 API 요청 형태 (OpenAI 형식)

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant...\n\n회사명을 입력하면 기업 개요..."
    },
    {
      "role": "user",
      "content": "삼성전자 정보 알려줘"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "deepsearch_company_snapshot",
        "description": "DeepSearch를 통해 기업 스냅샷을 조회합니다.",
        "parameters": {
          "type": "object",
          "properties": {
            "company_name": {
              "type": "string",
              "description": "(str) 회사명 (예: \"삼성전자\")"
            }
          },
          "required": ["company_name"]
        }
      }
    }
  ]
}
```

---

## 파일 경로 참조

| 기능 | 경로 |
|------|------|
| Storage Base | `/libs/agno/agno/db/base.py` |
| Memory Manager | `/libs/agno/agno/memory/manager.py` |
| Compression Manager | `/libs/agno/agno/compression/manager.py` |
| Tool Decorator | `/libs/agno/agno/tools/decorator.py` |
| Function Class | `/libs/agno/agno/tools/function.py` |

---

*이 문서는 Agno 프레임워크 내부 코드 분석을 기반으로 작성되었습니다.*
