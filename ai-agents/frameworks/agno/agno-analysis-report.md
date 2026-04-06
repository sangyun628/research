# Agno 프레임워크 심층 분석 보고서

> 이 문서는 Agno 프레임워크(https://github.com/agno-agi/agno)의 내부 구조를 분석하여, 공식 문서에서 다루지 않는 깊이 있는 활용 방법을 제공합니다.

---

## 목차

1. [프레임워크 개요](#1-프레임워크-개요)
2. [Agent 핵심 클래스](#2-agent-핵심-클래스)
3. [세션 관리 시스템](#3-세션-관리-시스템)
4. [메모리 관리 시스템](#4-메모리-관리-시스템)
5. [도구(Tool) 시스템](#5-도구tool-시스템)
6. [Hook 시스템](#6-hook-시스템)
7. [Reasoning 시스템](#7-reasoning-시스템)
8. [실전 활용 팁](#8-실전-활용-팁)

---

## 1. 프레임워크 개요

### 1.1 디렉토리 구조

```
libs/agno/agno/
├── agent/          # Agent 핵심 클래스 (12,499줄)
├── session/        # 세션 관리 (AgentSession, TeamSession, WorkflowSession)
├── memory/         # 메모리 시스템 (MemoryManager, strategies/)
├── tools/          # 도구 시스템 (Function, Toolkit, MCP 통합)
├── hooks/          # 훅 시스템 (@hook 데코레이터)
├── reasoning/      # 추론 시스템 (ReasoningManager, 모델별 구현)
├── models/         # 40+ LLM 모델 지원
├── db/             # 11개 데이터베이스 백엔드
├── knowledge/      # RAG 지식 베이스
├── vectordb/       # 18개 벡터 DB 지원
├── compression/    # 도구 결과 압축
├── guardrails/     # 입출력 가드레일
├── team/           # 멀티 에이전트 팀
└── workflow/       # 워크플로우 오케스트레이션
```

### 1.2 핵심 설계 원칙

1. **Dataclass 기반**: `@dataclass(init=False)` 패턴으로 타입 안전성 확보
2. **동기/비동기 이중 지원**: 모든 메서드에 `run()`/`arun()` 버전 제공
3. **Lazy Initialization**: 필요 시점에만 매니저 초기화
4. **전략 패턴**: 메모리 최적화, 추론 등에 교체 가능한 전략 적용
5. **훅 아키텍처**: 확장 가능한 pre/post 훅 시스템

---

## 2. Agent 핵심 클래스

### 2.1 Agent 클래스 구조

**파일 위치**: `/libs/agno/agno/agent/agent.py` (12,499줄)

```python
@dataclass(init=False)
class Agent:
    # 기본 설정 (총 70+ 속성)
    model: Optional[Model]
    name: Optional[str]
    id: Optional[str]

    # 세션 관리
    session_id: Optional[str]
    session_state: Optional[Dict[str, Any]]
    cache_session: bool = False

    # 메모리 관리
    memory_manager: Optional[MemoryManager]
    enable_agentic_memory: bool = False
    update_memory_on_run: bool = False

    # 도구
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]]
    tool_call_limit: Optional[int]

    # 훅
    pre_hooks: Optional[List[Union[Callable, BaseGuardrail, BaseEval]]]
    post_hooks: Optional[List[Union[Callable, BaseGuardrail, BaseEval]]]

    # 추론
    reasoning: bool = False
    reasoning_model: Optional[Model]
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10
```

### 2.2 실행 흐름 (run 메서드)

```
Agent.run(input)
    ↓
1. _initialize_session()          # 세션 생성/로드
2. initialize_agent()             # 매니저들 초기화
3. validate_input()               # 입력 검증
4. _execute_pre_hooks()           # Pre-훅 실행
5. get_tools()                    # 도구 목록 구성
6. _get_run_messages()            # 메시지 준비
7. _handle_reasoning()            # 추론 (선택)
8. model.response()               # 모델 호출
9. _update_run_response()         # 응답 처리
10. _execute_post_hooks()         # Post-훅 실행
11. _cleanup_and_store()          # 저장 및 정리
    ↓
RunOutput 반환
```

### 2.3 숨겨진 유용한 설정들

#### 성능 최적화

```python
agent = Agent(
    model=model,
    db=db,

    # 세션 캐싱 (메모리에 캐시하여 DB 접근 감소)
    cache_session=True,

    # 히스토리 제한 (컨텍스트 길이 관리)
    add_history_to_context=True,
    num_history_runs=3,  # 또는 num_history_messages=10
    max_tool_calls_from_history=5,

    # 도구 결과 압축 (토큰 절약)
    compress_tool_results=True,

    # 재시도 설정
    retries=3,
    delay_between_retries=2,
    exponential_backoff=True,  # 2초 → 4초 → 8초
)
```

#### 상태 관리

```python
agent = Agent(
    # 세션 상태 활성화
    session_state={"role": "admin", "counter": 0},
    add_session_state_to_context=True,  # 시스템 메시지에 추가

    # 에이전트가 상태 수정 가능
    enable_agentic_state=True,  # update_session_state 도구 제공

    # 상태 병합 정책
    overwrite_db_session_state=False,  # DB 상태와 병합 (기본)
)
```

#### 컨텍스트 자동 해석

```python
agent = Agent(
    # 변수 자동 치환
    system_message="Analyze {task} for {user_name}",
    resolve_in_context=True,  # {변수} 자동 대체

    # 추가 정보 포함
    add_datetime_to_context=True,
    timezone_identifier="Asia/Seoul",
    add_location_to_context=True,
    add_name_to_context=True,
)
```

### 2.4 확장 포인트 (서브클래싱)

```python
class EnterpriseAgent(Agent):
    """엔터프라이즈 기능 추가"""

    def __init__(self, *args, audit_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_enabled = audit_enabled

    # 메시지 구성 커스터마이징
    def _get_run_messages(self, run_response, run_context, ...) -> RunMessages:
        messages = super()._get_run_messages(...)
        # 커스텀 로직 추가
        return messages

    # 도구 선택 커스터마이징
    def get_tools(self, run_response, ...) -> List:
        tools = super().get_tools(...)
        if "admin" in self.role:
            tools.append(self._get_admin_tools())
        return tools

    # 세션 로드 커스터마이징
    def _read_or_create_session(self, session_id, user_id) -> AgentSession:
        session = super()._read_or_create_session(session_id, user_id)
        # 감사 로그 추가
        if self.audit_enabled:
            self._audit_log(f"Session loaded: {session_id}")
        return session
```

---

## 3. 세션 관리 시스템

### 3.1 세션 타입

```python
# 3가지 세션 타입
Session = Union[AgentSession, TeamSession, WorkflowSession]
```

**AgentSession** (`/session/agent.py`):
```python
@dataclass
class AgentSession:
    session_id: str                    # 필수: UUID
    agent_id: Optional[str]            # 연결된 Agent
    user_id: Optional[str]             # 사용자 ID (멀티유저 지원)

    # 데이터
    session_data: Dict[str, Any]       # session_state, metrics 포함
    runs: List[RunOutput]              # 실행 히스토리
    summary: Optional[SessionSummary]  # AI 생성 요약

    # 타임스탬프
    created_at: int
    updated_at: int
```

### 3.2 세션 생성 및 관리

```python
# 자동 세션 ID 생성
agent = Agent(model=model)
agent.run("query")  # UUID 자동 생성

# 명시적 세션 ID
agent.run("query", session_id="my-session-001")

# 사용자별 세션 격리
agent.run("query", session_id="session-1", user_id="user-123")
```

### 3.3 상태 병합 정책

```
우선순위: run_params > db_state > agent_defaults

# 예시
agent = Agent(session_state={"role": "user"})  # 기본값
# DB에 저장된 상태: {"role": "admin", "counter": 5}
# run() 호출 시 전달: {"counter": 10}

# 결과 (overwrite_db_session_state=False):
# {"role": "admin", "counter": 10}
```

### 3.4 지원 데이터베이스

| DB | 용도 | 특징 |
|----|------|------|
| SQLite | 개발/테스트 | 파일 기반, 경량 |
| PostgreSQL | 프로덕션 | pgvector 지원, 확장성 |
| MySQL | 프로덕션 | 엔터프라이즈 |
| MongoDB | NoSQL | 유연한 스키마 |
| Redis | 캐시 | 빠른 접근 |
| DynamoDB | AWS | 관리형 서비스 |
| Firestore | GCP | 관리형 서비스 |
| In-Memory | 테스트 | 영속성 없음 |

### 3.5 세션 요약 자동 생성

```python
agent = Agent(
    model=model,
    db=db,
    enable_session_summaries=True,        # 세션 요약 활성화
    add_session_summary_to_context=True,  # 컨텍스트에 추가
)
```

**SessionSummary 구조**:
```python
@dataclass
class SessionSummary:
    summary: str                # AI 생성 요약
    topics: List[str]          # 논의된 주제들
    updated_at: datetime
```

---

## 4. 메모리 관리 시스템

### 4.1 아키텍처

```
MemoryManager (최상위 관리자)
    ├── UserMemory (저장 단위)
    ├── MemoryOptimizationStrategy (최적화 전략)
    │   └── SummarizeStrategy
    └── CompressionManager (도구 결과 압축)
```

### 4.2 UserMemory 구조

```python
@dataclass
class UserMemory:
    memory: str                  # 메모리 내용 (필수)
    memory_id: str              # UUID
    topics: List[str]           # 분류 태그 ["name", "hobbies", "location"]
    user_id: str                # 사용자 ID
    agent_id: Optional[str]     # 에이전트별 메모리
    team_id: Optional[str]      # 팀 레벨 메모리
    input: str                  # 메모리 생성 원인
    created_at: int
    updated_at: int
```

### 4.3 메모리 검색 전략 (3가지)

```python
# 1. last_n: 최신 메모리 우선
memories = memory_manager.search_user_memories(
    user_id="user-123",
    retrieval_method="last_n",
    limit=10
)

# 2. first_n: 오래된 메모리 우선
memories = memory_manager.search_user_memories(
    retrieval_method="first_n",
    limit=10
)

# 3. agentic: LLM 기반 시맨틱 검색 (권장)
memories = memory_manager.search_user_memories(
    query="사용자의 취미와 관심사",
    retrieval_method="agentic",
    limit=5
)
```

### 4.4 메모리 최적화 (요약)

```python
# 메모리 압축 (여러 메모리 → 하나의 요약)
optimized = memory_manager.optimize_memories(
    user_id="user-123",
    strategy=MemoryOptimizationStrategyType.SUMMARIZE,
    apply=True  # DB에 바로 반영
)

# 결과: 10개 메모리 (5000 토큰) → 1개 요약 (200 토큰)
```

### 4.5 에이전트 메모리 vs 사용자 메모리

```python
# 사용자 메모리 (자동 생성)
agent = Agent(
    memory_manager=MemoryManager(db=db),
    update_memory_on_run=True,       # 실행 후 메모리 자동 생성
    add_memories_to_context=True,    # 컨텍스트에 추가
)

# 에이전트가 메모리 관리 (도구 제공)
agent = Agent(
    memory_manager=MemoryManager(db=db),
    enable_agentic_memory=True,      # update_user_memory 도구 제공
)
# Agent가 add_memory, update_memory, delete_memory 도구 사용 가능
```

### 4.6 도구 결과 압축 (CompressionManager)

```python
agent = Agent(
    model=model,
    compress_tool_results=True,
    compress_tool_results_limit=3,  # 3개 이상 도구 결과 시 압축
    # 또는
    compress_token_limit=4000,      # 토큰 임계값 초과 시 압축
)
```

**압축 전**:
```
"According to recent market analysis, OpenAI announced ChatGPT Atlas
on October 21, 2025, which is an AI-powered browser for macOS..."
```

**압축 후**:
```
"OpenAI - Oct 21 2025: ChatGPT Atlas (AI browser, macOS, search competitor)"
```

---

## 5. 도구(Tool) 시스템

### 5.1 도구 정의 방법

#### 방법 1: @tool 데코레이터 (권장)

```python
from agno.tools.decorator import tool

@tool
def search_web(query: str, max_results: int = 10) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
    """
    # 구현
    return results
```

#### 방법 2: 고급 옵션과 함께

```python
@tool(
    name="custom_search",
    description="고급 웹 검색",
    strict=True,                    # JSON Schema 엄격 모드
    show_result=True,               # 결과 출력
    stop_after_tool_call=False,     # 호출 후 계속 진행
    requires_confirmation=True,     # 사용자 확인 필요
    cache_results=True,             # 결과 캐싱
    cache_ttl=3600,                 # 1시간 캐시
    pre_hook=validate_input,        # 실행 전 훅
    post_hook=log_result,           # 실행 후 훅
    tool_hooks=[timing_interceptor] # 인터셉터
)
def advanced_search(query: str) -> str:
    pass
```

#### 방법 3: Toolkit 클래스

```python
from agno.tools import Toolkit

class DatabaseToolkit(Toolkit):
    def __init__(self, db_url: str):
        self.db = Database(db_url)
        super().__init__(
            name="database",
            tools=[self.query, self.insert, self.delete],
            requires_confirmation_tools=["delete"],
            cache_results=True,
        )

    def query(self, sql: str) -> str:
        """SQL 쿼리 실행"""
        return self.db.query(sql)

    def insert(self, table: str, data: dict) -> str:
        """데이터 삽입"""
        return self.db.insert(table, data)

    def delete(self, table: str, condition: str) -> str:
        """데이터 삭제 (확인 필요)"""
        return self.db.delete(table, condition)
```

### 5.2 프레임워크 자동 주입

```python
@tool
def my_tool(
    user_param: str,                  # 모델이 제공
    agent: Agent = None,              # 프레임워크 주입
    team: Team = None,                # 프레임워크 주입
    run_context: RunContext = None,   # 프레임워크 주입
    images: Sequence[Image] = None,   # 프레임워크 주입
) -> str:
    # run_context에서 세션 정보 접근
    if run_context:
        session_state = run_context.session_state
        user_id = run_context.user_id

    # agent에서 지식베이스 접근
    if agent and agent.knowledge:
        results = agent.knowledge.search(user_param)

    return "결과"
```

### 5.3 도구 훅 (인터셉터)

```python
# 로깅 인터셉터
def logging_hook(function_name: str, args: dict, func: Callable, **kwargs):
    print(f"[START] {function_name} with {args}")
    result = func(**args)  # 다음 함수 호출
    print(f"[END] {function_name}")
    return result

# 재시도 인터셉터
def retry_hook(function_name: str, args: dict, func: Callable, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(**args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

@tool(tool_hooks=[logging_hook, retry_hook])
def unreliable_api(endpoint: str) -> str:
    pass
```

### 5.4 MCP (Model Context Protocol) 통합

```python
from agno.tools.mcp import MCPTools

# STDIO 전송 (로컬 프로세스)
mcp_tools = MCPTools(
    command="my-mcp-server",
    transport="stdio",
)

# HTTP 전송 (원격 서버)
mcp_tools = MCPTools(
    url="https://mcp.example.com",
    transport="streamable-http",
    include_tools=["search", "summarize"],  # 특정 도구만
    tool_name_prefix="external_",           # 접두사 추가
)

# 동적 헤더 (멀티테넌트)
def get_tenant_headers(run_context: RunContext, **kwargs):
    return {
        "X-Tenant-ID": run_context.session_state.get("tenant_id"),
        "Authorization": f"Bearer {run_context.session_state.get('api_key')}"
    }

mcp_tools = MCPTools(
    url="https://mcp.example.com",
    header_provider=get_tenant_headers,  # 런타임마다 새 헤더
)

# Agent에 연결
agent = Agent(model=model, tools=[mcp_tools])
await mcp_tools.connect()
```

### 5.5 도구 제어 옵션

```python
agent = Agent(
    tools=[tool1, tool2, toolkit],

    # 도구 호출 제한
    tool_call_limit=5,  # 최대 5회 호출

    # 도구 선택 제어
    tool_choice="auto",      # 자동 선택 (기본)
    # tool_choice="none",    # 도구 호출 금지
    # tool_choice={"type": "function", "function": {"name": "search"}}  # 강제
)
```

---

## 6. Hook 시스템

### 6.1 훅 타입

```python
# 1. 일반 함수 훅
def simple_hook(run_output, agent, session, run_context, user_id, debug_mode):
    pass

# 2. @hook 데코레이터 (백그라운드 실행 지원)
from agno.hooks import hook

@hook(run_in_background=True)
def background_hook(run_output, agent):
    # 비동기로 실행, 메인 흐름 차단 안함
    send_notification(run_output.content)

# 3. 비동기 훅 (arun 메서드에서만 사용)
@hook
async def async_hook(run_output, agent):
    await send_notification(run_output.content)
```

### 6.2 훅 실행 시점

```
Agent.run()
    ↓
[PRE-HOOKS] ← run_input 수정 가능
    ↓
도구 결정 → 메시지 준비 → 모델 호출 → 응답 처리
    ↓
[POST-HOOKS] ← run_output 읽기/처리
    ↓
반환
```

### 6.3 입력 검증 훅

```python
from agno.exceptions import InputCheckError, CheckTrigger
from agno.hooks import hook

@hook
def validate_input(run_input, agent):
    """입력 검증"""
    content = run_input.input_content or ""

    # 빈 입력 검증
    if not content.strip():
        raise InputCheckError(
            message="입력이 비어있습니다",
            check_trigger=CheckTrigger.INPUT_NOT_ALLOWED
        )

    # PII 검증
    if contains_pii(content):
        raise InputCheckError(
            message="개인정보가 감지되었습니다",
            check_trigger=CheckTrigger.PII_DETECTED,
            additional_data={"detected": ["email", "phone"]}
        )

agent = Agent(model=model, pre_hooks=[validate_input])
```

### 6.4 출력 필터링 훅

```python
from agno.exceptions import OutputCheckError
import re

@hook
def filter_output(run_output, agent):
    """민감한 정보 필터링"""
    if run_output.content:
        # 신용카드 번호 마스킹
        run_output.content = re.sub(
            r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}',
            '[CARD REDACTED]',
            run_output.content
        )

        # 금지어 검사
        if "forbidden" in run_output.content.lower():
            raise OutputCheckError(
                message="금지된 콘텐츠가 포함되어 있습니다",
                check_trigger=CheckTrigger.OUTPUT_NOT_ALLOWED
            )

agent = Agent(model=model, post_hooks=[filter_output])
```

### 6.5 메트릭 수집 훅

```python
@hook(run_in_background=True)  # 비동기로 실행
def collect_metrics(run_output, agent, session, run_context):
    metrics = {
        "session_id": session.session_id,
        "response_length": len(run_output.content or ""),
        "tool_calls": len(run_output.tools or []),
        "status": run_output.status,
        "latency": run_output.metrics.response_time if run_output.metrics else None,
    }
    analytics.track(metrics)

agent = Agent(model=model, post_hooks=[collect_metrics])
```

### 6.6 자동 인자 필터링

```python
# 훅은 필요한 인자만 받을 수 있음 (자동 필터링)

@hook
def minimal_hook(run_output):  # run_output만 받음
    pass

@hook
def full_hook(run_output, agent, session, run_context, user_id, debug_mode):
    pass

@hook
def flexible_hook(run_output, **kwargs):  # 모든 인자 받음
    pass
```

---

## 7. Reasoning 시스템

### 7.1 네이티브 추론 모델 지원

| 모델 | 조건 |
|------|------|
| DeepSeek | deepseek-reasoner, deepseek-r1 |
| Anthropic Claude | Claude 3.7+ with thinking |
| OpenAI | o1, o3, o4 시리즈 |
| Google Gemini | 2.5+, thinking_budget > 0 |
| Groq | DeepSeek, Qwen3 32b |
| Ollama | qwq, deepseek-r1, openthinker |

### 7.2 추론 활성화

```python
# 네이티브 모델 사용 (자동 감지)
agent = Agent(
    model=DeepSeek(id="deepseek-reasoner"),
    reasoning=True,
)

# 별도 추론 모델 지정
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    reasoning=True,
    reasoning_model=DeepSeek(id="deepseek-reasoner"),
    reasoning_min_steps=1,
    reasoning_max_steps=10,
)

# 실행
response = agent.run("복잡한 수학 문제를 풀어주세요")
print(response.reasoning_content)   # 추론 과정
print(response.reasoning_steps)     # 단계별 분석
print(response.content)             # 최종 답변
```

### 7.3 추론 단계 구조

```python
@dataclass
class ReasoningStep:
    title: str                          # 단계 제목
    action: str                         # 수행 액션 (1인칭)
    result: str                         # 실행 결과
    reasoning: str                      # 사고 과정
    next_action: NextAction            # CONTINUE, VALIDATE, FINAL_ANSWER, RESET
    confidence: float                   # 0.0 - 1.0
```

### 7.4 기본 Chain-of-Thought (CoT)

네이티브 추론 미지원 모델용 6단계 프롬프트:

1. **문제 분석**: 핵심 요소 식별
2. **분해 및 전략**: 하위 문제로 분해
3. **의도 명확화**: 계획 수립
4. **실행**: 단계별 실행
5. **검증**: 결과 검증 (필수)
6. **최종 답변**: 종합 답변

### 7.5 추론 토큰 관리

```python
response = agent.run("query")

# 토큰 사용량 확인
if response.metrics:
    print(f"추론 토큰: {response.metrics.reasoning_tokens}")
    print(f"총 비용: ${response.metrics.cost}")
```

### 7.6 추론 결과 표시

```python
from agno.utils.print_response import print_run_output

# 추론 과정 표시
print_run_output(response, show_reasoning=True)

# 추론 과정 숨김 (최종 답변만)
print_run_output(response, show_reasoning=False)
```

---

## 8. 실전 활용 팁

### 8.1 성능 최적화 체크리스트

```python
agent = Agent(
    model=model,
    db=db,

    # ✅ 세션 캐싱
    cache_session=True,

    # ✅ 히스토리 제한
    add_history_to_context=True,
    num_history_runs=5,
    max_tool_calls_from_history=3,

    # ✅ 도구 결과 압축
    compress_tool_results=True,
    compress_token_limit=4000,

    # ✅ 재시도 설정
    retries=3,
    exponential_backoff=True,

    # ✅ 텔레메트리 비활성화 (프로덕션)
    telemetry=False,
)
```

### 8.2 멀티테넌트 설정

```python
# 테넌트별 격리
agent = Agent(
    model=model,
    db=db,
    session_state={
        "tenant_id": "tenant-123",
        "permissions": ["read", "write"],
    },
)

# 테넌트별 도구 권한
@tool
def admin_tool(param: str, run_context: RunContext = None) -> str:
    if "admin" not in run_context.session_state.get("permissions", []):
        raise PermissionError("관리자 권한이 필요합니다")
    return "결과"
```

### 8.3 메모리 관리 전략

```python
# 주기적 메모리 요약 (토큰 절약)
async def optimize_user_memory(user_id: str):
    if memory_manager.count_tokens(user_id) > 5000:
        await memory_manager.aoptimize_memories(
            user_id=user_id,
            strategy=MemoryOptimizationStrategyType.SUMMARIZE,
            apply=True
        )

# 메모리 검색과 컨텍스트 주입
relevant_memories = memory_manager.search_user_memories(
    query="사용자의 기술 스택",
    retrieval_method="agentic",
    limit=5,
    user_id=user_id
)
```

### 8.4 에러 핸들링 패턴

```python
from agno.exceptions import InputCheckError, OutputCheckError, AgentRunException

try:
    response = agent.run(user_input)

    if response.status == RunStatus.error:
        # 에이전트 레벨 에러
        handle_error(response.content)
    elif response.status == RunStatus.cancelled:
        # 취소됨
        handle_cancellation()
    else:
        # 성공
        process_response(response)

except InputCheckError as e:
    # Pre-hook 검증 실패
    notify_user(f"입력 오류: {e.message}")

except OutputCheckError as e:
    # Post-hook 검증 실패
    notify_user(f"출력 필터링: {e.message}")

except AgentRunException as e:
    # 도구 실행 실패
    log_error(e)
```

### 8.5 디버깅 설정

```python
agent = Agent(
    model=model,
    debug_mode=True,
    debug_level=2,  # 상세 로깅
)

# 런타임 디버깅
response = agent.run(
    input="query",
    debug_mode=True,
)

# 이벤트 스트리밍으로 진행 상황 확인
for event in agent.run_stream(input="query", stream_events=True):
    print(f"이벤트: {event.event_type}")
    if event.event_type == "tool_call_started":
        print(f"  도구: {event.tool_name}")
```

### 8.6 프로덕션 체크리스트

1. **데이터베이스**: PostgreSQL 또는 MySQL 사용
2. **세션 관리**: `cache_session=True`, 적절한 히스토리 제한
3. **메모리**: 주기적 최적화, 토큰 임계값 설정
4. **도구 압축**: `compress_tool_results=True`
5. **훅**: 입력 검증, 출력 필터링, 메트릭 수집
6. **재시도**: `retries=3`, `exponential_backoff=True`
7. **텔레메트리**: 프로덕션에서 비활성화
8. **로깅**: 적절한 레벨 설정

---

## 파일 경로 참조

| 기능 | 경로 |
|------|------|
| Agent 핵심 | `/libs/agno/agno/agent/agent.py` |
| Session | `/libs/agno/agno/session/` |
| Memory | `/libs/agno/agno/memory/` |
| Tools | `/libs/agno/agno/tools/` |
| Hooks | `/libs/agno/agno/hooks/` |
| Reasoning | `/libs/agno/agno/reasoning/` |
| Models | `/libs/agno/agno/models/` |
| Database | `/libs/agno/agno/db/` |
| Compression | `/libs/agno/agno/compression/` |

---

*이 보고서는 Agno v0.x 기준으로 작성되었습니다. 프레임워크 업데이트에 따라 내용이 변경될 수 있습니다.*
