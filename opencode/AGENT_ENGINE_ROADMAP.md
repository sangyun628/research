# Thronicle 자체 에이전트 엔진 구축 로드맵

> Agno 프레임워크를 제거하고, OpenCode(설계 철학) + OpenHarness(Python 참조 구현)를
> 벤치마킹하여 Thronicle 전용 에이전트 엔진을 직접 개발하는 단계별 계획.

---

## 전제: Agno가 현재 하는 일

Agno를 제거하려면, 이것이 현재 담당하는 기능을 모두 대체해야 합니다:

```
Agno가 해주는 것 (= 직접 구현해야 할 것)
├── 1. LLM API 호출 + 스트리밍
├── 2. 에이전트 루프 (도구 호출 → 실행 → 재호출)
├── 3. 도구 등록 + 스키마 생성 + 실행
├── 4. 세션/메시지 저장 (DB)
├── 5. 대화 기록 관리 (num_history_runs)
├── 6. 시스템 프롬프트 조립
├── 7. 토큰/비용 추적
├── 8. 세션 요약 (SessionSummaryManager)
└── 9. 스트리밍 이벤트를 HTTP 응답으로 변환

Agno를 유지할 것 (당장은 교체하지 않아도 되는 것)
├── AgentOS HTTP 라우팅 ← FastAPI로 직접 교체 가능
└── RunContext ← 직접 구현
```

---

## 개발 순서 총괄

```
Phase 0: 설계 + 데이터 모델 (1주)
    ↓
Phase 1: LLM 클라이언트 (1~2주)
    ↓
Phase 2: 메시지 시스템 + DB 영속 (1주)
    ↓
Phase 3: 도구 시스템 (1~2주)
    ↓
Phase 4: 에이전트 루프 (1주)
    ↓
Phase 5: 컨텍스트 관리 엔진 (1~2주)
    ↓
Phase 6: 세션 관리 + API 엔드포인트 (1주)
    ↓
Phase 7: 기존 도구 마이그레이션 (1~2주)
    ↓
Phase 8: Agno 제거 + 통합 테스트 (1주)
    ↓
Phase 9: 고도화 (지속)
```

---

## Phase 0: 설계 + 데이터 모델

### 목표
프로젝트 구조와 핵심 데이터 모델을 확정합니다.
코드를 쓰기 전에 "어떤 데이터가 어떻게 흐르는지"를 먼저 설계합니다.

### 디렉토리 구조 설계

```
server/src/thronicle/
├── engine/                    # ★ 새로 만드는 에이전트 엔진
│   ├── __init__.py
│   ├── client.py              # LLM API 클라이언트 (Phase 1)
│   ├── openai_client.py       # OpenAI 호환 클라이언트 (Phase 1)
│   ├── provider_registry.py   # 프로바이더 감지/라우팅 (Phase 1)
│   ├── messages.py            # 메시지 데이터 모델 (Phase 2)
│   ├── message_converter.py   # 내부 → LLM API 포맷 변환 (Phase 2)
│   ├── tool_base.py           # 도구 베이스 클래스 (Phase 3)
│   ├── tool_registry.py       # 도구 등록/필터링 (Phase 3)
│   ├── tool_executor.py       # 도구 실행 + truncation (Phase 3)
│   ├── agent_loop.py          # 핵심 에이전트 루프 (Phase 4)
│   ├── compact.py             # Pruning + Compaction (Phase 5)
│   ├── overflow.py            # Overflow 감지 (Phase 5)
│   ├── token_estimator.py     # 토큰 추정 (Phase 5)
│   ├── cost_tracker.py        # 비용 추적 (Phase 5)
│   ├── session.py             # 세션 관리 (Phase 6)
│   ├── prompt_builder.py      # 시스템 프롬프트 조립 (Phase 6)
│   └── stream_events.py       # 스트리밍 이벤트 타입 (Phase 1)
├── tools/                     # 기존 도구 (Phase 7에서 마이그레이션)
│   ├── cortex_toolkit.py      # 유지 (인터페이스만 변경)
│   ├── workspace_toolkit.py   # 유지 (인터페이스만 변경)
│   └── ...
└── agents/
    └── cortex_investor.py     # Phase 8에서 Agno → engine으로 교체
```

### 데이터 모델 설계

```python
# engine/messages.py — Phase 0에서 확정, Phase 2에서 구현

# ── 참조 ──
# OpenCode: session/message-v2.ts
# OpenHarness: engine/messages.py

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ToolState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class ToolTime(BaseModel):
    start: float | None = None
    end: float | None = None
    compacted: float | None = None  # pruning 시 타임스탬프

class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write

# 메시지 파트 (OpenCode 패턴)
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str
    synthetic: bool = False

class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    tool_name: str
    input: dict
    state: ToolState = ToolState.PENDING
    output: str = ""
    metadata: dict = Field(default_factory=dict)
    time: ToolTime = Field(default_factory=ToolTime)

class ReasoningPart(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str

Part = TextPart | ToolCallPart | ReasoningPart

# 메시지
class Message(BaseModel):
    id: str
    session_id: str
    role: MessageRole
    parts: list[Part] = Field(default_factory=list)
    model_id: str | None = None
    agent_id: str | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: float = 0.0
    summary: bool = False       # compaction 요약 메시지 여부
    created_at: float
```

### 벤치마크 참조

| 설계 항목 | OpenCode 참조 | OpenHarness 참조 |
|----------|-------------|-----------------|
| 메시지 파트 구조 | `session/message-v2.ts` (TextPart, ToolPart, ReasoningPart) | `engine/messages.py` |
| DB 스키마 | `session/session.sql.ts` (message + part 테이블) | 없음 (파일 기반) |
| 디렉토리 구조 | `session/`, `tool/`, `provider/` 분리 | `engine/`, `tools/`, `api/` 분리 |

---

## Phase 1: LLM 클라이언트

### 목표
Anthropic SDK와 OpenAI SDK를 직접 사용하는 스트리밍 클라이언트를 만듭니다.
이것이 Vercel AI SDK(OpenCode) / Agno의 LLM 호출 부분을 대체합니다.

### 구현

```python
# engine/stream_events.py — 통일된 스트리밍 이벤트 타입

class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str

class ToolCallStartEvent(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    tool_name: str

class ToolCallDeltaEvent(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    call_id: str
    input_json: str  # 스트리밍되는 JSON 조각

class ToolCallCompleteEvent(BaseModel):
    type: Literal["tool_call_complete"] = "tool_call_complete"
    call_id: str
    tool_name: str
    input: dict

class ReasoningDeltaEvent(BaseModel):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str

class FinishEvent(BaseModel):
    type: Literal["finish"] = "finish"
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"
    usage: dict

class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: str
    retryable: bool = False

StreamEvent = (TextDeltaEvent | ToolCallStartEvent | ToolCallDeltaEvent |
               ToolCallCompleteEvent | ReasoningDeltaEvent | FinishEvent | ErrorEvent)
```

```python
# engine/client.py — Anthropic 클라이언트

from typing import AsyncIterator, Protocol

class LLMClient(Protocol):
    """LLM 클라이언트 공통 인터페이스"""
    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        model: str,
        max_tokens: int = 32_000,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ...

class AnthropicClient:
    """anthropic SDK 직접 사용"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url)

    async def stream(self, messages, tools, system, model, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        # cache_control 마커 주입 (OpenCode 패턴)
        system_with_cache = self._apply_cache_markers(system)

        async with self._client.messages.stream(
            model=model,
            system=system_with_cache,
            messages=messages,
            tools=tools,
            max_tokens=kwargs.get("max_tokens", 32_000),
            temperature=kwargs.get("temperature"),
        ) as stream:
            async for event in stream:
                for converted in self._convert_event(event):
                    yield converted

    def _apply_cache_markers(self, system: str) -> list[dict]:
        """OpenCode 패턴: 시스템 프롬프트에 cache_control 마커 배치"""
        return [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
```

```python
# engine/openai_client.py — OpenAI 호환 클라이언트

class OpenAICompatibleClient:
    """openai SDK로 OpenAI-compatible 서버 전부 커버"""

    def __init__(self, api_key: str, base_url: str | None = None, model: str = ""):
        import openai
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url)

    async def stream(self, messages, tools, system, model, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        # Anthropic 형식 messages → OpenAI 형식으로 변환
        oai_messages = self._convert_messages(messages, system)
        oai_tools = self._convert_tools(tools)

        response = await self._client.chat.completions.create(
            model=model,
            messages=oai_messages,
            tools=oai_tools or openai.NOT_GIVEN,
            max_tokens=kwargs.get("max_tokens", 32_000),
            temperature=kwargs.get("temperature"),
            stream=True,
        )
        async for chunk in response:
            for converted in self._convert_chunk(chunk):
                yield converted
```

```python
# engine/provider_registry.py — 프로바이더 감지 + 클라이언트 생성

PROVIDER_CONFIGS = {
    "anthropic": {"sdk": "anthropic", "base_url": None},
    "openai": {"sdk": "openai", "base_url": None},
    "gemini": {"sdk": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    "deepseek": {"sdk": "openai", "base_url": "https://api.deepseek.com/v1"},
    "groq": {"sdk": "openai", "base_url": "https://api.groq.com/openai/v1"},
    "ollama": {"sdk": "openai", "base_url": "http://localhost:11434/v1"},
    # ... 필요한 만큼 추가
}

def create_client(provider_id: str, api_key: str) -> LLMClient:
    config = PROVIDER_CONFIGS[provider_id]
    if config["sdk"] == "anthropic":
        return AnthropicClient(api_key=api_key, base_url=config["base_url"])
    else:
        return OpenAICompatibleClient(api_key=api_key, base_url=config["base_url"])
```

### 벤치마크 참조

| 구현 항목 | OpenCode | OpenHarness |
|----------|----------|-------------|
| 클라이언트 인터페이스 | `streamText()` → 프로바이더별 어댑터 | `SupportsStreamingMessages` 프로토콜 |
| 이벤트 타입 | `text-delta`, `tool-call`, `finish-step` | `ApiTextDeltaEvent`, `ApiMessageCompleteEvent` |
| 캐시 전략 | `transform.ts::applyCaching()` — 4개 마커 | 없음 (여기서 OpenCode를 따름) |
| 재시도 | 프로바이더 SDK 내장 | 지수 백오프, 3회 재시도, 429/500/502/503 |

### 검증 기준
- [ ] Anthropic Claude에 메시지 보내고 스트리밍 응답 받기
- [ ] OpenAI GPT에 동일 인터페이스로 동작
- [ ] 도구 스키마 전달 + tool_use 이벤트 파싱
- [ ] Gemini에 OpenAI-compatible 엔드포인트로 동작

---

## Phase 2: 메시지 시스템 + DB 영속

### 목표
대화 기록을 PostgreSQL에 저장하고, LLM API 형식으로 변환하는 레이어를 만듭니다.
OpenCode의 "DB 원본 vs LLM 전달본 분리" 패턴을 구현합니다.

### 구현

```python
# engine/message_converter.py — 내부 메시지 → LLM API 형식 변환

def to_anthropic_messages(
    messages: list[Message],
    strip_media: bool = False,
) -> list[dict]:
    """
    내부 Message 리스트를 Anthropic API messages 배열로 변환.

    핵심 변환:
    1. ToolCallPart의 compacted된 출력 → "[Old tool result content cleared]"
    2. 미디어 제거 (compaction 시 strip_media=True)
    3. ReasoningPart → thinking 블록
    """
    result = []
    for msg in messages:
        if msg.role == MessageRole.USER:
            # TextPart → {"type": "text", "text": ...}
            # + compaction 요약 이후만 포함
            ...
        elif msg.role == MessageRole.ASSISTANT:
            # TextPart → text content
            # ToolCallPart → tool_use + tool_result
            #   ★ compacted된 것은 "[Old tool result content cleared]"
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    output = (
                        "[Old tool result content cleared]"
                        if part.time.compacted
                        else part.output
                    )
                    ...
    return result
```

```python
# DB 모델 확장 (기존 thronicle/db/models.py에 추가)
# OpenCode 패턴: message + part 분리 저장

class EngineMessage(Base):
    __tablename__ = "engine_messages"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    time_created: Mapped[float] = mapped_column(Float, nullable=False)
    time_updated: Mapped[float] = mapped_column(Float, nullable=False)

class EnginePart(Base):
    __tablename__ = "engine_parts"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(String, ForeignKey("engine_messages.id", ondelete="CASCADE"))
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    time_created: Mapped[float] = mapped_column(Float, nullable=False)
    time_updated: Mapped[float] = mapped_column(Float, nullable=False)
```

### 벤치마크 참조

| 구현 항목 | OpenCode | OpenHarness |
|----------|----------|-------------|
| DB 스키마 | message + part 테이블 (SQLite) | 없음 (파일 기반) → **OpenCode를 따름** |
| 메시지 변환 | `MessageV2.toModelMessages()` | `openai_client._convert_messages()` |
| Pruned 출력 교체 | `compacted → "[cleared]"` | `microcompact → "[cleared]"` — 동일 |
| 커서 페이지네이션 | 50개씩 로드 | 전체 로드 → **OpenCode를 따름** |

### 검증 기준
- [ ] Message + Part를 PostgreSQL에 저장/조회
- [ ] 내부 메시지 → Anthropic API 형식 변환
- [ ] 내부 메시지 → OpenAI API 형식 변환
- [ ] compacted된 도구 출력이 "[cleared]"로 변환되는지 확인

---

## Phase 3: 도구 시스템

### 목표
Agno의 `Toolkit.register()` + `BaseTool`을 자체 도구 시스템으로 교체합니다.

### 구현

```python
# engine/tool_base.py

class ToolContext(BaseModel):
    """도구 실행 컨텍스트"""
    session_id: str
    user_id: str | None = None
    message_id: str = ""
    call_id: str = ""

class ToolResult(BaseModel):
    """도구 실행 결과"""
    output: str = ""
    title: str = ""
    metadata: dict = Field(default_factory=dict)
    error: str | None = None

class BaseTool(ABC):
    """모든 도구의 베이스 클래스"""
    name: str
    description: str                    # LLM에게 보내는 설명 (★ 가장 중요)
    input_schema: type[BaseModel]       # Pydantic 모델 → JSON Schema 자동 생성
    read_only: bool = False             # 읽기 전용 여부 (권한 시스템)

    @abstractmethod
    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult:
        ...

    def get_schema(self) -> dict:
        """LLM에 전달할 도구 스키마 생성 (Anthropic 형식)"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }
```

```python
# engine/tool_registry.py

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tools(self, agent_id: str | None = None) -> list[BaseTool]:
        # 에이전트별 필터링 (permission 기반)
        ...

    def get_schemas(self, agent_id: str | None = None) -> list[dict]:
        return [t.get_schema() for t in self.get_tools(agent_id)]

    async def execute(self, tool_name: str, input: dict, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(error=f"Unknown tool: {tool_name}")

        # 1. 입력 검증 (Pydantic)
        validated = tool.input_schema.model_validate(input)

        # 2. 실행
        result = await tool.execute(validated, ctx)

        # 3. 자동 truncation (OpenCode 패턴)
        result = self._truncate_if_needed(result, tool_name)

        return result
```

### 도구 description 전략 (OpenCode 핵심 패턴)

```python
# tools/descriptions/ 디렉토리에 별도 텍스트 파일로 관리

# tools/descriptions/workspace_read.txt
"""워크스페이스에서 파일을 읽습니다.

- offset(1-indexed)과 limit으로 특정 구간만 읽을 수 있습니다
- 기본 2000줄 또는 50KB까지만 읽습니다
- 큰 파일은 먼저 workspace_grep으로 위치를 찾고, offset으로 해당 부분만 읽으세요
- 경로가 불확실하면 workspace_glob으로 먼저 찾으세요
- 여러 파일을 읽을 때는 한 번에 병렬로 호출하세요
- 30줄 단위로 자르지 마세요. 200줄 이상 넉넉하게 읽으세요"""

# 도구 클래스에서 로드:
class WorkspaceReadTool(BaseTool):
    name = "workspace_read"
    description = Path("tools/descriptions/workspace_read.txt").read_text()
```

### 벤치마크 참조

| 구현 항목 | OpenCode | OpenHarness |
|----------|----------|-------------|
| 도구 베이스 | `Tool.define()` + Zod | `BaseTool` + Pydantic |
| 스키마 생성 | `z.object()` → JSON Schema | `model_json_schema()` — **동일 패턴** |
| Truncation | 범용 서비스 (2000줄/50KB) | bash만 12KB → **OpenCode를 따름** |
| Description 관리 | 별도 `.txt` 파일 | 클래스 내 docstring → **OpenCode를 따름** |
| 자동 truncation | `Tool.define()` 래퍼에서 자동 | 없음 → **OpenCode를 따름** |

### 검증 기준
- [ ] BaseTool 상속으로 도구 정의 + Pydantic 입력 검증
- [ ] ToolRegistry에 등록 + LLM 스키마 생성
- [ ] 도구 실행 + 자동 truncation 적용
- [ ] 기존 WorkspaceToolkit의 도구 1개를 새 BaseTool로 변환

---

## Phase 4: 에이전트 루프

### 목표
핵심 에이전트 루프를 구현합니다. "LLM 호출 → 도구 실행 → 재호출" 반복.

### 구현

```python
# engine/agent_loop.py

DOOM_LOOP_THRESHOLD = 3
MAX_TURNS = 200

async def run_agent_loop(
    client: LLMClient,
    tools: ToolRegistry,
    messages: list[Message],
    system_prompt: str,
    model: str,
    agent_id: str,
    *,
    max_turns: int = MAX_TURNS,
    on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
    compact_engine: CompactEngine | None = None,
) -> Message:
    """
    OpenCode/OpenHarness 스타일 에이전트 루프.

    흐름:
    1. 매 turn: overflow 체크 → 필요 시 pruning/compaction
    2. LLM 스트리밍 호출
    3. 텍스트 + 도구 호출 파싱
    4. 도구 실행 (단일: 순차, 복수: asyncio.gather)
    5. 결과를 messages에 추가
    6. 도구 호출이 없으면 종료, 있으면 반복
    """
    recent_tool_calls: list[list[dict]] = []
    assistant_msg = _create_assistant_message(session_id, agent_id, model)

    for turn in range(max_turns):
        # ── 1. 자동 컨텍스트 관리 (Phase 5에서 구현) ──
        if compact_engine:
            await compact_engine.auto_compact_if_needed(messages)

        # ── 2. 메시지 → LLM API 형식 변환 ──
        api_messages = to_anthropic_messages(messages)  # or openai
        tool_schemas = tools.get_schemas(agent_id)

        # ── 3. LLM 스트리밍 호출 ──
        text_parts: list[str] = []
        tool_calls: list[ToolCallCompleteEvent] = []

        async for event in client.stream(
            messages=api_messages,
            tools=tool_schemas,
            system=system_prompt,
            model=model,
        ):
            # UI로 이벤트 전달
            if on_event:
                await on_event(event)

            match event:
                case TextDeltaEvent(text=text):
                    text_parts.append(text)
                    # assistant_msg에 TextPart 누적

                case ToolCallCompleteEvent() as tc:
                    tool_calls.append(tc)

                case ReasoningDeltaEvent(text=text):
                    # ReasoningPart 누적

                case FinishEvent(usage=usage, stop_reason=reason):
                    # 토큰/비용 집계
                    _update_usage(assistant_msg, usage, model)

                    # overflow 체크 (매 스텝)
                    if compact_engine and compact_engine.is_overflow(assistant_msg.tokens):
                        # compaction 필요 플래그
                        pass

        # ── 4. 도구 호출이 없으면 종료 ──
        if not tool_calls:
            messages.append(assistant_msg)
            return assistant_msg

        # ── 5. Doom Loop 감지 ──
        if _detect_doom_loop(recent_tool_calls, tool_calls):
            # 사용자에게 알리고 중단
            ...

        # ── 6. 도구 실행 ──
        if len(tool_calls) == 1:
            # 단일: 순차 실행
            result = await tools.execute(
                tool_calls[0].tool_name,
                tool_calls[0].input,
                ToolContext(session_id=..., user_id=...),
            )
        else:
            # 복수: asyncio.gather 병렬 실행
            results = await asyncio.gather(*[
                tools.execute(tc.tool_name, tc.input, ctx)
                for tc in tool_calls
            ])

        # ── 7. 결과를 messages에 추가 → 루프 반복 ──
        messages.append(assistant_msg)  # tool_call 포함
        messages.append(_make_tool_result_message(tool_calls, results))

    raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")
```

### 벤치마크 참조

| 구현 항목 | OpenCode | OpenHarness |
|----------|----------|-------------|
| 루프 구조 | `SessionProcessor.process()` → `Stream.tap` | `run_query()` → `async for` |
| 병렬 도구 실행 | `Promise.all()` | `asyncio.gather()` — **동일 패턴** |
| Doom Loop | 최근 3개 name+input 비교 | 최근 3개 비교 — 동일 |
| Max turns | Agent.steps (기본 없음) | 200 — **OpenHarness를 따름** |
| Overflow 체크 시점 | 매 finish-step | 매 turn 시작 — **OpenCode를 따름 (매 step)** |

### 검증 기준
- [ ] LLM 호출 → 텍스트 응답 → 종료 (도구 없이)
- [ ] LLM 호출 → 도구 호출 → 실행 → 결과 피드백 → LLM 재호출 → 종료
- [ ] 복수 도구 호출 시 asyncio.gather 병렬 실행
- [ ] Doom Loop 감지 (같은 도구 3회 연속)

---

## Phase 5: 컨텍스트 관리 엔진

### 목표
OpenCode의 4계층 방어를 구현합니다. 이것이 전체 시스템의 핵심 차별점입니다.

### 구현

```python
# engine/compact.py

CHARS_PER_TOKEN = 4
PRUNE_MINIMUM = 20_000
PRUNE_PROTECT = 40_000
PRUNE_PROTECTED_TOOLS = {"skill"}
COMPACTION_BUFFER = 20_000

class CompactEngine:
    def __init__(self, client: LLMClient, model: str, context_window: int = 200_000):
        self.client = client
        self.model = model
        self.context_window = context_window

    def is_overflow(self, tokens: TokenUsage) -> bool:
        """OpenCode overflow.ts 패턴"""
        max_output = 32_000
        reserved = min(COMPACTION_BUFFER, max_output)
        usable = self.context_window - max_output
        return tokens.total >= usable

    async def auto_compact_if_needed(self, messages: list[Message]):
        """매 turn 호출. 필요 시 pruning → compaction 순서로 실행"""
        estimated = self._estimate_tokens(messages)
        threshold = self.context_window - COMPACTION_BUFFER - 13_000

        if estimated < threshold:
            return

        # 1단계: Microcompact (Pruning)
        freed = self._microcompact(messages)
        if freed >= PRUNE_MINIMUM:
            return  # 충분히 확보됨

        # 2단계: Full Compaction (LLM 요약)
        await self._full_compact(messages)

    def _microcompact(self, messages: list[Message]) -> int:
        """OpenCode pruning 패턴 (토큰 기반 보호)"""
        protected_tokens = 0
        freed_tokens = 0

        for msg in reversed(messages):
            if msg.summary:
                break

            for part in reversed(msg.parts):
                if not isinstance(part, ToolCallPart):
                    continue
                if part.state != ToolState.COMPLETED:
                    continue
                if part.tool_name in PRUNE_PROTECTED_TOOLS:
                    continue
                if part.time.compacted:
                    continue

                tokens = len(part.output) // CHARS_PER_TOKEN

                if protected_tokens < PRUNE_PROTECT:
                    protected_tokens += tokens
                    continue

                # Prune!
                part.time.compacted = time.time()
                freed_tokens += tokens

        return freed_tokens

    async def _full_compact(self, messages: list[Message]):
        """OpenCode compaction 패턴 + OpenHarness XML 형식"""
        # 최근 6개 메시지는 보존
        older = messages[:-6]
        recent = messages[-6:]

        # LLM으로 요약 생성
        summary = await self._generate_summary(older)

        # 요약 메시지 생성 (summary=True 플래그)
        summary_msg = Message(
            id=generate_id(),
            session_id=messages[0].session_id,
            role=MessageRole.ASSISTANT,
            parts=[TextPart(text=summary)],
            summary=True,
            created_at=time.time(),
        )

        # messages를 [요약] + [최근 6개]로 교체
        messages.clear()
        messages.append(summary_msg)
        messages.extend(recent)
```

### 벤치마크 참조 (양쪽의 최선 조합)

| 항목 | 선택 | 이유 |
|------|------|------|
| Pruning 보호 기준 | **OpenCode (토큰 기반 40K)** | 개수 기반(OpenHarness 5개)보다 정밀 |
| Overflow 감지 | **OpenCode 공식** | `context_window - max_output` 계산 |
| Compaction 프롬프트 | **OpenCode 구조** | Goal/Instructions/Discoveries/Accomplished/Files |
| 코드 참조 | **OpenHarness** | Python이라 직접 참고 가능 |
| 자동 실행 시점 | **양쪽 결합** | 매 turn 시작(OpenHarness) + 매 step 체크(OpenCode) |

### 검증 기준
- [ ] 도구 출력이 20턴 후 자동으로 "[cleared]"로 교체되는지
- [ ] 최근 40K 토큰 범위의 도구 출력은 보호되는지
- [ ] Overflow 시 LLM 요약이 생성되고 messages가 교체되는지
- [ ] 요약 후에도 에이전트가 자연스럽게 대화를 이어가는지

---

## Phase 6: 세션 관리 + API 엔드포인트

### 목표
Agno의 `/agents/{id}/runs` 대신 자체 FastAPI 엔드포인트를 만듭니다.

### 구현

```python
# engine/session.py

class SessionManager:
    """세션 생성, 메시지 로드, 세션 재개"""

    async def create_session(self, user_id: str, agent_id: str) -> str: ...
    async def get_messages(self, session_id: str) -> list[Message]: ...
    async def save_message(self, message: Message): ...
    async def save_part(self, part: Part, message_id: str, session_id: str): ...
    async def update_part(self, part_id: str, data: dict): ...

# 새로운 FastAPI 라우터
# routers/engine_runs.py

@router.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, request: RunRequest):
    """Agno의 /agents/{id}/runs 대체"""
    session = await session_mgr.get_or_create(request.session_id, ...)
    messages = await session_mgr.get_messages(session.id)

    client = create_client(provider_id, api_key)
    tools = tool_registry.get_tools(agent_id)
    system = prompt_builder.build(agent_id, session)

    # 에이전트 루프 실행 → SSE 스트리밍 응답
    async def event_generator():
        async for event in run_agent_loop(
            client=client, tools=tools, messages=messages,
            system_prompt=system, model=model, agent_id=agent_id,
        ):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 검증 기준
- [ ] POST /agents/cortex-investor/run → SSE 스트리밍 응답
- [ ] 세션 재개 (같은 session_id로 이전 대화 이어가기)
- [ ] 메시지 + 파트가 PostgreSQL에 영속

---

## Phase 7: 기존 도구 마이그레이션

### 목표
Agno `Toolkit` 형식의 기존 도구들을 새 `BaseTool` 형식으로 변환합니다.

### 마이그레이션 순서 (의존성 + 중요도)

```
1순위: WorkspaceToolkit (13개 도구)
  → workspace_read, write, edit, edit_section, multiedit,
    glob, grep, read_section, extract_toc, backlinks,
    list, create_folder, delete
  → 이미 핵심 로직이 구현되어 있으므로 인터페이스만 교체

2순위: CortexToolkit (8개 도구)
  → memo_write/read/list, plan_create/step_done/status,
    context_summarize, reflect
  → 인터페이스 교체 + context_summarize를 CompactEngine과 통합

3순위: Domain Toolkits (DART, FMP, News 등)
  → 각 도구의 execute 로직은 그대로, 래퍼만 교체
  → 점진적 마이그레이션 가능

4순위: ReasoningTools (think, analyze)
  → Agno 내장이므로 직접 구현 필요 (단순)
```

### 마이그레이션 패턴

```python
# Before (Agno Toolkit):
class WorkspaceToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="workspace_toolkit")
        self.register(self.workspace_read)

    def workspace_read(self, path: str, offset: int = 1, limit: int = 2000,
                       run_context: RunContext | None = None) -> str:
        ...

# After (자체 BaseTool):
class WorkspaceReadInput(BaseModel):
    path: str = Field(description="파일 경로")
    offset: int = Field(default=1, description="읽기 시작 줄 번호 (1-indexed)")
    limit: int = Field(default=2000, description="읽을 최대 줄 수")

class WorkspaceReadTool(BaseTool):
    name = "workspace_read"
    description = WORKSPACE_READ_DESCRIPTION  # .txt 파일에서 로드
    input_schema = WorkspaceReadInput
    read_only = True

    async def execute(self, input: WorkspaceReadInput, ctx: ToolContext) -> ToolResult:
        # 기존 workspace_read 로직 재사용
        ...
```

---

## Phase 8: Agno 제거 + 통합 테스트

### 목표
Agno 의존성을 완전히 제거하고, 새 엔진으로 전환합니다.

### 전환 순서

```
1. cortex_investor.py에서 Agno Agent → engine.agent_loop 교체
2. app.py에서 AgentOS 라우팅 → FastAPI 라우터 교체
3. pyproject.toml에서 agno 의존성 제거
4. 통합 테스트 실행
```

### 검증 기준
- [ ] 기존 클라이언트(Next.js)에서 동일한 API 호출이 동작하는지
- [ ] 10턴 이상 대화에서 컨텍스트 관리가 정상 작동하는지
- [ ] 세션 재개가 되는지
- [ ] 비용 추적이 정확한지
- [ ] 스트리밍 응답이 끊김 없이 동작하는지

---

## Phase 9: 고도화

### 단기

| 항목 | 참조 |
|------|------|
| 프롬프트 캐싱 최적화 | OpenCode `transform.ts::applyCaching()` |
| Decimal 비용 계산 | OpenCode `session/index.ts::getUsage()` |
| 둠 루프 → 사용자 확인 UI | OpenCode `processor.ts` |

### 중기

| 항목 | 참조 |
|------|------|
| 멀티에이전트 (Swarm) | OpenHarness `swarm/` |
| IM 채널 통합 | OpenHarness `channels/` |
| MCP 프로토콜 지원 | OpenHarness `mcp/` |

### 장기

| 항목 | 참조 |
|------|------|
| 크론 스케줄링 | OpenHarness `cron_scheduler.py` |
| 플러그인 시스템 | OpenCode `plugin/`, OpenHarness `plugins/` |
| 커스텀 스킬 시스템 | OpenCode `skill/`, OpenHarness `skills/` |

---

## 요약: 각 Phase에서 뭘 참조하는가

```
Phase   OpenCode에서 가져올 것          OpenHarness에서 가져올 것
─────   ──────────────────────          ────────────────────────
0       메시지 데이터 모델 설계          디렉토리 구조 참조
1       캐시 마커 전략                  Python 클라이언트 코드
2       DB 원본/전달본 분리             메시지 변환 로직
3       도구 description 전략 (.txt)    BaseTool + Pydantic 패턴
4       에이전트 루프 설계              run_query() 실제 코드
5       Pruning (토큰 기반), Compaction  compact/ 엔진 코드
6       세션 관리 설계                  SessionStorage 코드
7       -                              도구 마이그레이션 패턴
8       -                              통합 테스트 패턴
9       프롬프트 캐싱, 비용 추적        Swarm, IM 채널
```
