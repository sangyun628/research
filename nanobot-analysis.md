# Nanobot 아키텍처 및 시스템 디자인 분석

> **분석 대상**: [HKUDS/nanobot](https://github.com/HKUDS/nanobot)
> **목적**: 에이전트 시스템 개발 및 설계 관점에서의 기술적 분석
> **분석 일자**: 2026-02-02

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [핵심 컴포넌트 분석](#3-핵심-컴포넌트-분석)
   - 3.1 [Agent Loop (에이전트 루프)](#31-agent-loop-에이전트-루프)
   - 3.2 [Message Bus (메시지 버스)](#32-message-bus-메시지-버스)
   - 3.3 [Tool System (도구 시스템)](#33-tool-system-도구-시스템)
   - 3.4 [Memory System (메모리 시스템)](#34-memory-system-메모리-시스템)
   - 3.5 [Skills System (스킬 시스템)](#35-skills-system-스킬-시스템)
   - 3.6 [Subagent System (서브에이전트 시스템)](#36-subagent-system-서브에이전트-시스템)
4. [시스템 연동 패턴](#4-시스템-연동-패턴)
   - 4.1 [Multi-Channel Integration](#41-multi-channel-integration)
   - 4.2 [Scheduling System](#42-scheduling-system)
   - 4.3 [LLM Provider Abstraction](#43-llm-provider-abstraction)
5. [설계 패턴 및 모범 사례](#5-설계-패턴-및-모범-사례)
6. [벤치마킹 핵심 인사이트](#6-벤치마킹-핵심-인사이트)
7. [구현 시 참고 사항](#7-구현-시-참고-사항)

---

## 1. 프로젝트 개요

### 1.1 Nanobot이란?

Nanobot은 홍콩대학교 HKUDS 팀이 개발한 **초경량 개인 AI 어시스턴트**입니다. [Clawdbot/OpenClaw](https://github.com/openclaw/openclaw)에서 영감을 받아 핵심 에이전트 기능을 **약 4,000줄**의 코드로 구현했습니다.

### 1.2 핵심 특징

| 특징 | 설명 |
|------|------|
| **Ultra-Lightweight** | 430k+ LOC → ~4,000 LOC (99% 감소) |
| **Research-Ready** | 깔끔하고 읽기 쉬운 코드베이스 |
| **Multi-Channel** | CLI, Telegram, WhatsApp 지원 |
| **Multi-Provider LLM** | OpenRouter, Anthropic, OpenAI, vLLM 등 |
| **Extensible** | 플러그인 형태의 Skills 시스템 |
| **Background Processing** | Subagent를 통한 비동기 작업 처리 |
| **Proactive Agent** | Heartbeat/Cron을 통한 능동적 작업 수행 |

### 1.3 기술 스택

```
Python 3.11+
├── typer          # CLI 프레임워크
├── litellm        # Multi-provider LLM 추상화
├── pydantic       # 설정 및 데이터 검증
├── asyncio        # 비동기 처리
├── websockets     # WhatsApp 브릿지 통신
├── httpx          # 비동기 HTTP 클라이언트
├── loguru         # 구조화된 로깅
├── croniter       # Cron 표현식 파싱
└── rich           # 터미널 UI
```

---

## 2. 전체 아키텍처

### 2.1 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NANOBOT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   CLI       │    │  Telegram   │    │  WhatsApp   │    < CHANNELS >      │
│  │  (typer)    │    │   Bot API   │    │   Bridge    │                      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                      │
│         │                  │                  │                              │
│         └──────────────────┼──────────────────┘                              │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       MESSAGE BUS                                    │    │
│  │  ┌─────────────────┐              ┌─────────────────┐               │    │
│  │  │  Inbound Queue  │◄────────────►│ Outbound Queue  │               │    │
│  │  └────────┬────────┘              └────────▲────────┘               │    │
│  └───────────┼────────────────────────────────┼────────────────────────┘    │
│              │                                │                              │
│              ▼                                │                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                        AGENT LOOP                                  │      │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │      │
│  │  │   Context   │───►│  LLM Call   │───►│   Tool      │            │      │
│  │  │   Builder   │    │  (litellm)  │    │ Execution   │            │      │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘            │      │
│  │         ▲                                     │                    │      │
│  │         │                                     ▼                    │      │
│  │  ┌──────┴──────┐    ┌─────────────┐    ┌─────────────┐            │      │
│  │  │   Memory    │    │   Skills    │    │    Tool     │            │      │
│  │  │   Store     │    │   Loader    │    │  Registry   │            │      │
│  │  └─────────────┘    └─────────────┘    └─────────────┘            │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│              │                                                               │
│              ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                      SUBAGENT MANAGER                              │      │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │      │
│  │  │  Subagent 1 │    │  Subagent 2 │    │  Subagent N │            │      │
│  │  │  (async)    │    │  (async)    │    │  (async)    │            │      │
│  │  └─────────────┘    └─────────────┘    └─────────────┘            │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     BACKGROUND SERVICES                            │      │
│  │  ┌─────────────────────────┐    ┌─────────────────────────┐       │      │
│  │  │     CRON SERVICE        │    │   HEARTBEAT SERVICE     │       │      │
│  │  │  (Scheduled Tasks)      │    │   (Periodic Wake-up)    │       │      │
│  │  └─────────────────────────┘    └─────────────────────────┘       │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     DATA PERSISTENCE                               │      │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │      │
│  │  │   Sessions    │  │    Memory     │  │   Cron Jobs   │          │      │
│  │  │   (JSONL)     │  │    (*.md)     │  │   (JSON)      │          │      │
│  │  └───────────────┘  └───────────────┘  └───────────────┘          │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
nanobot/
├── agent/              # 핵심 에이전트 로직
│   ├── loop.py         # 메인 에이전트 루프
│   ├── context.py      # 프롬프트/컨텍스트 빌더
│   ├── memory.py       # 영속적 메모리 시스템
│   ├── skills.py       # 스킬 로더
│   ├── subagent.py     # 서브에이전트 매니저
│   └── tools/          # 내장 도구들
│       ├── base.py     # Tool 추상 클래스
│       ├── registry.py # 동적 도구 레지스트리
│       ├── filesystem.py
│       ├── shell.py
│       ├── web.py
│       ├── message.py
│       └── spawn.py
├── skills/             # 번들된 스킬 (github, weather, tmux...)
├── channels/           # 채널 통합
│   ├── base.py
│   ├── manager.py
│   ├── telegram.py
│   └── whatsapp.py
├── bus/                # 메시지 라우팅
│   ├── queue.py
│   └── events.py
├── cron/               # 스케줄링
│   ├── service.py
│   └── types.py
├── heartbeat/          # 주기적 웨이크업
│   └── service.py
├── providers/          # LLM 프로바이더
│   ├── base.py
│   └── litellm_provider.py
├── session/            # 대화 세션 관리
│   └── manager.py
├── config/             # 설정
│   ├── schema.py
│   └── loader.py
└── cli/                # CLI 명령어
    └── commands.py
```

### 2.3 실행 흐름

#### CLI 모드 (단일 메시지)

```
User Input → CLI Handler → Load Config → Create Components
                                              ↓
                                    ┌─────────────────┐
                                    │   AgentLoop     │
                                    │ process_direct()│
                                    └────────┬────────┘
                                             ↓
                              ┌──────────────────────────┐
                              │  Context Builder         │
                              │  (system prompt +        │
                              │   history + memory +     │
                              │   skills)                │
                              └────────────┬─────────────┘
                                           ↓
                              ┌──────────────────────────┐
                              │  LLM Chat Call           │
                              │  (with tools)            │
                              └────────────┬─────────────┘
                                           ↓
                            ┌─────────────────────────────┐
                            │    Tool Execution Loop      │
                            │  while has_tool_calls:      │
                            │    - Execute tool           │
                            │    - Add result to msgs     │
                            │    - Call LLM again         │
                            └────────────┬────────────────┘
                                         ↓
                              ┌──────────────────────────┐
                              │  Save Session + Response │
                              └──────────────────────────┘
```

#### Gateway 모드 (서버)

```
Startup
   ↓
Load Config → Create Bus → Create Provider → Create AgentLoop
   ↓
Init Channels (Telegram, WhatsApp) → ChannelManager.start_all()
   ↓
Init CronService → Load jobs → Arm timer
   ↓
Init HeartbeatService → Start 30m interval timer
   ↓
┌─────────────────────────────────────────────────────────┐
│              Concurrent asyncio Tasks                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Agent Loop: consume_inbound → process → publish │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Telegram: listen → publish_inbound              │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ WhatsApp: WebSocket → publish_inbound           │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Channel Manager: consume_outbound → send        │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Cron Service: timer tick → execute_job          │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Heartbeat: timer tick → process_direct          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트 분석

### 3.1 Agent Loop (에이전트 루프)

**위치**: `nanobot/agent/loop.py`

Agent Loop는 시스템의 핵심 처리 엔진으로, ReAct(Reasoning + Acting) 패턴을 구현합니다.

#### 핵심 구조

```python
class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations

        # 핵심 컴포넌트 초기화
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(...)
```

#### Tool Execution Loop 패턴

```python
async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
    # 1. 세션 및 컨텍스트 준비
    session = self.sessions.get_or_create(msg.session_key)
    messages = self.context.build_messages(
        history=session.get_history(),
        current_message=msg.content
    )

    # 2. 에이전트 루프 (최대 max_iterations 반복)
    iteration = 0
    final_content = None

    while iteration < self.max_iterations:
        iteration += 1

        # LLM 호출
        response = await self.provider.chat(
            messages=messages,
            tools=self.tools.get_definitions(),
            model=self.model
        )

        # Tool calls 처리
        if response.has_tool_calls:
            # Assistant 메시지 추가 (tool_calls 포함)
            messages = self.context.add_assistant_message(
                messages, response.content, tool_call_dicts
            )

            # 각 도구 실행 및 결과 추가
            for tool_call in response.tool_calls:
                result = await self.tools.execute(tool_call.name, tool_call.arguments)
                messages = self.context.add_tool_result(
                    messages, tool_call.id, tool_call.name, result
                )
        else:
            # Tool calls 없음 = 최종 응답
            final_content = response.content
            break

    # 3. 세션 저장 및 응답 반환
    session.add_message("user", msg.content)
    session.add_message("assistant", final_content)
    self.sessions.save(session)

    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=final_content)
```

#### 설계 포인트

| 요소 | 설계 결정 | 이유 |
|------|----------|------|
| **Max Iterations** | 기본 20회 | 무한 루프 방지, 토큰 비용 제한 |
| **Tool Context Update** | 매 메시지마다 갱신 | 동적 채널/채팅 컨텍스트 반영 |
| **System Message 처리** | 별도 핸들러 | Subagent 결과 라우팅 분리 |
| **Session Key** | `{channel}:{chat_id}` | 채널별 대화 이력 분리 |

---

### 3.2 Message Bus (메시지 버스)

**위치**: `nanobot/bus/queue.py`, `nanobot/bus/events.py`

Message Bus는 채널과 에이전트 코어를 **느슨하게 결합(loose coupling)**하는 비동기 이벤트 큐입니다.

#### 이벤트 타입

```python
@dataclass
class InboundMessage:
    """채널에서 에이전트로 들어오는 메시지"""
    channel: str          # "telegram", "whatsapp", "cli", "system"
    sender_id: str        # 사용자 식별자
    chat_id: str          # 채팅방 식별자
    content: str          # 메시지 텍스트
    timestamp: datetime   # 타임스탬프
    media: list[str]      # 미디어 URL 목록
    metadata: dict        # 채널별 추가 데이터

    @property
    def session_key(self) -> str:
        """세션 식별을 위한 고유 키"""
        return f"{self.channel}:{self.chat_id}"

@dataclass
class OutboundMessage:
    """에이전트에서 채널로 나가는 메시지"""
    channel: str          # 대상 채널
    chat_id: str          # 대상 채팅방
    content: str          # 응답 텍스트
    reply_to: str | None  # 답장 대상 메시지 ID
    media: list[str]      # 첨부 미디어
    metadata: dict        # 채널별 추가 데이터
```

#### 버스 구현

```python
class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._outbound_subscribers: dict[str, list[Callable]] = {}

    # 채널 → 에이전트
    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    # 에이전트 → 채널
    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self.outbound.put(msg)

    # Pub/Sub 패턴
    def subscribe_outbound(self, channel: str, callback: Callable) -> None:
        if channel not in self._outbound_subscribers:
            self._outbound_subscribers[channel] = []
        self._outbound_subscribers[channel].append(callback)

    async def dispatch_outbound(self) -> None:
        """백그라운드 태스크로 실행 - 구독자에게 메시지 디스패치"""
        while self._running:
            msg = await self.outbound.get()
            for callback in self._outbound_subscribers.get(msg.channel, []):
                await callback(msg)
```

#### 아키텍처 장점

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Telegram   │         │  Message    │         │   Agent     │
│  Channel    │────────►│    Bus      │────────►│   Loop      │
└─────────────┘         └─────────────┘         └─────────────┘
                               ▲
┌─────────────┐               │
│  WhatsApp   │───────────────┘
│  Channel    │
└─────────────┘

장점:
1. 채널과 에이전트 코어의 완전한 분리
2. 새 채널 추가 시 Agent Loop 수정 불필요
3. 비동기 처리로 높은 처리량
4. System 채널을 통한 내부 메시지 라우팅 (Subagent 결과)
```

---

### 3.3 Tool System (도구 시스템)

**위치**: `nanobot/agent/tools/`

#### 추상 Tool 클래스

```python
class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools are capabilities that the agent can use to interact with
    the environment, such as reading files, executing commands, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters."""
        pass

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

#### Tool Registry (동적 레지스트리)

```python
class ToolRegistry:
    """Registry for agent tools. Allows dynamic registration and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        try:
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
```

#### 내장 도구 목록

| 도구 | 파일 | 기능 |
|------|------|------|
| `read_file` | filesystem.py | 파일 읽기 |
| `write_file` | filesystem.py | 파일 쓰기 |
| `edit_file` | filesystem.py | 파일 편집 (append/replace) |
| `list_dir` | filesystem.py | 디렉토리 목록 |
| `exec` | shell.py | 쉘 명령 실행 (60초 타임아웃, 10KB 출력 제한) |
| `web_search` | web.py | Brave Search API 검색 |
| `web_fetch` | web.py | 웹 페이지 가져오기 및 파싱 |
| `message` | message.py | 채널에 메시지 전송 |
| `spawn` | spawn.py | 백그라운드 서브에이전트 생성 |

#### 도구 구현 예시 (Spawn Tool)

```python
class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """현재 대화 컨텍스트 설정 (결과 라우팅용)"""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs) -> str:
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
```

---

### 3.4 Memory System (메모리 시스템)

**위치**: `nanobot/agent/memory.py`

Nanobot은 **마크다운 기반의 영속적 메모리 시스템**을 사용합니다.

#### 메모리 구조

```
~/.nanobot/workspace/
├── memory/
│   ├── MEMORY.md          # 장기 메모리 (핵심 정보)
│   ├── 2026-02-02.md      # 일일 노트
│   ├── 2026-02-01.md
│   └── ...
```

#### 메모리 스토어 구현

```python
class MemoryStore:
    """
    Memory system for the agent.
    Supports daily notes and long-term memory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"

    def get_today_file(self) -> Path:
        return self.memory_dir / f"{today_date()}.md"

    def read_today(self) -> str:
        """오늘의 노트 읽기"""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""

    def read_long_term(self) -> str:
        """장기 메모리 읽기"""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def get_recent_memories(self, days: int = 7) -> str:
        """최근 N일간의 메모리 조회 (7일 롤링 윈도우)"""
        memories = []
        today = datetime.now().date()
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            if file_path.exists():
                memories.append(file_path.read_text())
        return "\n\n---\n\n".join(memories)

    def get_memory_context(self) -> str:
        """에이전트 컨텍스트용 메모리 조합"""
        parts = []

        # 장기 메모리
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)

        # 오늘의 노트
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)

        return "\n\n".join(parts) if parts else ""
```

#### 메모리 활용 패턴

```python
# ContextBuilder에서 시스템 프롬프트 생성 시:
def build_system_prompt(self) -> str:
    parts = []
    parts.append(self._get_identity())
    parts.append(self._load_bootstrap_files())

    # 메모리 컨텍스트 주입
    memory = self.memory.get_memory_context()
    if memory:
        parts.append(f"# Memory\n\n{memory}")

    # 스킬 정보
    parts.append(self.skills.build_skills_summary())

    return "\n\n---\n\n".join(parts)
```

#### 장점

1. **Human-readable**: 마크다운 형식으로 사용자가 직접 확인/편집 가능
2. **Git-friendly**: 텍스트 기반으로 버전 관리 용이
3. **Lightweight**: DB 불필요, 파일 시스템만 사용
4. **Structured**: 장기/단기 메모리 분리로 컨텍스트 효율화

---

### 3.5 Skills System (스킬 시스템)

**위치**: `nanobot/agent/skills.py`

스킬 시스템은 **Progressive Loading** 패턴을 사용하여 에이전트 기능을 확장합니다.

#### 스킬 구조

```
nanobot/skills/
├── github/
│   └── SKILL.md
├── weather/
│   └── SKILL.md
├── summarize/
│   └── SKILL.md
├── tmux/
│   └── SKILL.md
└── skill-creator/
    └── SKILL.md

# 사용자 정의 스킬
~/.nanobot/workspace/skills/
└── my-custom-skill/
    └── SKILL.md
```

#### SKILL.md 형식 (YAML Frontmatter + Markdown)

```yaml
---
name: github
description: "Interact with GitHub using the `gh` CLI."
metadata: {
  "nanobot": {
    "emoji": "🐙",
    "requires": {
      "bins": ["gh"]              # 필요한 CLI 도구
    },
    "install": [                   # 설치 가이드
      {
        "id": "brew",
        "kind": "brew",
        "formula": "gh",
        "bins": ["gh"],
        "label": "Install GitHub CLI (brew)"
      }
    ]
  }
}
---

# GitHub Skill

Use the `gh` CLI to interact with GitHub...

## Pull Requests

```bash
gh pr checks 55 --repo owner/repo
```

...
```

#### Progressive Loading 패턴

```python
class SkillsLoader:
    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def build_skills_summary(self) -> str:
        """
        스킬 요약 생성 (XML 형식)
        - 에이전트가 필요할 때 read_file로 전체 스킬 로드
        """
        all_skills = self.list_skills(filter_unavailable=False)

        lines = ["<skills>"]
        for s in all_skills:
            available = self._check_requirements(self._get_skill_meta(s["name"]))
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f'    <name>{s["name"]}</name>')
            lines.append(f'    <description>{self._get_skill_description(s["name"])}</description>')
            lines.append(f'    <location>{s["path"]}</location>')

            if not available:
                missing = self._get_missing_requirements(skill_meta)
                lines.append(f'    <requires>{missing}</requires>')

            lines.append(f'  </skill>')
        lines.append("</skills>")

        return "\n".join(lines)

    def _check_requirements(self, skill_meta: dict) -> bool:
        """스킬 요구사항 충족 여부 확인"""
        requires = skill_meta.get("requires", {})

        # 필요한 바이너리 확인
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False

        # 필요한 환경 변수 확인
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False

        return True

    def get_always_skills(self) -> list[str]:
        """always=true로 마킹된 스킬 목록 (항상 컨텍스트에 포함)"""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            if skill_meta.get("always"):
                result.append(s["name"])
        return result
```

#### 컨텍스트에서의 스킬 표현

```
# Skills

The following skills extend your capabilities. To use a skill,
read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first.

<skills>
  <skill available="true">
    <name>github</name>
    <description>Interact with GitHub using the `gh` CLI</description>
    <location>/path/to/github/SKILL.md</location>
  </skill>
  <skill available="false">
    <name>summarize</name>
    <description>Summarize URLs, files, and videos</description>
    <location>/path/to/summarize/SKILL.md</location>
    <requires>CLI: summarize</requires>
  </skill>
</skills>
```

#### 설계 이점

1. **컨텍스트 효율성**: 요약만 포함, 필요 시 전체 로드
2. **확장성**: SKILL.md만 추가하면 새 기능 추가
3. **의존성 관리**: 요구사항 자동 체크 및 가용성 표시
4. **우선순위**: workspace 스킬 > builtin 스킬

---

### 3.6 Subagent System (서브에이전트 시스템)

**위치**: `nanobot/agent/subagent.py`

복잡하거나 시간이 오래 걸리는 작업을 **백그라운드에서 비동기 처리**합니다.

#### 아키텍처

```
Main Agent                          Subagent Manager
    │                                     │
    │  spawn("Research topic X")          │
    │────────────────────────────────────►│
    │  "Started (id: abc123)"             │
    │◄────────────────────────────────────│
    │                                     │
    │  (continues conversation)           │  ┌──────────────────┐
    │                                     │  │   Subagent       │
    │                                     │──│   (asyncio task) │
    │                                     │  │   - Focused prompt│
    │                                     │  │   - Limited tools │
    │                                     │  │   - 15 iterations │
    │                                     │  └────────┬─────────┘
    │                                     │           │
    │                                     │  ◄────────┘ Complete
    │  [System Message]                   │
    │  "Subagent 'Research' completed..." │
    │◄────────────────────────────────────│
    │                                     │
    │  (Incorporates result, responds)    │
    │                                     │
```

#### 서브에이전트 매니저

```python
class SubagentManager:
    """Manages background subagent execution."""

    def __init__(self, provider, workspace, bus, model, brave_api_key):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model
        self.brave_api_key = brave_api_key
        self._running_tasks: dict[str, asyncio.Task] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + "..."

        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        # 비동기 백그라운드 태스크 생성
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task

        # 완료 시 자동 정리
        bg_task.add_done_callback(
            lambda _: self._running_tasks.pop(task_id, None)
        )

        return f"Subagent [{display_label}] started (id: {task_id})."

    async def _run_subagent(self, task_id, task, label, origin):
        # 제한된 도구 세트 (message, spawn 제외)
        tools = ToolRegistry()
        tools.register(ReadFileTool())
        tools.register(WriteFileTool())
        tools.register(ListDirTool())
        tools.register(ExecTool(working_dir=str(self.workspace)))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())

        # 집중된 시스템 프롬프트
        system_prompt = self._build_subagent_prompt(task)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # 에이전트 루프 (최대 15회)
        max_iterations = 15
        iteration = 0
        final_result = None

        while iteration < max_iterations:
            iteration += 1
            response = await self.provider.chat(messages, tools.get_definitions(), self.model)

            if response.has_tool_calls:
                # 도구 실행 및 결과 추가
                ...
            else:
                final_result = response.content
                break

        # 결과 발표
        await self._announce_result(task_id, label, task, final_result, origin, "ok")

    async def _announce_result(self, task_id, label, task, result, origin, status):
        """System 채널을 통해 Main Agent에게 결과 전달"""
        announce_content = f"""[Subagent '{label}' completed]

Task: {task}

Result:
{result}

Summarize this naturally for the user."""

        msg = InboundMessage(
            channel="system",                              # System 채널
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",  # 원래 대화 참조
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)  # 메시지 버스로 주입
```

#### 서브에이전트 시스템 프롬프트

```python
def _build_subagent_prompt(self, task: str) -> str:
    return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages

## What You Cannot Do
- Send messages directly to users (no message tool)
- Spawn other subagents (no recursion)
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}

When completed, provide a clear summary of your findings."""
```

#### 설계 특징

| 특징 | 설명 |
|------|------|
| **Isolation** | 메인 에이전트와 독립된 컨텍스트 |
| **Limited Tools** | message, spawn 제외로 부작용 방지 |
| **Reduced Iterations** | 20 → 15로 제한하여 비용 절감 |
| **Async Execution** | asyncio.Task로 non-blocking 처리 |
| **Result Routing** | System 채널을 통한 결과 전달 |

---

## 4. 시스템 연동 패턴

### 4.1 Multi-Channel Integration

**위치**: `nanobot/channels/`

#### 채널 추상화

```python
class BaseChannel(ABC):
    """Base class for chat channels."""

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.is_running = False

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel."""
        pass

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """Send a message through this channel."""
        pass
```

#### 채널 매니저

```python
class ChannelManager:
    """Manages chat channels and coordinates message routing."""

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._init_channels()

    def _init_channels(self) -> None:
        # 설정에 따라 채널 동적 초기화
        if self.config.channels.telegram.enabled:
            from nanobot.channels.telegram import TelegramChannel
            self.channels["telegram"] = TelegramChannel(
                self.config.channels.telegram, self.bus
            )

        if self.config.channels.whatsapp.enabled:
            from nanobot.channels.whatsapp import WhatsAppChannel
            self.channels["whatsapp"] = WhatsAppChannel(
                self.config.channels.whatsapp, self.bus
            )

    async def start_all(self) -> None:
        # Outbound 디스패처 시작
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # 모든 채널 시작
        tasks = [asyncio.create_task(ch.start()) for ch in self.channels.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_outbound(self) -> None:
        """Outbound 메시지를 적절한 채널로 라우팅"""
        while True:
            msg = await self.bus.consume_outbound()
            channel = self.channels.get(msg.channel)
            if channel:
                await channel.send(msg)
```

#### 지원 채널

| 채널 | 난이도 | 설정 |
|------|--------|------|
| CLI | 기본 | 없음 |
| Telegram | 쉬움 | Bot token |
| WhatsApp | 중간 | QR 코드 스캔, Node.js 브릿지 |

---

### 4.2 Scheduling System

#### Cron Service

**위치**: `nanobot/cron/service.py`

```python
class CronService:
    """Service for managing and executing scheduled jobs."""

    def __init__(self, store_path: Path, on_job: Callable | None = None):
        self.store_path = store_path
        self.on_job = on_job  # 작업 실행 콜백
        self._store: CronStore | None = None
        self._timer_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._arm_timer()

    def _arm_timer(self) -> None:
        """다음 작업 시간에 타이머 설정"""
        next_wake = self._get_next_wake_ms()
        if not next_wake:
            return

        delay_s = max(0, next_wake - _now_ms()) / 1000

        async def tick():
            await asyncio.sleep(delay_s)
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        """타이머 틱 - 예정된 작업 실행"""
        now = _now_ms()
        due_jobs = [j for j in self._store.jobs
                    if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms]

        for job in due_jobs:
            await self._execute_job(job)

        self._save_store()
        self._arm_timer()  # 다음 타이머 설정
```

#### 스케줄 타입

```python
@dataclass
class CronSchedule:
    kind: str              # "at" | "every" | "cron"
    at_ms: int | None      # 일회성 실행 시간 (Unix timestamp ms)
    every_ms: int | None   # 주기 (밀리초)
    expr: str | None       # Cron 표현식 ("0 9 * * *")
    tz: str | None         # 타임존
```

#### Heartbeat Service

**위치**: `nanobot/heartbeat/service.py`

```python
class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.
    Agent reads HEARTBEAT.md and executes any listed tasks.
    """

    def __init__(self, workspace, on_heartbeat, interval_s=1800, enabled=True):
        self.workspace = workspace
        self.on_heartbeat = on_heartbeat
        self.interval_s = interval_s  # 기본 30분
        self.enabled = enabled

    async def _run_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.interval_s)
            if self._running:
                await self._tick()

    async def _tick(self) -> None:
        content = self._read_heartbeat_file()

        # HEARTBEAT.md가 비어있으면 스킵
        if _is_heartbeat_empty(content):
            return

        # 에이전트에게 작업 확인 요청
        response = await self.on_heartbeat(HEARTBEAT_PROMPT)

        if "HEARTBEAT_OK" in response.upper():
            logger.info("Heartbeat: OK (no action needed)")
        else:
            logger.info("Heartbeat: completed task")
```

#### Heartbeat 프롬프트

```python
HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
If nothing needs attention, reply with just: HEARTBEAT_OK"""
```

---

### 4.3 LLM Provider Abstraction

**위치**: `nanobot/providers/`

#### 추상 인터페이스

```python
@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        pass
```

#### LiteLLM Provider

```python
class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    Supports OpenRouter, Anthropic, OpenAI, Zhipu, vLLM.
    """

    def __init__(self, api_key, api_base, default_model="anthropic/claude-opus-4-5"):
        super().__init__(api_key, api_base)
        self.default_model = default_model

        # Provider 자동 감지
        self.is_openrouter = (
            (api_key and api_key.startswith("sk-or-")) or
            (api_base and "openrouter" in api_base)
        )
        self.is_vllm = bool(api_base) and not self.is_openrouter

        # API 키 설정
        if api_key:
            if self.is_openrouter:
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif self.is_vllm:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in default_model:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            # ... 기타 프로바이더

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        model = model or self.default_model

        # 모델명 프리픽스 처리
        if self.is_openrouter and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"
        elif self.is_vllm:
            model = f"hosted_vllm/{model}"

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}", finish_reason="error")
```

#### Provider 우선순위

```python
# Config.get_api_key()
def get_api_key(self) -> str | None:
    """API 키 우선순위: OpenRouter > Anthropic > OpenAI > Zhipu > vLLM"""
    return (
        self.providers.openrouter.api_key or
        self.providers.anthropic.api_key or
        self.providers.openai.api_key or
        self.providers.zhipu.api_key or
        self.providers.vllm.api_key or
        None
    )
```

---

## 5. 설계 패턴 및 모범 사례

### 5.1 핵심 설계 패턴

| 패턴 | 적용 위치 | 설명 |
|------|----------|------|
| **ReAct Loop** | AgentLoop | Reasoning + Acting 반복 |
| **Message Bus** | MessageBus | Producer-Consumer 비동기 큐 |
| **Registry** | ToolRegistry | 동적 도구 등록/실행 |
| **Strategy** | LLMProvider | LLM 제공자 추상화 |
| **Template Method** | BaseChannel | 채널 공통 인터페이스 |
| **Observer** | MessageBus.subscribe_outbound | Pub/Sub |
| **Factory** | ChannelManager._init_channels | 조건부 객체 생성 |

### 5.2 비동기 패턴

```python
# 1. 동시 실행 (asyncio.gather)
await asyncio.gather(
    agent.run(),
    channels.start_all(),
    return_exceptions=True
)

# 2. 타임아웃 처리
try:
    msg = await asyncio.wait_for(
        self.bus.consume_inbound(),
        timeout=1.0
    )
except asyncio.TimeoutError:
    continue

# 3. 백그라운드 태스크
bg_task = asyncio.create_task(self._run_subagent(...))
bg_task.add_done_callback(lambda _: cleanup())

# 4. 취소 처리
try:
    await self._timer_task
except asyncio.CancelledError:
    pass
```

### 5.3 에러 처리 패턴

```python
# Tool 실행 시 에러 -> 문자열 결과로 반환
async def execute(self, name: str, params: dict) -> str:
    tool = self._tools.get(name)
    if not tool:
        return f"Error: Tool '{name}' not found"
    try:
        return await tool.execute(**params)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"

# LLM 호출 시 에러 -> LLMResponse로 래핑
try:
    response = await acompletion(**kwargs)
    return self._parse_response(response)
except Exception as e:
    return LLMResponse(
        content=f"Error calling LLM: {str(e)}",
        finish_reason="error",
    )
```

### 5.4 설정 관리

```python
class Config(BaseSettings):
    """Root configuration using Pydantic."""

    agents: AgentsConfig
    channels: ChannelsConfig
    providers: ProvidersConfig
    gateway: GatewayConfig
    tools: ToolsConfig

    class Config:
        env_prefix = "NANOBOT_"           # 환경 변수 프리픽스
        env_nested_delimiter = "__"        # 중첩 구분자

# 사용 예: NANOBOT_AGENTS__DEFAULTS__MODEL=gpt-4
```

---

## 6. 벤치마킹 핵심 인사이트

### 6.1 에이전트 시스템 개발 시 참고할 패턴

#### 1) Agent Loop 설계

```
핵심 원칙:
- Max iterations로 무한 루프 방지
- Tool call → 결과 → 다시 LLM 호출 패턴
- 최종 응답은 tool_calls가 없을 때
- 세션 저장은 루프 완료 후
```

#### 2) Tool System 설계

```
핵심 원칙:
- 추상 Tool 클래스로 표준화
- JSON Schema 기반 파라미터 정의
- Registry로 동적 등록/조회
- 에러를 문자열로 반환 (LLM이 처리 가능하도록)
```

#### 3) 메시지 라우팅

```
핵심 원칙:
- Message Bus로 채널과 에이전트 분리
- System 채널로 내부 통신 (Subagent 결과)
- session_key = "{channel}:{chat_id}"로 대화 분리
```

#### 4) 메모리 시스템

```
핵심 원칙:
- 장기 메모리 (MEMORY.md) vs 단기 메모리 (일일 노트)
- 마크다운 = Human-readable + Git-friendly
- 7일 롤링 윈도우로 컨텍스트 제한
```

#### 5) Progressive Loading (Skills)

```
핵심 원칙:
- 요약만 시스템 프롬프트에 포함
- 필요 시 read_file로 전체 로드
- 요구사항 자동 체크 (bins, env)
- 우선순위: user > builtin
```

### 6.2 확장성 포인트

| 확장 포인트 | 방법 |
|------------|------|
| 새 도구 추가 | `Tool` 상속 → `ToolRegistry.register()` |
| 새 스킬 추가 | `workspace/skills/my-skill/SKILL.md` 생성 |
| 새 채널 추가 | `BaseChannel` 상속 → `ChannelManager` 등록 |
| 새 LLM 프로바이더 | `LLMProvider` 상속 (또는 LiteLLM 활용) |
| 커스텀 메모리 | `MemoryStore` 확장 또는 대체 |

### 6.3 성능/비용 최적화

| 최적화 | 구현 |
|--------|------|
| Context 효율화 | Progressive skill loading |
| 토큰 제한 | max_iterations, max_tokens |
| 세션 캐싱 | SessionManager._cache |
| 비동기 처리 | asyncio 전면 채택 |
| 경량화 | 4,000 LOC, 최소 의존성 |

---

## 7. 구현 시 참고 사항

### 7.1 자체 에이전트 구현 체크리스트

```
[ ] Agent Loop
    [ ] ReAct 패턴 (추론 → 행동 → 관찰 → 반복)
    [ ] Max iterations 제한
    [ ] Tool execution 에러 처리
    [ ] 세션/컨텍스트 관리

[ ] Tool System
    [ ] 추상 Tool 인터페이스
    [ ] JSON Schema 파라미터
    [ ] 동적 Registry
    [ ] OpenAI function calling 호환

[ ] Context Builder
    [ ] System prompt 조립
    [ ] 메모리 통합
    [ ] 대화 이력 관리
    [ ] Progressive loading

[ ] Communication
    [ ] Message Bus (async queue)
    [ ] Multi-channel 지원
    [ ] Internal routing (System channel)

[ ] Persistence
    [ ] Session storage (JSONL)
    [ ] Memory files (Markdown)
    [ ] Configuration (JSON + env vars)

[ ] Background Processing
    [ ] Subagent manager
    [ ] Cron scheduler
    [ ] Heartbeat service

[ ] LLM Integration
    [ ] Provider abstraction
    [ ] Multi-provider support
    [ ] Error handling
```

### 7.2 주의사항

1. **무한 루프 방지**: 반드시 max_iterations 설정
2. **비용 제어**: 토큰 제한, subagent iteration 제한
3. **에러 처리**: LLM이 이해할 수 있는 문자열로 반환
4. **컨텍스트 관리**: 불필요한 정보 제외, progressive loading
5. **보안**: allow_from으로 허용 사용자 제한

### 7.3 개선 가능한 영역

| 영역 | 현재 | 개선 방향 |
|------|------|----------|
| 메모리 | 파일 기반 | 벡터 DB 통합 (RAG) |
| 멀티모달 | 텍스트만 | 이미지/음성 처리 |
| 관찰성 | loguru | 분산 트레이싱 |
| 테스트 | 기본적 | 통합 테스트 강화 |
| 문서화 | README | API 문서화 |

---

## 참고 자료

- **GitHub**: https://github.com/HKUDS/nanobot
- **영감**: [OpenClaw/Clawdbot](https://github.com/openclaw/openclaw)
- **LiteLLM 문서**: https://docs.litellm.ai/
- **ReAct 논문**: [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

---

*이 문서는 에이전트 시스템 개발을 위한 기술적 참고 자료로 작성되었습니다.*
