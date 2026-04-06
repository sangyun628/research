# OpenCode 아키텍처 분석 — Python 보고서 AI 에디터 구현 가이드

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [에이전트 루프](#2-에이전트-루프)
3. [세션 및 대화 관리](#3-세션-및-대화-관리)
4. [메시지 데이터 구조](#4-메시지-데이터-구조)
5. [대용량 파일 읽기 전략](#5-대용량-파일-읽기-전략)
6. [도구(Tool) 시스템](#6-도구tool-시스템)
7. [도구 출력 Truncation](#7-도구-출력-truncation)
8. [컨텍스트 윈도우 관리 — Pruning & Compaction](#8-컨텍스트-윈도우-관리--pruning--compaction)
9. [토큰 카운팅 및 비용 추적](#9-토큰-카운팅-및-비용-추적)
10. [Provider 추상화 (다중 LLM 지원)](#10-provider-추상화-다중-llm-지원)
11. [스트리밍 응답 처리](#11-스트리밍-응답-처리)
12. [저장소(Storage) 시스템](#12-저장소storage-시스템)
13. [권한(Permission) 시스템](#13-권한permission-시스템)
14. [Python 구현 설계안](#14-python-구현-설계안)

---

## 1. 전체 아키텍처 개요

OpenCode는 터미널 기반 AI 코딩 어시스턴트로, 아래 핵심 컴포넌트로 구성된다:

```
┌──────────────────────────────────────────────────────┐
│                    CLI / TUI 진입점                    │
└────────┬─────────────────────────────────────────────┘
         │
    ┌────▼────┐     ┌────────────┐     ┌────────────┐
    │ Session │────▶│  Agent     │────▶│  LLM       │
    │ Manager │     │  Loop      │     │  Provider   │
    └────┬────┘     └────┬───────┘     └────────────┘
         │               │
    ┌────▼────┐     ┌────▼───────┐
    │ Storage │     │  Tool      │
    │ (SQLite)│     │  Registry  │
    └─────────┘     └────────────┘
```

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **스트리밍 우선** | 모든 LLM 응답은 스트리밍으로 처리, 실시간 토큰/비용 추적 |
| **지연 로딩** | 대화 기록은 커서 기반 페이지네이션으로 50개씩 로드 |
| **자동 컨텍스트 관리** | Pruning(도구 출력 제거) + Compaction(대화 요약)으로 컨텍스트 오버플로 방지 |
| **도구 출력 제한** | 모든 도구 출력은 2000줄/50KB로 제한, 초과분은 디스크 저장 |
| **이벤트 기반** | 세션/메시지 변경은 이벤트로 전파, UI 실시간 갱신 |

---

## 2. 에이전트 루프

### 2.1 에이전트 정의

OpenCode는 용도별로 분리된 에이전트를 사용한다:

```python
# Python 구현 예시
@dataclass
class AgentInfo:
    name: str                    # "build", "plan", "explore", "compaction" 등
    description: str
    mode: str                    # "primary" | "subagent" | "all"
    hidden: bool = False         # compaction, title, summary 에이전트는 숨김
    temperature: float = 0.0
    top_p: float | None = None
    permission: PermissionRuleset = field(default_factory=list)
    model: ModelRef | None = None
    prompt: str | None = None    # 커스텀 시스템 프롬프트
    steps: int | None = None     # 최대 실행 스텝 수
```

내장 에이전트:
- **build**: 기본 에이전트. 모든 도구 사용 가능
- **plan**: 계획 모드. 편집 도구 비활성화
- **general**: 범용 서브에이전트. 병렬 작업 실행
- **explore**: 탐색 전용. 읽기 전용 도구만 사용
- **compaction**: (숨김) 대화 요약 생성 전용
- **title**: (숨김) 세션 제목 자동 생성
- **summary**: (숨김) 세션 요약 생성

### 2.2 에이전트 루프 흐름

```
사용자 입력 → 메시지 생성 → LLM 스트리밍 호출 → 이벤트 처리 루프:
  ├─ text-delta      → 텍스트 파트 갱신
  ├─ reasoning-delta → 추론 파트 갱신 (Claude)
  ├─ tool-call       → 도구 실행 → 결과 반환
  ├─ finish-step     → 토큰/비용 집계
  └─ error           → 오류 처리

도구 결과 반환 후 → LLM 재호출 (도구 결과 포함)
                  → 반복 (도구 호출 없을 때까지)

완료 후 → 오버플로 확인 → 필요시 Pruning/Compaction
```

### 2.3 둠 루프(Doom Loop) 방지

동일한 도구를 동일한 인자로 3회 연속 호출하면 사용자에게 확인을 요청한다:

```python
DOOM_LOOP_THRESHOLD = 3

def check_doom_loop(recent_parts: list[ToolPart], new_call: ToolCall) -> bool:
    """최근 N개 파트가 모두 같은 도구+같은 입력이면 True"""
    if len(recent_parts) < DOOM_LOOP_THRESHOLD:
        return False
    return all(
        p.tool == new_call.tool_name and
        json.dumps(p.input, sort_keys=True) == json.dumps(new_call.input, sort_keys=True)
        for p in recent_parts[-DOOM_LOOP_THRESHOLD:]
    )
```

---

## 3. 세션 및 대화 관리

### 3.1 세션 구조

```python
@dataclass
class SessionInfo:
    id: str                        # 내림차순 ID (최신 세션이 먼저)
    slug: str                      # URL용 슬러그
    project_id: str
    parent_id: str | None = None   # 포크된 세션의 부모
    title: str = ""
    directory: str = ""            # 작업 디렉토리
    version: str = ""
    time: SessionTime = field(default_factory=SessionTime)
    summary: SessionSummary | None = None  # git diff 요약
    share: ShareInfo | None = None
    revert: RevertInfo | None = None
    permission: list[PermissionRule] | None = None

@dataclass
class SessionTime:
    created: float
    updated: float
    compacting: float | None = None   # 컴팩션 진행 중 타임스탬프
    archived: float | None = None
```

### 3.2 세션 연산

| 연산 | 설명 |
|------|------|
| `create()` | 새 세션 생성, 내림차순 ID 부여 |
| `fork(session_id, message_id?)` | 특정 시점에서 분기한 자식 세션 생성 |
| `touch(session_id)` | `time.updated` 갱신 |
| `messages(session_id, limit?)` | 커서 기반 페이지네이션으로 메시지 로드 |
| `share(session_id)` | 공유 URL 생성 |

### 3.3 메시지 로딩 — 커서 기반 페이지네이션

대화 기록은 한번에 전부 로드하지 않고, 50개씩 커서 기반으로 스트리밍한다:

```python
async def stream_messages(session_id: str, limit: int | None = None):
    """커서 기반 페이지네이션으로 메시지 로드"""
    BATCH_SIZE = 50
    cursor = None
    count = 0

    while True:
        query = (
            select(MessageTable)
            .where(MessageTable.session_id == session_id)
            .order_by(MessageTable.time_created.asc(), MessageTable.id.asc())
            .limit(BATCH_SIZE)
        )
        if cursor:
            query = query.where(
                or_(
                    MessageTable.time_created > cursor.time_created,
                    and_(
                        MessageTable.time_created == cursor.time_created,
                        MessageTable.id > cursor.id
                    )
                )
            )

        rows = await db.execute(query)
        if not rows:
            break

        for row in rows:
            parts = await load_parts(row.id)
            yield MessageWithParts(info=row.to_info(), parts=parts)
            count += 1
            if limit and count >= limit:
                return

        cursor = rows[-1]
        if len(rows) < BATCH_SIZE:
            break
```

---

## 4. 메시지 데이터 구조

### 4.1 메시지 타입

```python
@dataclass
class UserMessage:
    id: str
    session_id: str
    role: str = "user"           # 항상 "user"
    time: MessageTime
    agent: str = "build"
    model: ModelRef | None = None
    format: OutputFormat | None = None   # text | json_schema
    system: str | None = None    # 커스텀 시스템 프롬프트

@dataclass
class AssistantMessage:
    id: str
    session_id: str
    role: str = "assistant"      # 항상 "assistant"
    time: MessageTime
    parent_id: str               # 원본 사용자 메시지 ID
    model_id: str
    provider_id: str
    agent: str = "build"
    summary: bool = False        # 컴팩션 요약 메시지 여부
    cost: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    finish: str | None = None    # 완료 사유
    error: ErrorInfo | None = None
```

### 4.2 파트(Part) 타입

메시지는 여러 파트로 구성된다:

```python
# 텍스트 파트
@dataclass
class TextPart:
    type: str = "text"
    text: str = ""
    synthetic: bool = False      # AI가 생성한 플레이스홀더
    ignored: bool = False        # 모델 입력에서 제외
    time: PartTime | None = None
    metadata: dict | None = None  # 프로바이더별 메타데이터

# 도구 호출 파트
@dataclass
class ToolPart:
    type: str = "tool"
    call_id: str                 # 고유 도구 호출 ID
    tool: str                    # 도구 이름
    state: ToolState             # pending | running | completed | error

@dataclass
class ToolStateCompleted:
    status: str = "completed"
    input: dict                  # 도구 인자
    output: str                  # 도구 출력
    title: str = ""
    metadata: dict = field(default_factory=dict)
    time: ToolTime               # start, end, compacted(pruning 시 설정)
    attachments: list = field(default_factory=list)  # 이미지/PDF 등

# 추론(Reasoning) 파트 — Claude의 Extended Thinking
@dataclass
class ReasoningPart:
    type: str = "reasoning"
    text: str = ""
    time: PartTime | None = None

# 컴팩션 파트 — 대화 요약을 포함
@dataclass
class CompactionPart:
    type: str = "compaction"
    auto: bool = False           # 자동 트리거 여부
    overflow: bool | None = None # 미디어 오버플로 트리거

# 파일 파트 — 이미지, PDF 등 첨부
@dataclass
class FilePart:
    type: str = "file"
    mime: str                    # "image/png", "application/pdf" 등
    filename: str | None = None
    url: str                     # data: URL (base64 인코딩)
```

### 4.3 모델 메시지 변환

내부 `MessageV2` → LLM API 호출용 `ModelMessage` 변환 시 핵심 로직:

```python
async def to_model_messages(
    messages: list[MessageWithParts],
    model: Model,
    strip_media: bool = False
) -> list[dict]:
    """내부 메시지를 LLM API 형식으로 변환"""
    result = []

    for msg in messages:
        if msg.info.role == "user":
            parts = []
            for part in msg.parts:
                if part.type == "text" and not part.ignored:
                    parts.append({"type": "text", "text": part.text})
                elif part.type == "file":
                    if strip_media:
                        parts.append({"type": "text",
                                      "text": f"[Attached {part.mime}: {part.filename}]"})
                    else:
                        parts.append({"type": "image_url", "url": part.url})
                elif part.type == "compaction":
                    parts.append({"type": "text",
                                  "text": "What did we do so far?"})
            result.append({"role": "user", "content": parts})

        elif msg.info.role == "assistant":
            parts = []
            for part in msg.parts:
                if part.type == "text":
                    parts.append({"type": "text", "text": part.text})
                elif part.type == "tool" and part.state.status == "completed":
                    # ★ 핵심: Pruning된 도구 출력은 플레이스홀더로 교체
                    output = (
                        "[Old tool result content cleared]"
                        if part.state.time.compacted
                        else part.state.output
                    )
                    parts.append({
                        "type": "tool_use",
                        "id": part.call_id,
                        "name": part.tool,
                        "input": part.state.input,
                    })
                    # tool_result는 별도 메시지로
                    result.append({"role": "assistant", "content": parts})
                    result.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": part.call_id,
                            "content": output,
                        }]
                    })
                    parts = []
                    continue
            if parts:
                result.append({"role": "assistant", "content": parts})

    return result
```

---

## 5. 대용량 파일 읽기 전략

OpenCode는 대용량 파일을 **절대 한번에 전부 읽지 않는다**. 핵심 전략:

### 5.1 Read 도구 — 라인 기반 청킹 + 바이트 제한

```python
DEFAULT_READ_LIMIT = 2000      # 기본 최대 줄 수
MAX_LINE_LENGTH = 2000         # 줄당 최대 문자 수
MAX_BYTES = 50 * 1024          # 50KB 바이트 상한
MAX_LINE_SUFFIX = "... (truncated)"

async def read_file(
    file_path: str,
    offset: int = 1,           # 1-indexed 시작 줄
    limit: int = DEFAULT_READ_LIMIT,
) -> ReadResult:
    """파일을 라인 단위로 스트리밍하며 제한에 맞춰 반환"""
    lines = []
    total_bytes = 0
    line_count = 0
    has_more = False
    truncated_by_bytes = False
    start = offset - 1         # 0-indexed로 변환

    async with aiofiles.open(file_path, 'r') as f:
        async for raw_line in f:
            line_count += 1

            # offset 이전 줄은 건너뜀
            if line_count <= start:
                continue

            # limit 초과 시 중단
            if len(lines) >= limit:
                has_more = True
                continue

            # 긴 줄 자르기
            line = raw_line.rstrip('\n')
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + MAX_LINE_SUFFIX

            # 바이트 제한 확인
            line_bytes = len(line.encode('utf-8')) + (1 if lines else 0)
            if total_bytes + line_bytes > MAX_BYTES:
                truncated_by_bytes = True
                has_more = True
                break

            lines.append(line)
            total_bytes += line_bytes

    # 줄 번호 포함 출력 (cat -n 형식)
    numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(lines)]
    output = "\n".join(numbered)

    # 더 읽을 내용이 있으면 안내 메시지 추가
    if has_more:
        next_offset = start + len(lines) + 1
        hint = (
            f"\n\n... More content available. "
            f"Use offset={next_offset} to continue reading."
        )
        output += hint

    return ReadResult(
        output=output,
        total_lines=line_count,
        lines_read=len(lines),
        has_more=has_more,
        truncated_by_bytes=truncated_by_bytes,
    )
```

### 5.2 다른 도구의 출력 제한

| 도구 | 줄 제한 | 바이트 제한 | 항목 제한 |
|------|---------|-------------|-----------|
| **read** | 2000줄 | 50KB | - |
| **grep** | - | - | 100개 매치 |
| **glob** | - | - | 100개 파일 |
| **bash** | 2000줄 | 50KB | - |
| **webfetch** | - | 5MB | - |

### 5.3 보고서 에디터에서의 적용

보고서 편집기에서는 마크다운 파일이 주 대상이므로:

```python
# 보고서 전용 Read 설정
REPORT_READ_LIMIT = 500       # 보고서는 500줄 단위로
REPORT_MAX_BYTES = 30 * 1024  # 30KB
SECTION_DELIMITER = r'^#{1,3}\s'  # 마크다운 헤딩 기준 섹션 분할

async def read_report_section(
    file_path: str,
    section: str | None = None,  # 특정 섹션만 읽기
    offset: int = 1,
    limit: int = REPORT_READ_LIMIT,
) -> ReadResult:
    """마크다운 보고서를 섹션 단위로 읽기"""
    if section:
        # 헤딩 기준으로 해당 섹션만 추출
        return await read_section_by_heading(file_path, section)
    return await read_file(file_path, offset, limit)
```

---

## 6. 도구(Tool) 시스템

### 6.1 도구 정의 구조

```python
from pydantic import BaseModel
from typing import Any, Callable, Awaitable

class ToolDef:
    """도구 정의"""
    id: str                      # "read", "edit", "grep" 등
    description: str             # LLM에 보여줄 설명
    parameters: type[BaseModel]  # Pydantic 모델로 파라미터 스키마 정의
    execute: Callable[[Any, ToolContext], Awaitable[ToolResult]]

class ToolContext:
    """도구 실행 컨텍스트"""
    session_id: str
    message_id: str
    agent: str
    abort_signal: asyncio.Event
    messages: list[MessageWithParts]

    async def ask_permission(self, request: PermissionRequest) -> None:
        """사용자 권한 확인 (블로킹)"""
        ...

    def update_metadata(self, title: str = "", metadata: dict = None):
        """실행 중 메타데이터 갱신"""
        ...

class ToolResult:
    """도구 실행 결과"""
    title: str                   # 결과 제목
    output: str                  # 텍스트 출력
    metadata: dict               # 구조화된 메타데이터
    attachments: list[FilePart] = []  # 파일 첨부 (이미지/PDF)
```

### 6.2 보고서 에디터용 도구 목록

```python
TOOLS = {
    # --- 기본 도구 ---
    "read": ReadTool,           # 파일 읽기 — offset/limit 페이지네이션
    "write": WriteTool,         # 파일 쓰기 — 전체 덮어쓰기
    "edit": EditTool,           # 파일 편집 — old_string → new_string 치환
    "glob": GlobTool,           # 파일 검색 — glob 패턴 매칭
    "grep": GrepTool,           # 콘텐츠 검색 — 정규식 기반
    "bash": BashTool,           # 셸 명령 실행
    "question": QuestionTool,   # 사용자 질문
    "task": TaskTool,           # 서브에이전트 생성
    "webfetch": WebFetchTool,   # 웹 페이지 가져오기

    # --- 보고서 전용 도구 ---
    "read_section": ReadSectionTool,     # 마크다운 섹션 단위 읽기
    "extract_toc": ExtractTocTool,       # 보고서 목차(TOC) 추출
    "validate_report": ValidateReportTool,  # 마크다운 검증
    "insert_table": InsertTableTool,     # 표/차트 데이터 삽입
}
```

### 6.3 도구 등록 및 필터링

```python
class ToolRegistry:
    """도구 레지스트리"""
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self._tools[tool.id] = tool

    def get_tools(self, model: Model, agent: AgentInfo) -> dict[str, ToolDef]:
        """에이전트와 모델에 따라 사용 가능한 도구 필터링"""
        available = {}
        for id, tool in self._tools.items():
            if not agent.can_use(id):
                continue
            if not model.supports_tool(id):
                continue
            available[id] = tool
        return available
```

### 6.4 도구 실행 파이프라인

```python
async def execute_tool(
    tool_def: ToolDef,
    args: dict,
    ctx: ToolContext,
) -> ToolResult:
    """도구 실행 파이프라인"""
    # 1. 파라미터 검증 (Pydantic)
    try:
        validated = tool_def.parameters.model_validate(args)
    except ValidationError as e:
        return ToolResult(
            title="Error",
            output=f"Tool '{tool_def.id}' called with invalid arguments: {e}",
            metadata={"error": True},
        )

    # 2. 도구 실행
    result = await tool_def.execute(validated, ctx)

    # 3. 자동 Truncation (도구가 자체 처리하지 않은 경우)
    if result.metadata.get("truncated") is None:
        result = await truncate_output(result)

    return result
```

---

## 7. 도구 출력 Truncation

### 7.1 Truncation 정책

모든 도구 출력은 자동으로 크기 제한이 적용된다:

```python
MAX_TRUNCATE_LINES = 2000
MAX_TRUNCATE_BYTES = 50 * 1024   # 50KB
TRUNCATION_DIR = "/tmp/opencode-truncated"
RETENTION_DAYS = 7

async def truncate_output(
    result: ToolResult,
    max_lines: int = MAX_TRUNCATE_LINES,
    max_bytes: int = MAX_TRUNCATE_BYTES,
    direction: str = "head",     # "head" 또는 "tail"
) -> ToolResult:
    """도구 출력이 제한을 초과하면 디스크에 저장하고 미리보기 반환"""
    text = result.output
    lines = text.split('\n')
    total_bytes = len(text.encode('utf-8'))

    # 제한 이내면 그대로 반환
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        result.metadata["truncated"] = False
        return result

    # 전체 출력을 디스크에 저장
    output_path = os.path.join(TRUNCATION_DIR, generate_id())
    async with aiofiles.open(output_path, 'w') as f:
        await f.write(text)

    # 미리보기 생성
    if direction == "head":
        preview_lines = lines[:max_lines]
        removed = len(lines) - max_lines
        preview = '\n'.join(preview_lines)
        hint = f"Use Read tool with offset to see more, or Grep to search."
        output = f"{preview}\n\n...{removed} lines truncated...\n\n{hint}"
    else:
        preview_lines = lines[-max_lines:]
        removed = len(lines) - max_lines
        preview = '\n'.join(preview_lines)
        output = f"...{removed} lines truncated...\n\n{preview}"

    result.output = output
    result.metadata["truncated"] = True
    result.metadata["output_path"] = output_path
    return result
```

### 7.2 Truncation 흐름

```
도구 실행 완료
    │
    ▼
도구가 자체적으로 truncated 메타데이터를 설정했는가?
    ├─ Yes → 그대로 반환 (read, grep 등은 자체 제한 적용)
    └─ No  → truncate_output() 자동 적용
              │
              ├─ 제한 이내 → metadata.truncated = False
              └─ 초과      → 디스크 저장 + 미리보기 반환
                             metadata.truncated = True
                             metadata.output_path = "/tmp/..."
```

---

## 8. 컨텍스트 윈도우 관리 — Pruning & Compaction

이것이 OpenCode의 **가장 핵심적인 기능**이다. 긴 대화에서 컨텍스트 윈도우 초과를 방지한다.

### 8.1 전체 흐름

```
LLM 응답 완료
    │
    ▼
isOverflow() 확인 ──────────────── 정상 → 계속
    │ (오버플로)
    ▼
① Pruning 시도 (빠르고 가벼움)
    │
    ▼
충분한 공간 확보되었는가?
    ├─ Yes → 계속
    └─ No  → ② Compaction 실행 (LLM 호출로 요약 생성)
              │
              ▼
           요약 메시지 생성 + 기존 도구 출력 정리
              │
              ▼
           마지막 사용자 메시지 재전송 (자동)
```

### 8.2 오버플로 감지

```python
COMPACTION_BUFFER = 20_000     # 기본 안전 버퍼 (토큰)
OUTPUT_TOKEN_MAX = 32_000      # 기본 최대 출력 토큰

def is_overflow(
    tokens: TokenUsage,
    model: Model,
    config: CompactionConfig,
) -> bool:
    """컨텍스트 윈도우 오버플로 여부 판단"""
    if config.auto is False:
        return False
    if model.limit.context <= 0:
        return False

    # 총 사용 토큰 계산
    total = tokens.total or (
        tokens.input + tokens.output +
        tokens.cache_read + tokens.cache_write
    )

    # 사용 가능한 컨텍스트 계산
    max_output = min(model.limit.output, OUTPUT_TOKEN_MAX)
    reserved = config.reserved or min(COMPACTION_BUFFER, max_output)

    if model.limit.input:
        usable = model.limit.input - reserved
    else:
        usable = model.limit.context - max_output

    return total >= usable
```

### 8.3 Pruning — 도구 출력 선택적 제거

Pruning은 **LLM 호출 없이** 오래된 도구 출력을 제거하여 토큰을 확보한다.

```python
PRUNE_MINIMUM = 20_000         # 최소 이 정도 확보해야 실행
PRUNE_PROTECT = 40_000         # 최근 이만큼의 도구 출력은 보호
PRUNE_PROTECTED_TOOLS = {"skill"}  # 절대 제거하지 않는 도구

async def prune(
    messages: list[MessageWithParts],
    session: SessionService,
) -> int:
    """오래된 도구 출력을 제거하여 토큰 확보. 반환: 확보된 토큰 수"""
    protected_tokens = 0
    prunable: list[tuple[ToolPart, int]] = []  # (파트, 추정 토큰)
    user_turn_count = 0

    # 역순으로 순회
    for msg in reversed(messages):
        if msg.info.role == "user":
            user_turn_count += 1
            if user_turn_count < 2:
                continue  # 최근 2턴은 건드리지 않음

        if msg.info.summary:
            continue  # 요약 메시지는 건드리지 않음

        for part in msg.parts:
            if part.type != "tool" or part.state.status != "completed":
                continue
            if part.tool in PRUNE_PROTECTED_TOOLS:
                continue
            if part.state.time.compacted:
                continue  # 이미 정리됨

            tokens = estimate_tokens(part.state.output)

            if protected_tokens < PRUNE_PROTECT:
                protected_tokens += tokens
                continue  # 최근 것은 보호

            prunable.append((part, tokens))

    # 충분한 양이 모이면 실행
    total_prunable = sum(t for _, t in prunable)
    if total_prunable < PRUNE_MINIMUM:
        return 0

    # Pruning 실행: compacted 타임스탬프 설정
    for part, _ in prunable:
        part.state.time.compacted = time.time()
        await session.update_part(part)

    return total_prunable
```

**Pruning 후 모델 메시지 변환 시:**
```python
# compacted된 도구 출력은 플레이스홀더로 대체
if part.state.time.compacted:
    output = "[Old tool result content cleared]"
else:
    output = part.state.output
```

### 8.4 Compaction — 대화 요약 생성

Pruning으로 부족하면, LLM을 호출하여 전체 대화를 요약한다.

```python
DEFAULT_COMPACTION_PROMPT = """
Please provide a detailed summary of our conversation so far.
Structure it as follows:

## Goal
What is the user trying to accomplish?

## Instructions
What important instructions and guidelines has the user given?

## Discoveries
What notable things have been learned or discovered?

## Accomplished
What has been accomplished so far? What is still in progress or remaining?

## Relevant Files
What are the important files that have been read or modified?
List each file with a brief description of its purpose and any changes made.
"""

async def compact(
    messages: list[MessageWithParts],
    session_id: str,
    parent_id: str,
    model: Model,
    auto: bool = True,
) -> CompactionResult:
    """대화 요약을 생성하여 컨텍스트 초기화"""

    # 1. 요약 생성용 메시지 준비 (미디어 제거)
    model_messages = await to_model_messages(
        messages, model, strip_media=True
    )

    # 2. "compaction" 에이전트로 요약 생성
    summary_text = await call_llm(
        model=model,
        system_prompt=DEFAULT_COMPACTION_PROMPT,
        messages=model_messages,
        agent="compaction",
    )

    # 3. 요약을 CompactionPart로 저장
    compaction_part = CompactionPart(type="compaction", auto=auto)

    # 4. 요약을 포함한 새 어시스턴트 메시지 생성
    summary_msg = AssistantMessage(
        id=generate_id(),
        session_id=session_id,
        parent_id=parent_id,
        summary=True,  # ← 이 플래그가 핵심
    )

    # 5. 자동 모드: 마지막 사용자 메시지 재전송
    if auto:
        await replay_last_user_message(session_id, messages)

    return CompactionResult(summary=summary_text, message=summary_msg)
```

### 8.5 Compaction 후 메시지 구조

```
[컴팩션 전]
User: "파일 분석해줘"
Assistant: (도구 호출 20번, 텍스트 응답 10개)
User: "리팩토링 해줘"
Assistant: (도구 호출 15번, 텍스트 응답 5개)
User: "테스트 작성해줘"
... (컨텍스트 꽉 참)

[컴팩션 후]
User: "What did we do so far?"     ← CompactionPart가 이 텍스트 생성
Assistant: (요약)                   ← summary=True, 이전 전체 내용 요약
  "## Goal
   사용자가 코드 리팩토링과 테스트 작성을 요청함
   ## Accomplished
   - file_a.py 분석 완료
   - 클래스 구조 리팩토링 완료
   ## Relevant Files
   - file_a.py: 메인 로직
   - test_a.py: 테스트 파일"
User: "테스트 작성해줘"             ← 마지막 사용자 메시지 재전송
Assistant: (새로운 응답 시작)
```

---

## 9. 토큰 카운팅 및 비용 추적

### 9.1 빠른 토큰 추정

```python
CHARS_PER_TOKEN = 4  # 영어 기준 대략적 비율

def estimate_tokens(text: str) -> int:
    """빠른 토큰 추정 (pruning 판단 등에 사용)"""
    return max(0, round(len(text or "") / CHARS_PER_TOKEN))
```

> **참고**: 한글의 경우 1토큰 ≈ 1~2글자이므로, 보고서 에디터에서는 `CHARS_PER_TOKEN = 2`로 조정 권장.

### 9.2 정확한 토큰 사용량 (LLM 응답 기반)

```python
@dataclass
class TokenUsage:
    total: int = 0
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

def get_usage(model: Model, usage: dict, metadata: dict = None) -> Usage:
    """LLM 응답의 사용량 정보를 파싱"""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)

    # 프로바이더별 캐시 토큰 추출
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = 0
    if metadata:
        cache_write = (metadata.get("anthropic", {})
                       .get("cacheCreationInputTokens", 0))
        if not cache_write:
            cache_write = (metadata.get("bedrock", {})
                          .get("usage", {})
                          .get("cacheWriteInputTokens", 0))

    # 비용 계산 (Decimal으로 정밀 계산)
    from decimal import Decimal

    cost_info = model.cost
    if (cost_info.experimental_over_200k and
        input_tokens + cache_read > 200_000):
        cost_info = cost_info.experimental_over_200k

    cost = float(
        Decimal(str(input_tokens)) * Decimal(str(cost_info.input)) / Decimal("1000000")
        + Decimal(str(output_tokens)) * Decimal(str(cost_info.output)) / Decimal("1000000")
        + Decimal(str(cache_read)) * Decimal(str(cost_info.cache_read or 0)) / Decimal("1000000")
        + Decimal(str(cache_write)) * Decimal(str(cost_info.cache_write or 0)) / Decimal("1000000")
        + Decimal(str(reasoning_tokens)) * Decimal(str(cost_info.output)) / Decimal("1000000")
    )

    return Usage(
        tokens=TokenUsage(
            total=usage.get("total_tokens", 0),
            input=input_tokens - cache_read - cache_write,
            output=output_tokens,
            reasoning=reasoning_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        ),
        cost=cost,
    )
```

---

## 10. Provider 추상화 (다중 LLM 지원)

### 10.1 모델 정의

```python
@dataclass
class Model:
    id: str                      # "claude-sonnet-4-20250514"
    provider_id: str
    api: ModelAPI
    capabilities: ModelCapabilities
    limit: ModelLimits
    cost: ModelCost
    options: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)

@dataclass
class ModelLimits:
    context: int                 # 전체 컨텍스트 윈도우 크기
    input: int | None = None     # 입력 전용 제한 (일부 모델)
    output: int                  # 최대 출력 토큰

@dataclass
class ModelCost:
    input: float                 # 입력 토큰당 가격 (per 1M tokens)
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0
    experimental_over_200k: ModelCost | None = None
```

### 10.2 프로바이더별 메시지 변환

```python
class ProviderTransform:
    """프로바이더별 메시지/파라미터 변환"""

    @staticmethod
    def normalize_messages(messages: list[dict], model: Model) -> list[dict]:
        # Anthropic: 빈 메시지 제거, tool_call ID 정규화
        if model.api.sdk == "anthropic":
            messages = [m for m in messages if m.get("content")]
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for part in msg["content"]:
                        if part.get("id"):
                            part["id"] = re.sub(r'[^a-zA-Z0-9_-]', '_', part["id"])
        return messages

    @staticmethod
    def apply_caching(messages: list[dict], model: Model) -> list[dict]:
        """프롬프트 캐싱 힌트 — 시스템 처음 2개 + 대화 마지막 2개"""
        system_msgs = [m for m in messages if m["role"] == "system"][:2]
        final_msgs = [m for m in messages if m["role"] != "system"][-2:]
        cache_options = {
            "anthropic": {"cache_control": {"type": "ephemeral"}},
            "bedrock": {"cache_point": {"type": "default"}},
        }
        for msg in set(system_msgs + final_msgs):
            msg["provider_options"] = cache_options.get(model.provider_id, {})
        return messages

    @staticmethod
    def max_output_tokens(model: Model) -> int:
        return min(model.limit.output, OUTPUT_TOKEN_MAX)
```

### 10.3 Python Provider 인터페이스

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 32000,
        abort_signal: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ...

class AnthropicProvider(LLMProvider):
    async def stream(self, **kwargs) -> AsyncIterator[StreamEvent]:
        import anthropic
        client = anthropic.AsyncAnthropic()
        async with client.messages.stream(
            model=kwargs["model"],
            messages=kwargs["messages"],
            tools=kwargs["tools"],
            max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
        ) as stream:
            async for event in stream:
                yield self._convert_event(event)

class OpenAIProvider(LLMProvider):
    async def stream(self, **kwargs) -> AsyncIterator[StreamEvent]:
        import openai
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            tools=kwargs["tools"],
            max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
            stream=True,
        )
        async for chunk in response:
            yield self._convert_event(chunk)
```

---

## 11. 스트리밍 응답 처리

### 11.1 스트림 이벤트 타입

```python
@dataclass
class TextDeltaEvent:
    type: str = "text-delta"
    text: str = ""

@dataclass
class ReasoningDeltaEvent:
    type: str = "reasoning-delta"
    id: str = ""
    text: str = ""

@dataclass
class ToolCallEvent:
    type: str = "tool-call"
    tool_call_id: str = ""
    tool_name: str = ""
    input: Any = None

@dataclass
class ToolResultEvent:
    type: str = "tool-result"
    tool_call_id: str = ""
    output: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class FinishEvent:
    type: str = "finish"
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)

@dataclass
class ErrorEvent:
    type: str = "error"
    error: Exception = None

StreamEvent = (
    TextDeltaEvent | ReasoningDeltaEvent |
    ToolCallEvent | ToolResultEvent | FinishEvent | ErrorEvent
)
```

### 11.2 프로세서 — 이벤트 핸들링

```python
class SessionProcessor:
    """LLM 스트림 이벤트를 처리하고 세션 상태를 갱신"""

    def __init__(self, session: SessionService, tools: ToolRegistry):
        self.session = session
        self.tools = tools
        self.current_text: TextPart | None = None
        self.reasoning_map: dict[str, ReasoningPart] = {}
        self.needs_compaction = False

    async def process(
        self,
        stream: AsyncIterator[StreamEvent],
        assistant_msg: AssistantMessage,
        model: Model,
    ) -> ProcessResult:
        async for event in stream:
            match event:
                case TextDeltaEvent(text=text):
                    if not self.current_text:
                        self.current_text = TextPart(
                            time=PartTime(start=time.time()),
                        )
                        await self.session.add_part(assistant_msg.id, self.current_text)
                    self.current_text.text += text
                    await self.session.update_part_delta(self.current_text.id, text)

                case ToolCallEvent() as tc:
                    if self._check_doom_loop(tc):
                        await self._ask_user_permission("doom_loop")

                    tool_part = ToolPart(
                        call_id=tc.tool_call_id,
                        tool=tc.tool_name,
                        state=ToolStateRunning(input=tc.input),
                    )
                    await self.session.add_part(assistant_msg.id, tool_part)

                    result = await self.tools.execute(
                        tc.tool_name, tc.input,
                        ToolContext(session_id=assistant_msg.session_id)
                    )

                    tool_part.state = ToolStateCompleted(
                        input=tc.input, output=result.output,
                        title=result.title, metadata=result.metadata,
                        time=ToolTime(start=tool_part.state.time.start, end=time.time()),
                    )
                    await self.session.update_part(tool_part)

                case FinishEvent(usage=usage):
                    u = get_usage(model, usage)
                    assistant_msg.cost += u.cost
                    assistant_msg.tokens = u.tokens
                    await self.session.update_message(assistant_msg)

                    if is_overflow(u.tokens, model, self.config):
                        self.needs_compaction = True

        return ProcessResult(
            action="compact" if self.needs_compaction else "done"
        )
```

---

## 12. 저장소(Storage) 시스템

### 12.1 SQLite 설정

```python
import aiosqlite

PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA cache_size = -64000;
PRAGMA foreign_keys = ON;
PRAGMA wal_checkpoint(PASSIVE);
"""

async def init_database(db_path: str) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    await db.executescript(PRAGMAS)
    return db
```

### 12.2 스키마

```sql
CREATE TABLE project (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    directory TEXT NOT NULL,
    time_created REAL NOT NULL,
    time_updated REAL NOT NULL
);

CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES session(id),
    slug TEXT NOT NULL,
    directory TEXT NOT NULL,
    title TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    share_url TEXT,
    summary_additions INTEGER DEFAULT 0,
    summary_deletions INTEGER DEFAULT 0,
    summary_files INTEGER DEFAULT 0,
    summary_diffs TEXT,            -- JSON
    revert TEXT,                   -- JSON
    permission TEXT,               -- JSON
    time_created REAL NOT NULL,
    time_updated REAL NOT NULL,
    time_compacting REAL,
    time_archived REAL
);
CREATE INDEX idx_session_project ON session(project_id);

CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    data TEXT NOT NULL,             -- JSON (메시지 직렬화)
    time_created REAL NOT NULL,
    time_updated REAL NOT NULL
);
CREATE INDEX idx_message_session ON message(session_id, time_created, id);

CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,       -- 비정규화 (쿼리 최적화)
    data TEXT NOT NULL,             -- JSON (파트 직렬화)
    time_created REAL NOT NULL,
    time_updated REAL NOT NULL
);
CREATE INDEX idx_part_message ON part(message_id, id);
CREATE INDEX idx_part_session ON part(session_id);
```

---

## 13. 권한(Permission) 시스템

```python
@dataclass
class PermissionRule:
    permission: str    # "read", "edit", "bash" 등
    pattern: str       # 파일 패턴 또는 와일드카드
    action: str        # "allow" | "deny" | "ask"

PERMISSION_TYPES = {
    "read", "edit", "write", "bash", "grep", "glob",
    "webfetch", "websearch", "external_directory",
    "task", "doom_loop",
}

class PermissionService:
    def __init__(self, rules: list[PermissionRule]):
        self.rules = rules
        self.session_rules: list[PermissionRule] = []

    def evaluate(self, permission: str, pattern: str) -> str:
        for rule in self.session_rules + self.rules:
            if rule.permission == permission:
                if fnmatch.fnmatch(pattern, rule.pattern):
                    return rule.action
        return "ask"

    async def ask(self, request: PermissionRequest) -> None:
        action = self.evaluate(request.permission, request.patterns[0])
        if action == "allow":
            return
        if action == "deny":
            raise PermissionDenied(request)
        approved = await self._prompt_user(request)
        if not approved:
            raise PermissionDenied(request)
```

---

## 14. Python 구현 설계안

### 14.1 프로젝트 구조

```
report-editor/
├── pyproject.toml
├── src/
│   └── report_editor/
│       ├── __init__.py
│       ├── main.py                  # CLI 진입점
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── agent.py             # 에이전트 정의 및 루프
│       │   └── prompt.py            # 시스템 프롬프트 생성
│       ├── session/
│       │   ├── __init__.py
│       │   ├── session.py           # 세션 CRUD
│       │   ├── message.py           # 메시지 데이터 구조
│       │   ├── processor.py         # 스트림 이벤트 처리
│       │   ├── compaction.py        # Pruning + Compaction
│       │   └── overflow.py          # 오버플로 감지
│       ├── tool/
│       │   ├── __init__.py
│       │   ├── registry.py          # 도구 등록/필터링
│       │   ├── tool.py              # 도구 베이스 클래스
│       │   ├── truncate.py          # 출력 Truncation
│       │   ├── read.py
│       │   ├── write.py
│       │   ├── edit.py
│       │   ├── glob_tool.py
│       │   ├── grep.py
│       │   ├── bash.py
│       │   ├── webfetch.py
│       │   ├── question.py
│       │   ├── task.py
│       │   └── report/              # 보고서 전용 도구
│       │       ├── read_section.py
│       │       ├── extract_toc.py
│       │       ├── validate.py
│       │       └── insert_table.py
│       ├── provider/
│       │   ├── __init__.py
│       │   ├── provider.py          # 추상 인터페이스
│       │   ├── anthropic.py
│       │   ├── openai_provider.py
│       │   └── transform.py         # 메시지 변환
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py          # SQLite 연결 관리
│       │   ├── schema.py
│       │   └── migration.py
│       ├── permission/
│       │   ├── __init__.py
│       │   └── permission.py
│       └── config/
│           ├── __init__.py
│           └── config.py
└── tests/
```

### 14.2 핵심 의존성

```toml
[project]
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "pydantic>=2.0",
    "aiosqlite>=0.20.0",
    "aiofiles>=24.0",
    "rich>=13.0",
    "click>=8.0",
    "httpx>=0.27.0",
]
```

### 14.3 메인 에이전트 루프 (완전한 예시)

```python
class AgentLoop:
    """OpenCode 스타일의 에이전트 루프"""

    def __init__(
        self,
        session: SessionService,
        provider: LLMProvider,
        tools: ToolRegistry,
        compaction: CompactionService,
        model: Model,
        agent: AgentInfo,
    ):
        self.session = session
        self.provider = provider
        self.tools = tools
        self.compaction = compaction
        self.model = model
        self.agent = agent

    async def run(self, session_id: str, user_input: str) -> str:
        """사용자 입력을 받아 에이전트 루프 실행"""

        # 1. 사용자 메시지 저장
        user_msg = await self.session.create_user_message(
            session_id, user_input, self.agent.name
        )

        # 2. 어시스턴트 메시지 초기화
        assistant_msg = await self.session.create_assistant_message(
            session_id, user_msg.id, self.model
        )

        # 3. 에이전트 루프
        max_iterations = self.agent.steps or 100
        for iteration in range(max_iterations):

            # 3a. 전체 대화 히스토리 로드
            messages = await self.session.get_messages(session_id)

            # 3b. 모델 메시지로 변환
            model_messages = await to_model_messages(messages, self.model)

            # 3c. 시스템 프롬프트 구성
            system = self._build_system_prompt()

            # 3d. 도구 스키마 구성
            tool_schemas = self.tools.get_schemas(self.model, self.agent)

            # 3e. LLM 스트리밍 호출
            stream = self.provider.stream(
                messages=model_messages,
                tools=tool_schemas,
                model=self.model.id,
                temperature=self.agent.temperature,
                max_tokens=ProviderTransform.max_output_tokens(self.model),
            )

            # 3f. 스트림 처리
            processor = SessionProcessor(self.session, self.tools)
            result = await processor.process(stream, assistant_msg, self.model)

            # 3g. 결과에 따른 후속 처리
            if result.action == "done":
                break
            elif result.action == "compact":
                pruned = await self.compaction.prune(messages, self.session)
                if pruned < PRUNE_MINIMUM:
                    await self.compaction.compact(
                        messages, session_id, user_msg.id, self.model
                    )
                continue
            elif result.action == "tool_calls_pending":
                continue

        return assistant_msg.get_final_text()
```

### 14.4 구현 우선순위

| 단계 | 컴포넌트 | 설명 |
|------|----------|------|
| **Phase 1** | Storage + Session + Message | SQLite 기반 세션/메시지 CRUD |
| **Phase 2** | Provider + Streaming | Anthropic/OpenAI 스트리밍 |
| **Phase 3** | Tool System | read, write, edit, glob, grep 기본 도구 |
| **Phase 4** | Agent Loop + Processor | 완전한 에이전트 루프 |
| **Phase 5** | Truncation | 도구 출력 자동 제한 |
| **Phase 6** | Pruning + Compaction | 컨텍스트 윈도우 관리 |
| **Phase 7** | Report Tools | 보고서 전용 도구 |
| **Phase 8** | Permission + TUI | 권한 시스템 + 터미널 UI |

---

## 부록: OpenCode 소스 파일 참조

| 컴포넌트 | 파일 경로 |
|----------|-----------|
| 에이전트 정의 | `packages/opencode/src/agent/agent.ts` |
| 세션 관리 | `packages/opencode/src/session/index.ts` |
| 메시지 구조 | `packages/opencode/src/session/message-v2.ts` |
| 컴팩션 | `packages/opencode/src/session/compaction.ts` |
| 오버플로 감지 | `packages/opencode/src/session/overflow.ts` |
| 스트림 처리 | `packages/opencode/src/session/processor.ts` |
| LLM 통합 | `packages/opencode/src/session/llm.ts` |
| 도구 베이스 | `packages/opencode/src/tool/tool.ts` |
| 도구 Truncation | `packages/opencode/src/tool/truncate.ts` |
| 파일 읽기 | `packages/opencode/src/tool/read.ts` |
| 파일 편집 | `packages/opencode/src/tool/edit.ts` |
| Grep | `packages/opencode/src/tool/grep.ts` |
| Glob | `packages/opencode/src/tool/glob.ts` |
| Bash | `packages/opencode/src/tool/bash.ts` |
| Provider 추상화 | `packages/opencode/src/provider/provider.ts` |
| Provider 변환 | `packages/opencode/src/provider/transform.ts` |
| DB 스키마 | `packages/opencode/src/session/session.sql.ts` |
| 토큰 추정 | `packages/opencode/src/util/token.ts` |
| 설정 | `packages/opencode/src/config/config.ts` |
| 권한 | `packages/opencode/src/permission/index.ts` |
