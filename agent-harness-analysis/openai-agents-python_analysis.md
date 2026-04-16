# OpenAI Agents Python SDK 심층 분석 보고서

> **분석 관점**: opencode, Claude Code와 같은 에이전트 플랫폼/하네스 개발팀의 시각에서 벤치마킹 가능한 설계, 추상화, 패턴을 도출한다.
>
> **분석 대상**: [openai/openai-agents-python](https://github.com/openai/openai-agents-python) v0.14.1 (MIT License, Python ≥ 3.10)
>
> **작성일**: 2026-04-16

---

## 0. TL;DR — 하네스 팀이 챙겨야 할 핵심 5가지

1. **`NextStep*` Sum Type 기반 에이전트 루프** — `NextStepRunAgain | NextStepFinalOutput | NextStepHandoff | NextStepInterruption` 4개 타입으로 에이전트 루프의 다음 동작을 모델링. `match`/`isinstance` 분기만으로 깔끔한 상태머신을 구성.
2. **이중 계층 스트리밍 이벤트** — `RawResponsesStreamEvent`(토큰 단위 raw 패스스루) + `RunItemStreamEvent`(시맨틱 이벤트). 같은 데이터를 두 추상도로 동시 노출해 "토큰 UI"와 "이벤트 UI"를 모두 지원.
3. **Python 함수 시그니처 → 엄격 JSON 스키마 자동 생성** — `inspect.signature` + `pydantic.create_model` + `griffe` 도큐스트링 파서를 결합. `Annotated[T, "desc"]`로 인라인 설명 가능. `ensure_strict_json_schema()` 단일 함수가 OpenAI Structured Outputs 호환 형태로 정규화.
4. **Protocol 기반 미니멀 Session 인터페이스** — `get_items / add_items / pop_item / clear_session` 단 4개 메서드. SDK 내부 타입이 아니라 **wire format**(Responses API input items)을 저장하므로 백엔드 구현이 단순.
5. **`Handoff` vs `agent.as_tool()` 이원화** — 같은 "함수 호출" wire format을 사용하되, 전자는 대화 주도권 양도(라우팅), 후자는 중첩된 서브 호출(매니저-워커). 설계 레벨에서 명확히 구분.

이 외에도 **트레이싱(스팬 타입을 의미별로 세분화)**, **HIL 인터럽션(직렬화/재개 가능한 RunState)**, **MCP 통합(도구를 `FunctionTool`로 흡수)**, **Lazy optional extras(코어를 가볍게 유지)**, **Prompt cache key 자동 해싱** 등은 모두 하네스 설계에 즉시 적용 가능한 패턴들이다.

---

## 1. 리포지토리 개요

| 항목 | 값 |
|------|-----|
| 레포 | https://github.com/openai/openai-agents-python |
| 버전 | 0.14.1 |
| 라이선스 | MIT |
| Python | ≥ 3.10 |
| 핵심 의존성 | `openai>=2.26`, `pydantic>=2.12`, `griffelib>=2`, `mcp>=1.19`, `websockets>=15` |
| Optional Extras | voice, viz(graphviz), litellm, any-llm, realtime, sqlalchemy, redis, dapr, encrypt, s3, temporal, sandbox(docker/e2b/daytona/modal/runloop/blaxel/vercel/cloudflare) |
| 코어 LOC | `src/agents/` ≈ 87k LOC (top-level만 ≈ 16k) |
| 다국어 docs | 한/일/중 번역 포함 (MkDocs) |
| 예제 | 20+ 디렉터리 (basic, agent_patterns, handoffs, customer_service, research_bot, financial_research_agent, mcp, memory, model_providers, realtime, sandbox, tools, voice, hosted_mcp, …) |

### 1.1 디렉터리 구조

```
src/agents/
├── agent.py             (941 LOC)   Agent dataclass
├── run.py              (1,859 LOC)  Runner / AgentRunner (public)
├── run_state.py        (3,304 LOC)  RunState 직렬화/재개
├── run_internal/       (~12k LOC)   실제 루프 구현체 (22+ 파일)
├── tool.py             (1,938 LOC)  Tool 타입들 + @function_tool
├── items.py              (829 LOC)  RunItem 계층
├── result.py             (896 LOC)  RunResult / RunResultStreaming
├── handoffs/                        Handoff 메커니즘
├── guardrail.py                     Input/Output guardrails
├── lifecycle.py                     RunHooks / AgentHooks
├── memory/                          Session 구현체들
├── mcp/                             MCP 통합 (stdio/sse/streamable-http)
├── tracing/                         Span/Processor/Exporter
├── models/             (~6k LOC)    Provider 추상화
├── extensions/models/  (~2k LOC)    LiteLLM, any-llm 어댑터
├── voice/                           STT→LLM→TTS 파이프라인
├── realtime/                        gpt-realtime WebSocket
├── sandbox/                         원격 샌드박스 클라이언트들
├── stream_events.py                 스트리밍 이벤트 타입
├── usage.py                         토큰/요청 집계
└── exceptions.py                    예외 계층
```

> **핵심 관찰**: public API는 `run.py`, `agent.py`, `tool.py`, `handoffs/`, `guardrail.py`, `memory/`, `mcp/`, `tracing/` 정도. 무거운 로직은 전부 `run_internal/`에 격리되어 있고, `AgentRunner` 자체가 "**experimental, not part of public API**"로 명시됨 (`run.py:429`).

### 1.2 의존성 전략

코어 패키지는 **OpenAI + Pydantic + griffelib + MCP** 정도로 가볍게 유지하고, 나머지는 모두 optional extras로 분리한다. `MultiProvider._create_fallback_provider()` (`models/multi_provider.py:147`)에서 보듯 LiteLLM은 첫 호출 시점에 lazy import.

> **하네스 팀 시사점**: TUI/IDE 통합 하네스는 종종 무거운 의존성(트레이싱 백엔드, 벡터 DB, 여러 LLM SDK)을 떠안기 쉽다. 코어는 최소화하고 **optional extras + lazy import**로 모듈화하는 패턴이 유지보수성에 결정적이다.

---

## 2. 핵심 추상화

### 2.1 Agent — Generic Dataclass

```python
@dataclass
class Agent(AgentBase, Generic[TContext]):
    name: str
    instructions: str | Callable[[RunContextWrapper, Agent], MaybeAwaitable[str]] | None
    prompt: Prompt | DynamicPromptFunction | None     # OpenAI Responses API Prompt 템플릿
    handoffs: list[Agent | Handoff]
    model: str | Model | None
    model_settings: ModelSettings
    tools: list[Tool]
    mcp_servers: list[MCPServer]
    mcp_config: MCPConfig
    input_guardrails: list[InputGuardrail]
    output_guardrails: list[OutputGuardrail]
    output_type: type | AgentOutputSchemaBase | None
    hooks: AgentHooks | None
    tool_use_behavior: "run_llm_again" | "stop_on_first_tool" | StopAtTools | ToolsToFinalOutputFunction
    reset_tool_choice: bool = True
```
(`agent.py:223-322`)

**설계 포인트**:

- **Plain dataclass** — DSL/YAML 없음. `dataclasses.replace`로 1-line `clone()`.
- **`Generic[TContext]`** — 사용자 컨텍스트(예: 세션 객체, DB 핸들)가 **타입 안전하게** 모든 콜백/도구에 전파됨.
- **`__post_init__`에서 적극적 타입 체크** (`agent.py:324-455`) — 잘못 구성하면 즉시 실패. mypy strict 와중에도 런타임 검증을 추가.
- **`reset_tool_choice=True`** — 도구 호출 후 `tool_choice`를 자동으로 리셋. 무한 루프 방지를 위한 기본값. 의외로 많은 자작 하네스가 빠뜨리는 디테일.

### 2.2 동적 instructions / 동적 prompts

```python
agent = Agent(
    name="...",
    instructions=lambda ctx, agent: f"You are helping user {ctx.context.user_id}",
)
```

`get_system_prompt()`는 callable의 인자 개수(2개) 검증, sync/async 모두 지원 (`agent.py:902-929`). 단순하지만 상태 의존적 시스템 프롬프트를 깔끔하게 표현.

### 2.3 `Agent.as_tool()` — 에이전트의 도구화

```python
booker = Agent(name="Booker", ...)
parent = Agent(
    name="Concierge",
    tools=[booker.as_tool(tool_name="book_flight", tool_description="...")]
)
```
(`agent.py:472-900`)

내부 동작:
1. 사용자가 `parameters`로 Pydantic 타입을 주거나, 기본 `AgentAsToolInput` 사용
2. 엄격 JSON 스키마 자동 생성
3. 도구 호출 시 **중첩된 `Runner.run`** 실행 (`resolve_agent_tool_input`)
4. **부모의 승인 상태가 중첩 컨텍스트로 전파** (`_apply_nested_approvals`) — "Remember my choice" UX가 nesting을 거쳐도 동작
5. `on_stream` 콜백으로 중첩 실행의 이벤트를 부모 스트림으로 forward

**시사점**: 서브 에이전트를 호출하는 방법이 두 가지(Handoff vs as_tool)임을 분리하고, 각각의 wire format/의미론을 명확히 정의한 점이 인상적. 자체 하네스에서도 "라우팅 vs 사브루틴 호출"을 같은 메커니즘으로 뭉뚱그리지 말고 분리할 가치가 있다.

---

## 3. Runner / 에이전트 루프

### 3.1 진입점 구조

```python
class Runner:
    @classmethod
    async def run(...) -> RunResult: ...
    @classmethod
    def run_sync(...) -> RunResult: ...
    @classmethod
    def run_streamed(...) -> RunResultStreaming: ...
```
(`run.py:192-424`)

`Runner`는 얇은 facade. 실제 구현은 모듈-레벨 `DEFAULT_AGENT_RUNNER: AgentRunner`에 위임되며, `set_default_agent_runner()`로 교체 가능 (실험적). 이 분리 덕분에 public API 표면적이 작게 유지된다.

### 3.2 실행 루프의 4단 상태머신

루프의 한 turn은 `process_model_response()` → `execute_tools_and_side_effects()`로 분리되고, 그 결과로 다음 4가지 중 하나의 step을 반환한다:

```python
@dataclass class NextStepHandoff:    new_agent: Agent[Any]
@dataclass class NextStepFinalOutput: output: Any
@dataclass class NextStepRunAgain:    pass
@dataclass class NextStepInterruption: interruptions: list[ToolApprovalItem]
```
(`run_internal/run_steps.py:143-208`)

루프 본체는 이 4개를 `isinstance` 분기하여 처리:
- `NextStepFinalOutput` → output guardrails 실행 → `RunResult` 반환
- `NextStepHandoff` → `current_agent` 교체, `on_agent_start` 훅 재호출
- `NextStepRunAgain` → 다음 turn으로
- `NextStepInterruption` → `RunResult(interruptions=[...], state=...)` 반환, 호출자가 `RunState`로 재개

> **하네스 시사점**: 상태머신을 boolean flag(`is_done`, `needs_handoff`, `paused`)로 누덕누덕 표현하면 곧 망가진다. **Sum type + match**가 정답. Python에서도 dataclass + `isinstance`로 충분히 깔끔하게 표현 가능.

### 3.3 Turn별 처리 흐름

`run_single_turn()` (`run_internal/run_loop.py:1684`):

1. `asyncio.gather()`로 system prompt + Prompt 설정을 병렬 해석
2. `get_output_schema()`, `get_handoffs()` (가능 도구 필터링)
3. `RunConfig.call_model_input_filter` 적용 (사용자가 입력을 잘라내거나 변형 가능)
4. `deduplicate_input_items_preferring_latest()` — 재개/세션 머지 시 중복 제거
5. `model.get_response()` 또는 `stream_response()` 호출
6. `process_model_response()` (`turn_resolution.py:1420`) — 응답 항목 분류:
   - `ResponseFunctionToolCall` → `ToolRunFunction` 또는 `ToolRunHandoff` (이름이 handoff_map에 있으면)
   - 기타 hosted tool, computer, shell, apply_patch, file_search, web_search, MCP 호출 → 각 타입의 `ToolRun*`
   - reasoning, message, MCP list, MCP approval → 단순 `RunItem` 기록
7. `execute_tools_and_side_effects()` (`turn_resolution.py:547`):
   - `_build_plan_for_fresh_turn` — 승인 상태 해석
   - `_execute_tool_plan` — function tool은 `asyncio.gather`로 **병렬**, computer/shell/apply_patch는 **직렬**
   - 승인 인터럽션 → `NextStepInterruption`
   - handoff → `execute_handoffs()` → `NextStepHandoff`
   - `_maybe_finalize_from_tool_results` — `tool_use_behavior`에 따라 short-circuit
   - 더 이상 도구 없고 마지막이 텍스트 메시지면 `output_schema` 검증 후 `NextStepFinalOutput`
   - 그 외 `NextStepRunAgain`

### 3.4 `tool_use_behavior` — 종료 조건의 4가지 모드

```python
"run_llm_again"             # 기본: 도구 결과를 LLM에 다시 넣음
"stop_on_first_tool"        # 첫 도구 호출 결과가 final output
StopAtTools(stop_at_tool_names=["save_record"])  # 특정 도구 호출 시 stop
ToolsToFinalOutputFunction  # 사용자 callable로 최종 출력 결정
```

`ToolsToFinalOutputFunction`은 `(ctx, list[FunctionToolResult]) -> ToolsToFinalOutputResult(is_final_output, final_output)` 시그니처. 도메인-특수한 종료 정책을 깔끔하게 주입할 수 있다.

> **시사점**: 코딩 에이전트(Claude Code 류)는 종종 "특정 도구 결과 = 종료"를 원할 때가 있다(예: `submit_pr`). 이를 루프 본체가 아니라 **에이전트 정의 시점에 declaratively 표현**하는 디자인이 깔끔하다.

### 3.5 max_turns와 우아한 탈출구

```python
DEFAULT_MAX_TURNS = 10
```
초과 시 `MaxTurnsExceeded` 발생. 단, `RunErrorHandlers["max_turns"]`를 등록하면 **예외 대신 합성된 final output**을 반환할 수 있는 escape hatch가 있다 (`run.py:1040-1126`). 인터랙티브 하네스에서 "강제 종료해도 사용자에게는 마지막 답변을 보여주고 싶다"는 요구를 깔끔하게 수용.

### 3.6 병렬 실행과 격리

- function tools: 동일 turn 내 모든 호출이 `asyncio.gather`로 **병렬**
- computer/shell/apply_patch/local_shell: **직렬** (상태 변경, 순서 중요)
- guardrails + hooks: 각 도구마다 병렬 실행
- `isolate_parallel_failures` 플래그로 한 sibling 실패가 다른 호출 cancel 여부 제어

### 3.7 에러 계층

```
AgentsException (base, .run_data: RunErrorDetails | None)
├── MaxTurnsExceeded
├── ModelBehaviorError       # JSON 파싱 실패, 알 수 없는 도구명 등
├── UserError                # SDK 오용
├── MCPToolCancellationError
├── ToolTimeoutError(tool_name, timeout_seconds)
├── InputGuardrailTripwireTriggered(guardrail_result)
├── OutputGuardrailTripwireTriggered(guardrail_result)
├── ToolInputGuardrailTripwireTriggered(guardrail, output)
└── ToolOutputGuardrailTripwireTriggered(guardrail, output)
```
(`exceptions.py`)

`RunErrorDetails`는 `input, new_items, raw_responses, last_agent, context_wrapper, input_guardrail_results, output_guardrail_results`를 모두 포함. 예외가 발생해도 **어디까지 진행되었는지 부분 상태를 보존**하므로 디버깅과 재시도 UX가 가능하다.

> **시사점**: 하네스에서 가장 중요한 디버깅 정보는 "어디까지 갔다가 어떻게 깨졌는가"이다. 단순한 `Exception(msg)`는 운영 환경에서 무용. **부분 상태를 첨부하는 예외**가 표준이 되어야 한다.

---

## 4. Tool 시스템

### 4.1 Tool 타입 분류

`Tool` union (`tool.py`):

| 타입 | 용도 | 실행 위치 |
|------|------|-----------|
| `FunctionTool` | 사용자 정의 Python 함수 | 클라이언트 |
| `FileSearchTool, WebSearchTool, CodeInterpreterTool, ImageGenerationTool, HostedMCPTool, ToolSearchTool` | OpenAI Hosted | 서버 |
| `ComputerTool` | 컴퓨터 사용 (Anthropic computer-use 류) | 클라이언트 (브리지) |
| `LocalShellTool, ShellTool, ApplyPatchTool` | 셸/패치 실행 | 클라이언트 |
| `CustomTool` | Responses의 "custom tool" 타입 | 서버 |

각 hosted tool은 그저 **설정만 담은 dataclass**(예: `FileSearchTool(vector_store_ids=[...])`). 실제 실행은 OpenAI 측에서 이루어지므로 클라이언트가 할 일이 없다.

### 4.2 `FunctionTool` 구조

```python
@dataclass
class FunctionTool:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_tool: Callable[[ToolContext, str], Awaitable[Any]]
    strict_json_schema: bool = True
    is_enabled: bool | Callable[[ctx, agent], MaybeAwaitable[bool]] = True
    tool_input_guardrails: list[ToolInputGuardrail] = []
    tool_output_guardrails: list[ToolOutputGuardrail] = []
    needs_approval: bool | Callable[[ctx, params, call_id], Awaitable[bool]] = False
    timeout_seconds: float | None = None
    timeout_behavior: "error_as_result" | "raise_exception" = "error_as_result"
    timeout_error_function: Callable | None = None
    defer_loading: bool = False  # hosted tool search용
    # ...metadata: _tool_origin, _mcp_title, _tool_namespace, _is_agent_tool, _is_codex_tool, _agent_instance
```
(`tool.py:281-416`)

**주목할 필드들**:

- `is_enabled`: callable로 받아 **턴별 동적 활성화** 가능. 예: 사용자 권한에 따라 도구 노출 변경.
- `needs_approval`: 호출 단위 HIL 게이트. callable이면 인자값까지 보고 결정 가능 ("이 SQL이 DROP을 포함하면 승인 필요").
- `timeout_seconds + timeout_behavior`: timeout을 LLM-visible 에러로 줄지, 예외로 raise 할지 선택. **기본이 `error_as_result`**라는 점이 코딩 에이전트 친화적.
- `defer_loading`: hosted tool search와 함께 쓰일 때만 정의를 노출 (토큰 절약).

### 4.3 `@function_tool` 데코레이터의 마법

```python
@function_tool
async def list_files(ctx: RunContextWrapper[Workspace],
                      glob: Annotated[str, "Glob pattern, e.g. '**/*.py'"]) -> list[str]:
    """List files in the workspace matching the glob."""
    return ctx.context.find(glob)
```
(`tool.py:1725-1870`)

내부 처리 (`function_schema.py:246-424`):
1. `inspect.signature` + `get_type_hints(include_extras=True)`로 시그니처 분석
2. **griffe**로 docstring 파싱 — google/numpy/sphinx 스타일 자동 감지
3. 첫 인자가 `RunContextWrapper`/`ToolContext` 이면 schema에서 제거 (`takes_context=True` 마킹)
4. `Annotated[T, "desc"]` 메타데이터에서 인자 설명 추출
5. `pydantic.create_model()`로 동적 모델 생성, JSON schema 추출
6. `ensure_strict_json_schema()`로 정규화

**호출 시점** (`_on_invoke_tool_impl`):
1. JSON 파싱
2. `schema.params_pydantic_model(**json_data)` — 실패 시 `ModelBehaviorError`
3. `schema.to_call_args(parsed)` — `*args`/`**kwargs` 재구성
4. sync 함수면 `asyncio.to_thread`로 자동 오프로딩
5. 예외 시 `failure_error_function` 호출 (기본은 일반화된 메시지를 LLM에 반환, `None` 설정 시 raise)

> **하네스 시사점**:
> - **"코드가 곧 도구 정의"** — 별도 manifest 파일이나 JSON 스키마 직접 작성 없음. 개발 속도와 일관성 모두에 유리.
> - **`ensure_strict_json_schema()` 한 함수**가 모든 strict 보정을 책임 (`additionalProperties: false`, 모든 prop required, 비호환 스키마 거부). 자체 하네스에서도 도구 등록 후 단일 정규화 단계는 필수.
> - **sync 자동 to_thread**: 사용자가 async 학습 부담 없이 `def`로 도구를 짜도 동작. 진입 장벽을 크게 낮춤.

### 4.4 도구 실행 흐름 (`run_internal/tool_execution.py`)

각 function tool 호출은 다음 순서를 거친다:

```
tool_input_guardrails → on_tool_start hooks → invoke_function_tool
   → tool_output_guardrails → on_tool_end hooks
   → wrap as ToolCallOutputItem (with ToolOrigin)
```

`_FunctionToolBatchExecutor.execute()`가 모든 호출을 `asyncio.gather`로 묶어 병렬 실행. guardrail/hook도 호출별 병렬.

---

## 5. Handoff 메커니즘

### 5.1 정의

```python
@dataclass
class Handoff(Generic[TContext, TAgent]):
    tool_name: str                                # 기본: "transfer_to_<agent_name>"
    tool_description: str
    input_json_schema: dict
    on_invoke_handoff: Callable[..., Awaitable[Agent]]
    agent_name: str
    input_filter: HandoffInputFilter | None       # 대화 히스토리 재작성
    nest_handoff_history: bool | None             # 이전 대화를 요약 메시지로 wrap
    strict_json_schema: bool = True
    is_enabled: bool | Callable = True
```
(`handoffs/__init__.py:93-180`)

### 5.2 Wire format

LLM에게 handoff는 그냥 함수 호출처럼 보인다 — `transfer_to_billing_agent({reason: "user wants refund"})`. Runner가 응답의 함수 이름을 `handoff_map`에서 lookup하여 일반 도구 호출과 분기한다 (`process_model_response`).

전송 메시지는 `{"assistant": "billing_agent"}` 한 줄 (`Handoff.get_transfer_message`).

### 5.3 `input_filter` — 외과적 히스토리 제어

```python
def remove_all_tools(data: HandoffInputData) -> HandoffInputData:
    return data.clone(input_history=..., new_items=[item for item in data.new_items if not is_tool_item(item)])
```
(`extensions/handoff_filters.py:33-108`)

`HandoffInputData`는 `input_history, pre_handoff_items, new_items, run_context, input_items`를 모두 노출하는 frozen dataclass + `.clone()`. 사용자 정의 필터로 다음 에이전트의 시야를 정밀하게 제어 가능.

### 5.4 Handoff vs `as_tool()` — 어떤 차이?

| 측면 | Handoff | `agent.as_tool()` |
|------|---------|-------------------|
| 제어 흐름 | 새 에이전트가 **대화 주도권 인계** | 부모가 도구 호출로 **중첩 실행**, 이후 계속 |
| 입력 | 전체 대화 히스토리 (filter로 가공 가능) | 사용자 정의 스키마 |
| 출력 | 타깃 에이전트가 final output 생성 | 도구 결과를 부모에게 반환 |
| 트레이스 | `HandoffCallItem` + `HandoffOutputItem` | `ToolCallItem` + `ToolCallOutputItem` (origin=AGENT_AS_TOOL) |
| 컨텍스트 | 동일 `RunContextWrapper` | 새 `ToolContext` (승인 상태는 mirroring) |

**선택 기준**:
- **Handoff**: "이제부터 이 분야 전문가가 대답해야 함" (라우팅, triage)
- **as_tool**: "이 작업은 위임하되 결과는 내가 받아서 계속" (매니저-워커, planning)

---

## 6. Guardrail 시스템

### 6.1 InputGuardrail

```python
@dataclass
class InputGuardrail(Generic[TContext]):
    guardrail_function: Callable[[ctx, agent, input], MaybeAwaitable[GuardrailFunctionOutput]]
    name: str | None = None
    run_in_parallel: bool = True

@dataclass
class GuardrailFunctionOutput:
    output_info: Any
    tripwire_triggered: bool
```
(`guardrail.py:19-343`)

루프 통합:
- **Turn 0의 첫 에이전트에서만** 실행. 핸드오프된 후속 에이전트의 input은 다시 검사하지 않음 (의도된 디자인 결정).
- `run_in_parallel=False` (sequential): 모델 호출 전 완료. tripwire 시 LLM 호출 스킵 → **토큰 비용 0**.
- `run_in_parallel=True` (default): 모델 호출과 race. tripwire 시 모델 task를 cancel 가능 (`should_cancel_parallel_model_task_on_input_guardrail_trip`).

### 6.2 OutputGuardrail

`NextStepFinalOutput`이 만들어진 후에만 실행 (LLM이 더 이상 작업 중이 아니므로 병렬 실행 옵션 없음).

### 6.3 ToolInput/Output Guardrails — 별도 개념

함수 도구마다 첨부되는 정책 검사. 단순 boolean이 아니라:
- **skip**: 도구 호출 자체를 건너뜀
- **replace output**: 결과를 임의 값으로 대체
- **tripwire**: 예외 발생

> **시사점**: 가드레일은 "막거나/통과시키거나"의 이분법으로 자주 모델링되지만, 실제로는 **수정/대체** 케이스가 많다. Return type을 다양화하면 표현력이 커진다.

---

## 7. Session / Memory

### 7.1 미니멀 Protocol

```python
@runtime_checkable
class Session(Protocol):
    session_id: str
    session_settings: SessionSettings | None

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]: ...
    async def add_items(self, items: list[TResponseInputItem]) -> None: ...
    async def pop_item(self) -> TResponseInputItem | None: ...
    async def clear_session(self) -> None: ...
```
(`memory/session.py:14-54`)

**핵심 결정**: 저장 단위가 **wire format (`TResponseInputItem`)**, 즉 OpenAI Responses API의 input item 형태. SDK 내부 `RunItem` 타입이 아니다.

이로 인해:
- 백엔드 구현이 단순 (JSON blob 저장만 하면 됨)
- SDK 내부 타입 변경이 저장소 호환성을 깨뜨리지 않음
- 다른 도구로도 손쉽게 검사/마이그레이션 가능

### 7.2 빌트인 구현

| 구현 | 위치 | 비고 |
|------|------|------|
| `SQLiteSession` | `memory/sqlite_session.py` | WAL 모드, thread-local connection (file), shared connection (`:memory:`). 테이블 2개: `agent_sessions`, `agent_messages` (FK + cascade) |
| `OpenAIConversationsSession` | `memory/openai_conversations_session.py` | 첫 호출 시 lazy하게 server-side conversation 생성 |
| `OpenAIResponsesCompactionSession` | `memory/openai_responses_compaction_session.py` | 서버 측 history compaction 트리거 (`run_compaction({response_id, mode, store, force})`) |
| Optional: Redis, SQLAlchemy(Postgres), Dapr, S3, 암호화, 파일 | extras로 분리 | 코어 의존성 오염 방지 |

### 7.3 루프 통합

- `validate_session_conversation_settings`: Session과 `conversation_id`/`previous_response_id`가 **상호 배타적**. 문서화 + 런타임 enforce.
- `prepare_input_with_session`: 매 run 시작 시 history prepend
- `save_result_to_session`: turn 후 새 항목 저장 (재개 turn의 중복 방지를 위해 `_current_turn_persisted_item_count` 사용)
- `session_input_callback`(RunConfig): 사용자 머지 로직 주입

> **하네스 시사점**:
> - Protocol + 4개 메서드로 충분. 무거운 ABC 강제하지 않음.
> - **Wire format 저장**이 정답. 자체 SDK 타입을 저장하면 마이그레이션 지옥.
> - "서버 측 대화 영속화 vs 클라이언트 측 세션"의 **상호 배타성**을 일찍 enforce할 것.

---

## 8. Tracing — 운영 가능한 관측성

### 8.1 모델

```
Trace (workflow_name, trace_id="trace_<32alnum>", group_id, metadata)
└── Span[TSpanData] (started_at, ended_at, parent_id)
        │
        └── SpanData 서브타입:
            AgentSpanData, TaskSpanData, TurnSpanData, FunctionSpanData,
            GenerationSpanData, GuardrailSpanData, HandoffSpanData,
            MCPListToolsSpanData, ResponseSpanData, SpeechGroupSpanData,
            SpeechSpanData, TranscriptionSpanData, CustomSpanData
```
(`tracing/span_data.py:28-450`)

> **시사점**: 일반화된 단일 Span 타입이 아니라 **의미별로 SpanData를 세분화**한 것이 인상적. UI/대시보드가 도메인 지식을 가진 채로 렌더링 가능 (예: "이 span은 도구 호출이니 도구명·인자·결과 컬럼을 보여줘").

### 8.2 Processor / Exporter 인터페이스

```python
class TracingProcessor(abc.ABC):
    on_trace_start / on_trace_end / on_span_start / on_span_end / shutdown / force_flush

class TracingExporter(abc.ABC):
    export(items: list[Trace | Span])
```
(`tracing/processor_interface.py`)

빌트인 (`tracing/processors.py`, ~650 LOC):
- `ConsoleSpanExporter` — stdout
- `BackendSpanExporter` — `https://api.openai.com/v1/traces/ingest` (`OpenAI-Beta: traces=v1`), 지수 백오프, 100k byte 필드 truncation, 사용량 allowlist 필터
- `BatchTraceProcessor` (default) — 백그라운드 스레드 + 큐 + 주기적 flush + `flush_traces()`

### 8.3 Sensitive Data 분리

```python
# RunConfig
trace_include_sensitive_data: bool = True     # env로 override 가능
# 모델 단위
ModelTracing.DISABLED / ENABLED / ENABLED_WITHOUT_DATA
```

> **시사점**: "트레이스의 **모양**"과 "트레이스의 **내용**"을 분리하는 플래그는 프라이버시/규제 환경에서 결정적. 자체 하네스에서도 필수 옵션.

### 8.4 자동 nesting via ContextVar

`tracing/scope.py`의 ContextVar가 span 부모-자식 관계를 자동으로 추적. 사용자가 `with trace("foo"):`로 여러 `Runner.run`을 묶으면 **하나의 trace로 collapse**.

---

## 9. Streaming — 이중 계층 이벤트

### 9.1 단 3개의 이벤트 타입

```python
@dataclass class RawResponsesStreamEvent:
    data: TResponseStreamEvent       # OpenAI SDK에서 그대로 패스스루
    type = "raw_response_event"

@dataclass class RunItemStreamEvent:
    name: Literal[
        "message_output_created", "handoff_requested", "handoff_occured",
        "tool_called", "tool_search_called", "tool_search_output_created",
        "tool_output", "reasoning_item_created",
        "mcp_approval_requested", "mcp_approval_response", "mcp_list_tools",
    ]
    item: RunItem
    type = "run_item_stream_event"

@dataclass class AgentUpdatedStreamEvent:
    new_agent: Agent[Any]
    type = "agent_updated_stream_event"
```
(`stream_events.py:10-62`, 단 62 LOC)

### 9.2 이중 계층의 의의

- **Raw layer** — 토큰 단위. 채팅 UI가 typing effect를 구현하기 위함.
- **Semantic layer** — "도구가 호출됨", "에이전트가 바뀜" 같은 의미적 사건. 진행 인디케이터/로그/이벤트소싱에 적합.

같은 데이터를 두 추상도로 동시에 노출. 비용은 거의 0(애초에 같은 응답을 두 번 보내는 게 아니라 같은 큐에서 두 종류의 이벤트가 흘러나옴).

### 9.3 우아한 취소

```python
result.cancel(mode="immediate")   # 즉시 task cancel + 큐 drain
result.cancel(mode="after_turn")  # 현재 turn 완료 후 종료 (도구 호출, 세션 저장 보장)
```

`RunResultStreaming.cancel()` (`result.py:670+`). **인터랙티브 하네스**에서 사용자가 ESC를 눌렀을 때, "지금 당장" vs "이번 작업만 끝내고"를 구분하는 UX가 자연스럽게 가능.

> **시사점**: opencode/Claude Code에서 사용자가 stop을 눌렀을 때 실제 의미가 무엇인지 ambiguity가 크다. 이 두 가지 모드를 분리해 노출하는 것 자체가 디자인 결정.

---

## 10. Lifecycle Hooks

```python
# 글로벌 (Runner.run에 전달)
class RunHooksBase[TContext, TAgent]:
    on_llm_start, on_llm_end,
    on_agent_start, on_agent_end,
    on_handoff,
    on_tool_start, on_tool_end

# 에이전트 단위 (Agent.hooks)
class AgentHooksBase[TContext, TAgent]:
    # 동일 메서드 — 단, 이 에이전트가 active일 때만 firing
```
(`lifecycle.py`)

루프는 두 hook 세트를 `asyncio.gather`로 **병렬 호출** (`tool_execution.py:1681`, `run_loop.py:1717`). `AgentHookContext`는 `RunContextWrapper`를 확장하여 `turn_input`을 포함.

> **시사점**: 글로벌 + 에이전트별 두 레벨로 나눈 것이 합리적. 하네스에서도 "전역 로깅/메트릭" vs "특정 에이전트의 동작 관측"이 자주 필요.

---

## 11. Model Provider 추상화

### 11.1 ABCs

```python
class Model(abc.ABC):
    async def get_response(system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *,
                            previous_response_id, conversation_id, prompt) -> ModelResponse
    def stream_response(...) -> AsyncIterator[TResponseStreamEvent]
    async def close(self) -> None
    def get_retry_advice(request) -> ModelRetryAdvice | None

class ModelProvider(abc.ABC):
    def get_model(model_name: str | None) -> Model
    async def aclose(self) -> None
```
(`models/interface.py`)

### 11.2 MultiProvider — 접두사 라우팅

```python
"openai/gpt-4.1"      # OpenAI Responses
"litellm/anthropic/claude-3-5-sonnet"  # LiteLLM 경유
"any-llm/..."         # any-llm SDK
```
(`models/multi_provider.py`)

설정:
- `unknown_prefix_mode: "error" | "model_id"` — 미지의 접두사를 에러 처리하거나 OpenAI provider에 그대로 전달 (OpenRouter 등 호환 endpoint용)
- `openai_prefix_mode: "alias" | "model_id"` — `openai/` 접두사 처리
- `MultiProviderMap`로 사용자 접두사 등록 가능
- LiteLLM/any-llm는 첫 요청 시 lazy import

### 11.3 Chat Completions ↔ Responses 변환

`models/chatcmpl_converter.py` (873 LOC)와 `chatcmpl_stream_handler.py`가 두 형식 간 변환을 책임. LiteLLM/any-llm 어댑터가 이를 활용.

> **시사점**: Provider-agnostic을 표방하면서 내부적으로 한쪽 wire format(여기서는 Responses API)을 정규형으로 두고 나머지를 그쪽으로 변환하는 패턴. 자체 하네스에서도 **하나의 정규형을 정하고 어댑터로 흡수**하는 전략이 유지보수에 유리.

### 11.4 Prompt Cache Key

```python
# run_internal/prompt_cache_key.py
PromptCacheKeyResolver.resolve()
# → "agents-sdk:<kind>:<sha256-32>"  (conversation/session/group id 기반)
# → "agents-sdk:run:<uuid>"          (단일 run scoped)
```

사용자가 이미 `model_settings.extra_args["prompt_cache_key"]`를 설정했으면 **건드리지 않음** (graceful fallback). 자체 cache key 정책이 있는 사용자를 존중.

### 11.5 Usage 추적

```python
@dataclass
class Usage:
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokensDetails  # cached_tokens
    output_tokens_details: OutputTokensDetails  # reasoning_tokens
    request_usage_entries: list[RequestUsage]  # 요청별 상세
```
(`usage.py:60-319`)

`RunContextWrapper.usage`가 라이브 누적기. 단, **빌트인 사용량 limit 강제 없음** — SDK는 보고만, 정책은 애플리케이션 책임.

---

## 12. MCP 통합

### 12.1 ABC 및 구현

```python
class MCPServer(abc.ABC):
    connect, cleanup, list_tools, call_tool, list_prompts, get_prompt
    # default NotImplementedError: list_resources, list_resource_templates, read_resource
```
(`mcp/server.py:223-478`)

3가지 구체 구현:
- `MCPServerStdio` (subprocess + stdio)
- `MCPServerSse` (Server-Sent Events)
- `MCPServerStreamableHttp` (신형 transport, **특정 서버 알림 버그를 흡수하는** `_InitializedNotificationTolerantStreamableHTTPTransport` 포함)

공통 베이스 `_MCPServerWithClientSession`이 `mcp.ClientSession` 라이프사이클, anyio memory streams, 특정 에러에 대한 isolated session 재시도(`_SharedSessionRequestNeedsIsolation`)를 처리.

### 12.2 도구 통합

`MCPUtil` (`mcp/util.py`)이 MCP 도구 정의를 `FunctionTool`로 변환. `convert_schemas_to_strict` 옵션으로 strict 모드 best-effort 적용. 결과적으로 **MCP 도구가 일반 도구와 동일한 파이프라인** (guardrails, approval, failure handling, tracing)을 거친다.

### 12.3 승인 정책의 다양한 표현

```python
require_approval = "always"
require_approval = "never"
require_approval = {"always": {"tool_names": ["delete_file"]}, "never": {...}}
require_approval = {"delete_file": True, "list_files": False}
require_approval = lambda ctx, agent, tool: ...
```

`_normalize_needs_approval`이 모든 형태를 dict 또는 callable로 정규화. **사용자가 가장 편한 형태로 쓰게 하고, 내부에서 통일**하는 DX 패턴.

### 12.4 ToolFilter

`create_static_tool_filter(allowed_tool_names, blocked_tool_names)` 등으로 MCP 서버의 도구 중 일부만 노출 가능.

### 12.5 라이프사이클 — `MCPServerManager`

`mcp/manager.py`가 connect/cleanup을 같은 asyncio task에 묶어줌 (MCP 세션은 task-bound). **anyio + MCP 통합 시 흔한 함정**을 SDK 레벨에서 흡수.

> **시사점**: MCP 통합은 단순히 wire 호환만으로 끝나지 않는다. **현실의 다양한 MCP 서버 버그/quirk를 흡수하는 layer**가 필요. 자체 하네스에서도 transport-level patch 지점을 일찍 만들어두면 두고두고 도움이 된다.

---

## 13. 인터럽션 / HIL — RunState 직렬화

### 13.1 메커니즘

도구 호출 중 `needs_approval`이 트리거되면:
1. Runner는 `NextStepInterruption(interruptions=[ToolApprovalItem(...)])` 반환
2. `RunResult.state = result.to_state()` — 전체 실행 상태 직렬화
3. 사용자가 승인/거부 후 `Runner.run(agent, state)`로 재개

`RunState`는 `run_state.py`에 무려 **3,304 LOC**. 컨텍스트, 승인 상태, 트레이스 상태, 샌드박스 상태 모두 직렬화. `to_json()` / `from_json()` 지원.

### 13.2 중첩 승인 미러링

`agent.as_tool()`로 호출된 중첩 에이전트의 승인 상태가 **부모 컨텍스트와 동기화** (`_apply_nested_approvals`, `agent.py:658-722`). "Remember my choice"가 nesting을 거쳐도 의도대로 동작.

> **시사점**: HIL을 **사후 첨가물이 아니라 1급 시민**으로 설계. 채팅 UI에서 "이 도구 실행을 승인하시겠습니까?" 다이얼로그를 띄우고 다음 세션에서 재개하는 시나리오가 가능. 코딩 에이전트(예: 파일 쓰기, 외부 명령 실행)에 결정적인 패턴.

---

## 14. Voice / Realtime / Sandbox — 별도 트랙

### 14.1 VoicePipeline

```
STT → workflow (any async text generator) → TTS
```
(`voice/pipeline.py:15-200`)

`VoiceWorkflowBase`는 단순 async 이터레이터 contract. 텍스트 에이전트 위에 음성 layer를 얹는 구조.

### 14.2 Realtime

`RealtimeAgent`는 **별도의 단순화된 dataclass** (model, model_settings, output_type, tool_use_behavior 없음). `RealtimeRunner` + `RealtimeSession`이 `gpt-realtime-*` WebSocket을 양방향으로 운용. 이벤트 시스템(`realtime/events.py`)도 텍스트 에이전트의 `StreamEvent`와 다르다.

### 14.3 Sandbox (0.14.0 신규)

`src/agents/sandbox/`가 큰 디렉터리로 추가됨. `SandboxAgent`, `Manifest`, 그리고 Docker, E2B, Daytona, Modal, Runloop, Cloudflare, Dapr, Vercel, Blaxel 클라이언트들. 파일시스템 backed workspace에서 에이전트가 동작. **코어 루프 위에 layer로 얹은 구조** — 코어를 오염시키지 않음.

> **시사점**: 음성/실시간/샌드박스 같은 큰 기능은 **별도 추상화 트랙**으로 분리. 텍스트 에이전트 코어를 단순하게 유지.

---

## 15. 핵심 설계 철학 — 정리

1. **Python-first, few abstractions** — 단 6개 primitive (Agent, Runner, Handoff, Guardrail, Session, Tool). DSL/YAML 없음. docs/index.md 인용: "use the language you already know."
2. **Dataclass-heavy, Pydantic at boundaries** — 런타임 핫 패스는 dataclass + OpenAI SDK 타입. Pydantic은 (1) 도구 입력 검증, (2) 도구 스키마 생성, (3) LLM 구조화 출력 검증, (4) RunState 직렬화에만.
3. **Strict JSON schema by default** — `ensure_strict_json_schema()` 한 함수가 모든 정규화 책임. OpenAI Structured Outputs를 전제로 설계.
4. **Provider-agnostic via 접두사 라우팅** — MultiProvider + LiteLLM/any-llm. 코어는 Responses API 형식을 정규형으로 사용.
5. **Async-first, sync는 그저 `asyncio.run` wrapper** — 사용자는 sync 함수도 짤 수 있음 (`asyncio.to_thread` 자동). 그러나 진짜 async-aware.
6. **세션 vs 서버 대화 ID 상호 배타** — 두 영속화 모델을 명시적으로 분리, runtime enforce.
7. **Anti-bloat via optional extras** — MCP, Redis, SQLAlchemy, LiteLLM, sandbox 클라이언트들 전부 extras. Lazy import.
8. **가드레일은 별개 concern** — 도구 입력 검증(Pydantic)과 정책 검사(guardrail)를 명확히 분리.
9. **HIL은 1급 시민** — Approval 인터럽션, RunState 직렬화/재개 모두 코어 기능.
10. **Tracing-by-default, 끄기 쉬움** — 모든 turn/tool/guardrail/handoff에 자동 span. `OPENAI_AGENTS_DISABLE_TRACING=1`로 즉시 disable.
11. **`RunConfig`는 평면적** — 에이전트별 설정이 handoff로 자동 상속되지 않음. 글로벌 정책은 RunConfig에.
12. **`AgentRunner`는 명시적 experimental** — public 표면적을 의도적으로 작게 유지.

---

## 16. 하네스 팀이 즉시 차용할 수 있는 패턴 정리

opencode/Claude Code 같은 **인터랙티브 코딩 에이전트 하네스** 관점에서 직접 적용 가능한 항목들:

### 16.1 도구 시스템

- [ ] **Python 시그니처 → 엄격 JSON 스키마 자동 생성** (`@function_tool` 패턴): `inspect` + `pydantic.create_model` + 도큐스트링 파서
- [ ] **`Annotated[T, "desc"]` 인라인 설명** — 별도 schema 파일 필요 없음
- [ ] **첫 인자 컨텍스트 자동 주입 + 스키마에서 제외** — 사용자 코드를 깔끔하게
- [ ] **`ensure_strict_json_schema()` 단일 정규화 함수** — 모든 도구 등록 후 정규화
- [ ] **sync 함수 자동 to_thread offload** — 진입 장벽 낮춤
- [ ] **`is_enabled` callable** — 턴별 동적 도구 노출 (예: 권한별)
- [ ] **`needs_approval` per-call HIL** — 인자 검사 후 승인 요청 결정 가능
- [ ] **`timeout_behavior="error_as_result"`** — LLM이 timeout을 회복 가능한 에러로 인식
- [ ] **`failure_error_function`** — 도구 예외를 LLM-visible 에러 메시지로 변환

### 16.2 루프 구조

- [ ] **`NextStep*` Sum Type** — boolean flag 누덕누덕 대신 sum type + match
- [ ] **Two-phase processing**: classify (process_model_response) → execute (execute_tools_and_side_effects)
- [ ] **Tool 카테고리별 동시성 정책**: 순수 함수 도구는 병렬, 상태 변경(shell, patch) 도구는 직렬
- [ ] **`tool_use_behavior` declarative 종료 조건** — `stop_on_first_tool`, `StopAtTools`, custom function
- [ ] **`max_turns` + `RunErrorHandlers["max_turns"]` escape hatch**
- [ ] **부분 상태를 첨부하는 예외** (`RunErrorDetails`) — 디버깅/재시도 UX의 핵심

### 16.3 스트리밍

- [ ] **이중 계층 이벤트** (raw + semantic). 토큰 단위 + 의미 단위 동시 노출
- [ ] **`cancel(mode="immediate"|"after_turn")`** — UX-friendly 취소 의미 분리
- [ ] **Background task + asyncio.Queue** — 단순한 스트리밍 구현 패턴

### 16.4 세션 / 메모리

- [ ] **Protocol + 4개 메서드 미니멀 인터페이스** — ABC 강제 X
- [ ] **Wire format 저장** (SDK 내부 타입 X) — 마이그레이션 친화적
- [ ] **Session vs server-managed conversation 상호 배타** 명시
- [ ] **`pop_item()` 노출** — 부분 롤백/재시도 시나리오에 유용

### 16.5 Tracing

- [ ] **의미별 SpanData 서브타입** (Function/Generation/Guardrail/Handoff/MCP/...) — 일반화된 단일 Span 회피
- [ ] **`TracingProcessor` (6 메서드) + `TracingExporter` (1 메서드)** 작은 표면적
- [ ] **`BatchTraceProcessor` 기본 + `flush_traces()` for graceful shutdown**
- [ ] **`trace_include_sensitive_data` 분리** — 모양 vs 내용
- [ ] **ContextVar 기반 자동 nesting** — `with trace():`로 여러 run을 묶기

### 16.6 가드레일

- [ ] **Input(parallel/sequential) vs Output 분리**
- [ ] **Tool guardrail의 풍부한 return**: skip/replace/tripwire — boolean을 넘어
- [ ] **Tripwire 시 race 중인 모델 호출 cancel** — 토큰 비용 절감

### 16.7 Multi-agent

- [ ] **Handoff vs as_tool 명시적 이원화**
- [ ] **Handoff `input_filter`** — 다음 에이전트 시야 외과적 제어
- [ ] **`HandoffInputData` frozen + `.clone()`** — 일반적이고 유용한 패턴
- [ ] **중첩 승인 미러링** — 서브 호출에서도 HIL 일관성

### 16.8 Provider 추상화

- [ ] **접두사 기반 라우팅** (`MultiProvider`)
- [ ] **하나의 정규형 + 어댑터** (chatcmpl_converter)
- [ ] **Lazy import for optional providers**
- [ ] **`get_retry_advice()`** — provider별 재시도 힌트 인터페이스
- [ ] **자동 prompt cache key 해싱**, 사용자 override 존중

### 16.9 MCP

- [ ] **MCP 도구를 `FunctionTool`로 변환** — 단일 파이프라인 재사용
- [ ] **Approval 정책의 다양한 입력 형태 정규화** — DX
- [ ] **`MCPServerManager` task-binding**
- [ ] **Transport-level workaround 흡수 layer**
- [ ] **`ToolFilter`로 노출 도구 화이트/블랙리스트**

### 16.10 HIL

- [ ] **Approval interruption을 1급 시민으로**
- [ ] **`RunState` 직렬화/재개 (to_json/from_json)**
- [ ] **중첩 컨텍스트로 승인 상태 propagation**

### 16.11 패키징

- [ ] **코어는 ~7개 의존성으로 작게** + Optional extras 광범위
- [ ] **Lazy import everywhere** — 안 쓰는 기능은 import 비용 0
- [ ] **`AgentRunner` 같은 내부 클래스 명시적 experimental 마킹** — public 표면적 보호
- [ ] **다국어 docs (한/일/중)** — 글로벌 채택을 노린다면 차이를 만든다

---

## 17. 한계 / 비판적 관점

벤치마킹 시 무비판 수용을 막기 위한 메모:

1. **`run.py` 1,859 LOC, `run_state.py` 3,304 LOC** — 코어 루프가 결코 작지 않다. "few abstractions"라는 표방과 달리 실제 코드는 무겁다. 추상화는 적어도 케이스 처리는 많다는 뜻.
2. **`RunState` 직렬화의 복잡성** — HIL/재개를 1급으로 만든 대가. 대부분의 하네스는 이 정도까지 필요하지 않을 수 있음. 점진적 도입을 권장.
3. **OpenAI 종속성이 명확** — Multi-provider 표방하지만 정규형이 Responses API. Anthropic 네이티브 기능(예: extended thinking, prompt caching의 세부)에는 어댑터가 손해.
4. **사용량 제한이 SDK 차원에 없음** — 정책으로 위임. 멀티테넌트 SaaS 하네스라면 직접 구현 필요.
5. **Guardrail이 turn 0 + 첫 에이전트에만** 적용 — handoff 후 모델 입력에는 다시 검사하지 않음. 의도이긴 하나 운영 정책에 따라 한계.
6. **Voice/Realtime/Sandbox는 별도 추상화** — 텍스트 에이전트와 일관성이 부족. 통합 모델을 원한다면 추가 layer 필요.
7. **`AgentRunner` 교체 가능성을 막아둠** — 확장성보다 안정성을 택한 결정. 실험적 라우터/플래너를 끼우려면 fork 수준의 작업 필요.

---

## 18. 결론

OpenAI Agents Python SDK는 **"개발자 친화적인 코드 우선 + OpenAI 생태계 강결합 + HIL/Tracing/MCP 1급 시민화"**를 동시에 추구한 SDK이다. 코어 표면적을 일부러 작게 두고, 무거운 기능은 optional extras + lazy import로 분리한 패키징 전략은 인상적이다.

opencode/Claude Code 류 **인터랙티브 코딩 하네스** 팀이 가장 즉시 차용할 가치가 있는 것은:

1. **`@function_tool` 도구 등록 패턴** — Python 시그니처에서 strict JSON schema까지의 자동화
2. **`NextStep*` 기반 루프 상태머신** — 깔끔한 상태 전이
3. **이중 계층 스트리밍 이벤트** — UI 자유도 확보
4. **`Session` Protocol + wire-format 저장** — 백엔드 교체 자유
5. **의미별 SpanData 서브타입** — 운영 가능한 트레이싱
6. **`ensure_strict_json_schema()` 단일 정규화 + Pydantic을 boundary에서만 사용** — 핫 패스 가벼움 유지
7. **HIL approval + RunState 재개** — 코딩 에이전트의 destructive action 안전장치

반면 **무비판적으로 따라가지 않을 부분**은:

- Multi-provider의 정규형 선택 (각자의 wire format 전략)
- Guardrail을 첫 turn/첫 에이전트에만 적용하는 디자인
- HIL을 위한 거대한 RunState 직렬화 (점진적 도입)

자체 하네스의 도메인 (코딩 task, 인터랙티브 TUI, 멀티테넌트 등)에 따라 선별적으로 차용하면 좋다.

---

## 부록 A. 핵심 파일:라인 인덱스

> 실제 분석 시점의 v0.14.1 기준. 차용 시 코드 직접 확인 권장.

### 공개 진입점
- Runner: `src/agents/run.py:192-424`
- Agent dataclass: `src/agents/agent.py:223-322`
- `Agent.clone()`: `src/agents/agent.py:457-470`
- `Agent.as_tool()`: `src/agents/agent.py:472-900`
- `function_tool` 데코레이터: `src/agents/tool.py:1725-1870`
- `FunctionTool`: `src/agents/tool.py:281-416`
- 스키마 생성: `src/agents/function_schema.py:246-424`
- Strict 정규화: `src/agents/strict_schema.py:17-80`

### 루프 내부
- 메인 루프: `src/agents/run.py:750-1450`
- NextStep 타입: `src/agents/run_internal/run_steps.py:143-208`
- `run_single_turn`: `src/agents/run_internal/run_loop.py:1684-1770`
- `process_model_response`: `src/agents/run_internal/turn_resolution.py:1420-1860`
- `execute_tools_and_side_effects`: `src/agents/run_internal/turn_resolution.py:547-716`
- 병렬 도구 실행: `src/agents/run_internal/tool_execution.py:1873-1892`

### Handoff / Guardrail / Lifecycle
- Handoff: `src/agents/handoffs/__init__.py:42-335`
- Handoff filters: `src/agents/extensions/handoff_filters.py:33-108`
- Guardrails: `src/agents/guardrail.py:19-343`
- Hooks: `src/agents/lifecycle.py:13-199`

### Memory
- Protocol: `src/agents/memory/session.py:14-54`
- SQLite: `src/agents/memory/sqlite_session.py:17-331`
- OpenAI Conversations: `src/agents/memory/openai_conversations_session.py:23-126`

### Models
- ABC: `src/agents/models/interface.py:20-150`
- MultiProvider: `src/agents/models/multi_provider.py:60-249`
- ChatCompletion converter: `src/agents/models/chatcmpl_converter.py`
- LiteLLM: `src/agents/extensions/models/litellm_model.py`

### Streaming / Result
- Stream events: `src/agents/stream_events.py:10-62`
- RunItem 계층: `src/agents/items.py:90-625`
- RunResult: `src/agents/result.py:332-896`

### Tracing
- Processor/Exporter ABC: `src/agents/tracing/processor_interface.py:9-143`
- 빌트인 processors: `src/agents/tracing/processors.py:21-650`
- SpanData 서브타입: `src/agents/tracing/span_data.py:28-450`

### MCP
- ABC: `src/agents/mcp/server.py:223-478`
- 변환 유틸: `src/agents/mcp/util.py`
- Manager: `src/agents/mcp/manager.py`

### Context / State / Cache / Usage
- RunContextWrapper: `src/agents/run_context.py:42-120+`
- ToolContext: `src/agents/tool_context.py:35-50`
- RunState: `src/agents/run_state.py:183-320, 655, 1061`
- Prompt cache key: `src/agents/run_internal/prompt_cache_key.py:17-130`
- Usage: `src/agents/usage.py:60-319`

### Config / 예외
- RunConfig: `src/agents/run_config.py:140-303`
- Public API: `src/agents/__init__.py:1-538`
- 예외: `src/agents/exceptions.py:1-154`

### 추천 예제
- Customer service + handoff: `examples/customer_service/main.py`
- Agents-as-tools: `examples/agent_patterns/agents_as_tools.py`
- HIL/approval/state 직렬화: `examples/agent_patterns/human_in_the_loop.py`
- 스트리밍 시맨틱 이벤트: `examples/basic/stream_items.py`
- Manager 패턴: `examples/research_bot/manager.py`
