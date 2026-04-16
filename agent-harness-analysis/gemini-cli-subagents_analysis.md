# Gemini CLI 서브에이전트 심층 분석

> **분석 대상**: [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) (TypeScript 모노레포)
>
> **분석 관점**: opencode, Claude Code, OpenAI Agents SDK 같은 에이전트 하네스를 만드는 팀의 시각에서, Gemini CLI가 새로 출시한 **서브에이전트 멀티에이전트 아키텍처**의 실제 구현을 코드 레벨로 파헤치고, 설계 결정 / 강점 / 약점 / 차용 가능 패턴을 정리한다.
>
> **작성일**: 2026-04-16

---

## 0. TL;DR — 한 페이지 요약

Gemini CLI 서브에이전트의 핵심 디자인은 다음 5가지로 압축된다.

1. **단 하나의 `invoke_agent` 디스패처 도구** — `Agent.as_tool()` (OpenAI Agents SDK)처럼 에이전트마다 별개 함수 선언을 만들지 않고, **공용 함수 1개 + `agent_name` 인자**로 모든 서브에이전트를 호출. LLM에게는 시스템 프롬프트의 `<available_subagents>` XML 리스트로 노출.
2. **격리된 레지스트리 인스턴스 + 공유 ContentGenerator** — 호출마다 `ToolRegistry/PromptRegistry/ResourceRegistry`를 새로 만들고 `MessageBus.derive(name)`으로 서브 메시지 버스 파생. 단, 네트워크 레이어 (`GeminiClient`)는 공유 → 글로벌 rate-limit/auth 일관성.
3. **재귀 금지 (1단계 위임)** — 서브에이전트의 도구 화이트리스트에서 `kind === Kind.Agent`인 도구를 명시적으로 제외. 무한 위임 footgun 차단.
4. **`complete_task` 종료 프로토콜 + 60초 grace period** — 서브에이전트가 텍스트로 끝나는 게 아니라 **반드시** `complete_task` 도구를 호출해야 정상 종료. 안 부르면 `ERROR_NO_COMPLETE_TASK_CALL` → 60초 grace period로 "마지막 한 번의 기회" 부여.
5. **병렬은 스케줄러 레벨, 에이전트 레벨이 아님** — 메인 에이전트의 `Scheduler`가 `Promise.all`로 연속된 parallelizable 도구 호출을 묶어 실행. `invoke_agent`도 일반 도구이므로 자동으로 병렬화. 단, 동시성 상한이나 워커 풀은 없다.

이 외에도 **마크다운 + YAML frontmatter 파일 포맷**, **프로젝트 에이전트의 sha256 acknowledgement 보안 게이트**, **workspace directory의 AsyncLocalStorage 기반 추가 스코핑**, **OTel GenAI 시맨틱 컨벤션 준수 트레이싱** 등은 자체 하네스 설계에 즉시 차용 가능한 패턴들이다.

---

## 1. 코드 위치 및 구조

### 1.1 핵심 디렉터리

서브에이전트 코드는 거의 전부 `core` 패키지에 있고, UI 부분만 `cli` 패키지에 있다.

```
packages/core/src/agents/                  # 핵심 (~30 파일)
├── types.ts                                # AgentDefinition, AgentTerminateMode 등
├── agentLoader.ts                          # 마크다운 + YAML 파서, zod 스키마
├── registry.ts                             # AgentRegistry (716 LOC)
├── agent-tool.ts                           # invoke_agent 도구 + 디스패처
├── local-executor.ts                       # LocalAgentExecutor (메인 루프)
├── local-invocation.ts                     # 부모 → 자식 결과 변환
├── remote-invocation.ts                    # A2A 원격 에이전트
├── agent-scheduler.ts                      # 서브에이전트 도구 호출 스케줄링
├── generalist-agent.ts                     # 빌트인: generalist
├── cli-help-agent.ts                       # 빌트인: cli_help
├── codebase-investigator.ts                # 빌트인: codebase_investigator
├── memory-manager-agent.ts                 # 빌트인: save_memory
├── browser/browserAgentDefinition.ts       # 빌트인: browser_agent
└── acknowledgedAgents.ts                   # 프로젝트 에이전트 승인 캐시

packages/core/src/scheduler/scheduler.ts    # 도구 호출 스케줄러 (병렬 실행)
packages/core/src/tools/complete-task.ts    # 종료용 필수 도구
packages/core/src/tools/tool-names.ts       # AGENT_TOOL_NAME = 'invoke_agent'
packages/core/src/prompts/snippets.ts       # renderSubAgents (시스템 프롬프트 주입)
packages/core/src/config/agent-loop-context.ts  # AgentLoopContext

packages/cli/src/ui/commands/agentsCommand.ts   # /agents 슬래시 커맨드
packages/cli/src/ui/hooks/atCommandProcessor.ts # @agent 파서
packages/cli/src/ui/hooks/useAgentStream.ts     # 스트리밍 훅
packages/cli/src/ui/components/messages/Subagent*  # 진행 상황 UI
```

### 1.2 외부 노출 포인트

- **메인 에이전트의 도구 레지스트리에 등록되는 단 하나의 도구**: `AgentTool` (`packages/core/src/config/config.ts:3643-3646`)
- **시스템 프롬프트 템플릿 변수**: `${SubAgents}` (`packages/core/src/prompts/utils.ts:75-85`)
- **사용자 명령**: `/agents`, `@agent`

---

## 2. 핵심 추상화

### 2.1 AgentDefinition — local vs remote

`packages/core/src/agents/types.ts:189-255`. discriminated union.

```ts
export interface BaseAgentDefinition<TOutput extends z.ZodTypeAny = z.ZodUnknown> {
  name: string;
  displayName?: string;
  description: string;
  experimental?: boolean;
  inputConfig: InputConfig;
  outputConfig?: OutputConfig<TOutput>;
  metadata?: { hash?: string; filePath?: string };
}

export interface LocalAgentDefinition<TOutput> extends BaseAgentDefinition<TOutput> {
  kind: 'local';
  promptConfig: PromptConfig;          // systemPrompt + initialMessages + query
  modelConfig: ModelConfig;            // model (or 'inherit'), temperature, thinking, ...
  runConfig: RunConfig;                // maxTurns, maxTimeMinutes
  toolConfig?: ToolConfig;             // 도구 화이트리스트 (이름/인스턴스/선언)
  workspaceDirectories?: string[];     // 추가 작업 디렉터리 (예: ~/.gemini)
  mcpServers?: Record<string, MCPServerConfig>;
  processOutput?: (output: z.infer<TOutput>) => string;
  onBeforeTurn?: (chat: GeminiChat, signal?: AbortSignal) => Promise<void> | void;
}
```

기본값:
```ts
DEFAULT_QUERY_STRING = 'Get Started!'
DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_TIME_MINUTES = 10
```

종료 모드:
```ts
export enum AgentTerminateMode {
  ERROR, TIMEOUT, GOAL, MAX_TURNS, ABORTED, ERROR_NO_COMPLETE_TASK_CALL
}
```

부모에게 반환되는 결과 객체:
```ts
export interface OutputObject {
  result: string;
  terminate_reason: AgentTerminateMode;
}
```

> **포인트**: 두 가지 종료 모드(`GOAL`, `ERROR_NO_COMPLETE_TASK_CALL`)가 상징하듯, **"종료" 자체가 별도 도구 호출 이벤트**다. 단순 텍스트 응답으로 끝나는 통상적 LLM 패턴과 결을 달리한다.

### 2.2 마크다운 + YAML frontmatter 파일 포맷

zod 스키마 (`agentLoader.ts:87-111`):

```ts
const localAgentSchema = z.object({
  kind: z.literal('local').optional().default('local'),
  name: z.string().regex(/^[a-z0-9-_]+$/, 'Name must be a valid slug'),
  description: z.string().min(1),
  display_name: z.string().optional(),
  tools: z.array(
    z.string().refine(v => isValidToolName(v, { allowWildcards: true }))
  ).optional(),
  mcp_servers: z.record(mcpServerSchema).optional(),
  model: z.string().optional(),                  // 미지정 시 'inherit'
  temperature: z.number().optional(),
  max_turns: z.number().int().positive().optional(),
  timeout_mins: z.number().int().positive().optional(),
}).strict();
```

파일 형식:
```markdown
---
name: frontend-specialist
description: Senior Frontend Specialist for UI/UX review
tools: [read_file, grep_search, glob, list_directory, web_fetch, google_web_search]
model: inherit
---

You are a Senior Frontend Specialist...

## Core Principles
- Modular architecture
- Core Web Vitals based performance optimization
- WCAG 2.1+ accessibility compliance
...
```

frontmatter 위는 **YAML 검증 + zod 매핑**, frontmatter 아래 마크다운 본문은 **그대로 시스템 프롬프트**가 된다 (`agentLoader.ts:394-401`).

원격(A2A) 에이전트는 별도 frontmatter 스키마(`agent_card_url` XOR `agent_card_json`, OAuth/Bearer/Basic 인증 옵션)를 가진다 (`agentLoader.ts:120-194`).

### 2.3 도구 와일드카드 표현

frontmatter `tools:` 필드는 단순 이름 목록이 아니라:
- `"read_file"` — 정확한 도구 이름
- `"*"` — 모든 부모 도구 복사
- `"mcp__*"` — 모든 MCP 도구
- `"mcp__<server>__*"` — 특정 MCP 서버의 모든 도구
- `FunctionDeclaration` 객체 — 모델 선언만 (실행 불가)

이 와일드카드 처리는 executor에서 수행 (`local-executor.ts:185-255`).

### 2.4 모델 'inherit'

`model: inherit` 또는 미지정 시 (`agentLoader.ts:532`), registry가 현재 메인 모델을 동적으로 조회 (`registry.ts:651-655`):

```ts
let model = modelConfig.model;
if (model === 'inherit') {
  model = this.config.getModel();
}
```

또한 `CoreEvent.ModelChanged` 이벤트에 reattach해서 사용자가 `/model`로 메인 모델을 바꾸면 `inherit` 에이전트도 자동 reconcile (`registry.ts:67-79`).

### 2.5 `query` 템플릿

frontmatter나 옵션에서 `query` 문자열은 `AgentInputs`로 `${var}` 보간된다. 빌트인의 경우:
- `generalist`: `query: '${request}'`
- `cli_help`: `query: '${question}'`
- `codebase_investigator`: `query: '${objective}'`

또 자동으로 inject되는 변수 (`local-executor.ts:582-587`):
- `cliVersion`
- `activeModel`
- `today`

→ `cli_help` 에이전트의 시스템 프롬프트가 `${cliVersion}`, `${activeModel}`, `${today}`를 참조하는 이유.

---

## 3. 로더 / 레지스트리 / 우선순위

### 3.1 로더 (`agentLoader.ts:620-677`)

```ts
export async function loadAgentsFromDirectory(dir: string): Promise<AgentLoadResult> {
  const dirEntries = await fs.readdir(dir, { withFileTypes: true });
  const files = dirEntries.filter(e => e.isFile() && !e.name.startsWith('_') && e.name.endsWith('.md'));
  for (const entry of files) {
    const content = await fs.readFile(path.join(dir, entry.name), 'utf-8');
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    const agentDefs = await parseAgentMarkdown(filePath, content);
    for (const def of agentDefs)
      result.agents.push(markdownToAgentDefinition(def, { hash, filePath }));
  }
  return result;
}
```

특징:
- **Shallow 스캔** (재귀 X)
- `_`로 시작하는 파일 무시
- `.md` 확장자만
- 파일별 sha256 해시 계산 (acknowledgement 검증용)
- 파일 단위 에러는 `result.errors`에 누적, 전체 스캔은 abort하지 않음

### 3.2 우선순위 (`registry.ts:117-263`)

| 순위 | 출처 | 조건 |
|------|------|------|
| 0 | 빌트인 (TypeScript) | 항상 |
| 1 | `~/.gemini/agents/*.md` | 항상 |
| 2 | `.gemini/agents/*.md` (프로젝트) | `!folderTrust \|\| isTrustedFolder` |
| 3 | Extension `agents/` | active 확장만 |

**같은 `name`이면 뒤에 등록되는 게 덮어쓴다** (`agents` Map). 순서: `extension > project > user > built-in`.

`allDefinitions` Map은 **disable된 에이전트까지 포함한 전체 발견 목록**으로, `/agents config <name>` 등에서 사용된다.

### 3.3 프로젝트 에이전트 acknowledgement (보안 게이트)

`registry.ts:169-211`. 프로젝트의 `.gemini/agents/*.md`는 PR을 통해 임의 시스템 프롬프트를 들여올 수 있어 위험하다. 이를 방지하기 위해:

1. 로더가 파일 sha256을 계산 → AgentDefinition.metadata.hash에 저장
2. `AcknowledgedAgentsService`가 `(projectRoot, agentName, hash)` 튜플을 영속화
3. 미승인 에이전트가 있으면 `coreEvents.emitAgentsDiscovered()` 발화 → UI에서 `NewAgentsNotification` + `AgentConfigDialog` 표시
4. 사용자가 승인하면 `acknowledgeAgent()`가 해시 저장 + `registerAgent()` 호출

**파일이 수정되면 해시가 바뀌어 acknowledgement 무효화** → 재승인 요구. 공급망 공격에 대한 명시적 방어선.

원격 에이전트의 경우 (`registry.ts:177-189`), `agentCardUrl` 자체나 `sha256(agentCardJson)`이 해시로 사용됨.

### 3.4 핫 리로드

`agentRegistry.reload()`가 진입점 (`registry.ts:84-91`):
- A2A 클라이언트 캐시 클리어
- `config.reloadAgents()` 호출
- 전체 디렉터리 재스캔
- `coreEvents.emitAgentsRefreshed()` 발화

트리거:
- `/agents reload`
- `/agents enable|disable`
- Acknowledgement flow
- `CoreEvent.ModelChanged` (단, 이때는 `local` 에이전트만 — `inherit` 재바인딩 목적)

### 3.5 settings 기반 오버라이드

`config.getAgentsSettings().overrides[agentName]`로 frontmatter를 변경하지 않고도 다음 항목들을 덮어쓸 수 있다:
- `enabled`
- `runConfig`
- `modelConfig`
- `tools`
- `mcpServers`

오버라이드는 getter로 lazy-merge되므로 (`registry.ts:574-646`), 향후 모델 변경도 `inherit` 경로로 흘러들어옴.

---

## 4. 실행 모델 — 호출에서 결과까지

### 4.1 메인 LLM이 서브에이전트를 보는 방식

**두 채널**:

#### 채널 1: 시스템 프롬프트 주입

`prompts/snippets.ts:252-290`의 `renderSubAgents`가 활성 서브에이전트들을 XML 리스트로 시스템 프롬프트에 추가:

```ts
export function renderSubAgents(subAgents?: SubAgentOptions[]): string {
  if (!subAgents || subAgents.length === 0) return '';
  const subAgentsXml = subAgents
    .map(a => `  <subagent>\n    <name>${a.name}</name>\n    <description>${a.description}</description>\n  </subagent>`)
    .join('\n');
  return `
# Available Sub-Agents
Sub-agents are specialized expert agents. You can invoke them using the ${formatToolName(AGENT_TOOL_NAME)} tool ...

**Concurrency Safety and Mandate:** You should NEVER run multiple subagents in a single turn if their abilities mutate the same files or resources. ...

<available_subagents>
${subAgentsXml}
</available_subagents>
...`;
}
```

`prompts/utils.ts:75-85`에서 시스템 프롬프트 템플릿의 `${SubAgents}` 변수가 이 XML 블록으로 치환된다.

#### 채널 2: 단 하나의 `invoke_agent` 도구

`agent-tool.ts:50-73`:

```ts
super(
  AGENT_TOOL_NAME,           // 'invoke_agent'
  'Invoke Subagent',
  'Invoke a subagent to perform a specific task or investigation.',
  Kind.Agent,
  {
    type: 'object',
    properties: {
      agent_name: { type: 'string', description: 'Name of the subagent to invoke' },
      prompt:     {
        type: 'string',
        description: 'The COMPLETE query to send the subagent. MUST be comprehensive and detailed. ' +
                     'Include all context, background, questions, and expected output format. ' +
                     'Do NOT send brief or incomplete instructions.',
      },
    },
    required: ['agent_name', 'prompt'],
  },
  messageBus,
  true, true,  // isOutputMarkdown, canUpdateOutput
);
```

> **결정적 디자인 차이**: OpenAI Agents SDK의 `Agent.as_tool()`은 에이전트마다 별개 함수 선언을 만든다. Gemini CLI는 **단 하나의 디스패처** + `agent_name` 인자. 시스템 프롬프트의 description으로 LLM이 라우팅하는 방식.
>
> 장점: 함수 선언이 적어 토큰 절약, 동적 추가/제거 용이
> 단점: 입력 스키마가 단일 `prompt` 문자열로 약화

### 4.2 "스마트 파라미터 매퍼"

LLM이 항상 `prompt` 키로 보내지만, 실제 서브에이전트의 input schema는 `objective`, `request`, `question` 같은 다른 키일 수 있다. `agent-tool.ts:106-119`:

```ts
private mapParams(prompt: string, schema: unknown): AgentInputs {
  const properties = schema['properties'];
  if (isRecord(properties)) {
    const keys = Object.keys(properties);
    if (keys.length === 1) return { [keys[0]]: prompt };
  }
  return { prompt };
}
```

**단일 프로퍼티 케이스에 한해서만** 자동 매핑. 다중 입력 에이전트는 LLM이 직접 구조화 인자를 보내야 함 — 잠재적 발 걸림돌.

### 4.3 부모 → 자식 디스패치

메인 에이전트가 `invoke_agent`를 호출하면 (`agent-tool.ts:76-213`):

1. `AgentTool.createInvocation()` → `definition = registry.getDefinition(agent_name)` 조회 (없으면 throw)
2. `DelegateInvocation.execute()`가 `definition.name + kind`로 분기:
   - `browser_agent` → `BrowserAgentInvocation` (도구가 동적으로 결정됨)
   - `kind === 'remote'` → `RemoteAgentInvocation` (A2A 클라이언트)
   - 그 외 → `LocalSubagentInvocation`
3. `runInDevTraceSpan({ operation: AgentCall, attributes: { GEN_AI_AGENT_NAME, GEN_AI_AGENT_DESCRIPTION } })`로 OTel GenAI 컨벤션 트레이싱

### 4.4 LocalAgentExecutor — 내부 루프

핵심 파일: `local-executor.ts`.

#### 4.4.1 셋업 단계 (`create`, L152-281)

```ts
static async create<TOutput>(definition, context, onActivity) {
  // 1. 서브 메시지 버스 파생
  const subagentMessageBus = context.messageBus.derive(definition.name);

  // 2. 격리된 레지스트리 신규 생성
  const agentToolRegistry = new ToolRegistry(context.config, subagentMessageBus);
  const agentPromptRegistry = new PromptRegistry();
  const agentResourceRegistry = new ResourceRegistry();

  // 3. 에이전트 전용 MCP 서버 검색 → 격리된 레지스트리에만 등록
  if (definition.mcpServers) {
    for (const [n, c] of Object.entries(definition.mcpServers))
      await globalMcpManager.maybeDiscoverMcpServer(n, c, {
        toolRegistry: agentToolRegistry,
        promptRegistry: agentPromptRegistry,
        resourceRegistry: agentResourceRegistry,
      });
  }

  // 4. 도구 화이트리스트 복사 — Kind.Agent는 명시적으로 제외 (재귀 금지)
  // ... toolConfig 처리 ...

  // 5. 필수 complete_task 도구 등록
  agentToolRegistry.registerTool(new CompleteTaskTool(
    subagentMessageBus, definition.outputConfig, definition.processOutput
  ));

  return new LocalAgentExecutor(definition, context, agentToolRegistry, ...);
}
```

**핵심 결정**:
- 매 호출마다 **fresh ToolRegistry/PromptRegistry/ResourceRegistry**. 격리 보장.
- `MessageBus.derive(name)`: 모든 confirmation 요청/activity가 서브에이전트 이름으로 태깅됨.
- **`kind === Kind.Agent` 도구는 화이트리스트에서 제외** (`local-executor.ts:187-190`) → 서브에이전트가 다른 서브에이전트를 호출할 수 없음. **무한 위임 방지**.
- MCP 서버는 에이전트 종료 시 `globalMcpManager.removeRegistries(...)`로 정리 (L701-707).

#### 4.4.2 채팅 객체 (`createChatObject`, L1001-1044)

```ts
const chat = new GeminiChat(
  this.executionContext,         // 에이전트 전용 레지스트리를 가진 새 AgentLoopContext
  systemInstruction,             // buildSystemPrompt(inputs)
  [{ functionDeclarations: tools }],
  startHistory,                  // 템플릿 적용된 initialMessages
  undefined, undefined,
);
await chat.initialize(undefined, 'subagent');
```

**각 서브에이전트 호출은 자체 `GeminiChat` 인스턴스를 가진다**. 메인 에이전트와 같은 `GeminiChat` 클래스이지만 fresh history와 자체 `ChatRecordingService`를 가진다. 단, `ContentGenerator` (HTTP/SSE 클라이언트)는 `config.geminiClient`로 **공유** — 글로벌 rate-limit과 auth 일관성을 위함.

`chat.initialize(undefined, 'subagent')`로 recording service가 메인과 서브 세션을 구별 가능.

#### 4.4.3 시스템 프롬프트 합성 (`buildSystemPrompt`, L1311-1368)

```ts
let finalPrompt = templateString(promptConfig.systemPrompt, inputs);
// + skills block (ACTIVATE_SKILL_TOOL_NAME 사용 가능 시)
// + user memory
// + environment context (cwd + folder structure)
// + non-interactive rules
// + mandatory complete_task instructions
```

마지막에 강제로 추가되는 "Important Rules" 블록:

> * You are running in a non-interactive mode. You CANNOT ask the user for input or clarification.
> * Work systematically using available tools to complete your task.
> * Always use absolute paths for file operations.
> * If a tool call is rejected by the user, acknowledge... rethink... do NOT retry.
> * When you have completed your task, you MUST call `complete_task` ...

이 strict 가이드라인은 **사용자가 작성한 시스템 프롬프트를 신뢰하지 않고 fallback 정책을 강제**하는 형태. 보안과 일관성에 유리.

#### 4.4.4 메인 루프 (`runInternal`, L543-859)

```ts
while (true) {
  if (turnCounter >= maxTurns) { terminate = MAX_TURNS; break; }
  if (combinedSignal.aborted)  { terminate = (deadline ? TIMEOUT : ABORTED); break; }

  const turnResult = await executeTurn(
    chat, currentMessage, turnCounter++, combinedSignal, deadline,
    onWaitingForConfirmation
  );

  if (turnResult.status === 'stop') {
    terminate = turnResult.terminateReason;
    finalResult = turnResult.finalResult;
    break;
  }

  currentMessage = turnResult.nextMessage;
  // user_steering 및 background_completion 큐 드레인
}
```

각 턴 (`executeTurn`, L317-396):

1. `tryCompressChat(chat, promptId, signal)` — `ChatCompressionService`로 컨텍스트 압축. `COMPRESSION_FAILED_INFLATED_TOKEN_COUNT` 감지 시 `hasFailedCompressionAttempt`로 재시도 회피.
2. (옵션) `definition.onBeforeTurn(chat, signal)` — 정의가 history 변형 가능. 브라우저 에이전트가 `supersedeStaleSnapshots`로 오래된 스크린샷 제거에 사용.
3. `callModel` (L903-998) — `chat.sendMessageStream({ model, overrideScope: definition.name }, ...)` 스트리밍. `overrideScope`로 model config service에 에이전트별 오버라이드 적용.
4. 모델이 함수 호출을 **하나도** emit 안 하면 → `ERROR_NO_COMPLETE_TASK_CALL`.
5. `processFunctionCalls` (L1051-1283) — 인자 파싱, 화이트리스트 검증 (`allowedToolNames = toolRegistry.getAllToolNames()`), `scheduleAgentTools(...)`로 위임.
6. `complete_task` 응답이 `data.taskCompleted === true` + `data.submittedOutput: string` 포함하면 (L1192-1205) 루프 종료, `terminate = GOAL`.

#### 4.4.5 모델 라우팅 (L923-946)

`modelConfig.model`이 `auto`로 해결되면 메인 에이전트와 같은 `config.getModelRouterService().route(routingContext)`를 호출. **각 서브에이전트가 턴마다 다른 모델로 동적으로 land 가능**. 다른 SDK에서는 보기 드문 기능.

#### 4.4.6 Grace Period 회복 (L430-516, L714-761)

`TIMEOUT`, `MAX_TURNS`, `ERROR_NO_COMPLETE_TASK_CALL`로 종료될 때, `executeFinalWarningTurn`이 **60초 grace period** (`GRACE_PERIOD_MS`, L90)와 함께 하드코딩된 회복 메시지를 발송:

> "You have exceeded the time limit. You have one final chance to complete the task with a short grace period. You MUST call `complete_task` immediately with your best answer and explain that your investigation was interrupted. Do not call any other tools."

회복 성공하면 `terminate = GOAL`로 업그레이드. `RecoveryAttemptEvent`로 로깅.

> **시사점**: 하드 종료 직전에 "마지막 한 번의 기회"를 부여하는 패턴은 비동기 시스템의 graceful degradation 패턴을 LLM 컨텍스트에 적용한 인상적인 예. 자체 하네스에서도 차용 가치 큼.

#### 4.4.7 워크스페이스 디렉터리 추가 스코핑 (L525-541)

`definition.workspaceDirectories`가 있으면 전체 실행을 `runWithScopedWorkspaceContext`로 감싼다:

```ts
runWithScopedWorkspaceContext(
  createScopedWorkspaceContext(parent, dirs),
  () => runInternal(...)
)
```

**AsyncLocalStorage 기반**이므로 `config.getWorkspaceContext()`가 서브에이전트의 도구 안에서 확장된 컨텍스트를 반환. **공유 `Config`를 mutate하지 않으면서 에이전트별 추가 디렉터리 접근**을 부여하는 깔끔한 방식.

`save_memory` 에이전트가 이 메커니즘으로 `~/.gemini/GEMINI.md`에 접근.

### 4.5 결과를 부모에게 반환

`local-invocation.ts:108-381`. Executor의 `OutputObject`를 `ToolResult`로 wrap:

```ts
const resultContent = `Subagent '${this.definition.name}' finished.
Termination Reason: ${output.terminate_reason}
Result:
${output.result}`;
return {
  llmContent: [{ text: resultContent }],
  returnDisplay: progress,           // SubagentProgress 객체 (UI용)
  data: { agentId: executor.agentId },
};
```

**부모의 function-response part는 텍스트 한 블록**. 서브에이전트의 전체 루프가 부모 history에서 단 하나의 도구 반환으로 collapse. 시스템 프롬프트 (snippets.ts:271)의 광고 그대로:

> "When you delegate, the sub-agent's entire execution is consolidated into a single summary in your history, keeping your main loop lean."

---

## 5. 병렬 실행 — 스케줄러 레벨

### 5.1 자동 병렬화

병렬화는 **에이전트 레벨이 아니라 도구 호출 스케줄러 레벨**에서 일어난다. 메인 에이전트의 LLM이 한 턴에 여러 `invoke_agent` 함수 호출을 emit하면 메인의 `Scheduler` (`packages/core/src/scheduler/scheduler.ts`)가 자동으로 `Promise.all`로 묶는다.

**병렬화 게이트** (`scheduler.ts:537-547`):

```ts
private _isParallelizable(request: ToolCallRequestInfo): boolean {
  if (request.args) {
    const wait = request.args['wait_for_previous'];
    if (typeof wait === 'boolean') return !wait;
  }
  return true;  // 기본은 병렬
}
```

**배칭 + 팬아웃** (`scheduler.ts:448-495`):

```ts
if (this._isParallelizable(next.request)) {
  while (this.state.queueLength > 0) {
    const peeked = this.state.peekQueue();
    if (peeked && this._isParallelizable(peeked.request)) this.state.dequeue();
    else break;
  }
}
// ...
if (validatingCalls.length > 0)
  await Promise.all(validatingCalls.map(c => this._processValidatingCall(c, signal)));
// ...
if (allReady && scheduledCalls.length > 0) {
  const execResults = await Promise.all(scheduledCalls.map(c => this._execute(c, signal)));
}
```

메인 에이전트가 `[invoke_agent(A), invoke_agent(B), invoke_agent(C)]`를 emit하면 세 `LocalAgentExecutor.run()` Promise가 동시에 race — 각자 자체 `ToolRegistry`, `PromptRegistry`, `ResourceRegistry`, `MessageBus`, `GeminiChat`을 가짐.

`scheduler_parallel.test.ts:475-510`의 `"should execute [Agent, Agent, Sequential, Parallelizable] in three waves"` 테스트가 이를 확인: 두 에이전트 호출이 wave 1에서, 순차 쓰기가 wave 2에서, 읽기가 wave 3에서 실행.

### 5.2 동시성 제한 — 없음

**명시적 세마포어/워커 풀이 없다**. 병렬도는 모델이 한 턴에 emit하는 함수 호출 수로만 제한된다. 시스템 프롬프트(snippets.ts:273)가 **모델에게** 명시적으로 경고:

> "You should NEVER run multiple subagents in a single turn if their abilities mutate the same files or resources."

> **한계**: 멀티테넌트 SaaS나 글로벌 quota 강제가 필요한 환경에선 이 디자인은 부족. 자체 하네스라면 worker pool/semaphore 추가가 필요할 가능성 높음.

### 5.3 취소 시멘틱

각 서브에이전트는 `combinedSignal = AbortSignal.any([externalSignal, deadlineTimer.signal])` 수신 (`local-executor.ts:571`).
- Ctrl+C → external signal → 전파
- 데드라인 도달 → deadline timer signal → 전파
- 어느 쪽이든 `AgentTerminateMode.ABORTED`, invocation이 `AbortError` re-throw (`local-invocation.ts:285-300`)

### 5.4 타임아웃 + 사용자 확인 일시정지

`maxTimeMinutes` (기본 10분)는 `DeadlineTimer` (L556-559)로 강제. 매우 흥미로운 디테일 (L562-568):

```ts
const onWaitingForConfirmation = (waiting: boolean) => {
  if (waiting) deadlineTimer.pause();
  else         deadlineTimer.resume();
};
```

**서브에이전트가 도구 호출 confirmation을 기다리는 동안에는 deadline timer가 pause**. 사용자가 고민하는 시간이 에이전트 클럭을 갉아먹지 않게 하는 UX 배려.

> **시사점**: 인터랙티브 하네스에서 timeout 정책 설계 시 자주 놓치는 디테일. 차용 가치 큼.

---

## 6. 라우팅 & 명시적 호출

### 6.1 자동 라우팅

오케스트레이터 LLM이 시스템 프롬프트의 description text를 보고 어떤 서브에이전트를 호출할지 결정한다. `prompts/utils.ts:75-85`:

```ts
const subAgentsContent = activeSnippets.renderSubAgents(
  context.config.getAgentRegistry().getAllDefinitions()
    .map((d) => ({ name: d.name, description: d.description })),
);
result = result.replace(/\${SubAgents}/g, subAgentsContent);
```

라우팅 정확도는 description의 품질에 크게 의존. 빌트인 에이전트들의 description이 **"Use this for X. Excellent for: A, B, C"** 형식인 이유.

### 6.2 `@agent` 멘션 파싱

`packages/cli/src/ui/hooks/atCommandProcessor.ts:141-173`. `@agent`를 `@file`/`@resource`와 **구문적으로 구별하지 않고**, 토큰화 후 어떤 레지스트리가 인식하는지로 판단:

```ts
const name = part.content.substring(1);
if (agentRegistry?.getDefinition(name))      agentParts.push(part);
else if (resourceRegistry.findResourceByUri(name)) resourceParts.push(part);
else                                          fileParts.push(part);
```

에이전트 멘션이 발견되면 사용자 쿼리 끝에 `<system_note>` 가이드를 append (`atCommandProcessor.ts:702-708`):

```ts
const agentNudge = `\n<system_note>\nThe user has explicitly selected the following agent(s): ${agentNames.join(', ')}. Please use the following tool(s) to delegate the task: ${toolsList}.\n</system_note>\n`;
```

**short-circuit 아님** — 메인 오케스트레이터가 여전히 호출 여부/방법을 결정. `@agent`는 강한 힌트일 뿐.

탭 자동완성은 `buildAgentCandidates` (`useAtCompletion.ts:142-152`)로 활성 에이전트 나열.

### 6.3 `/agents` 슬래시 커맨드

`packages/cli/src/ui/commands/agentsCommand.ts`. 서브커맨드:

| 커맨드 | 동작 |
|--------|------|
| `/agents list` (기본, autoExecute) | `HistoryItemAgentsList` UI 항목 추가 — 모든 정의 나열 |
| `/agents reload` (alt: `refresh`) | `agentRegistry.reload()` — 전체 재스캔 + A2A 캐시 클리어 |
| `/agents enable <name>` | `agents.overrides[name].enabled = true` 설정 → `reload()` |
| `/agents disable <name>` | 위와 같지만 `false` |
| `/agents config <name>` | `AgentConfigDialog` 열어 정의 인스펙트/오버라이드 |

Enable/disable는 `SettingScope.Workspace` 또는 `SettingScope.User`로 영속화.

---

## 7. 빌트인 에이전트들

### 7.1 generalist (`generalist-agent.ts:20-68`)

```ts
export const GeneralistAgent = (context) => ({
  kind: 'local',
  name: 'generalist',
  displayName: 'Generalist Agent',
  description: 'A general-purpose AI agent with access to all tools. ' +
    'Highly recommended for tasks that are turn-intensive or involve processing large amounts of data. ' +
    'Use this to keep the main session history lean and efficient. ' +
    'Excellent for: batch refactoring/error fixing across multiple files, ' +
    'running commands with high-volume output, and speculative investigations.',
  inputConfig: { inputSchema: { type: 'object', properties: { request: { type: 'string' } }, required: ['request'] } },
  outputConfig: { outputName: 'result', schema: z.object({ response: z.string() }) },
  modelConfig: { model: 'inherit' },
  get toolConfig() { return { tools: context.toolRegistry.getAllToolNames() }; },
  get promptConfig() {
    return {
      systemPrompt: getCoreSystemPrompt(context.config, undefined, /*interactiveOverride=*/ false),
      query: '${request}',
    };
  },
  runConfig: { maxTimeMinutes: 10, maxTurns: 20 },
});
```

**핵심**: 메인 에이전트와 **동일한 `getCoreSystemPrompt`** 사용 + `interactiveOverride: false`로 비대화 모드 강제. 모델은 `'inherit'`, 도구는 모든 부모 도구를 동적으로 (`get toolConfig() { ... }`).

→ **"나 자신, 단 fresh context window에서 + 의무적인 `complete_task`로 끝내는"** 에이전트. Claude Code의 Task tool, OpenAI Agents SDK의 `Agent.as_tool()`에 해당.

### 7.2 cli_help (`cli-help-agent.ts:26-94`)

| 항목 | 값 |
|------|-----|
| 모델 | `GEMINI_MODEL_ALIAS_FLASH` (하드코드, inherit 아님) |
| 도구 | `[GetInternalDocsTool]` 단 1개 — 번들된 문서 접근만 |
| Thinking | `{ includeThoughts: true, thinkingBudget: -1 }` (Flash + 무제한 thinking) |
| 한계 | 3분 / 10턴 |
| 출력 | `{ answer: string, sources: string[] }` JSON |

시스템 프롬프트 일부:
> "You are **CLI Help Agent**, an expert on Gemini CLI. ...
> ### Runtime Context
> - CLI Version: ${cliVersion}
> - Active Model: ${activeModel}
> - Today's Date: ${today}
> ### Instructions
> 1. Explore Documentation: Use `get_internal_docs` ...
> 2. Be Precise ...
> 3. Cite Sources ...
> 4. Non-Interactive ..."

### 7.3 codebase_investigator (`codebase-investigator.ts:51-193`)

| 항목 | 값 |
|------|-----|
| 모델 | `PREVIEW_GEMINI_FLASH_MODEL` (modern 지원 시) 또는 `DEFAULT_GEMINI_MODEL` |
| Thinking | `ThinkingLevel.HIGH` (modern) 또는 `DEFAULT_THINKING_MODE` |
| 도구 | `ls`, `read_file`, `glob`, `grep` — **read-only만** |
| 한계 | 10분 / 50턴 |
| 출력 | `{ SummaryOfFindings, ExplorationTrace[], RelevantLocations[] }` 구조화 |

> **안전 디자인**: `edit`, `write_file`, `shell` 명시적 제외 → 코드베이스 조사 중 사고 가능성 0.

시스템 프롬프트 핵심:
> "You are **Codebase Investigator**, a hyper-specialized AI agent and an expert in reverse-engineering complex software projects. ... Your **SOLE PURPOSE** is to build a complete mental model of the code...
> - **DO:** Find the key modules ...
> - **DO:** Foresee the ripple effects of a change.
> - **DO NOT:** Write the final implementation code yourself.
> - **DO NOT:** Stop at the first relevant file.
>
> ## Scratchpad Management
> ... you MUST create the `<scratchpad>` section. ... Explicitly log questions in `Questions to Resolve`. ... Do not consider your investigation complete until this list is empty."

OS-aware 템플릿 비트도 포함 (`process.platform === 'win32' ? 'dir /s' : 'ls -R'`).

### 7.4 마케팅에 안 들어간 빌트인들

- **browser_agent**: `config.getBrowserAgentConfig().enabled` 시에만 등록. `BrowserAgentFactory`가 vision 활성/`analyze_screenshot` 도구 존재 여부에 따라 도구 동적 생성. `onBeforeTurn`으로 stale snapshot 제거.
- **save_memory**: `config.isMemoryManagerEnabled()` 시 레거시 `save_memory` 도구를 대체. `workspaceDirectories: [globalGeminiDir]`로 `~/.gemini/GEMINI.md` 접근. Flash 하드코드.
- **skill-extraction-agent**: skills 파이프라인용, 사용자 직접 호출 X.

---

## 8. 에러 / 한계 / 관측성

### 8.1 한계

| 항목 | 기본값 | 비고 |
|------|---------|------|
| `maxTurns` | 30 | codebase_investigator: 50, generalist: 20, cli_help: 10 |
| `maxTimeMinutes` | 10 | DeadlineTimer로 강제, confirmation 시 pause |
| `GRACE_PERIOD_MS` | 60_000 | TIMEOUT/MAX_TURNS/NO_COMPLETE_TASK 시 마지막 기회 |

### 8.2 도구 실패 처리

- **인자 파싱 실패** → 스케줄러 안 거치고 동기적 `functionResponse.error` (`local-executor.ts:1080-1099`)
- **권한 없는 도구 호출** → `Unauthorized tool call: 'xxx' is not available to this agent.` 합성 응답 (L1126-1148). **모델이 hallucinate한 도구도 안전하게 거부**.
- **사용자 거부 (soft Cancel)**: 도구 응답이 instructional error로 변환 (L1218-1237):
  > "User rejected this operation. Please acknowledge this, rethink your strategy, and try a different approach. If you cannot proceed without the rejected operation, summarize the issue and use `complete_task` to report your findings and the blocker."
  서브에이전트는 계속 실행됨.
- **하드 abort (Ctrl+C)** → `aborted = true` → `runInternal`이 `ABORTED` 반환 → `LocalSubagentInvocation`이 `AbortError` re-throw

### 8.3 관측성

- `telemetry/types.ts`에 `AgentStartEvent`, `AgentFinishEvent`, `RecoveryAttemptEvent`, `LlmRole.SUBAGENT`
- `runInDevTraceSpan({ operation: AgentCall, attributes: { gen_ai.agent.name, gen_ai.agent.description } })` — **OTel GenAI 시맨틱 컨벤션 준수**
- 스트리밍 활동: `SUBAGENT_ACTIVITY` 메시지가 메시지 버스로 → UI (`SubagentProgressDisplay.tsx`)가 구독, thought/tool call chip 렌더링
- 스트리밍 전 sanitization: `sanitizeThoughtContent`, `sanitizeToolArgs`, `sanitizeErrorMessage` (시크릿 등 제거)

### 8.4 정책 엔진

`PolicyEngine` 룰 (`registry.ts:377-404`):
- **Local 에이전트**: `Kind.Agent` blanket-allow at `PRIORITY_SUBAGENT_TOOL`
- **Remote 에이전트**: `argsPattern: /"agent_name":\s*"<name>"/`로 `ASK_USER` 룰 동적 삽입 (priority `+0.1`) → A2A 호출은 기본적으로 사용자 확인 필요

---

## 9. 진행 상황 스트리밍 — UI 통합

`SubagentActivityEvent` 유니언 (`types.ts:82-106`):

```ts
export interface SubagentActivityEvent {
  isSubagentActivityEvent: true;
  agentName: string;
  type: 'TOOL_CALL_START' | 'TOOL_CALL_END' | 'THOUGHT_CHUNK' | 'ERROR';
  data: Record<string, unknown>;
}
export interface SubagentProgress {
  isSubagentProgress: true;
  agentName: string;
  recentActivity: SubagentActivityItem[];
  state?: 'running' | 'completed' | 'error' | 'cancelled';
  result?: string;
  terminateReason?: AgentTerminateMode;
}
```

`onActivity` 콜백 (`local-invocation.ts:127-275`)이 이를 `updateOutput`에 전달 → UI는 `running` 상태의 thought를 mutable buffer로 유지하여 **긴 thinking 세그먼트를 단일 버블로 부드럽게 렌더**.

> **시사점**: 서브에이전트의 깊은 활동(수십 번의 도구 호출)을 사용자에게 압축적으로 보여주는 것은 UX 핵심. 활동 이벤트를 작은 enum 4개(`TOOL_CALL_START`, `TOOL_CALL_END`, `THOUGHT_CHUNK`, `ERROR`)로 분류한 것은 깔끔한 모델링.

---

## 10. `complete_task` 도구 — 종료 프로토콜

`tools/complete-task.ts`. 서브에이전트의 정상 종료를 강제하는 핵심 메커니즘.

스키마 생성 (`tools/complete-task.ts:50-80`):

```ts
private static buildParameterSchema(outputConfig?: OutputConfig<z.ZodTypeAny>): unknown {
  if (outputConfig) {
    const jsonSchema = zodToJsonSchema(outputConfig.schema);
    const { $schema: _, definitions: __, ...schema } = jsonSchema;
    return {
      type: 'object',
      properties: { [outputConfig.outputName]: schema },
      required: [outputConfig.outputName],
    };
  }
  return {
    type: 'object',
    properties: { result: { type: 'string', description: 'Your final results ...' } },
    required: ['result'],
  };
}
```

각 서브에이전트의 `outputConfig`가 zod 스키마라면 `zodToJsonSchema`로 LLM-visible JSON schema 변환. 이로써:
- LLM이 잘못된 형식으로 종료 못 함
- 사용자가 정의한 출력 구조를 강제 가능 (예: `codebase_investigator`의 `RelevantLocations[]`)

> **시사점**: 텍스트 응답으로 끝내지 않고 **명시적 도구 호출로 종료**시키는 패턴은 다음 이점을 가진다:
> 1. 구조화 출력 강제
> 2. 종료 시점 명확화 (LLM이 "다 했어" 텍스트 후 추가 도구 호출하는 모호함 제거)
> 3. 종료 자체가 관측 가능한 이벤트 (telemetry/UI)
>
> 자체 하네스에서 차용할 가치 매우 큼.

---

## 11. 다른 SDK와의 비교

| 항목 | OpenAI Agents SDK | Claude Code (Task tool) | **Gemini CLI 서브에이전트** |
|------|---|---|---|
| 부모 → 자식 wire format | 에이전트별 별도 함수 선언 (`as_tool()`) | 단일 `Task` 도구 + agent type | **단일 `invoke_agent` 도구 + `agent_name`** |
| 위임 깊이 | 무제한 (`Handoff` chain) | 1단계 | **1단계 (Kind.Agent 화이트리스트 차단)** |
| 컨텍스트 격리 | RunContextWrapper 공유 (가능), 새 Runner.run | 새 컨텍스트 윈도우 | **새 GeminiChat + fresh registries** |
| 도구 화이트리스트 | 에이전트 정의 시 `tools=[...]` | 에이전트 type별 정해짐 | **frontmatter `tools:` + 와일드카드 (`*`, `mcp__*`)** |
| MCP 스코핑 | 에이전트별 `mcp_servers=[...]` | 글로벌 | **에이전트 전용 MCP 서버 추가 검색 + 격리 레지스트리** |
| 종료 프로토콜 | 텍스트 응답 + `output_type` 검증 | 텍스트 응답 | **`complete_task` 도구 호출 강제 + 60s grace** |
| 병렬 실행 | function tool 자동 `asyncio.gather` | 한 메시지 다중 Task 호출 시 병렬 | **메인 Scheduler가 다중 invoke_agent를 Promise.all 배칭** |
| 동시성 상한 | 없음 | 알려지지 않음 | **없음** |
| 사용자 확인 중 timeout pause | 없음 | 없음 (알려진 한) | **있음 (DeadlineTimer.pause)** |
| 정의 파일 포맷 | Python 코드 | (내장 type) | **마크다운 + YAML frontmatter** |
| 글로벌/프로젝트 분리 | N/A | N/A | **`~/.gemini/agents/` vs `.gemini/agents/` + extensions** |
| 보안 게이트 | N/A | N/A | **sha256 acknowledgement (프로젝트 에이전트)** |
| Workspace 추가 스코핑 | 없음 | 없음 (알려진 한) | **AsyncLocalStorage 기반 추가 디렉터리** |
| 모델 routing per turn | 정적 | 정적 | **per-turn ModelRouterService 호출 가능 (`auto`)** |

---

## 12. 강점 정리

### 12.1 설계 수준

1. **단일 `invoke_agent` 디스패처** — 함수 선언 폭발 회피, 동적 등록/제거 용이
2. **재귀 명시적 차단** — `Kind.Agent` 화이트리스트 제외로 무한 위임 footgun 차단
3. **격리 레지스트리 + 공유 ContentGenerator** — 에이전트별 history/도구 격리하면서도 글로벌 rate-limit/auth는 일관성 유지
4. **`complete_task` 종료 프로토콜** — 구조화 출력 + 명시적 종료 이벤트 + grace period 회복
5. **AsyncLocalStorage 기반 workspace 추가 스코핑** — 공유 Config를 mutate하지 않고 에이전트별 권한 확장
6. **per-agent ModelRouterService** — 서브에이전트가 턴마다 다른 모델로 land 가능
7. **Confirmation 동안 timeout pause** — 사용자 고민 시간이 에이전트 클럭을 잠식하지 않음
8. **OTel GenAI 시맨틱 컨벤션 준수** — `gen_ai.agent.name`, `gen_ai.agent.description` 표준 attribute

### 12.2 운영/배포 수준

1. **마크다운 + YAML 정의** — Git-friendly, PR 리뷰 친화적
2. **3-tier 우선순위** (extension > project > user > built-in) — 명시적이고 디버깅 가능
3. **sha256 acknowledgement** — 공급망 공격 방어선
4. **frontmatter 와일드카드** (`*`, `mcp__*`, `mcp__<srv>__*`) — 도구 그룹 관리 유연
5. **settings 오버라이드** — 파일 수정 없이 `runConfig`/`tools`/`mcpServers` 조정
6. **`/agents` 슬래시 커맨드** — list/reload/enable/disable/config 모두 지원
7. **활동 이벤트 4종 + sanitization** — UI에 안전하고 부드럽게 진행 표시

### 12.3 사용자 보호 수준

1. **codebase_investigator의 read-only 도구 세트** — 조사 중 사고 0
2. **시스템 프롬프트 강제 append** (`complete_task` 의무, 사용자 거부 처리, non-interactive 모드)
3. **사용자 거부를 instructional error로 변환** — 에이전트가 "다른 접근법" 학습 가능
4. **원격 에이전트는 ASK_USER 정책 자동 삽입**

---

## 13. 한계 / 약점

1. **`prompt` 단일 문자열 wire format** — 다중 입력 에이전트는 LLM이 직접 구조화 인자 작성 필요. "스마트 매퍼"는 단일 프로퍼티만 지원.
2. **동시성 상한 없음** — 모델이 10개 `invoke_agent` emit하면 10개 동시 실행. MCP 서버 에이전트의 경우 각자 MCP discovery 발생 → 자원 폭증 위험.
3. **재귀 금지 = 1단계 위임만 가능** — 계층적 오케스트레이션 (서브에이전트가 더 작은 서브에이전트 spawn) 불가능. OpenAI Agents SDK의 unbounded handoff와 대비.
4. **per-subagent 토큰 budget 강제 없음** — turn + wall-clock만. `COMPRESSION_FAILED_INFLATED_TOKEN_COUNT` 발생 후에야 인지.
5. **`onBeforeTurn`만 확장 훅** — 일반적 middleware/before-after-tool 훅 없음. 글로벌 hook-utils.ts는 적용되지만 서브에이전트 전용 행동 정의 어려움.
6. **자동 라우팅 정확도가 description text 품질에 100% 의존** — 잘못 적힌 description은 에이전트 미선택 또는 오선택 유발.
7. **서브에이전트 결과가 단일 텍스트 블록** — 부모 LLM이 구조화 출력을 다시 파싱해야 할 수 있음.
8. **`@agent`가 short-circuit 아님** — 사용자가 명시적으로 호출해도 메인이 거절 가능. 의도된 디자인이지만 UX 혼란 가능.

---

## 14. 하네스 팀이 차용 가능한 패턴 — 체크리스트

opencode/Claude Code 류 인터랙티브 코딩 하네스 관점에서:

### 14.1 즉시 차용 가치 ★★★

- [ ] **단일 디스패처 도구 + 시스템 프롬프트 description 라우팅** — 함수 선언 토큰 절약
- [ ] **`Kind.Agent` 화이트리스트 명시적 차단** — 무한 위임 방지
- [ ] **`complete_task` 종료 프로토콜** — 구조화 출력 강제, 종료 이벤트 명확화
- [ ] **Grace period 회복** — TIMEOUT/MAX_TURNS 직전 "마지막 기회" 부여
- [ ] **사용자 확인 동안 timeout pause** — 인터랙티브 하네스의 필수 디테일
- [ ] **마크다운 + YAML frontmatter 정의 포맷** — Git-friendly, PR 리뷰 가능
- [ ] **3-tier 우선순위 (extension/project/user/builtin)** — 명시적 충돌 해결
- [ ] **sha256 acknowledgement** — 프로젝트 에이전트 공급망 공격 방어
- [ ] **활동 이벤트 4종** (TOOL_CALL_START/END/THOUGHT_CHUNK/ERROR) — UI 친화적 progress 모델

### 14.2 차용 가치 ★★

- [ ] **AsyncLocalStorage 기반 workspace 추가 스코핑** — 공유 Config 오염 방지
- [ ] **격리 레지스트리 인스턴스 (호출별 fresh)** — 도구/MCP 격리
- [ ] **`MessageBus.derive(name)` 패턴** — 모든 활동에 에이전트 이름 자동 태깅
- [ ] **frontmatter 도구 와일드카드** (`*`, `mcp__*`, `mcp__<srv>__*`) — 그룹 관리
- [ ] **사용자 거부를 instructional error로 변환** — 에이전트 학습 가능
- [ ] **시스템 프롬프트 강제 append 정책** — 사용자 정의 프롬프트를 신뢰하지 않고 fallback 강제
- [ ] **OTel GenAI 시맨틱 컨벤션** — `gen_ai.agent.name`/`description` attribute
- [ ] **활동 sanitization** (`sanitizeThoughtContent` 등) — 시크릿 누출 방지
- [ ] **/agents 슬래시 커맨드 패턴** — list/reload/enable/disable/config

### 14.3 신중한 차용 ★

- [ ] **모델 'inherit'** — 메인 모델 변경 시 자동 reconcile. 의도하지 않은 모델 폭발 위험 가능.
- [ ] **per-turn ModelRouterService** — 강력하지만 비용/예측성 trade-off
- [ ] **단일 `prompt` 문자열 + 스마트 매퍼** — 단순하지만 다중 입력 에이전트에 약함
- [ ] **재귀 금지** — 안전하지만 표현력 제한. 도메인에 따라 결정.

### 14.4 차용 비추 ✗ (하네스 도메인에 따라)

- [ ] **동시성 상한 없음** — 멀티테넌트/quota 환경에서는 worker pool/semaphore 추가 필요
- [ ] **부모 결과가 단일 텍스트 블록** — 계층적 워크플로우라면 구조화 결과가 더 유리
- [ ] **`@agent` short-circuit 안 하는 정책** — UX 명확성 측면에서 도메인별 결정

---

## 15. 결론

Gemini CLI의 서브에이전트는 **"안전하고 단순한 1단계 멀티에이전트"** 시스템이다. OpenAI Agents SDK의 자유도 높은 `Handoff`와 달리, 다음 디자인 결정으로 **footgun을 적극적으로 차단**한다:

- 1개의 디스패처 도구
- 1단계 위임 (`Kind.Agent` 차단)
- 명시적 종료 프로토콜 (`complete_task`)
- 시스템 프롬프트 강제 append
- 프로젝트 에이전트 sha256 acknowledgement

대신 **다중 서브에이전트의 자동 병렬 실행**, **AsyncLocalStorage 기반 workspace 스코핑**, **사용자 확인 동안 timeout pause**, **OTel GenAI 컨벤션 준수** 같은 운영 디테일에서 인상적인 완성도를 보여준다.

가장 가져갈 만한 핵심 디자인 5가지를 다시 정리하면:

1. **`complete_task` 종료 프로토콜 + grace period 회복** — 구조화 출력 강제 + 명시적 종료 이벤트 + 마지막 기회 부여
2. **단일 `invoke_agent` 디스패처 + description 라우팅** — 토큰 절약 + 동적 관리
3. **격리 레지스트리 + 공유 ContentGenerator** — 격리와 공유의 균형
4. **마크다운 + YAML 정의 + sha256 acknowledgement** — Git-friendly + 공급망 방어
5. **사용자 확인 중 timeout pause** — 인터랙티브 UX 결정적 디테일

하지만 **"진짜 계층적 멀티에이전트 오케스트레이션"이 필요한 도메인에는 부족**하다. 1단계 위임 제약, 결과의 단일 텍스트 블록 collapse, 동시성 상한 부재 등은 SaaS 멀티테넌트 환경이나 깊은 에이전트 트리를 다루는 시스템에는 별도 layer가 필요함을 의미한다.

opencode/Claude Code 같은 인터랙티브 코딩 하네스 팀이라면, **Gemini CLI의 안전 중심 디자인은 거의 그대로 차용 가치**가 있고, 자체 도메인이 더 깊은 오케스트레이션을 필요로 한다면 그 위에 워커 풀, 계층 위임, 구조화 결과 전송 layer를 추가하는 형태가 적절할 것이다.

---

## 부록 A. 핵심 파일:라인 인덱스

> v 분석 시점 기준. 차용 시 코드 직접 확인 권장.

### 타입 / 정의
- `packages/core/src/agents/types.ts:24-31` — `AgentTerminateMode`
- `packages/core/src/agents/types.ts:36-39` — `OutputObject`
- `packages/core/src/agents/types.ts:44-54` — 기본 한계값
- `packages/core/src/agents/types.ts:82-115` — `SubagentActivityEvent`, `SubagentProgress`
- `packages/core/src/agents/types.ts:189-255` — `LocalAgentDefinition`
- `packages/core/src/agents/types.ts:283-354` — `PromptConfig`/`ToolConfig`/`InputConfig`/`OutputConfig`/`RunConfig`

### 로더
- `packages/core/src/agents/agentLoader.ts:50-111` — `localAgentSchema`
- `packages/core/src/agents/agentLoader.ts:198-234` — 원격 에이전트 스키마
- `packages/core/src/agents/agentLoader.ts:319-402` — `parseAgentMarkdown`
- `packages/core/src/agents/agentLoader.ts:482-610` — `markdownToAgentDefinition`
- `packages/core/src/agents/agentLoader.ts:620-677` — `loadAgentsFromDirectory`

### 레지스트리
- `packages/core/src/agents/registry.ts:47-56` — 클래스 헤더
- `packages/core/src/agents/registry.ts:60-70` — `initialize`
- `packages/core/src/agents/registry.ts:117-263` — `loadAgents` (우선순위)
- `packages/core/src/agents/registry.ts:265-304` — `loadBuiltInAgents`
- `packages/core/src/agents/registry.ts:337-404` — `registerLocalAgent` + 정책
- `packages/core/src/agents/registry.ts:420-572` — `registerRemoteAgent`
- `packages/core/src/agents/registry.ts:574-646` — `applyOverrides`
- `packages/core/src/agents/registry.ts:648-679` — `registerModelConfigs`

### 도구 wrapper
- `packages/core/src/agents/agent-tool.ts:40-120` — `AgentTool`
- `packages/core/src/agents/agent-tool.ts:122-252` — `DelegateInvocation`
- `packages/core/src/tools/tool-names.ts:191` — `AGENT_TOOL_NAME = 'invoke_agent'`
- `packages/core/src/config/config.ts:3643-3646` — `AgentTool` 등록

### Invocation / Executor
- `packages/core/src/agents/local-invocation.ts:50-382` — `LocalSubagentInvocation`
- `packages/core/src/agents/local-executor.ts:114-281` — `LocalAgentExecutor.create`
- `packages/core/src/agents/local-executor.ts:317-396` — `executeTurn`
- `packages/core/src/agents/local-executor.ts:430-516` — `executeFinalWarningTurn` (grace)
- `packages/core/src/agents/local-executor.ts:525-859` — `run` / `runInternal`
- `packages/core/src/agents/local-executor.ts:903-998` — `callModel`
- `packages/core/src/agents/local-executor.ts:1001-1044` — `createChatObject`
- `packages/core/src/agents/local-executor.ts:1051-1283` — `processFunctionCalls`
- `packages/core/src/agents/local-executor.ts:1311-1368` — `buildSystemPrompt`

### 스케줄러 / 병렬
- `packages/core/src/scheduler/scheduler.ts:448-495` — 병렬 배칭
- `packages/core/src/scheduler/scheduler.ts:537-547` — `_isParallelizable`
- `packages/core/src/agents/agent-scheduler.ts:50-93` — `scheduleAgentTools`

### 빌트인
- `packages/core/src/agents/generalist-agent.ts:20-68`
- `packages/core/src/agents/cli-help-agent.ts:26-94`
- `packages/core/src/agents/codebase-investigator.ts:51-193`
- `packages/core/src/agents/browser/browserAgentDefinition.ts`
- `packages/core/src/agents/memory-manager-agent.ts:37-157`

### UI / 커맨드
- `packages/cli/src/ui/commands/agentsCommand.ts:1-373` — `/agents`
- `packages/cli/src/ui/hooks/atCommandProcessor.ts:141-173` — `@agent` 분류
- `packages/cli/src/ui/hooks/atCommandProcessor.ts:702-708` — `<system_note>` 가이드
- `packages/cli/src/ui/hooks/useAtCompletion.ts:142-152` — `buildAgentCandidates`
- `packages/cli/src/ui/hooks/useAgentStream.ts` — 스트리밍 프로토콜
- `packages/cli/src/ui/components/messages/SubagentProgressDisplay.tsx`

### 프롬프트 / 발견
- `packages/core/src/prompts/snippets.ts:252-290` — `renderSubAgents`
- `packages/core/src/prompts/utils.ts:75-85` — `${SubAgents}` 치환
- `packages/core/src/config/storage.ts:55-107, 163-298` — 에이전트 디렉터리 경로
- `packages/core/src/config/agent-loop-context.ts:19-46` — `AgentLoopContext`

### 기타
- `packages/core/src/tools/complete-task.ts:28-160` — `CompleteTaskTool`
- `packages/core/src/agents/acknowledgedAgents.ts` — 프로젝트 acknowledgement
- `packages/core/src/agents/remote-invocation.ts` — A2A 원격 path
- `packages/core/src/agents/utils.ts` — `templateString`
- `packages/core/src/utils/agent-sanitization-utils.ts` — sanitization
- `packages/test-utils/src/fixtures/agents.ts:1-153` — frontmatter 예시

---

## 부록 B. 빌트인 에이전트 frontmatter 등가물

빌트인은 TS 팩토리이지만, frontmatter로 표현하면 다음과 같다:

### generalist
```yaml
---
name: generalist
description: A general-purpose AI agent with access to all tools. Highly recommended for tasks that are turn-intensive or involve processing large amounts of data...
tools: ["*"]
model: inherit
max_turns: 20
timeout_mins: 10
---
{{ getCoreSystemPrompt with interactiveOverride: false }}
```

### cli_help (frontmatter 변환 어려운 케이스 — 도구 인스턴스 + thinking 설정)
```yaml
---
name: cli_help
description: Expert agent for Gemini CLI. Use this to answer questions about Gemini CLI features, configuration, and current state.
tools: [get_internal_docs]
model: gemini-flash-latest
max_turns: 10
timeout_mins: 3
---
You are CLI Help Agent...
```

### codebase_investigator
```yaml
---
name: codebase_investigator
description: Hyper-specialized AI agent for reverse-engineering complex software projects. Use this to map codebase architecture, find root causes of bugs, identify cross-system dependencies.
tools: [ls, read_file, glob, grep]
model: gemini-flash-preview
max_turns: 50
timeout_mins: 10
---
You are Codebase Investigator...
```
