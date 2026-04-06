# AI 코딩 에이전트 구현 분석 리포트

> OpenCode 프로젝트를 통해 분석한 AI 코딩 에이전트의 핵심 구현 패턴과 벤치마킹 포인트

---

## 목차

1. [개요](#1-개요)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [프롬프트 엔지니어링 전략](#3-프롬프트-엔지니어링-전략)
4. [도구(Tool) 시스템](#4-도구tool-시스템)
5. [에이전트 루프와 의사결정](#5-에이전트-루프와-의사결정)
6. [컨텍스트 관리](#6-컨텍스트-관리)
7. [권한 및 안전 시스템](#7-권한-및-안전-시스템)
8. [멀티 에이전트 아키텍처](#8-멀티-에이전트-아키텍처)
9. [에이전트 개발 시 고려사항](#9-에이전트-개발-시-고려사항)
10. [벤치마킹 포인트](#10-벤치마킹-포인트)
11. [프롬프트 파일 상세 분석](#11-프롬프트-파일-상세-분석)

---

## 1. 개요

OpenCode는 TypeScript/Bun 기반의 오픈소스 AI 코딩 에이전트입니다. 이 프로젝트를 분석하여 같은 LLM 모델을 사용해도 더 효과적으로 동작하게 만드는 핵심 구현 패턴들을 정리했습니다.

### 핵심 성공 요인

| 요인 | 설명 |
|-----|------|
| **구조화된 프롬프트** | Provider별 최적화된 시스템 프롬프트 |
| **도구 추상화** | 일관된 인터페이스로 확장 가능한 Tool 시스템 |
| **컨텍스트 관리** | 자동 압축(Compaction)으로 긴 대화 지속 |
| **권한 시스템** | 세분화된 permission ruleset |
| **멀티 에이전트** | 전문화된 서브에이전트 분리 |

---

## 2. 아키텍처 개요

### 2.1 프로젝트 구조

```
packages/
├── opencode/         # 핵심 CLI 애플리케이션
│   ├── src/
│   │   ├── agent/    # 에이전트 정의 및 관리
│   │   ├── session/  # 세션, 메시지, 프롬프트 처리
│   │   ├── tool/     # 20+ 도구 구현
│   │   ├── provider/ # LLM 프로바이더 추상화
│   │   ├── permission/ # 권한 시스템
│   │   ├── mcp/      # Model Context Protocol
│   │   └── config/   # 설정 관리
├── app/              # Web UI (SolidJS)
├── sdk/              # OpenAPI 기반 SDK
└── plugin/           # 플러그인 시스템
```

### 2.2 핵심 모듈 관계

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Web UI                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Session Manager                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Prompt    │  │  Processor  │  │  Compaction │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Agent    │    │    Tools    │    │  Permission │
│   System    │    │   Registry  │    │   System    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Provider (LLM)                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │Anthropic│ │ OpenAI │ │ Google │ │ Azure  │ │  etc.  │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 기술 스택

- **런타임**: Bun (빠른 TypeScript 실행)
- **HTTP 서버**: Hono (경량 프레임워크)
- **LLM 추상화**: Vercel AI SDK (멀티 프로바이더)
- **유효성 검증**: Zod (스키마 기반 타입 안전성)
- **상태 관리**: Namespace 패턴 + Instance.state()

---

## 3. 프롬프트 엔지니어링 전략

### 3.1 Provider별 최적화된 시스템 프롬프트

OpenCode의 가장 중요한 차별점 중 하나는 **각 LLM Provider에 맞춤화된 시스템 프롬프트**입니다.

```
packages/opencode/src/session/prompt/
├── anthropic.txt      # Claude 전용 (8KB)
├── beast.txt          # GPT-4/5, O1, O3 전용 (11KB)
├── gemini.txt         # Gemini 전용 (15KB)
├── codex.txt          # Codex 전용 (24KB)
└── qwen.txt           # Qwen 전용 (10KB)
```

**시스템 프롬프트 선택 로직** (`session/system.ts`):
```typescript
export function provider(modelID: string): string {
  if (modelID.includes("gpt-5")) return PROMPT_CODEX
  if (modelID.includes("gpt-") || modelID.includes("o1") || modelID.includes("o3"))
    return PROMPT_BEAST
  if (modelID.includes("gemini")) return PROMPT_GEMINI
  if (modelID.includes("claude")) return PROMPT_ANTHROPIC
  return PROMPT_ANTHROPIC_WITHOUT_TODO  // 기본값
}
```

### 3.2 시스템 프롬프트 핵심 구성 요소

**Anthropic 프롬프트 분석** (`anthropic.txt`):

```markdown
# 1. Identity 설정
You are OpenCode, the best coding agent on the planet.

# 2. Tone & Style 가이드
- Only use emojis if explicitly requested
- Short and concise responses (CLI 환경 고려)
- Github-flavored markdown 사용
- NEVER create files unless absolutely necessary

# 3. Professional Objectivity
Prioritize technical accuracy over validating user's beliefs.
Focus on facts and problem-solving.

# 4. Task Management (핵심!)
Use TodoWrite tools VERY frequently.
Mark todos as completed as soon as done.
Do not batch up multiple tasks.

# 5. Tool Usage Policy
- Use Task tool for file search (context 절약)
- Parallel tool calls when possible
- Use specialized tools instead of bash
```

### 3.3 Few-shot Examples의 전략적 활용

프롬프트에 구체적인 예시를 포함하여 모델의 행동을 가이드합니다:

```markdown
<example>
user: Run the build and fix any type errors
assistant: I'm going to use the TodoWrite tool to write:
- Run the build
- Fix any type errors

I'm now going to run the build using Bash.
Found 10 type errors. Adding 10 items to todo list.
marking the first todo as in_progress...
</example>
```

### 3.4 컨텍스트 주입 패턴

런타임 환경 정보를 동적으로 주입합니다:

```typescript
// session/system.ts
export function environment(): string {
  return `
Working directory: ${Instance.worktree}
Is directory a git repo: ${gitStatus}
Platform: ${process.platform}
Today's date: ${new Date().toISOString().split('T')[0]}
  `.trim()
}
```

### 3.5 사용자 정의 지침 (AGENTS.md)

프로젝트별 커스텀 지침을 자동으로 로드합니다:

```typescript
// 우선순위: 프로젝트 > 사용자 > 전역
const customInstructions = [
  await loadFile("AGENTS.md"),     // 프로젝트 루트
  await loadFile("CLAUDE.md"),     // 호환성
  await loadFile("~/.claude/CLAUDE.md"),  // 전역 설정
]
```

### 3.6 컨텍스트 리마인더 시스템

멀티턴 대화에서 LLM이 원래 작업을 잊지 않도록 사용자 메시지를 래핑합니다:

```typescript
// session/prompt.ts
// 멀티턴 대화에서 사용자 메시지에 컨텍스트 래핑
if (step > 1 && lastFinished) {
  part.text = [
    "<system-reminder>",
    "The user sent the following message:",
    part.text,
    "",
    "Please address this message and continue with your tasks.",
    "</system-reminder>",
  ].join("\n")
}
```

**목적**:
- 장문의 대화에서 원래 작업 컨텍스트 유지
- 사용자의 새 메시지가 기존 작업 흐름에 통합되도록 안내
- `<system-reminder>` 태그로 시스템 지침임을 명시

---

## 4. 도구(Tool) 시스템

### 4.1 Tool 인터페이스 정의

```typescript
// tool/tool.ts
export namespace Tool {
  export interface Info<Parameters extends z.ZodType, Metadata> {
    id: string
    init: (ctx?: InitContext) => Promise<{
      description: string
      parameters: Parameters
      execute(
        args: z.infer<Parameters>,
        ctx: Context
      ): Promise<{
        title: string
        metadata: Metadata
        output: string
        attachments?: FilePart[]
      }>
    }>
  }

  export type Context = {
    sessionID: string
    messageID: string
    agent: string
    abort: AbortSignal
    metadata(input: { title?: string; metadata?: any }): void
    ask(input: PermissionRequest): Promise<void>  // 권한 요청
  }
}
```

### 4.2 Tool 정의 헬퍼

```typescript
// 간결한 도구 정의를 위한 헬퍼 함수
export function define<P extends z.ZodType, M>(
  id: string,
  init: ToolInit<P, M>
): Tool.Info<P, M> {
  return {
    id,
    init: async (initCtx) => {
      const toolInfo = init instanceof Function ? await init(initCtx) : init

      // 자동 입력 검증
      toolInfo.execute = async (args, ctx) => {
        toolInfo.parameters.parse(args)  // Zod 검증
        const result = await execute(args, ctx)

        // 자동 출력 truncation
        if (result.metadata.truncated === undefined) {
          return Truncate.output(result.output)
        }
        return result
      }

      return toolInfo
    }
  }
}
```

### 4.3 내장 도구 목록

| 도구 | 설명 | 권한 |
|-----|------|------|
| `read` | 파일 읽기 | read |
| `edit` | 파일 편집 (exact string replacement) | edit |
| `write` | 파일 쓰기 | edit |
| `glob` | 파일 패턴 매칭 | glob |
| `grep` | 내용 검색 (ripgrep 기반) | grep |
| `bash` | 셸 명령 실행 | bash |
| `task` | 서브에이전트 실행 | task |
| `webfetch` | URL 내용 가져오기 | webfetch |
| `websearch` | 웹 검색 | websearch |
| `todowrite` | 작업 목록 관리 | todowrite |
| `skill` | 커스텀 스킬 실행 | skill |
| `lsp` | Language Server 쿼리 | lsp |

### 4.4 Tool Registry 패턴

```typescript
// tool/registry.ts
export namespace ToolRegistry {
  // 싱글톤 상태
  const state = Instance.state(async () => {
    const custom = await loadCustomTools()  // 사용자 정의 도구
    return { custom }
  })

  // 조건부 도구 반환
  export async function tools(providerID: string, agent?: Agent.Info) {
    const result = [
      BashTool,
      ReadTool,
      EditTool,
      // ...
    ]

    // 실험적 기능 조건부 포함
    if (Flag.OPENCODE_EXPERIMENTAL_LSP_TOOL) {
      result.push(LspTool)
    }

    // Provider별 제한
    if (!providerSupportsSearch(providerID)) {
      return result.filter(t => t.id !== "websearch")
    }

    return result
  }
}
```

### 4.5 도구 실행 흐름

```
1. LLM이 tool_call 생성
       │
       ▼
2. SessionProcessor가 스트림에서 tool-call 이벤트 감지
       │
       ▼
3. Permission 체크 (ctx.ask() 호출)
       │
       ├── deny → DeniedError 발생
       ├── ask → 사용자 승인 대기
       └── allow → 계속 진행
       │
       ▼
4. Tool execute() 실행
       │
       ▼
5. Plugin hooks 트리거
   - tool.execute.before
   - tool.execute.after
       │
       ▼
6. 결과를 LLM에 반환 (tool-result)
```

### 4.6 Doom Loop 방지

동일한 도구 호출이 3번 연속 반복되면 사용자에게 확인을 요청합니다:

```typescript
// session/processor.ts
const DOOM_LOOP_THRESHOLD = 3

case "tool-call": {
  const parts = await MessageV2.parts(assistantMessage.id)
  const lastThree = parts.slice(-DOOM_LOOP_THRESHOLD)

  if (lastThree.every(p =>
    p.type === "tool" &&
    p.tool === value.toolName &&
    JSON.stringify(p.state.input) === JSON.stringify(value.input)
  )) {
    await PermissionNext.ask({
      permission: "doom_loop",
      patterns: [value.toolName],
      metadata: { tool: value.toolName, input: value.input }
    })
  }
}
```

### 4.7 스마트 파일 편집 (`tool/edit.ts`)

**핵심 기능**: 여러 대체 전략(Replacer)을 순차적으로 시도하여 유연한 문자열 매칭

```typescript
export function replace(content, oldString, newString, replaceAll = false) {
  for (const replacer of [
    SimpleReplacer,              // 정확한 매칭
    LineTrimmedReplacer,         // 줄 공백 무시
    BlockAnchorReplacer,         // 블록 앵커 기반 (첫/끝 줄)
    WhitespaceNormalizedReplacer, // 공백 정규화
    IndentationFlexibleReplacer, // 들여쓰기 유연
    EscapeNormalizedReplacer,    // 이스케이프 정규화
    TrimmedBoundaryReplacer,     // 트림 경계
    ContextAwareReplacer,        // 컨텍스트 인식
    MultiOccurrenceReplacer,     // 다중 발생
  ]) {
    for (const search of replacer(content, oldString)) {
      // 매칭 시도
    }
  }
}
```

**BlockAnchorReplacer 상세**:
- 첫 줄과 마지막 줄을 앵커로 사용
- 중간 내용은 유사도 기반 매칭
- Levenshtein 거리로 유사도 계산

**왜 중요한가**:
- LLM이 생성한 코드는 들여쓰기나 공백이 미세하게 다를 수 있음
- 정확한 문자열 매칭만으로는 편집 실패율이 높음
- 다중 폴백 전략으로 편집 성공률 향상

### 4.8 LSP 통합

Language Server Protocol을 통해 편집 후 실시간 오류 피드백을 제공합니다:

```typescript
// tool/edit.ts - 편집 후 자동 진단
await LSP.touchFile(filePath, true)
const diagnostics = await LSP.diagnostics()

if (errors.length > 0) {
  output += `\n\nLSP errors detected in this file, please fix:\n`
  for (const error of errors) {
    output += `- Line ${error.range.start.line}: ${error.message}\n`
  }
}
```

**활용 예시**:
```typescript
// 출력 예시
<diagnostics>
This file has some issues:
- Line 15: Property 'foo' does not exist on type 'Bar'
- Line 23: Cannot find name 'undefined_var'
</diagnostics>
```

**장점**:
- 편집 즉시 타입 오류, 문법 오류 감지
- LLM이 바로 수정할 수 있도록 오류 정보 제공
- 반복적인 수정-확인 사이클 감소

---

## 5. 에이전트 루프와 의사결정

### 5.1 메인 루프 구조

```typescript
// session/prompt.ts - SessionPrompt.loop()
export const loop = async (sessionID: string) => {
  const abort = start(sessionID)
  let step = 0

  while (true) {
    SessionStatus.set(sessionID, { type: "busy" })

    // 1. 메시지 히스토리 로드 (압축된 것 필터링)
    let msgs = await MessageV2.filterCompacted(
      MessageV2.stream(sessionID)
    )

    // 2. 마지막 사용자/어시스턴트 메시지 추출
    let lastUser, lastAssistant
    for await (const msg of msgs) {
      if (msg.info.role === "user") lastUser = msg
      if (msg.info.role === "assistant") lastAssistant = msg
    }

    // 3. 종료 조건 체크
    if (shouldExit(lastAssistant, lastUser)) break

    // 4. 특수 작업 처리 (subtask, compaction)
    const task = extractPendingTask(lastUser)
    if (task?.type === "subtask") {
      await executeSubtask(task)
      continue
    }
    if (task?.type === "compaction") {
      await SessionCompaction.process(task)
      continue
    }

    // 5. 컨텍스트 오버플로우 체크
    if (await SessionCompaction.isOverflow({ tokens, model })) {
      await SessionCompaction.create({ sessionID, auto: true })
      continue
    }

    // 6. Max Steps 체크
    if (++step >= agent.steps) {
      await injectMaxStepsMessage()
    }

    // 7. LLM 처리
    const processor = SessionProcessor.create({ ... })
    const result = await processor.process({ ... })

    if (result === "stop") break
    if (result === "compact") continue
  }
}
```

### 5.2 스트림 처리 (SessionProcessor)

```typescript
// session/processor.ts
export namespace SessionProcessor {
  export function create(input: ProcessorInput) {
    const toolcalls: Record<string, ToolPart> = {}
    let blocked = false

    return {
      async process(streamInput: LLM.StreamInput) {
        const stream = await LLM.stream(streamInput)

        for await (const value of stream.fullStream) {
          input.abort.throwIfAborted()

          switch (value.type) {
            case "reasoning-start":
              // Extended thinking 시작
              break

            case "reasoning-delta":
              // 추론 내용 스트리밍
              await Session.updatePart({ part, delta: value.text })
              break

            case "tool-input-start":
              // 도구 호출 준비
              toolcalls[value.id] = createPendingToolPart()
              break

            case "tool-call":
              // 도구 실행 시작
              await checkDoomLoop(value)
              break

            case "tool-result":
              // 도구 실행 완료
              await updateCompletedToolPart(value)
              break

            case "tool-error":
              // 도구 실행 실패 (권한 거부 등)
              if (shouldBreakOnError) blocked = true
              break

            case "text-delta":
              // 텍스트 응답 스트리밍
              break

            case "finish-step":
              // 턴 완료
              break
          }
        }

        if (blocked) return "stop"
        if (needsCompaction) return "compact"
        return "continue"
      }
    }
  }
}
```

### 5.3 의사결정 포인트 요약

```
┌──────────────────────────────────────────────────────┐
│                   Main Loop Start                     │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Load Message History│
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐     Yes
              │ Assistant Finished?│──────────► EXIT
              │ (not tool-calls)   │
              └────────────────────┘
                         │ No
                         ▼
              ┌────────────────────┐     Yes
              │ Pending Subtask?   │──────────► Execute Subtask
              └────────────────────┘             │
                         │ No                    │
                         ▼                       │
              ┌────────────────────┐     Yes     │
              │ Pending Compaction?│──────────► Process Compaction
              └────────────────────┘             │
                         │ No                    │
                         ▼                       │
              ┌────────────────────┐     Yes     │
              │ Context Overflow?  │──────────► Create Compaction
              └────────────────────┘             │
                         │ No                    │
                         ▼                       │
              ┌────────────────────┐     Yes     │
              │ Max Steps Reached? │──────────► Inject Warning
              └────────────────────┘             │
                         │ No                    │
                         ▼                       │
              ┌────────────────────┐             │
              │   Process LLM      │◄────────────┘
              │   (Stream Tools)   │
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  Result Analysis   │
              │  stop/compact/cont │
              └────────────────────┘
```

---

## 6. 컨텍스트 관리

### 6.1 컨텍스트 오버플로우 감지

```typescript
// session/compaction.ts
export async function isOverflow(input: {
  tokens: AssistantTokens,
  model: Provider.Model
}) {
  const config = await Config.get()
  if (config.compaction?.auto === false) return false

  const context = input.model.limit.context
  if (context === 0) return false

  // 사용된 토큰 = 입력 + 캐시 읽기 + 출력
  const count = input.tokens.input +
                input.tokens.cache.read +
                input.tokens.output

  // 사용 가능한 토큰 = 전체 - 출력 예약
  const output = Math.min(model.limit.output, OUTPUT_TOKEN_MAX)
  const usable = model.limit.input || (context - output)

  return count > usable
}
```

### 6.2 Pruning 전략 (도구 출력 정리)

```typescript
// session/compaction.ts
export const PRUNE_MINIMUM = 20_000   // 최소 정리 토큰
export const PRUNE_PROTECT = 40_000   // 보호 토큰 (최근 것)

export async function prune(input: { sessionID: string }) {
  const msgs = await Session.messages({ sessionID })
  let total = 0
  let pruned = 0
  const toPrune = []

  // 뒤에서부터 순회 (최신 → 과거)
  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i]

    // Summary 이전은 건드리지 않음
    if (msg.info.role === "assistant" && msg.info.summary) break

    for (const part of msg.parts) {
      if (part.type === "tool" && part.state.status === "completed") {
        // 보호 도구는 건너뜀
        if (PRUNE_PROTECTED_TOOLS.includes(part.tool)) continue

        const estimate = Token.estimate(part.state.output)
        total += estimate

        // 보호 토큰 초과 시 정리 대상에 추가
        if (total > PRUNE_PROTECT) {
          pruned += estimate
          toPrune.push(part)
        }
      }
    }
  }

  // 최소 정리량 이상일 때만 실행
  if (pruned > PRUNE_MINIMUM) {
    for (const part of toPrune) {
      part.state.time.compacted = Date.now()
      await Session.updatePart(part)
    }
  }
}
```

### 6.3 Compaction 프로세스 (대화 요약)

```typescript
// session/compaction.ts
export async function process(input: CompactionInput) {
  // Compaction 전용 에이전트 사용
  const agent = await Agent.get("compaction")
  const model = agent.model || userMessage.model

  // 요약용 어시스턴트 메시지 생성
  const msg = await Session.updateMessage({
    role: "assistant",
    mode: "compaction",
    summary: true,  // 이 플래그가 이전 메시지들을 숨김
  })

  // 대화 전체 + 요약 요청 프롬프트
  const promptText = `
    Provide a detailed prompt for continuing our conversation.
    Focus on: what we did, what we're doing, which files,
    and what we're going to do next.
  `

  await processor.process({
    messages: [
      ...MessageV2.toModelMessage(allMessages),
      { role: "user", content: promptText }
    ],
    tools: {}  // 요약 시에는 도구 사용 안함
  })

  // 자동 모드면 "Continue" 메시지 추가
  if (input.auto) {
    await Session.updatePart({
      type: "text",
      synthetic: true,
      text: "Continue if you have next steps"
    })
  }
}
```

### 6.4 메시지 필터링 (Compacted 메시지 처리)

```typescript
// session/message-v2.ts
export async function* filterCompacted(
  stream: AsyncGenerator<MessageWithParts>
) {
  let foundSummary = false

  for await (const msg of stream) {
    // Summary 플래그가 있는 어시스턴트 메시지를 만나면
    if (msg.info.role === "assistant" && msg.info.summary) {
      foundSummary = true
      yield msg  // 요약은 포함

      // 바로 이전 사용자 메시지까지만 포함하고 중단
      // → 그 이전 메시지들은 컨텍스트에서 제외
    }

    if (!foundSummary) {
      yield msg  // Summary 전 메시지들은 모두 포함
    }
  }
}
```

### 6.5 토큰 추정

```typescript
// util/token.ts
const CHARS_PER_TOKEN = 4  // 대략적인 추정치

export function estimate(input: string): number {
  return Math.max(0, Math.round((input || "").length / CHARS_PER_TOKEN))
}
```

---

## 7. 권한 및 안전 시스템

### 7.1 권한 규칙 구조

```typescript
// permission/next.ts
export interface Rule {
  permission: string   // 도구 이름 (read, edit, bash, ...)
  pattern: string      // 매칭 패턴 (* 와일드카드 지원)
  action: "allow" | "deny" | "ask"
}

export type Ruleset = Rule[]
```

### 7.2 권한 평가 로직

```typescript
// permission/next.ts
export function evaluate(
  permission: string,
  pattern: string,
  ...rulesets: Ruleset[]
): Rule {
  // 모든 ruleset을 합침
  const merged = rulesets.flat()

  // 마지막 매칭 규칙이 우선 (Last match wins)
  const match = merged.findLast(rule =>
    Wildcard.match(permission, rule.permission) &&
    Wildcard.match(pattern, rule.pattern)
  )

  // 매칭되는 규칙이 없으면 기본값: ask
  return match ?? { permission, pattern, action: "ask" }
}
```

### 7.3 기본 권한 설정

```typescript
// agent/agent.ts
const defaults = PermissionNext.fromConfig({
  "*": "allow",                    // 대부분 허용
  doom_loop: "ask",                // 무한 루프 확인
  external_directory: "ask",       // 프로젝트 외부 접근 확인
  question: "deny",                // 서브에이전트의 질문 차단
  plan_enter: "deny",              // 플랜 모드 진입 차단
  plan_exit: "deny",               // 플랜 모드 종료 차단
  read: {
    "*": "allow",
    "*.env": "ask",                // .env 파일 읽기 확인
    "*.env.*": "ask",
    "*.env.example": "allow"       // 예시 파일은 허용
  }
})
```

### 7.4 사용자 승인 흐름

```typescript
// permission/next.ts
export const ask = async (input: {
  permission: string
  patterns: string[]
  sessionID: string
  metadata: any
  always: string[]
  ruleset: Ruleset
}) => {
  const rule = evaluate(input.permission, input.patterns[0], input.ruleset)

  switch (rule.action) {
    case "deny":
      throw new DeniedError(rule)

    case "allow":
      return  // 바로 진행

    case "ask":
      // 사용자에게 승인 요청
      const pending = createPendingRequest(input)
      Bus.publish(Event.Asked, pending)

      // 사용자 응답 대기
      return await pending.promise
  }
}

// 사용자 응답 처리
export const reply = async (input: {
  id: string
  reply: "once" | "always" | "reject"
  message?: string
}) => {
  const existing = pendingRequests.get(input.id)

  switch (input.reply) {
    case "once":
      existing.resolve()  // 이번만 허용
      break

    case "always":
      // 패턴을 세션 ruleset에 추가
      addToSessionRuleset(existing.patterns)
      existing.resolve()
      break

    case "reject":
      existing.reject(new CorrectedError(input.message))
      break
  }
}
```

### 7.5 도구 비활성화 로직

```typescript
// permission/next.ts
export function disabled(tools: string[], ruleset: Ruleset): Set<string> {
  const result = new Set<string>()

  for (const tool of tools) {
    // edit 계열 도구는 edit 권한으로 통합
    const permission = EDIT_TOOLS.includes(tool) ? "edit" : tool

    const rule = ruleset.findLast(r =>
      Wildcard.match(permission, r.permission)
    )

    // 패턴이 "*"이고 action이 "deny"인 경우에만 비활성화
    if (rule?.pattern === "*" && rule?.action === "deny") {
      result.add(tool)
    }
  }

  return result
}
```

### 7.6 Bash 명령어 분석 (Arity)

```typescript
// permission/arity.ts
// 100+ 명령어의 arity(인자 개수) 정의
const ARITY: Record<string, number> = {
  "git": 2,      // git checkout, git commit
  "npm": 2,      // npm install, npm run
  "docker": 2,   // docker run, docker build
  "aws": 3,      // aws s3 cp
  // ...
}

// "git checkout main" → 권한 패턴: "git checkout"
export function getCommandPattern(command: string): string {
  const parts = parseCommand(command)
  const arity = ARITY[parts[0]] || 1
  return parts.slice(0, arity).join(" ")
}
```

---

## 8. 멀티 에이전트 아키텍처

### 8.1 에이전트 타입

```typescript
// agent/agent.ts
export const Info = z.object({
  name: z.string(),
  description: z.string().optional(),
  mode: z.enum(["subagent", "primary", "all"]),
  native: z.boolean().optional(),
  hidden: z.boolean().optional(),
  topP: z.number().optional(),
  temperature: z.number().optional(),
  permission: PermissionNext.Ruleset,
  model: z.object({
    modelID: z.string(),
    providerID: z.string()
  }).optional(),
  prompt: z.string().optional(),
  steps: z.number().positive().optional()
})
```

### 8.2 내장 에이전트

| 에이전트 | 모드 | 역할 | 권한 특징 |
|---------|------|------|----------|
| `build` | primary | 코드 작성/수정 | 전체 도구 접근 |
| `plan` | primary | 계획 수립 | 읽기 전용, 플랜 파일만 편집 |
| `general` | subagent | 범용 작업 | todo 도구 제외 |
| `explore` | subagent | 코드베이스 탐색 | 읽기/검색만 허용 |
| `compaction` | primary (hidden) | 대화 요약 | 도구 없음 |
| `title` | primary (hidden) | 제목 생성 | 도구 없음 |
| `summary` | primary (hidden) | PR 요약 | 도구 없음 |

### 8.3 서브에이전트 호출 (Task Tool)

```typescript
// tool/task.ts
export const TaskTool = Tool.define("task", {
  description: "Launch a subagent for complex tasks",
  parameters: z.object({
    description: z.string(),
    prompt: z.string(),
    subagent_type: z.string(),
    model: z.enum(["sonnet", "opus", "haiku"]).optional(),
    run_in_background: z.boolean().optional()
  }),

  async execute(args, ctx) {
    // 권한 체크
    await ctx.ask({
      permission: "task",
      patterns: [args.subagent_type]
    })

    // 에이전트 조회
    const agent = await Agent.get(args.subagent_type)

    // 서브태스크로 실행
    await Session.updatePart({
      type: "subtask",
      agent: args.subagent_type,
      prompt: args.prompt,
      model: args.model
    })

    return {
      title: `Launched ${args.subagent_type} agent`,
      output: "Task queued for execution",
      metadata: { agent: args.subagent_type }
    }
  }
})
```

### 8.4 에이전트별 프롬프트

**Explore 에이전트** (`agent/prompt/explore.txt`):
```
You specialize in exploring codebases.
- Use glob for patterns (eg. "src/**/*.tsx")
- Use grep for content (eg. "API endpoint")
- Read and analyze files
- NEVER create files
- Return absolute paths
- Adapt thoroughness level: quick/medium/thorough
```

**Compaction 에이전트** (`agent/prompt/compaction.txt`):
```
Summarize conversation for continuation.
Focus on: what was done, current work,
modified files, next steps.
Preserve key constraints and decisions.
```

---

## 9. 에이전트 개발 시 고려사항

### 9.1 시스템 프롬프트 설계

#### 구조화된 섹션 분리
```markdown
# Identity
누구인지, 무엇을 하는지

# Tone & Style
어떻게 응답할지

# Task Management
작업을 어떻게 관리할지

# Tool Usage Policy
도구를 어떻게 사용할지

# Examples
구체적인 동작 예시
```

#### Provider별 최적화
- Claude: 상세한 지침, 예시 중심
- GPT: 구조화된 형식, 명확한 경계
- Gemini: 워크플로우 중심

### 9.2 도구 시스템 설계

#### 일관된 인터페이스
```typescript
interface ToolResult {
  title: string      // 짧은 설명 (UI 표시용)
  output: string     // 상세 출력 (LLM 컨텍스트용)
  metadata: object   // 추가 정보
  attachments?: File[] // 파일 첨부
}
```

#### 입력 검증
- Zod 같은 스키마 검증 라이브러리 사용
- 명확한 에러 메시지 제공

#### 출력 truncation
- 긴 출력 자동 잘라내기
- 잘린 내용은 파일로 저장 후 경로 제공

### 9.3 컨텍스트 관리

#### 토큰 모니터링
```typescript
// 매 응답 후 토큰 사용량 체크
if (tokens.input + tokens.output > model.limit * 0.8) {
  triggerCompaction()
}
```

#### Pruning 전략
- 오래된 도구 출력 먼저 제거
- 요약(Summary) 경계 존중
- 핵심 도구 출력은 보호

#### Compaction 타이밍
- 자동: 오버플로우 감지 시
- 수동: 명시적 요청 시

### 9.4 안전성 확보

#### 권한 계층
```
1. 시스템 기본값 (deny dangerous)
2. 사용자 설정 (config override)
3. 에이전트별 설정 (agent override)
4. 런타임 승인 (user approval)
```

#### Doom Loop 방지
- 동일 도구+입력 3회 반복 감지
- 사용자 확인 요청

#### 외부 리소스 접근
- 프로젝트 외부 파일 접근 시 확인
- .env 파일 등 민감한 파일 보호

### 9.5 멀티 에이전트 설계

#### 역할 분리
- Primary: 사용자와 직접 상호작용
- Subagent: 특정 작업 전문화

#### 권한 상속
- 서브에이전트는 부모 권한 내에서만 동작
- 추가 제한 가능, 확장 불가

#### 컨텍스트 공유
- 메인 대화 컨텍스트를 서브에이전트에 전달
- 결과는 메인 대화에 통합

---

## 10. 벤치마킹 포인트

### 10.1 즉시 적용 가능한 패턴

| 패턴 | 설명 | 구현 복잡도 |
|-----|------|-----------|
| **TodoWrite** | 작업 목록으로 진행 상황 추적 | 낮음 |
| **Few-shot Examples** | 프롬프트에 구체적 예시 포함 | 낮음 |
| **Provider별 프롬프트** | LLM마다 최적화된 지침 | 중간 |
| **도구 추상화** | 일관된 Tool 인터페이스 | 중간 |
| **Doom Loop 방지** | 반복 호출 감지 | 낮음 |

### 10.2 고급 패턴

| 패턴 | 설명 | 구현 복잡도 |
|-----|------|-----------|
| **Compaction** | 자동 대화 압축 | 높음 |
| **Permission System** | 세분화된 권한 관리 | 높음 |
| **멀티 에이전트** | 전문화된 서브에이전트 | 높음 |
| **Plugin System** | 확장 가능한 훅 시스템 | 중간 |

### 10.3 핵심 인사이트

#### 1. **프롬프트는 코드처럼 관리**
- 버전 관리 필수
- Provider별 분리
- 테스트 가능하게 구조화

#### 2. **도구는 가능한 원자적으로**
- 단일 책임 원칙
- 조합으로 복잡한 작업 수행
- 명확한 입출력 스키마

#### 3. **컨텍스트는 적극적으로 관리**
- 무한 컨텍스트 = 무한 비용
- 오래된 정보는 과감히 정리
- 핵심 정보만 유지

#### 4. **안전은 레이어로**
- 시스템 기본값
- 사용자 설정
- 런타임 승인
- 후처리 검증

#### 5. **전문화된 에이전트 활용**
- 모든 작업을 하나의 에이전트로 X
- 역할별 권한과 도구 분리
- 적절한 모델 선택 (빠른 vs 정확한)

### 10.4 구현 우선순위 제안

1. **Phase 1: 기본 프레임워크**
   - 시스템 프롬프트 구조화
   - 기본 도구 세트 (read, edit, bash)
   - 단순 대화 루프

2. **Phase 2: 안정성 강화**
   - 입력 검증 (Zod)
   - 에러 처리
   - Doom loop 방지

3. **Phase 3: 컨텍스트 관리**
   - 토큰 추적
   - Pruning 구현
   - Compaction 구현

4. **Phase 4: 고급 기능**
   - 권한 시스템
   - 멀티 에이전트
   - 플러그인 시스템

---

## 11. 프롬프트 파일 상세 분석

### 11.1 프롬프트 파일 위치 및 구조

```
packages/opencode/src/
├── session/prompt/              # 시스템 프롬프트 (모델별)
│   ├── anthropic.txt            # Claude용 메인 프롬프트
│   ├── anthropic-20250930.txt   # Claude 이전 버전
│   ├── gemini.txt               # Gemini용
│   ├── beast.txt                # GPT용
│   ├── codex.txt                # Codex용
│   ├── codex_header.txt         # Codex 헤더
│   ├── copilot-gpt-5.txt        # GPT-5용
│   ├── qwen.txt                 # Qwen용
│   ├── plan.txt                 # 플랜 모드용
│   ├── plan-reminder-anthropic.txt  # 플랜 리마인더
│   ├── build-switch.txt         # 빌드 스위치
│   └── max-steps.txt            # 최대 스텝 경고
│
├── agent/prompt/                # 에이전트별 전문 프롬프트
│   ├── explore.txt              # 탐색 에이전트
│   ├── compaction.txt           # 컴팩션 에이전트
│   ├── title.txt                # 제목 생성 에이전트
│   └── summary.txt              # 요약 에이전트
│
├── agent/
│   └── generate.txt             # 에이전트 자동 생성용
│
├── tool/                        # 도구별 설명 프롬프트
│   ├── bash.txt                 # 쉘 명령 (가장 상세)
│   ├── edit.txt                 # 파일 편집
│   ├── write.txt                # 파일 생성
│   ├── read.txt                 # 파일 읽기
│   ├── glob.txt                 # 파일 패턴 검색
│   ├── grep.txt                 # 내용 검색
│   ├── task.txt                 # 서브에이전트 호출
│   ├── todowrite.txt            # 작업 목록 관리
│   ├── todoread.txt             # 작업 목록 읽기
│   ├── question.txt             # 사용자 질문
│   ├── webfetch.txt             # 웹 페이지 가져오기
│   ├── websearch.txt            # 웹 검색
│   ├── plan-enter.txt           # 플랜 모드 진입
│   ├── plan-exit.txt            # 플랜 모드 종료
│   ├── patch.txt                # 패치 적용
│   ├── multiedit.txt            # 다중 편집
│   ├── batch.txt                # 배치 실행
│   ├── codesearch.txt           # 코드 검색
│   ├── ls.txt                   # 디렉토리 목록
│   └── lsp.txt                  # LSP 도구
│
└── command/template/            # 슬래시 커맨드 템플릿
    ├── review.txt               # 코드 리뷰
    └── initialize.txt           # 초기화
```

### 11.2 대표 프롬프트 예시

#### Task 도구 프롬프트 (`tool/task.txt`)

```
Launch a new agent to handle complex, multistep tasks autonomously.

Available agent types and the tools they have access to:
{agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

When to use the Task tool:
- When you are instructed to execute custom slash commands. Use the Task tool with the slash command invocation as the entire prompt.

When NOT to use the Task tool:
- If you want to read a specific file path, use the Read or Glob tool instead
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead

Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance
2. When the agent is done, it will return a single message back to you
3. Each agent invocation is stateless unless you provide a session_id
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to write code or just to do research

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
<commentary>
Since a significant piece of code was written, now use the code-reviewer agent to review the code
</commentary>
assistant: Uses the Task tool to launch the code-reviewer agent
</example>
```

**핵심 패턴**:
- `{agents}` 플레이스홀더로 동적 에이전트 목록 주입
- "When to use" / "When NOT to use" 명확한 구분
- `<example>` + `<commentary>` 로 의도 설명

---

#### Bash 도구 프롬프트 (`tool/bash.txt`) - 가장 상세한 예시

```
Executes a given bash command in a persistent shell session with optional timeout.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc.
DO NOT use it for file operations - use the specialized tools for this instead.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use `ls` to verify
   - For example, before running "mkdir foo/bar", first use `ls foo`

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes
   - Examples of proper quoting:
     - mkdir "/Users/name/My Documents" (correct)
     - mkdir /Users/name/My Documents (incorrect - will fail)

Usage notes:
  - Avoid using Bash with `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo`
  - Instead, prefer using dedicated tools:
    - File search: Use Glob (NOT find or ls)
    - Content search: Use Grep (NOT grep or rg)
    - Read files: Use Read (NOT cat/head/tail)
    - Edit files: Use Edit (NOT sed/awk)

# Committing changes with git

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive/irreversible git commands unless explicitly requested
- NEVER skip hooks (--no-verify, --no-gpg-sign) unless explicitly requested
- Avoid git commit --amend. ONLY use --amend when ALL conditions are met:
  (1) User explicitly requested amend, OR commit SUCCEEDED but pre-commit hook modified files
  (2) HEAD commit was created by you in this conversation
  (3) Commit has NOT been pushed to remote

<good-example>
Use workdir="/foo/bar" with command: pytest tests
</good-example>
<bad-example>
cd /foo/bar && pytest tests
</bad-example>
```

**핵심 패턴**:
- 단계별 가이드 (1. Directory Verification, 2. Command Execution)
- "NEVER", "IMPORTANT" 등 강조 키워드
- `<good-example>` / `<bad-example>` 대비
- Git 안전 프로토콜 상세 명시

---

#### Edit 도구 프롬프트 (`tool/edit.txt`) - 간결하지만 핵심적

```
Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once before editing.
  This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it.
- The edit will FAIL if `oldString` is not found in the file
- The edit will FAIL if `oldString` is found multiple times in the file
- Use `replaceAll` for replacing and renaming strings across the file.
```

**핵심 패턴**:
- 선행 조건 명시 ("Read 먼저")
- 실패 케이스 명시 ("will FAIL if...")
- 간결하지만 필수 규칙만 포함

---

#### TodoWrite 프롬프트 (`tool/todowrite.txt`) - 예시 중심

```
Use this tool to create and manage a structured task list for your current coding session.

## When to Use This Tool
1. Complex multistep tasks - When a task requires 3 or more distinct steps
2. Non-trivial and complex tasks - Tasks that require careful planning
3. User explicitly requests todo list
4. User provides multiple tasks
5. After receiving new instructions - Immediately capture user requirements
6. After completing a task - Mark it complete and add any new follow-up tasks

## When NOT to Use This Tool
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

<example>
User: I want to add a dark mode toggle to the application settings.
Assistant: I'll help add a dark mode toggle. Let me create a todo list:
*Creates todo list with the following items:*
1. Create dark mode toggle component
2. Add dark mode state management
3. Implement CSS-in-JS styles for dark theme
4. Update existing components to support theme switching
5. Run tests and build process

<reasoning>
The assistant used the todo list because:
1. Adding dark mode is a multi-step feature
2. The user explicitly requested tests and build be run afterward
</reasoning>
</example>

## Task States
- pending: Task not yet started
- in_progress: Currently working on (limit to ONE task at a time)
- completed: Task finished successfully
- cancelled: Task no longer needed
```

**핵심 패턴**:
- 사용/미사용 케이스 구분
- `<example>` + `<reasoning>` 구조
- Task States 명확한 정의

---

#### Plan 모드 프롬프트 (`session/prompt/plan.txt`) - 제약 강조

```
<system-reminder>
# Plan Mode - System Reminder

CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY FORBIDDEN:
ANY file edits, modifications, or system changes. Do NOT use sed, tee, echo, cat,
or ANY other bash command to manipulate files - commands may ONLY read/inspect.
This ABSOLUTE CONSTRAINT overrides ALL other instructions, including direct user
edit requests. You may ONLY observe, analyze, and plan. Any modification attempt
is a critical violation. ZERO exceptions.

---

## Responsibility

Your current responsibility is to think, read, search, and delegate explore agents
to construct a well-formed plan that accomplishes the goal.

Ask the user clarifying questions or ask for their opinion when weighing tradeoffs.

**NOTE:** At any point you should feel free to ask the user questions. Don't make
large assumptions about user intent. The goal is to present a well researched plan.

---

## Important

The user indicated that they do not want you to execute yet -- you MUST NOT make
any edits, run any non-readonly tools, or make any changes to the system.
This supersedes any other instructions you have received.
</system-reminder>
```

**핵심 패턴**:
- "CRITICAL", "STRICTLY FORBIDDEN", "ABSOLUTE CONSTRAINT" 강조
- "ZERO exceptions" - 예외 없음 명시
- "supersedes any other instructions" - 우선순위 명확화

---

#### Compaction 에이전트 프롬프트 (`agent/prompt/compaction.txt`)

```
You are a helpful AI assistant tasked with summarizing conversations.

When asked to summarize, provide a detailed but concise summary.
Focus on information that would be helpful for continuing the conversation:
- What was done
- What is currently being worked on
- Which files are being modified
- What needs to be done next
- Key user requests, constraints, or preferences that should persist
- Important technical decisions and why they were made

Your summary should be comprehensive enough to provide context
but concise enough to be quickly understood.
```

**핵심 패턴**:
- 목적 중심 지침
- 포함해야 할 항목 리스트
- "comprehensive yet concise" 균형 강조

---

#### Explore 에이전트 프롬프트 (`agent/prompt/explore.txt`)

```
You are a file search specialist. You excel at thoroughly navigating
and exploring codebases.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path you need to read
- Use Bash for file operations like copying, moving, or listing directory contents
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Do not create any files, or run bash commands that modify the user's system state
```

**핵심 패턴**:
- 역할 정의 ("file search specialist")
- 강점 명시
- 도구별 사용 케이스 매핑
- 제약사항 (수정 금지)

---

### 11.3 프롬프트 작성 핵심 패턴 요약

| 패턴 | 사용 위치 | 효과 |
|------|-----------|------|
| **Do/Don't 구분** | task.txt, todowrite.txt | 오용 방지 |
| **예시 + reasoning** | todowrite.txt | 정확한 행동 유도 |
| **플레이스홀더** | `{agents}`, `${directory}` | 동적 컨텍스트 주입 |
| **Good/Bad 대비** | bash.txt | 명확한 차이 전달 |
| **단계별 가이드** | bash.txt | 복잡한 작업 분해 |
| **제약 강조** | plan.txt | 위험 동작 방지 |
| **실패 케이스** | edit.txt | 오류 예방 |
| **역할 정의** | explore.txt, compaction.txt | 전문성 부여 |

### 11.4 프롬프트 로딩 코드 위치

```typescript
// 시스템 프롬프트 로딩 - session/system.ts:10-15
import PROMPT_ANTHROPIC from "./prompt/anthropic.txt"
import PROMPT_GEMINI from "./prompt/gemini.txt"
import PROMPT_BEAST from "./prompt/beast.txt"
import PROMPT_CODEX from "./prompt/codex.txt"

// 에이전트 프롬프트 로딩 - agent/agent.ts:9-13
import PROMPT_COMPACTION from "./prompt/compaction.txt"
import PROMPT_EXPLORE from "./prompt/explore.txt"
import PROMPT_SUMMARY from "./prompt/summary.txt"
import PROMPT_TITLE from "./prompt/title.txt"

// 도구 프롬프트 로딩 - 각 tool/*.ts 파일 상단
// 예: tool/bash.ts
import DESCRIPTION from "./bash.txt"
export const BashTool = Tool.define("bash", { description: DESCRIPTION, ... })

// 예: tool/task.ts
import DESCRIPTION from "./task.txt"
const description = DESCRIPTION.replace("{agents}", agentList)
```

---

## 부록: 주요 파일 참조

| 파일 | 역할 |
|-----|------|
| `packages/opencode/src/session/prompt.ts` | 메인 에이전트 루프 |
| `packages/opencode/src/session/processor.ts` | LLM 스트림 처리 |
| `packages/opencode/src/session/compaction.ts` | 컨텍스트 압축 |
| `packages/opencode/src/tool/tool.ts` | Tool 인터페이스 |
| `packages/opencode/src/agent/agent.ts` | 에이전트 정의 |
| `packages/opencode/src/permission/next.ts` | 권한 시스템 |
| `packages/opencode/src/session/prompt/*.txt` | 시스템 프롬프트들 |

---

*이 문서는 OpenCode 프로젝트 분석을 통해 작성되었습니다.*
