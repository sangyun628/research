# Claude Code 아키텍처 심층 분석서

> Claude Code 소스코드(2026-03-31 유출본) 기반 기술 분석
> 목적: 자체 AI 에이전트 구축을 위한 핵심 설계 패턴 및 기술 문서화

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [핵심 엔진: 대화 루프와 쿼리 엔진](#2-핵심-엔진-대화-루프와-쿼리-엔진)
3. [컨텍스트 관리 시스템](#3-컨텍스트-관리-시스템)
4. [메모리 시스템](#4-메모리-시스템)
5. [도구(Tool) 시스템](#5-도구tool-시스템)
6. [에이전트 시스템](#6-에이전트-시스템)
7. [권한 및 안전 시스템](#7-권한-및-안전-시스템)
8. [세션 관리 및 상태 지속성](#8-세션-관리-및-상태-지속성)
9. [통신 아키텍처 (Transport Layer)](#9-통신-아키텍처-transport-layer)
10. [브릿지 시스템 (원격 실행)](#10-브릿지-시스템-원격-실행)
11. [피처 플래그 시스템](#11-피처-플래그-시스템)
12. [자체 에이전트 구축 가이드](#12-자체-에이전트-구축-가이드)

---

## 1. 전체 아키텍처 개요

### 1.1 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                          │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│   UI Layer  │  Core Engine  │  Tool System  │  Service Layer   │
│  (Ink/React)│  (Coordinator)│  (18+ Tools)  │  (API/MCP/LSP)  │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│                    State & Persistence Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ AppState │ │ Memory   │ │ Context  │ │ Session Storage  │  │
│  │ (Global) │ │ (Memdir) │ │ Window   │ │ (NDJSON)         │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                    Transport / Bridge Layer                     │
│  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐   │
│  │ WebSocket │ │ SSE+POST │ │ Hybrid  │ │ REPL Bridge    │   │
│  └───────────┘ └──────────┘ └─────────┘ └────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **메시지 기반 아키텍처** | 모든 상호작용이 구조화된 메시지(SDKMessage)로 흐름 |
| **도구 중심 실행** | LLM이 직접 작업하지 않고, 도구를 호출하여 간접 실행 |
| **컨텍스트 윈도우 관리** | 자동 압축으로 무한 대화 가능 |
| **안전 우선** | 모든 도구 실행에 권한 검증 필수 |
| **멀티 트랜스포트** | 로컬/원격/IDE 등 다양한 환경 지원 |

### 1.3 디렉토리 구조 (전체 ~512K 라인 TypeScript)

```
claude-code/
├── coordinator/          # 대화 오케스트레이션 루프
├── QueryEngine.ts        # Claude API 호출 엔진
├── context/              # 컨텍스트 윈도우 관리
├── tools/                # 18+ 도구 구현체
│   ├── BashTool/
│   ├── FileReadTool/
│   ├── FileEditTool/
│   ├── AgentTool/        # 서브에이전트 생성
│   ├── WebSearchTool/
│   └── ...
├── services/
│   ├── api/              # Anthropic API 클라이언트
│   ├── mcp/              # Model Context Protocol
│   ├── compact/          # 대화 압축 서비스
│   ├── extractMemories/  # 메모리 자동 추출
│   └── lsp/              # Language Server Protocol
├── state/                # 전역 상태 관리
├── memdir/               # 영구 메모리 시스템
├── bridge/               # 원격 실행 브릿지
├── cli/                  # CLI 입출력 처리
├── commands/             # 80+ 슬래시 커맨드
├── skills/               # 재사용 가능 프롬프트 템플릿
├── hooks/                # 도구 권한 검증 훅
├── components/           # UI 컴포넌트
└── ink/                  # 커스텀 터미널 렌더링 엔진
```

---

## 2. 핵심 엔진: 대화 루프와 쿼리 엔진

### 2.1 Coordinator 패턴 (핵심 오케스트레이션)

Claude Code의 심장은 `coordinator/coordinatorMode.ts`에 있는 **대화 루프**이다. 이것은 전형적인 ReAct(Reasoning + Acting) 에이전트 루프를 구현한다:

```
사용자 입력 → LLM 추론 → 도구 호출 → 결과 수집 → LLM 재추론 → ... → 최종 응답
```

#### 대화 루프 의사코드

```typescript
async function coordinatorLoop(userMessage: string) {
  // 1. 컨텍스트 구성
  const messages = contextManager.buildMessages(userMessage)
  
  // 2. 시스템 프롬프트 구성 (동적)
  const systemPrompt = buildSystemPrompt({
    tools: getAvailableTools(),
    memory: await loadRelevantMemories(),
    gitStatus: await getGitStatus(),
    environment: getEnvironmentInfo(),
    permissions: getCurrentPermissionMode(),
  })
  
  while (true) {
    // 3. LLM에 쿼리
    const response = await queryEngine.ask(messages, systemPrompt)
    
    // 4. 응답 처리
    for (const block of response.content) {
      if (block.type === 'text') {
        // 텍스트 응답 → 사용자에게 표시
        yield { type: 'text', content: block.text }
      }
      else if (block.type === 'tool_use') {
        // 5. 권한 확인
        const permission = await checkPermission(block.name, block.input)
        if (permission === 'denied') {
          messages.push(toolDeniedResult(block))
          continue
        }
        
        // 6. 도구 실행
        const result = await executeTool(block.name, block.input)
        messages.push(toolResult(block.id, result))
      }
    }
    
    // 7. 도구 호출이 없었으면 루프 종료
    if (!hasToolUse(response)) break
    
    // 8. 컨텍스트 윈도우 체크 → 필요시 압축
    if (contextManager.isNearLimit()) {
      messages = await compactService.compress(messages)
    }
  }
}
```

### 2.2 QueryEngine (API 통신)

`QueryEngine.ts`는 Claude API와의 통신을 담당한다:

```
┌─────────────────────────────────────────────┐
│              QueryEngine.ask()               │
├─────────────────────────────────────────────┤
│ 1. 메시지 직렬화                             │
│ 2. 스트리밍 API 호출 (SSE)                   │
│ 3. content_block_delta 실시간 처리           │
│ 4. tool_use 블록 감지 시 도구 실행 트리거     │
│ 5. 토큰 사용량 추적                          │
│ 6. 레이트 리밋 핸들링                        │
│ 7. 에러 복구 (재시도 + 백오프)               │
└─────────────────────────────────────────────┘
```

**핵심 기법들:**

- **스트리밍 응답 처리**: Server-Sent Events(SSE) 기반으로 토큰 단위 실시간 출력
- **레이트 리밋 대응**: `rate_limit_event` 메시지를 클라이언트에 전파, 자동 대기
- **모델 폴백**: 주 모델 실패 시 대체 모델로 자동 전환
- **토큰 카운팅**: 입출력 토큰을 실시간 추적하여 비용 계산

---

## 3. 컨텍스트 관리 시스템

이것이 Claude Code의 **가장 핵심적인 기술적 차별점**이다. 무한한 대화를 유한한 컨텍스트 윈도우에서 처리하는 방법.

### 3.1 컨텍스트 윈도우 구성

```
┌──────────────────────────────────────────────────────────────┐
│                    컨텍스트 윈도우 (1M 토큰)                   │
├──────────────────────────────────────────────────────────────┤
│ [System Prompt]         ~2K-5K 토큰                          │
│  ├─ 기본 행동 지침                                           │
│  ├─ 환경 정보 (OS, shell, git status)                        │
│  ├─ 메모리 (MEMORY.md 인덱스)                                │
│  ├─ 사용 가능 도구 목록                                       │
│  └─ 권한 모드                                                │
├──────────────────────────────────────────────────────────────┤
│ [Conversation History]  가변 (자동 압축)                      │
│  ├─ 사용자 메시지                                             │
│  ├─ 어시스턴트 응답                                           │
│  ├─ 도구 호출 + 결과                                          │
│  └─ 시스템 리마인더 (동적 주입)                                │
├──────────────────────────────────────────────────────────────┤
│ [Current Turn]          현재 턴의 메시지들                     │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 자동 압축 (Compaction Service)

대화가 길어지면 `services/compact/` 서비스가 자동으로 컨텍스트를 압축한다:

```
자동 압축 트리거 조건:
  ├─ 컨텍스트 사용률 > 임계값 (보통 80%)
  ├─ /compact 명령어 수동 실행
  └─ 턴 사이 자동 체크

압축 전략:
  1. 최근 N턴은 보존 (원본 유지)
  2. 이전 대화를 요약으로 대체
  3. 도구 실행 결과는 핵심만 추출
  4. 파일 내용은 참조(경로)로 대체
  5. 압축된 요약 + 최근 턴 = 새 컨텍스트

압축 후 메시지 구조:
  [System Prompt]
  [압축 요약: "이전 대화에서 X, Y, Z를 진행했음..."]
  [최근 N턴 원본]
  [현재 턴]
```

### 3.3 시스템 리마인더 (System Reminders)

Claude Code는 **동적 시스템 리마인더**를 대화 중간에 주입하는 독특한 패턴을 사용한다:

```typescript
// 시스템 리마인더는 도구 결과나 사용자 메시지에 태그로 삽입됨
<system-reminder>
현재 날짜: 2026-04-01
사용 가능한 도구: Read, Edit, Bash, ...
권한 모드: auto-accept
</system-reminder>
```

**목적:**
- LLM의 "잊어버림"을 방지 (긴 대화에서 초반 시스템 프롬프트 효과 감소)
- 동적 상태 업데이트 (날짜, 가용 도구, 권한 변경 등)
- 컨텍스트 윈도우의 어느 위치에서든 정보 주입 가능

### 3.4 메시지 중복 제거

```typescript
// Circular Buffer 기반 UUID 추적 (용량 10,000)
class MessageDeduplicator {
  private buffer: string[] = []
  private set = new Set<string>()
  private readonly capacity = 10_000
  
  track(uuid: string): boolean {
    if (this.set.has(uuid)) return false // 중복
    this.set.add(uuid)
    this.buffer.push(uuid)
    if (this.buffer.length > this.capacity) {
      const old = this.buffer.shift()!
      this.set.delete(old)
    }
    return true // 새 메시지
  }
}
```

### 3.5 세션 히스토리 페이지네이션

```typescript
// 커서 기반 페이지네이션으로 과거 대화 로드
type HistoryPage = {
  events: SDKMessage[]      // 시간순 이벤트
  firstId: string | null    // 이전 페이지 커서
  hasMore: boolean          // 더 이전 이벤트 존재 여부
}

// 사용 패턴:
// 1. fetchLatestEvents(ctx, 100) → 최근 100개 이벤트
// 2. fetchOlderEvents(ctx, firstId) → 이전 100개
// 3. hasMore === false 까지 반복
```

---

## 4. 메모리 시스템

Claude Code의 메모리는 **파일 기반 영구 저장소**로, 대화 간 컨텍스트를 유지한다.

### 4.1 메모리 아키텍처

```
~/.claude/projects/<project-hash>/memory/
├── MEMORY.md                    # 메모리 인덱스 (항상 컨텍스트에 로드)
├── user_role.md                 # 사용자 프로필
├── feedback_testing.md          # 피드백/교정 사항
├── project_auth_rewrite.md      # 프로젝트 진행 상황
└── reference_linear.md          # 외부 자원 참조

CLAUDE.md (프로젝트 루트)         # 프로젝트별 지침 (git 추적 가능)
```

### 4.2 메모리 유형

| 유형 | 목적 | 저장 시점 | 사용 시점 |
|------|------|----------|----------|
| **user** | 사용자 역할, 선호도, 전문분야 | 사용자 정보 학습 시 | 응답 맞춤화 시 |
| **feedback** | 접근 방식에 대한 교정/확인 | 사용자가 교정하거나 비전형적 접근 확인 시 | 작업 방향 결정 시 |
| **project** | 프로젝트 목표, 마감, 결정 사항 | 비코드 프로젝트 정보 학습 시 | 제안/결정 맥락 제공 시 |
| **reference** | 외부 시스템 위치 정보 | 외부 자원 위치 학습 시 | 외부 정보 참조 시 |

### 4.3 메모리 파일 포맷

```markdown
---
name: 테스트 피드백
description: 통합 테스트에서 DB 모킹 금지 규칙
type: feedback
---

통합 테스트에서는 실제 DB를 사용해야 함, 모킹 금지.

**Why:** 지난 분기 모킹된 테스트가 통과했지만 프로덕션 마이그레이션이 실패한 사고 발생
**How to apply:** 테스트 코드 작성/수정 시 DB 모킹 대신 테스트 DB 연결 사용
```

### 4.4 자동 메모리 추출 (Extract Memories)

`services/extractMemories/` 서비스는 대화에서 자동으로 중요 정보를 추출:

```
대화 종료 시:
  1. 대화 전체 분석
  2. 사용자 프로필 정보 추출 → user 메모리
  3. 교정/확인 패턴 추출 → feedback 메모리
  4. 프로젝트 결정 사항 추출 → project 메모리
  5. 기존 메모리와 중복 체크
  6. 새 메모리 저장 또는 기존 메모리 업데이트
```

### 4.5 MEMORY.md 인덱스

```markdown
- [사용자 역할](user_role.md) — 시니어 백엔드 개발자, Go 전문
- [테스트 피드백](feedback_testing.md) — 통합 테스트에서 DB 모킹 금지
- [인증 리라이트](project_auth_rewrite.md) — 법적 컴플라이언스로 인한 인증 미들웨어 재작성
```

**핵심 설계 결정:**
- MEMORY.md는 **항상** 시스템 프롬프트에 포함됨 (200줄 제한)
- 각 메모리 파일은 필요 시에만 읽음 (온디맨드)
- 메모리는 "주장"일 뿐 — 실제 코드/파일 상태와 충돌 시 현재 상태를 신뢰

### 4.6 CLAUDE.md (프로젝트 지침)

```
프로젝트 루트/CLAUDE.md → 프로젝트별 영구 지침
~/.claude/CLAUDE.md → 전역 사용자 지침

로드 우선순위:
  1. 전역 CLAUDE.md
  2. 프로젝트 CLAUDE.md
  3. MEMORY.md 인덱스
  4. 개별 메모리 파일 (온디맨드)
```

---

## 5. 도구(Tool) 시스템

### 5.1 도구 아키텍처

```typescript
// 도구 인터페이스 (간략화)
interface Tool {
  name: string
  description: string           // LLM이 도구를 이해하는 데 사용
  parameters: JSONSchema        // 입력 파라미터 스키마
  
  execute(input: ToolInput): Promise<ToolResult>
  
  // 권한 관련
  permissionLevel: 'read' | 'write' | 'dangerous'
  requiresConfirmation: boolean
}

// 도구 실행 파이프라인
async function executeTool(name: string, input: any): Promise<ToolResult> {
  // 1. 도구 조회
  const tool = toolRegistry.get(name)
  
  // 2. 입력 검증 (Zod 스키마)
  const validated = tool.schema.parse(input)
  
  // 3. 권한 확인
  const perm = await checkPermission(tool, validated)
  if (perm.denied) return { error: perm.reason }
  
  // 4. 훅 실행 (pre-tool hooks)
  await runPreToolHooks(tool, validated)
  
  // 5. 도구 실행
  const result = await tool.execute(validated)
  
  // 6. 훅 실행 (post-tool hooks)
  await runPostToolHooks(tool, validated, result)
  
  // 7. 결과 반환
  return result
}
```

### 5.2 내장 도구 목록 및 기능

| 도구 | 권한 수준 | 핵심 기능 |
|------|----------|----------|
| **Read** | read | 파일 읽기, 이미지/PDF 지원, 라인 범위 지정 |
| **Glob** | read | 파일 패턴 매칭 (`**/*.ts`), 수정 시간순 정렬 |
| **Grep** | read | ripgrep 기반 콘텐츠 검색, 정규식/멀티라인 지원 |
| **Bash** | dangerous | 셸 명령 실행, 타임아웃, 백그라운드 실행 |
| **Edit** | write | 정확한 문자열 치환 기반 파일 수정 |
| **Write** | write | 파일 생성/전체 덮어쓰기 |
| **NotebookEdit** | write | Jupyter 노트북 셀 편집 |
| **Agent** | write | 서브에이전트 생성 (독립 컨텍스트) |
| **WebSearch** | read | 웹 검색 |
| **WebFetch** | read | URL 콘텐츠 가져오기 |
| **TaskCreate** | write | 백그라운드 작업 생성 |
| **ToolSearch** | read | 지연 로드 도구 스키마 검색 |
| **Skill** | write | 재사용 스킬 프롬프트 실행 |
| **SendMessage** | write | 실행 중인 서브에이전트에 메시지 전송 |
| **EnterPlanMode** | read | 계획 수립 모드 전환 |
| **MCPTool** | varies | MCP 서버 도구 호출 |
| **LSPTool** | read | Language Server 통합 |

### 5.3 도구 지연 로드 (Deferred Tools)

모든 도구를 한번에 로드하면 시스템 프롬프트가 너무 커진다. Claude Code는 **지연 로드** 패턴을 사용:

```
초기 로드:
  ├─ 핵심 도구 (Read, Edit, Bash, Grep, Glob, Write) → 즉시 사용 가능
  └─ 나머지 도구 → 이름만 노출, 스키마는 숨김

사용자가 특수 도구 필요 시:
  1. LLM이 ToolSearch 도구로 스키마 검색
  2. 스키마가 컨텍스트에 주입됨
  3. 이후 해당 도구 호출 가능

장점:
  - 시스템 프롬프트 토큰 절약 (~50% 감소)
  - 불필요한 도구의 "유혹" 방지
  - MCP 도구 동적 확장 가능
```

### 5.4 MCP (Model Context Protocol) 통합

```
Claude Code
    │
    ├─ MCP Client
    │   ├─ connect(serverConfig) → stdio/SSE transport
    │   ├─ listTools() → 도구 스키마 가져오기
    │   ├─ callTool(name, args) → 결과 반환
    │   └─ listResources() → 리소스 목록
    │
    └─ 서버 설정 (.claude/settings.json)
        ├─ project scope: 프로젝트별 MCP 서버
        ├─ user scope: 사용자별 MCP 서버
        └─ local scope: 로컬 전용 MCP 서버
```

---

## 6. 에이전트 시스템

### 6.1 서브에이전트 아키텍처

Claude Code의 **AgentTool**은 독립적인 서브에이전트를 생성하여 복잡한 작업을 병렬 처리한다:

```
메인 에이전트 (부모)
    │
    ├─ Agent("코드베이스 탐색", type=Explore)
    │   ├─ 독립 컨텍스트 윈도우
    │   ├─ 제한된 도구 세트 (Read, Grep, Glob만)
    │   └─ 결과를 부모에게 요약 반환
    │
    ├─ Agent("테스트 실행", type=general-purpose)
    │   ├─ 전체 도구 사용 가능
    │   ├─ 독립 컨텍스트
    │   └─ 결과를 부모에게 반환
    │
    └─ Agent("PR 리뷰", type=Plan)
        ├─ 읽기 전용 도구만
        ├─ 독립 컨텍스트
        └─ 계획/분석 결과 반환
```

### 6.2 에이전트 유형

| 유형 | 목적 | 사용 가능 도구 | 특징 |
|------|------|-------------|------|
| **general-purpose** | 범용 작업 | 전체 | 가장 유연, 기본값 |
| **Explore** | 코드베이스 탐색 | Read, Grep, Glob, WebFetch | 빠르고 가벼움 |
| **Plan** | 설계/계획 수립 | Read, Grep, Glob (편집 불가) | 읽기 전용 |
| **statusline-setup** | 설정 변경 | Read, Edit | 특수 목적 |
| **claude-code-guide** | 도움말/가이드 | Glob, Grep, Read, WebFetch | 문서 참조 특화 |

### 6.3 에이전트 격리 모드

```typescript
// Git worktree 격리: 서브에이전트가 독립된 코드 복사본에서 작업
Agent({
  prompt: "리팩토링 수행",
  isolation: "worktree"  // 임시 git worktree 생성
})

// Worktree 격리 흐름:
// 1. git worktree add .claude-worktree-{id} → 독립 복사본
// 2. 서브에이전트가 복사본에서 작업
// 3. 변경 없으면 → 자동 정리
// 4. 변경 있으면 → worktree 경로 + 브랜치 반환
```

### 6.4 에이전트 간 통신

```
부모 → 자식: Agent 도구로 프롬프트 전달
자식 → 부모: 실행 완료 시 결과 텍스트 반환
부모 → 실행 중 자식: SendMessage 도구로 추가 지시

핵심 제약:
  - 자식은 부모의 컨텍스트를 모름
  - 부모는 자식에게 충분한 맥락을 제공해야 함
  - 자식의 결과는 부모에게만 보임 (사용자에게 직접 표시 안 됨)
```

### 6.5 백그라운드 작업 (Tasks)

```typescript
// 백그라운드 에이전트 실행
Agent({
  prompt: "전체 테스트 스위트 실행",
  run_in_background: true
})

// 백그라운드 작업 관리:
// TaskCreate → 작업 생성
// TaskGet → 진행 상태 확인
// TaskList → 전체 작업 목록
// TaskUpdate → 상태 업데이트
// TaskStop → 작업 중지
// TaskOutput → 결과 조회
```

---

## 7. 권한 및 안전 시스템

### 7.1 권한 계층

```
┌─────────────────────────────────────────────┐
│              Permission Modes                │
├─────────────────────────────────────────────┤
│ plan         : 읽기 전용, 편집/실행 불가      │
│ default      : 읽기 허용, 쓰기/실행은 확인     │
│ auto-accept  : 대부분 자동 승인               │
│ full-auto    : 모든 도구 자동 승인             │
└─────────────────────────────────────────────┘
```

### 7.2 권한 검증 파이프라인

```
도구 호출 요청
    │
    ├─ hasPermissionsToUseTool() [동기]
    │   ├─ 'allow'           → 즉시 실행
    │   ├─ 'deny'            → 거부 반환
    │   └─ 'requires_action' → 아래로 진행
    │
    ├─ createCanUseTool() [비동기, 레이스 조건 패턴]
    │   ├─ 훅 실행 (settings.json의 hooks)  ─┐
    │   └─ SDK 권한 프롬프트 (사용자에게 질문) ─┤
    │                                        ├─ 먼저 결정된 쪽이 승리
    │                                        └─ 나머지는 취소
    │
    └─ 결과
        ├─ 승인: 도구 실행 + 권한 캐시 업데이트
        └─ 거부: 거부 사유 포함 결과 반환
```

### 7.3 훅 시스템

```json
// settings.json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "./scripts/validate-bash-command.sh"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "command": "./scripts/log-tool-usage.sh"
      }
    ]
  }
}
```

**훅 유형:**
- `PreToolUse`: 도구 실행 전 검증 (거부 가능)
- `PostToolUse`: 도구 실행 후 로깅/알림
- `UserPromptSubmit`: 사용자 입력 전처리

### 7.4 샌드박스 실행

```
Bash 도구 실행 시:
  ├─ macOS: Apple Sandbox (sandbox-exec) 활용
  ├─ Linux: 네임스페이스 격리 / seccomp
  └─ 공통:
      ├─ 파일시스템 접근 제한
      ├─ 네트워크 접근 제한 (설정 가능)
      ├─ 프로세스 생성 제한
      └─ 타임아웃 강제 (기본 120초, 최대 600초)
```

---

## 8. 세션 관리 및 상태 지속성

### 8.1 세션 저장소

```
~/.claude/projects/<project-hash>/
├── sessions/
│   ├── <session-id>/
│   │   ├── transcript.ndjson    # 전체 대화 기록 (턴별 NDJSON)
│   │   ├── state.json           # 세션 상태 (모델, 권한 등)
│   │   └── metadata.json        # 메타데이터 (시작 시간, 프로젝트 등)
│   └── ...
├── memory/                      # 영구 메모리 (위 참조)
└── settings.json                # 프로젝트별 설정
```

### 8.2 세션 복구

```
세션 복구 시나리오:
  1. 명시적 재개: claude --resume <session-id>
  2. 마지막 세션 이어가기: claude --continue
  3. 중단된 턴 자동 복구: CLAUDE_CODE_RESUME_INTERRUPTED_TURN
  4. CCR v2 워커 복구: restoredWorkerState → 외부 메타데이터에서 복원

복구 과정:
  1. 세션 ID로 transcript.ndjson 로드
  2. 메시지 히스토리 재구성
  3. 파일 상태 동기화 (readFileState)
  4. 중단점부터 재개
```

### 8.3 전역 상태 (AppState)

```typescript
// 전역 상태 구조 (간략화)
type AppState = {
  // 대화 상태
  messages: Message[]
  currentTurn: 'user' | 'assistant'
  isStreaming: boolean
  
  // 도구 상태
  pendingToolUses: Map<string, ToolUse>
  completedToolUses: Map<string, ToolResult>
  
  // 권한 상태
  permissionMode: PermissionMode
  toolPermissions: Map<string, PermissionDecision>
  
  // UI 상태
  footerSelection: 'input' | 'companion' | 'settings'
  isFullscreen: boolean
  
  // 컴패니언 상태 (버디 시스템)
  companionReaction: string | null
  companionPetAt: number | null
}
```

### 8.4 NDJSON 직렬화

```typescript
// 안전한 NDJSON 직렬화 (U+2028/U+2029 이스케이프)
function ndjsonSafeStringify(obj: unknown): string {
  return JSON.stringify(obj).replace(/\u2028|\u2029/g, (ch) => 
    `\\u${ch.charCodeAt(0).toString(16).padStart(4, '0')}`
  )
}

// 이유: JSON.stringify는 U+2028/U+2029를 그대로 출력하지만,
// 이들은 JavaScript에서 줄 종결자로 취급됨
// → NDJSON 파서가 메시지를 잘못 분리할 수 있음
```

---

## 9. 통신 아키텍처 (Transport Layer)

### 9.1 트랜스포트 계층 구조

```
┌───────────────────────────────────────────────────┐
│                StructuredIO (기반)                  │
│  - NDJSON 메시지 파싱                              │
│  - control_request/response 프로토콜               │
│  - 권한 결정 레이스 조건 패턴                       │
│  - UUID 기반 중복 제거                              │
├───────────────────────────────────────────────────┤
│                RemoteIO (확장)                      │
│  - 트랜스포트 선택 로직                             │
│  - CCR v2 워커 라이프사이클                         │
│  - Keep-alive (120초 간격)                          │
│  - 세션 상태 복원                                   │
├───────────────┬───────────────┬───────────────────┤
│  WebSocket    │  Hybrid       │  SSE+POST         │
│  Transport    │  Transport    │  Transport        │
│  (양방향 WS)  │  (WS읽기+POST)│  (SSE읽기+POST)   │
└───────────────┴───────────────┴───────────────────┘
```

### 9.2 트랜스포트 선택 우선순위

```
getTransportForUrl():
  CCR v2 (SSE+POST)  >  HybridTransport (WS+POST)  >  WebSocketTransport (WS)
```

### 9.3 WebSocket Transport 상세

```
기능:
├─ 런타임 감지: Bun 네이티브 WS > ws npm 패키지
├─ 메시지 버퍼링: CircularBuffer (1000 용량)
├─ 재연결 로직:
│   ├─ 지수 백오프: 1초 → 30초 (지터 포함)
│   ├─ 수면 감지: 60초 이상 갭 → 예산 초기화
│   ├─ 영구 종료 코드: 1002, 4001, 4003 (재시도 안 함)
│   └─ 재연결 예산: 최대 10분
├─ Ping/Pong: 10초 간격 헬스체크
└─ Keep-alive: 5분 간격 프레임 전송

상태 머신:
  idle → reconnecting → connected → closing → closed
```

### 9.4 Hybrid Transport (읽기 WS + 쓰기 POST)

```
쓰기 최적화:
  stream_event → 100ms 지연 버퍼 → 배치 POST
  기타 메시지  → 버퍼 즉시 플러시 → POST → 플러시

백프레셔:
  큐 최대 크기: 100,000
  enqueue() 큐 가득 시 블로킹
  drain 진행 시 해제
```

### 9.5 SSE Transport (Server-Sent Events)

```
읽기: SSE 스트림 (시퀀스 번호 기반 재개)
쓰기: HTTP POST

이벤트 합치기:
  text_delta 이벤트 100ms 동안 축적
  전체-현재까지 텍스트 스냅샷으로 방출
  중간 재연결 시 완전한 텍스트 수신 가능

라이브니스 감지:
  45초 SSE 침묵 시 타임아웃
  keepalive 코멘트도 라이브니스로 인정
```

### 9.6 SerialBatchEventUploader

```
직렬 배칭 업로더:

생산자 → enqueue() → 대기열 → takeBatch() → POST → 결과 처리
                       ↑                          │
                       └──────── 재시도 ───────────┘

배치 제한:
  maxBatchSize: 100 아이템
  maxBatchBytes: 10MB
  maxQueueSize: 100,000

재시도 전략:
  지수 백오프: baseDelay × 2^(failures-1)
  지터: ±[0, jitterMs]
  Retry-After 헤더 지원
  최대 연속 실패 후 배치 드롭
```

---

## 10. 브릿지 시스템 (원격 실행)

### 10.1 세 가지 브릿지 구현

```
1. Environment Bridge (bridgeMain.ts, 3500줄)
   ├─ 멀티세션 독립 브릿지
   ├─ Environments API 등록
   ├─ 폴링으로 작업 수신
   └─ 동시 세션 관리 (최대 32+)

2. REPL Bridge (replBridge.ts, 2400줄)
   ├─ 직접 WebSocket 세션 브릿지
   ├─ Environment 계층 없음
   └─ CCR v1/v2 트랜스포트 지원

3. Envless Bridge (remoteBridgeCore.ts)
   ├─ 경량 OAuth→JWT 브릿지
   ├─ Environment API 불필요
   └─ REPL 전용 세션
```

### 10.2 세션 라이프사이클

```
┌─────────────────────────────────────────────────┐
│ 1. 초기화                                        │
│    ├─ registerBridgeEnvironment() → env_id       │
│    ├─ QR 코드 출력                                │
│    └─ 상태 표시 타이머 시작                        │
│                                                   │
│ 2. 폴링 루프 (loopSignal 종료까지 반복)            │
│    ├─ pollForWork(envId, secret)                  │
│    │   ├─ null → 대기, 계속                       │
│    │   └─ work → 처리                             │
│    ├─ healthcheck → 확인 응답                      │
│    └─ session →                                   │
│        ├─ worktree 생성 (격리 모드)                │
│        ├─ 워커 등록 (CCR v2)                       │
│        ├─ 자식 프로세스 생성                        │
│        ├─ 토큰 갱신 스케줄링                        │
│        └─ 세션 종료 핸들러 등록                     │
│                                                   │
│ 3. 세션 완료                                      │
│    ├─ stopWorkWithRetry()                         │
│    ├─ archiveSession()                            │
│    ├─ worktree 정리                                │
│    └─ 싱글: 종료 / 멀티: 폴링 계속                  │
│                                                   │
│ 4. 셧다운                                         │
│    ├─ loopSignal abort                            │
│    ├─ 모든 세션 종료 (30초 유예 → SIGKILL)          │
│    └─ deregisterEnvironment()                     │
└─────────────────────────────────────────────────┘
```

### 10.3 인증 및 보안

```
JWT 토큰 관리:
├─ 만료 5분 전 사전 갱신
├─ 세대 카운터로 오래된 비동기 작업 무효화
├─ 최대 3회 재시도 (60초 간격)
└─ 후속 갱신 30분마다 (장기 세션 대비)

신뢰 장치 토큰:
├─ macOS 키체인에 저장
├─ 90일 롤링 만료
├─ GrowthBook 피처 게이트로 강제
└─ 2FA 강화 세션에 필수

Work Secret 디코딩:
├─ base64url → JSON → 검증
├─ session_ingress_token (JWT)
├─ api_base_url
├─ git 소스, 환경 변수, MCP 설정 포함
└─ 자식 프로세스에 전달
```

### 10.4 중복 제거 (BoundedUUIDSet)

```typescript
class BoundedUUIDSet {
  private _set = new Set<string>()
  private _queue: string[] = []
  
  add(uuid: string): void {
    if (this._set.has(uuid)) return
    this._set.add(uuid)
    this._queue.push(uuid)
    if (this._queue.length > MAX_SIZE) {
      const old = this._queue.shift()!
      this._set.delete(old)
    }
  }
  
  has(uuid: string): boolean {
    return this._set.has(uuid)
  }
}

// 두 가지 중복 제거 전략 동시 사용:
// 1. recentPostedUUIDs: 자신이 보낸 메시지의 에코 방지
// 2. recentInboundUUIDs: 히스토리 재전송 중복 방지
```

---

## 11. 피처 플래그 시스템

### 11.1 컴파일 타임 피처 플래그

```javascript
// node_modules/bundle/index.js
const ENABLED_FEATURES = new Set([
  // 'KAIROS',                 // 어시스턴트 / 데일리 로그 모드
  // 'PROACTIVE',              // 자율 선행 실행 모드
  // 'BRIDGE_MODE',            // IDE 브릿지
  // 'VOICE_MODE',             // 음성 입력
  // 'COORDINATOR_MODE',       // 멀티에이전트 스웜 코디네이터
  // 'BUDDY',                  // 컴패니언 스프라이트
  // 'WEB_BROWSER_TOOL',       // 인프로세스 웹 브라우저
  // 'CHICAGO_MCP',            // 컴퓨터 사용 (화면 제어)
  // 'AGENT_TRIGGERS',         // 스케줄 크론 에이전트
  // 'ULTRAPLAN',              // 울트라 상세 계획 모드
  // 'EXTRACT_MEMORIES',       // 백그라운드 메모리 추출
  // 'TEAMMEM',                // 팀 공유 메모리
  // 기타 20+ 플래그
])

function feature(name) {
  return ENABLED_FEATURES.has(name)
}
```

### 11.2 런타임 피처 게이트 (GrowthBook)

```
GrowthBook 통합:
├─ 서버에서 피처 플래그 가져오기
├─ 캐시: 디스크 + 메모리 (5분 갱신)
├─ checkGate_CACHED_OR_BLOCKING(): 처음엔 블로킹, 이후 캐시
├─ getFeatureValue_CACHED_WITH_REFRESH(): 캐시 값 반환 + 백그라운드 갱신
└─ 기본값 폴백: 서버 실패 시 하드코딩 기본값 사용
```

---

## 12. 자체 에이전트 구축 가이드

Claude Code 분석에서 추출한 핵심 패턴을 기반으로 자체 에이전트를 구축하는 가이드.

### 12.1 최소 실행 가능 에이전트 (MVP) 구조

```
my-agent/
├── src/
│   ├── main.ts                  # 진입점
│   ├── coordinator.ts           # ReAct 대화 루프
│   ├── queryEngine.ts           # LLM API 통신
│   ├── contextManager.ts        # 컨텍스트 윈도우 관리
│   ├── tools/                   # 도구 구현체
│   │   ├── registry.ts          # 도구 레지스트리
│   │   ├── read.ts
│   │   ├── write.ts
│   │   ├── bash.ts
│   │   └── search.ts
│   ├── memory/                  # 영구 메모리
│   │   ├── memoryManager.ts
│   │   └── types.ts
│   ├── permissions/             # 권한 시스템
│   │   └── permissionChecker.ts
│   └── types.ts                 # 공통 타입
├── package.json
└── tsconfig.json
```

### 12.2 핵심 구현 체크리스트

#### A. ReAct 대화 루프 (최우선)

```typescript
// coordinator.ts
async function* agentLoop(userMessage: string): AsyncGenerator<AgentEvent> {
  const messages: Message[] = []
  messages.push({ role: 'user', content: userMessage })
  
  while (true) {
    // 1. 컨텍스트 윈도우 체크 + 필요시 압축
    if (estimateTokens(messages) > MAX_CONTEXT * 0.8) {
      messages = await compressContext(messages)
    }
    
    // 2. LLM 호출
    const response = await queryEngine.stream({
      system: buildSystemPrompt(),
      messages,
      tools: toolRegistry.getSchemas(),
    })
    
    // 3. 응답 처리
    const toolCalls = extractToolCalls(response)
    messages.push({ role: 'assistant', content: response.content })
    
    if (toolCalls.length === 0) {
      yield { type: 'final_response', content: response.text }
      break
    }
    
    // 4. 도구 실행
    for (const call of toolCalls) {
      const permission = await checkPermission(call)
      if (!permission.allowed) {
        messages.push(toolDenied(call))
        continue
      }
      
      yield { type: 'tool_start', tool: call.name }
      const result = await executeTool(call)
      messages.push(toolResult(call.id, result))
      yield { type: 'tool_end', tool: call.name, result }
    }
  }
}
```

#### B. 컨텍스트 관리 (핵심 차별점)

```typescript
// contextManager.ts
class ContextManager {
  private maxTokens: number
  private reservedForResponse: number = 4096
  
  async compress(messages: Message[]): Promise<Message[]> {
    // 전략 1: 최근 N턴 보존 + 나머지 요약
    const recentCount = 4 // 최근 4턴 보존
    const recent = messages.slice(-recentCount * 2)
    const older = messages.slice(0, -recentCount * 2)
    
    // LLM으로 이전 대화 요약
    const summary = await this.summarize(older)
    
    return [
      { role: 'user', content: `[이전 대화 요약]\n${summary}` },
      { role: 'assistant', content: '이전 대화를 이해했습니다. 계속하겠습니다.' },
      ...recent,
    ]
  }
  
  // 전략 2: 도구 결과 압축 (큰 파일 내용 → 요약)
  compressToolResults(messages: Message[]): Message[] {
    return messages.map(msg => {
      if (msg.role === 'tool' && msg.content.length > 5000) {
        return { ...msg, content: truncateWithSummary(msg.content, 2000) }
      }
      return msg
    })
  }
  
  // 전략 3: 시스템 리마인더 주입
  injectReminders(messages: Message[], reminders: string[]): Message[] {
    // 대화 중간에 <system-reminder> 태그로 주입
    // 긴 대화에서 LLM의 "잊어버림" 방지
  }
}
```

#### C. 메모리 시스템

```typescript
// memoryManager.ts
class MemoryManager {
  private memoryDir: string // ~/.my-agent/memory/
  
  // 메모리 저장
  async save(memory: Memory): Promise<void> {
    const filename = slugify(memory.name) + '.md'
    const content = `---
name: ${memory.name}
description: ${memory.description}
type: ${memory.type}
---

${memory.content}`
    
    await writeFile(join(this.memoryDir, filename), content)
    await this.updateIndex(memory)
  }
  
  // 인덱스 로드 (항상 시스템 프롬프트에 포함)
  async loadIndex(): Promise<string> {
    return readFile(join(this.memoryDir, 'MEMORY.md'), 'utf-8')
  }
  
  // 특정 메모리 로드 (온디맨드)
  async load(filename: string): Promise<Memory> {
    const raw = await readFile(join(this.memoryDir, filename), 'utf-8')
    return parseMemoryFile(raw)
  }
  
  // 중복 체크 후 업데이트 또는 생성
  async upsert(memory: Memory): Promise<void> {
    const existing = await this.findSimilar(memory)
    if (existing) {
      await this.update(existing.filename, memory)
    } else {
      await this.save(memory)
    }
  }
}
```

#### D. 도구 시스템

```typescript
// registry.ts
interface ToolDefinition {
  name: string
  description: string
  parameters: z.ZodSchema  // Zod 스키마로 입력 검증
  permissionLevel: 'read' | 'write' | 'dangerous'
  execute: (input: any) => Promise<ToolResult>
}

class ToolRegistry {
  private tools = new Map<string, ToolDefinition>()
  
  register(tool: ToolDefinition): void {
    this.tools.set(tool.name, tool)
  }
  
  // LLM에 제공할 도구 스키마
  getSchemas(): ToolSchema[] {
    return Array.from(this.tools.values()).map(t => ({
      name: t.name,
      description: t.description,
      input_schema: zodToJsonSchema(t.parameters),
    }))
  }
  
  // 지연 로드: 기본 도구만 스키마 제공, 나머지는 이름만
  getSchemasWithDeferred(): { eager: ToolSchema[]; deferred: string[] } {
    const eager = ['Read', 'Edit', 'Bash', 'Grep', 'Glob']
    return {
      eager: this.getSchemas().filter(t => eager.includes(t.name)),
      deferred: Array.from(this.tools.keys()).filter(n => !eager.includes(n)),
    }
  }
}
```

#### E. 권한 시스템

```typescript
// permissionChecker.ts
type PermissionMode = 'strict' | 'normal' | 'auto'

class PermissionChecker {
  private mode: PermissionMode
  private cache = new Map<string, boolean>()
  
  async check(tool: ToolDefinition, input: any): Promise<PermissionResult> {
    // auto 모드: 모두 허용
    if (this.mode === 'auto') return { allowed: true }
    
    // read 도구: 항상 허용
    if (tool.permissionLevel === 'read') return { allowed: true }
    
    // 캐시 확인
    const cacheKey = `${tool.name}:${hashInput(input)}`
    if (this.cache.has(cacheKey)) {
      return { allowed: this.cache.get(cacheKey)! }
    }
    
    // 사용자에게 확인
    const decision = await this.promptUser(tool, input)
    this.cache.set(cacheKey, decision.allowed)
    return decision
  }
}
```

### 12.3 고급 패턴 (Claude Code에서 차용)

#### A. 서브에이전트 패턴

```typescript
async function spawnSubAgent(task: string, type: AgentType): Promise<string> {
  // 독립 컨텍스트로 서브에이전트 실행
  const subAgent = new Agent({
    systemPrompt: getAgentTypePrompt(type),
    tools: getAgentTypeTools(type),  // 유형별 도구 제한
    maxTokens: 100_000,  // 부모보다 작은 컨텍스트
  })
  
  const result = await subAgent.run(task)
  return result.summary  // 요약만 부모에게 반환
}
```

#### B. 메시지 중복 제거 (BoundedSet)

```typescript
class BoundedSet<T> {
  private set = new Set<T>()
  private queue: T[] = []
  
  constructor(private capacity: number) {}
  
  add(item: T): boolean {
    if (this.set.has(item)) return false // 이미 존재
    this.set.add(item)
    this.queue.push(item)
    if (this.queue.length > this.capacity) {
      this.set.delete(this.queue.shift()!)
    }
    return true // 새 아이템
  }
}
```

#### C. 지수 백오프 재시도

```typescript
async function withRetry<T>(
  fn: () => Promise<T>,
  opts: { maxRetries: number; baseDelay: number; maxDelay: number }
): Promise<T> {
  for (let i = 0; i <= opts.maxRetries; i++) {
    try {
      return await fn()
    } catch (e) {
      if (i === opts.maxRetries) throw e
      const delay = Math.min(
        opts.baseDelay * Math.pow(2, i) + Math.random() * 1000,
        opts.maxDelay
      )
      await sleep(delay)
    }
  }
  throw new Error('unreachable')
}
```

#### D. 레이스 조건 패턴 (권한 결정)

```typescript
// 두 소스 중 먼저 결정된 쪽이 승리
async function raceDecision(
  hookDecision: Promise<Decision>,
  userDecision: Promise<Decision>,
): Promise<Decision> {
  const controller = new AbortController()
  
  return Promise.race([
    hookDecision.then(d => { controller.abort(); return d }),
    userDecision.then(d => { controller.abort(); return d }),
  ])
}
```

#### E. Flush Gate 패턴

```typescript
// 초기 플러시 동안 메시지를 큐잉하여 순서 보장
class FlushGate<T> {
  private queue: T[] = []
  private active = false
  
  start(): void { this.active = true }
  
  enqueue(item: T): boolean {
    if (!this.active) return false
    this.queue.push(item)
    return true
  }
  
  end(): T[] {
    this.active = false
    const items = this.queue
    this.queue = []
    return items
  }
}
```

### 12.4 시스템 프롬프트 설계 패턴

Claude Code의 시스템 프롬프트에서 배울 수 있는 핵심 패턴:

```
1. 역할 정의: "You are X, a tool that helps with Y"
2. 도구 사용 규칙: 각 도구를 언제 사용하고 언제 사용하지 말아야 하는지
3. 행동 제약: "NEVER do X", "ALWAYS do Y"
4. 출력 형식: "Keep responses short", "Use markdown"
5. 안전 지침: 보안, 파괴적 작업 방지
6. 환경 정보: OS, 작업 디렉토리, git 상태 등
7. 메모리 인덱스: MEMORY.md 내용 (항상 포함)
8. 동적 리마인더: 대화 중간에 <system-reminder>로 주입
```

### 12.5 핵심 교훈 요약

| # | 교훈 | 상세 |
|---|------|------|
| 1 | **컨텍스트 관리가 핵심** | 무한 대화의 비결은 자동 압축 + 시스템 리마인더 주입 |
| 2 | **도구 지연 로드** | 모든 도구를 즉시 노출하지 말고, 필요 시 스키마 로드 |
| 3 | **메모리는 파일 기반** | DB 불필요, 마크다운 파일 + 인덱스로 충분 |
| 4 | **서브에이전트로 병렬화** | 독립 컨텍스트의 서브에이전트로 복잡 작업 분할 |
| 5 | **권한은 레이스 패턴** | 훅과 사용자 프롬프트를 병렬 실행, 먼저 결정된 쪽 승리 |
| 6 | **멀티 트랜스포트** | WS, SSE+POST, Hybrid를 전략 패턴으로 선택 |
| 7 | **UUID 중복 제거** | BoundedSet으로 메모리 누수 없이 중복 방지 |
| 8 | **지수 백오프** | 모든 네트워크 호출에 재시도 + 백오프 적용 |
| 9 | **NDJSON 스트리밍** | 구조화된 메시지를 줄 단위로 스트리밍 |
| 10 | **피처 플래그** | 컴파일 타임 + 런타임 이중 게이트로 기능 제어 |

---

## 부록 A: 주요 설계 패턴 정리

| 패턴 | 사용 위치 | 목적 |
|------|----------|------|
| Message Broker | StructuredIO.outbound | 모든 stdout 메시지의 중앙 큐 |
| Race Condition | createCanUseTool() | 훅 vs SDK 프롬프트 중 승자 결정 |
| Producer-Consumer | SerialBatchEventUploader | 백프레셔 기반 직렬 드레인 |
| Strategy | getTransportForUrl() | 환경/프로토콜별 트랜스포트 선택 |
| Adapter | HybridTransport | WebSocket 래핑 + POST 레이어 |
| Lazy Loading | handlers/, ToolSearch | 사용 시점에 로드하여 시작 시간 단축 |
| Generation Counter | 토큰 갱신 스케줄러 | 오래된 비동기 작업 무효화 |
| Flush Gate | replBridge 초기화 | 초기 플러시 동안 메시지 큐잉 |
| Bounded Set | UUID 중복 제거 | 고정 메모리로 무한 중복 제거 |
| Dependency Injection | Bridge 모듈 전체 | 테스트 용이성, 구현 교체 |

## 부록 B: 성능 수치 참고

| 작업 | 지연 시간 | 비고 |
|------|----------|------|
| 폴링 작업 | 50-100ms | HTTP 라운드트립 |
| 워커 등록 (v2) | 100-200ms | 서버 상태 업데이트 |
| Worktree 생성 | 1-2초 | Git clone/checkout |
| 세션 생성 | 100-500ms | 자식 프로세스 시작 |
| 토큰 갱신 | 50-100ms | OAuth 서버 |
| 메시지 중복 체크 | O(1) | HashSet 조회 |

## 부록 C: 보안 체크리스트

- [ ] 16자 이상 토큰은 로그에서 마스킹 (앞 8자 + 뒤 4자만 표시)
- [ ] 브릿지 ID 검증으로 경로 순회 방지
- [ ] Worktree로 동시 세션 파일 충돌 방지
- [ ] JWT는 exp 클레임만 읽고 서명 검증은 백엔드에 위임
- [ ] 권한 에스컬레이션 없이 부모 자격증명으로 실행
- [ ] 샌드박스 실행으로 파일/네트워크 접근 제한

---

> 이 문서는 Claude Code 유출 소스코드(2026-03-31)를 기반으로 작성되었습니다.
> 실제 구현과 차이가 있을 수 있으며, 자체 에이전트 구축을 위한 참고 자료로 활용하시기 바랍니다.
