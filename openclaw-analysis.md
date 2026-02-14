# OpenClaw 오픈소스 심층 분석 보고서

> 분석 대상: [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
> 분석 관점: 에이전트 시스템 설계자를 위한 벤치마킹 및 학습 포인트
> 분석 일자: 2026-02-14

---

## 1. 프로젝트 개요

### 1.1 배경

OpenClaw은 오스트리아 개발자 Peter Steinberger가 2025년 11월에 공개한 오픈소스 자율 AI 에이전트 플랫폼이다. 원래 "Clawdbot"이라는 이름으로 시작했으나, Anthropic의 상표권 이슈로 "Moltbot"을 거쳐 2026년 1월 "OpenClaw"로 최종 리네이밍되었다. 2026년 2월 기준 GitHub 스타 145,000+를 돌파하며 2026년 초반 가장 주목받는 AI 레포지토리로 자리잡았다.

주목할 만한 바이럴 요소로, 2026년 1월 말 출시된 "Moltbook" 프로젝트 — 수천 개의 OpenClaw 에이전트가 자율적으로 게시물을 작성하고, 댓글을 달고, 업보트하는 "Dead Internet" 실험 — 이 프로젝트의 폭발적 성장에 기여했다.

### 1.2 핵심 컨셉

OpenClaw은 **"개인 AI 비서를 내 기기에서, 내가 이미 사용하는 채널에서 실행한다"**는 컨셉의 에이전트 프레임워크다. LLM을 통해 자율적으로 태스크를 수행하며, 메시징 플랫폼(WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Teams 등 20+개)을 주요 인터페이스로 사용한다.

### 1.3 기술 스택 요약

| 구분 | 기술 | 비고 |
|------|------|------|
| **런타임** | Node.js 22+ | ESM modules |
| **언어** | TypeScript 5.9 (ES2023) | 354,000+ LOC |
| **패키지** | pnpm 10.23 | 모노레포 |
| **CLI** | Commander 14 | 서브커맨드 체계 |
| **서버** | Express 5 + WebSocket | 게이트웨이 |
| **에이전트 코어** | @mariozechner/pi-agent-core 0.52 | 내장 에이전트 런타임 |
| **스키마** | Zod 4 + TypeBox | 런타임 검증 |
| **빌드** | tsdown + rolldown | Rust 기반 번들러 |
| **테스트** | Vitest 4 | unit/e2e/gateway/live 분리 |
| **린팅** | oxfmt + oxlint | Rust 기반 고속 린터 |
| **벡터DB** | sqlite-vec 0.1.7 | 임베딩 기반 검색 |
| **브라우저** | Playwright Core 1.58 | 브라우저 자동화 |

### 1.4 최신 버전 (v2026.2.6)

2026년 2월 7일 릴리스. Anthropic Opus 4.6, OpenAI GPT-5.3-Codex, xAI Grok 지원 추가. 스킬/플러그인 코드 안전 스캐너와 설정 응답에서의 자격 증명 마스킹 기능 포함.

---

## 2. 아키텍처 분석

### 2.1 전체 아키텍처 (High-Level)

```
                        ┌──────────────────────────────┐
                        │       CLI Entry Point        │
                        │    (Commander 서브커맨드)      │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │   Gateway (WebSocket + HTTP)  │
                        │  - 세션, 인증, 채널, 스케줄링    │
                        │  - 40+ RPC 메서드              │
                        │  - 실시간 이벤트 브로드캐스트     │
                        └──────────────┬───────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
    ┌──────────▼──────────┐ ┌──────────▼──────────┐ ┌──────────▼──────────┐
    │   Agent Runtime     │ │   Channel Dock      │ │   Plugin System     │
    │  - 모델 선택/폴백    │ │  - 20+ 메시징 채널   │ │  - 툴, 채널, 메모리   │
    │  - 툴 실행 파이프라인 │ │  - 라우팅 해석       │ │  - Provider, Hook    │
    │  - 세션 컨텍스트     │ │  - 메시지 정규화     │ │  - 서비스, CLI 확장   │
    │  - 서브에이전트      │ │  - 인바운드/아웃바운드│ │  - Gateway 메서드     │
    └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
               │                       │                       │
    ┌──────────▼──────────┐ ┌──────────▼──────────┐ ┌──────────▼──────────┐
    │   Skill System      │ │   Session Store     │ │   Exec Approvals    │
    │  - 48+ 번들 스킬    │ │  - 파일 기반 영속화  │ │  - allowlist 기반    │
    │  - ClawHub 레지스트리│ │  - 컨텍스트 압축    │ │  - 소켓 기반 승인    │
    │  - 6단계 우선순위    │ │  - 쓰기 잠금        │ │  - 샌드박스 격리     │
    └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

### 2.2 부팅 시퀀스

```
openclaw.mjs (CLI 실행 파일, Node 옵션 설정 + 리스폰)
  ↓
src/entry.ts (프로세스 부트스트랩, 프로필 파싱)
  ↓
src/index.ts (로거 초기화, 설정 로딩)
  ↓
src/cli/program/build-program.ts (Commander 프로그램 빌더)
  ↓
register.*.ts (서브커맨드 등록: gateway, agent, onboard, send, doctor, config, skills)
```

### 2.3 핵심 디렉토리 구조

```
openclaw/
├── src/              # 코어 소스 (52개 모듈, 354K LOC)
│   ├── agents/       # 에이전트 런타임, 툴, 스킬, 인증 프로필
│   │   ├── pi-embedded-runner/  # 내장 에이전트 실행 엔진
│   │   ├── auth-profiles/       # OAuth + API 키 관리
│   │   ├── skills/              # 스킬 디스커버리 & 로딩
│   │   ├── tools/               # 도구 정의
│   │   ├── pi-tools.ts          # 도구 합성 (21KB)
│   │   ├── pi-tools.policy.ts   # 도구 정책 필터링
│   │   └── model-selection.ts   # 모델 해석 & 폴백
│   ├── gateway/      # WebSocket 제어 평면
│   │   ├── server.impl.ts       # 게이트웨이 구현 (25KB)
│   │   ├── server-http.ts       # HTTP/WS 서버 (20KB)
│   │   ├── server-methods/      # 40+ RPC 핸들러
│   │   └── protocol/            # 프로토콜 정의 (19KB)
│   ├── channels/     # 멀티채널 메시징 추상화
│   │   ├── dock.ts              # Channel Dock 인터페이스 (17KB)
│   │   ├── registry.ts          # 채널 메타데이터 & 순서
│   │   └── plugins/             # 채널 플러그인 인터페이스
│   ├── plugins/      # 플러그인 시스템
│   │   ├── registry.ts          # 플러그인 레지스트리 (14KB)
│   │   ├── loader.ts            # 동적 로딩 (14KB)
│   │   ├── install.ts           # 설치 플로우 (16KB)
│   │   └── types.ts             # 플러그인 타입 정의
│   ├── routing/      # 인바운드 메시지 라우팅
│   │   └── resolve-route.ts     # 라우팅 로직 (10KB)
│   ├── sessions/     # 세션 영속화, 컨텍스트 관리
│   ├── hooks/        # 라이프사이클 훅 시스템
│   ├── auto-reply/   # 응답 디스패치, 청킹, 큐잉
│   ├── config/       # 설정 시스템 (30+ 타입 모듈)
│   └── infra/        # 인프라 (승인, 디스커버리, 60+ 유틸)
├── extensions/       # 38개 확장 플러그인 (별도 npm 패키지)
├── skills/           # 48+ 번들 스킬 (YAML + Markdown)
├── apps/             # 네이티브 앱 (macOS/Swift, iOS, Android/Kotlin)
├── ui/               # WebChat UI (React)
├── packages/         # 모노레포 워크스페이스 패키지
└── docs/             # Mintlify 문서 사이트
```

---

## 3. 핵심 설계 패턴 심층 분석

### 3.1 멀티에이전트 라우팅 시스템

OpenClaw의 가장 독창적인 설계 중 하나는 **설정 기반(Config-Driven) 멀티에이전트 라우팅**이다.

**라우팅 해석 우선순위 (높은 것이 우선):**

```
1. binding.peer         ← 특정 사용자/그룹 직접 바인딩
2. binding.peer.parent  ← 스레드 부모의 바인딩 상속
3. binding.guild+roles  ← 길드 + Discord 역할 기반
4. binding.guild        ← 길드(서버) 단위
5. binding.team         ← Teams 팀 단위
6. binding.account      ← 계정 단위
7. binding.channel      ← 채널 종류 단위
8. default              ← 기본 에이전트
```

**설정 예시:**
```json
{
  "agents": {
    "list": [
      { "id": "personal", "workspaceDir": "~/ai/personal" },
      { "id": "work", "workspaceDir": "~/ai/work" }
    ]
  },
  "bindings": [
    { "channel": "slack", "accountId": "work-slack", "agent": "work" },
    { "channel": "telegram", "peer": "user:12345", "agent": "personal" },
    { "channel": "discord", "guildId": "G123", "roles": ["admin"], "agent": "work" },
    { "channel": "*", "agent": "personal" }
  ]
}
```

**라우팅 결과 타입:**
```typescript
type ResolvedAgentRoute = {
  agentId: string;
  channel: string;
  accountId: string;
  sessionKey: string;       // 내부 세션 키 (영속화 + 동시성 제어용)
  mainSessionKey: string;   // DM 통합용 편의 별칭
  matchedBy:                // 디버깅/로깅용 매칭 기술
    | "binding.peer" | "binding.peer.parent"
    | "binding.guild+roles" | "binding.guild"
    | "binding.team" | "binding.account"
    | "binding.channel" | "default";
};
```

**핵심 인사이트:**
- 코드 변경 없이 JSON 설정만으로 에이전트 토폴로지를 변경
- 에이전트 간 완전 격리 (워크스페이스, 세션, 도구 정책)
- `matchedBy` 필드로 라우팅 결정 과정을 투명하게 추적 가능

### 3.2 플러그인 아키텍처

OpenClaw의 플러그인 시스템은 **7가지 확장 포인트**을 제공한다.

**확장 포인트:**

| 확장 유형 | 설명 | 예시 |
|-----------|------|------|
| **Tool** | 에이전트 도구 추가 | memory_search, llm_task, voice_call |
| **Channel** | 새 메시징 채널 | Discord, Slack, Telegram, Matrix |
| **Provider** | LLM/인증 백엔드 | Google Gemini OAuth, Copilot Proxy |
| **Hook** | 라이프사이클 인터셉트 | before_agent_start, agent_end, message:sending |
| **Service** | 백그라운드 프로세스 | Voice Call 런타임, OpenTelemetry |
| **Gateway Method** | RPC 핸들러 추가 | voicecall.initiate, voicecall.speak |
| **CLI Command** | CLI 서브커맨드 추가 | openclaw memory, openclaw voicecall |

**플러그인 레지스트리 구조:**
```typescript
type PluginRegistry = {
  plugins: PluginRecord[];                    // 메타데이터
  tools: PluginToolRegistration[];            // 도구
  hooks: PluginHookRegistration[];            // 훅 핸들러
  channels: PluginChannelRegistration[];      // 채널
  providers: PluginProviderRegistration[];    // LLM/인증
  gatewayHandlers: GatewayRequestHandlers;   // RPC 메서드
  httpHandlers: routes;                       // HTTP 미들웨어
  services: PluginServiceRegistration[];      // 백그라운드 서비스
};

type PluginRecord = {
  id: string;
  source: string;                // 파일 경로 또는 레지스트리 URL
  origin: "bundled" | "installed" | "workspace";
  enabled: boolean;
  status: "loaded" | "disabled" | "error";
  toolNames: string[];           // 제공하는 도구
  channelIds: string[];          // 지원하는 채널
  configSchema?: Record<string, unknown>;
};
```

**플러그인 등록 패턴:**
```typescript
// 표준 채널 플러그인
export default {
  id: "discord",
  name: "Discord",
  configSchema: myConfigSchema,
  register(api: OpenClawPluginApi) {
    setDiscordRuntime(api.runtime);
    api.registerChannel({ plugin: discordPlugin });
  }
};

// 도구 + 훅 + 서비스 복합 플러그인 (memory-lancedb)
export default {
  id: "memory-lancedb",
  register(api: OpenClawPluginApi) {
    // 도구 등록
    api.registerTool((ctx) => [
      createMemoryRecallTool(db),
      createMemoryStoreTool(db),
      createMemoryForgetTool(db)
    ], { names: ["memory_recall", "memory_store", "memory_forget"] });

    // 라이프사이클 훅 (에이전트 시작 전 관련 메모리 주입)
    api.on("before_agent_start", async (event) => {
      const memories = await db.search(embedding, 3, 0.3);
      return { prependContext: formatMemories(memories) };
    });

    // 에이전트 종료 후 자동 캡처
    api.on("agent_end", async (event) => {
      const textToCapture = filterCapturable(event.messages);
      await Promise.all(textToCapture.map(text => db.store(...)));
    });

    // 백그라운드 서비스
    api.registerService({ id: "memory-lancedb", start, stop });
  }
};
```

**플러그인 로딩 메커니즘:**
- JIT 컴파일러(jiti)로 ESM/CommonJS 모두 동적 로딩
- 플러그인은 별도 npm 패키지로 격리 (의존성 충돌 방지)
- openclaw.plugin.json 매니페스트로 메타데이터 선언
- 설정 스키마의 런타임 검증 (JSON Schema + Zod)
- 플러그인 로딩 실패 시 graceful 비활성화 (status='error')

### 3.3 채널 독(Channel Dock) 추상화

20+개 메시징 플랫폼을 **하나의 인터페이스**로 통합하는 Channel Dock 패턴:

```typescript
ChannelDock = {
  id: string;                            // "discord", "slack", ...
  capabilities: ChannelCapabilities;     // 기능 명세

  // 필수 어댑터 (모든 채널이 구현)
  config: ChannelConfigAdapter;          // 계정/설정 관리 (멀티 계정 지원)
  security: ChannelSecurityAdapter;      // DM 정책, 허용 목록
  messaging: ChannelMessagingAdapter;    // 타겟 정규화
  outbound: ChannelOutboundAdapter;      // 메시지 발송, 청킹
  actions: ChannelActionsAdapter;        // 리액션, 편집, 삭제
  directory: ChannelDirectoryAdapter;    // 피어/그룹 목록
  gateway: ChannelGatewayAdapter;        // 시작/중지, QR 인증
  status: ChannelStatusAdapter;          // 헬스체크, 진단

  // 선택적 어댑터 (채널 역량에 따라)
  commands?: ChannelCommandAdapter;
  groups?: ChannelGroupAdapter;
  mentions?: ChannelMentionAdapter;
  threading?: ChannelThreadingAdapter;
  agentPrompt?: ChannelAgentPromptAdapter;
  streaming?: BlockStreamingCoalesceDefaults;
  elevated?: ChannelElevatedAdapter;
};
```

**채널별 차이점 처리 예시:**

| 채널 | 텍스트 한도 | 청킹 전략 | 인증 방식 | 스레딩 |
|------|-----------|-----------|----------|--------|
| **Discord** | 2,000자 | 직접 분할 | Bot Token | 스레드 지원 |
| **Slack** | 4,000자 | 직접 분할 | Bot+User Token | 스레드 지원 |
| **Telegram** | 4,000자 | 마크다운 인식 분할 | Bot Token + 프록시 | 토픽 스레드 |
| **WhatsApp** | — | 게이트웨이 모드 | QR 코드 스캔 | 미지원 |

**인바운드 메시지 플로우:**
```
Raw Channel Message
  → 정규화 (채널별 핸들러)
  → 라우팅 해석 (resolve-route.ts)
  → 세션 조회
  → 에이전트 라우팅 (채널/계정/피어/길드/역할별)
  → Auto-Reply 큐
  → 응답 디스패치 (아웃바운드 어댑터)
```

### 3.4 에이전트 런타임 & 도구 실행 파이프라인

**에이전트 실행 플로우:**

```
Input Message
  → Route Resolution (resolve-route.ts)
  → Session Load (sessions/)
  → Tool Resolution (pi-tools.ts)
  → Agent Run (pi-embedded-runner)
    → Attempt Loop (attempt.ts)
      → LLM API Call
      → Tool Call Parsing
      → Tool Policy Check (owner/group/subagent)
      → Sandbox + Lock Acquisition
      → Tool Execution
      → Result Handling + Event Emission
      → Stream to Subscribers
    → Context Overflow? → Compaction
    → Provider Error? → Fallback Model
  → Reply Dispatch (auto-reply/)
  → Channel Output
```

**도구 합성 계층:**
```typescript
function createOpenClawCodingTools(options) {
  // 1. 내장 도구: exec, bash, process, apply-patch
  // 2. 파일 도구: read, write, edit (샌드박스 적용)
  // 3. 채널 도구: messaging, threading
  // 4. OpenClaw 도구: session 관리, subagent 스포닝
  // 5. 플러그인 도구: getPluginToolMeta()로 동적 등록
}
```

**도구 정책 필터링 파이프라인:**
```
1. resolveEffectiveToolPolicy()   ← 글로벌 + 에이전트 병합
2. resolveGroupToolPolicy()       ← 그룹/채널별 제한
3. resolveSubagentToolPolicy()    ← 서브에이전트 범위 제한
4. applyOwnerOnlyToolPolicy()     ← 소유자 전용 도구 게이팅
5. filterToolsByPolicy()          ← 최종 인가 레이어
```

**도구 스키마 어댑테이션:**
- Claude: 전체 스키마 + 파라미터 그룹
- Gemini: 정규화된 타입 + 정리된 스키마
- OpenAI: 호환 서브셋
- 프로바이더별 자동 변환으로 동일 도구가 여러 LLM에서 동작

**서브에이전트 메커니즘:**
```
Parent Agent                    Child Agent
    │                               │
    ├─ sessions/spawn ──────────────┤
    │                               │
    │  [자식이 독립적으로 실행]         │
    │                               │
    │                               ├─ subagent-announce (완료)
    │  [부모가 결과 읽기] ◄────────── │
    │                               │
    └─ 사용자에게 반환                │
```

- 인메모리 Map + 디스크 영속화로 서브에이전트 상태 추적
- 라이프사이클: pending → started → completed → cleanup → archived
- 자식 세션은 부모의 전달 컨텍스트를 상속하되 별도 에이전트 스코프로 실행

### 3.5 스킬(Skill) 시스템

**6단계 우선순위 기반 스킬 디스커버리:**

```
우선순위 (낮음 → 높음):
1. extraDirs          ← 설정으로 지정한 추가 경로
2. bundledSkillsDir   ← OpenClaw 내장 스킬 (48+개)
3. managedSkillsDir   ← ~/.openclaw/skills (ClawHub 설치)
4. personalAgentsDir  ← ~/.agents/skills (개인)
5. projectAgentsDir   ← ./.agents/skills (프로젝트)
6. workspaceSkills    ← ./skills (워크스페이스)
→ 나중 소스가 같은 이름의 스킬을 오버라이드
```

**스킬 포맷 (SKILL.md):**

```yaml
---
name: github
description: "GitHub CLI로 이슈, PR, CI 관리"
metadata:
  openclaw:
    emoji: "🐙"
    requires:
      bins: ["gh"]          # 필수 바이너리 (전부 필요)
      anyBins: ["npm", "pnpm"]  # 선택 바이너리 (하나만 필요)
      env: ["GH_TOKEN"]     # 필수 환경변수
      config: ["channels.slack"]  # 필수 설정 경로
    os: ["darwin", "linux"]  # 지원 OS
    install:
      - kind: brew
        formula: gh
        bins: ["gh"]
        label: "Install GitHub CLI (brew)"
      - kind: apt
        package: gh
        bins: ["gh"]
        label: "Install GitHub CLI (apt)"
---

[마크다운 형식의 상세 사용법과 도구 정의]
```

**8단계 적격성 필터링:**

```
1. 명시적 비활성화 체크     → config.skills.entries[name].enabled === false?
2. 번들 허용 목록 체크       → config.skills.allowBundled 필터
3. OS 호환성 체크           → metadata.os 배열 매칭 (원격 노드 포함)
4. "always" 플래그 체크      → 무조건 포함
5. 필수 바이너리 존재 체크    → requires.bins (전부 존재해야 함)
6. 선택 바이너리 존재 체크    → requires.anyBins (하나만 존재하면 됨)
7. 환경변수 설정 체크         → requires.env (process.env 또는 skillConfig.env)
8. 설정 경로 존재 체크        → requires.config (e.g., channels.slack 존재 여부)
```

**스킬 스냅샷:**
```typescript
type SkillSnapshot = {
  prompt: string;                        // 시스템 프롬프트에 주입할 포맷된 스킬 텍스트
  skills: Array<{name, primaryEnv}>;     // 활성화된 스킬 메타
  resolvedSkills?: Skill[];              // 실제 스킬 정의 (에이전트 사용용)
  version?: number;                      // 변경 추적 (타임스탬프)
};
```

**스킬의 3단계 Progressive Disclosure:**
1. **항상 로딩** (~100 토큰): `name` + `description` (메타데이터만)
2. **트리거 시** (<5K 토큰): SKILL.md 본문 전체
3. **필요 시** (무제한): scripts/ (실행 가능), references/ (참조 문서), assets/

**ClawHub 레지스트리:**
- `clawhub search "postgres backups"` — 스킬 검색
- `clawhub install my-skill --version 1.2.3` — 버전 지정 설치
- `clawhub update --all` — 전체 업데이트
- `clawhub publish ./my-skill --slug my-skill` — 스킬 퍼블리시
- 기본 레지스트리: `https://clawhub.com`
- 설치 방법: brew, node(npm/pnpm/yarn/bun), go, uv, download

### 3.6 세션 관리 및 컨텍스트 압축

**세션 키 설계:**
```
{agentId}:{channel}:{accountId}:{peer}

예시:
- main:webchat:user123              # 웹챗 세션
- main:telegram:user123:topic456    # 텔레그램 토픽
- support:discord:user456:thread789 # 디스코드 스레드
```

**DM 세션 스코프 옵션:**

| 스코프 | 설명 |
|--------|------|
| `main` | 모든 DM을 하나의 세션으로 통합 |
| `per-peer` | 사용자별 분리 세션 |
| `per-channel-peer` | 채널+사용자별 분리 |
| `per-account-channel-peer` | 계정+채널+사용자별 분리 |

**세션 저장소:**
```typescript
// 에이전트별 세션 저장소 (JSON 파일)
type SessionStore = Record<string, SessionEntry>;
// 경로: {storePath}/{agentId}/sessions.json

// 45초 TTL 캐시
SESSION_STORE_CACHE = Map<storePath, {
  store: SessionStore;
  loadedAt: number;
  mtimeMs?: number;    // 디스크 mtime 변경 감지
}>;

// 파일 기반 뮤텍스 (쓰기 잠금)
// - 동시 읽기: 허용
// - 동시 쓰기: 직렬화
// - 타임아웃: 30초 기본
```

**컨텍스트 압축 전략:**
```
1. 컨텍스트 윈도우 한계 도달 시:
   → 예약 토큰 계산 (압축 바닥 + 도구 정의)
   → 오래된 메시지/도구 결과에서 제거 후보 식별
   → LLM을 사용한 요약으로 교체
   → 도구 결과 트렁케이션 (앞 500자 + ... + 뒤 500자)
   → 최대 3회 재시도
```

### 3.7 실행 승인 및 보안 모델

**3계층 보안 아키텍처:**

```
Layer 1: Tool Policy (도구 정책)
  ├─ owner-only: 관리자 전용 도구
  ├─ group policy: 그룹/채널별 도구 제한
  └─ subagent policy: 서브에이전트 도구 범위 제한

Layer 2: Exec Approval (실행 승인)
  ├─ deny: 모든 실행 차단
  ├─ allowlist: 패턴 매칭된 명령만 허용
  └─ full: 전체 허용 (위험, 명시적 옵트인 필요)

Layer 3: Sandbox (샌드박스)
  ├─ 워크스페이스 경계 강제
  ├─ 읽기 전용 접근 (워크스페이스 외부)
  └─ 경로 변수 ($WORKSPACE, $AGENT_DIR)
```

**Exec Approval 상세:**
```json
{
  "version": 1,
  "socket": { "path": "~/.openclaw/exec-approvals.sock", "token": "secret" },
  "defaults": {
    "security": "allowlist",
    "ask": "on-miss",
    "autoAllowSkills": false
  },
  "agents": {
    "main": {
      "allowlist": [
        {
          "id": "uuid",
          "pattern": "ls.*",
          "lastUsedAt": 1234567890,
          "lastUsedCommand": "ls -la",
          "lastResolvedPath": "/usr/bin/ls"
        }
      ]
    }
  }
}
```

**승인 워크플로우:**
```
도구가 exec/bash 호출
  → allowlist 패턴 매칭 검사
  → 미매칭 + ask:"on-miss" → 사용자 승인 요청
  → Gateway RPC 또는 대화형 프롬프트로 승인
  → 승인 시: allowlist에 패턴 추가 + lastUsedAt 업데이트
```

**소켓 기반 원격 승인:**
- 게이트웨이가 Unix 소켓 + 토큰으로 승인 서버 운영
- CLI 프로세스가 소켓에 연결하여 `{token, command}` 전송
- `{approved, pattern?}` 응답 수신
- allowlist 영속적 업데이트 가능

### 3.8 게이트웨이 프로토콜

**서버 구성:**
- 기본 포트: 18789 (루프백 바인딩)
- 인증: 토큰 또는 패스워드
- 프로토콜: WebSocket + HTTP 이중
- TypeBox 기반 스키마 검증

**40+ RPC 메서드:**

| 카테고리 | 메서드 예시 |
|----------|-----------|
| **Agent** | agent, agents (list/add/delete/identity) |
| **Chat** | chat (세션 업데이트 구독) |
| **Sessions** | sessions (list, send, create, delete, patch, reset, compact) |
| **Config** | config (apply patch/reload) |
| **Channels** | channels (status, start, stop) |
| **Models** | models (list, catalog) |
| **Health** | health (서버 + 채널) |
| **Cron** | cron (schedule, list) |
| **Tools** | tools (HTTP 호출) |
| **Skills** | skills (list, install, update) |
| **System** | update, logs, voicewake, talk, tts |
| **Exec** | exec-approval.* (보안 승인 워크플로우) |

---

## 4. 확장 플러그인 생태계 분석

### 4.1 38개 확장 분류

**메시징 채널 (20개):**
Discord, Slack, Telegram, WhatsApp, Microsoft Teams, Mattermost, Google Chat, Nextcloud Talk, Matrix, IRC, Signal, Twitch, Zalo, Zalo User, LINE, iMessage, BlueBubbles, Feishu/Lark, Tlon/Urbit, Nostr

**인증 프로바이더 (4개):**
Google Antigravity OAuth, Google Gemini CLI OAuth, MiniMax Portal OAuth, Qwen Portal OAuth

**메모리 플러그인 (2개):**
memory-core (파일 기반), memory-lancedb (LanceDB 벡터 DB)

**에이전트 도구 (3개):**
llm-task (JSON 전용 LLM 태스크), lobster (타입드 파이프라인 워크플로우), copilot-proxy

**특수 유틸리티 (5+개):**
voice-call (Telnyx/Twilio/Plivo), diagnostics-otel (OpenTelemetry), open-prose (VM 스킬 팩) 등

### 4.2 채널 어댑터 패턴 (Discord 예시)

```typescript
// extensions/discord/index.ts
export default {
  id: "discord",
  name: "Discord",
  register(api: OpenClawPluginApi) {
    setDiscordRuntime(api.runtime);
    api.registerChannel({ plugin: discordPlugin });
  }
};

// discordPlugin implements ChannelPlugin<DiscordAccount>
const discordPlugin = {
  config: {
    listAccountIds(cfg),        // 멀티 계정 목록
    resolveAccount(cfg, id),    // 계정 해석
    isConfigured(account),      // 설정 완료 여부
  },
  security: {
    resolveDmPolicy(params),    // DM 정책: "pairing" | "open" | "closed"
    collectWarnings(params),    // 보안 경고
  },
  messaging: {
    normalizeTarget(to),        // "user:123" → 정규화
  },
  outbound: {
    deliveryMode: "direct",     // 직접 전송
    textChunkLimit: 2000,       // Discord 문자 제한
    sendText(params),           // 텍스트 전송
    sendMedia(params),          // 미디어 전송
  },
  actions: {
    listActions(ctx),           // react, edit, delete, pin, unpin...
    handleAction(ctx),          // 액션 처리
  },
  gateway: {
    startAccount(ctx),          // 봇 연결 시작
  },
  status: {
    probeAccount(account, timeout),  // 헬스 체크
    auditAccount(account),           // 권한 감사
  }
};
```

### 4.3 메모리 플러그인 패턴 (memory-lancedb)

```typescript
// 3개 도구: memory_recall, memory_store, memory_forget
class MemoryDB {
  async store(entry: { text, vector, importance, category }): Promise<void>
  async search(vector, limit, minScore): Promise<MemorySearchResult[]>
  async delete(id): Promise<boolean>  // GDPR 준수
}

// 카테고리: preference, decision, entity, fact, other

// 라이프사이클 훅으로 자동 메모리 관리:
// - before_agent_start: 관련 메모리를 에이전트 컨텍스트에 주입
// - agent_end: 중요 정보를 자동 캡처하여 저장
```

---

## 5. 장점 및 차별성

### 5.1 멀티채널 통합의 깊이

대부분의 에이전트 프레임워크가 API/CLI 인터페이스에 집중하는 반면, OpenClaw은 **일상적으로 사용하는 메시징 플랫폼**을 주요 인터페이스로 삼는다. 이는 "에이전트를 사용하기 위해 새로운 도구를 배울 필요가 없다"는 접근이다.

| 차별점 | 설명 |
|--------|------|
| **20+ 채널 지원** | 산업 최고 수준의 메시징 플랫폼 커버리지 |
| **통합 추상화** | Channel Dock 패턴으로 어댑터 인터페이스 일관성 확보 |
| **멀티 계정** | 하나의 채널에서 여러 계정 동시 운영 |
| **실시간 스트리밍** | 채널별 coalescing 전략으로 최적화된 응답 전달 |

### 5.2 설정 기반 멀티에이전트 오케스트레이션

```
단일 게이트웨이 인스턴스
  ├─ Agent "personal" ← 개인 텔레그램/시그널 메시지
  ├─ Agent "work"     ← 회사 슬랙/팀즈 메시지
  └─ Agent "support"  ← 디스코드 서버 멘션
```

코드 변경 없이 **JSON 설정만으로** 에이전트 토폴로지를 변경할 수 있다. 대부분의 에이전트 프레임워크가 에이전트 정의를 코드로 하는 것과 대비된다.

### 5.3 Progressive Disclosure 기반 스킬 시스템

48+개 스킬이 있어도 기본 시스템 프롬프트를 합리적 크기로 유지하면서, 필요할 때만 상세 정보를 로딩한다. 선언적 의존성 명세로 환경 자동 감지, chokidar 기반 실시간 워치로 스킬 변경 감지. 이는 토큰 효율성과 개발자 경험을 동시에 달성한다.

### 5.4 레이어드 보안 모델

많은 에이전트 프레임워크가 보안을 사후적으로 처리하는 반면, OpenClaw은 **설계 단계부터 3계층 보안**을 내장했다. 특히 `ask: "on-miss"` 패턴은 **"알려지지 않은 명령은 인간에게 묻고, 한번 허용되면 allowlist에 추가"**라는 점진적 신뢰 구축 모델로 실용적이다.

### 5.5 로컬 퍼스트 + 원격 노드 하이브리드

게이트웨이가 로컬에서 실행되면서도 원격 macOS 노드와 페어링하여:
- 원격 노드의 바이너리 가용성을 프로빙
- 원격에서만 실행 가능한 스킬(macOS 전용 등)을 활성화
- 디바이스 페어링/mDNS 디스커버리로 자동 연결

이는 "로컬 프라이버시 + 원격 능력"을 결합하는 접근이다.

### 5.6 7-Way 플러그인 확장성

Tool, Channel, Provider, Hook, Service, Gateway Method, CLI Command의 7가지 확장 포인트은 에이전트 프레임워크 중에서도 가장 넓은 커버리지다. 이로 인해 커뮤니티가 거의 모든 형태의 기능을 플러그인으로 추가할 수 있다.

---

## 6. 에이전트 시스템 설계자가 배울 점

### 6.1 아키텍처 패턴

#### (1) Config-Driven Agent Routing

에이전트 매핑을 코드가 아닌 설정으로 관리하면, 운영 시 유연성이 극대화된다.

```
적용 방안:
- 에이전트 ↔ 인터페이스 매핑을 설정 파일로 추출
- 라우팅 우선순위 체인을 명시적으로 설계 (peer > guild > account > channel > default)
- 세션 키에 라우팅 컨텍스트를 인코딩하여 별도 매핑 테이블 제거
- matchedBy 같은 디버깅 필드로 라우팅 결정의 투명성 확보
```

#### (2) 7-Way Plugin Extension Points

확장 포인트이 충분히 다양해야 커뮤니티 기여가 활성화된다.

```
적용 방안:
- Tool, Channel, Provider, Hook, Service, Gateway, CLI 수준의 확장 포인트 설계
- 플러그인을 독립 패키지로 격리 (의존성 충돌 방지)
- JIT 컴파일러로 ESM/CJS 모두 동적 로딩
- 플러그인 설정 스키마의 런타임 검증
- 확장 포인트 간 조합 가능 (e.g., memory-lancedb = Tool + Hook + Service)
```

#### (3) Progressive Skill Loading

모든 도구를 항상 시스템 프롬프트에 넣지 않아도 된다.

```
적용 방안:
- 스킬 메타데이터(이름/설명)만 기본 로딩 → 트리거 시 전체 로딩
- 선언적 의존성 명세 (바이너리, 환경변수, 설정 경로)
- 환경 자동 감지로 적격한 스킬만 활성화
- 파일 워치로 스킬 변경 실시간 반영
- 6단계 우선순위로 오버라이드 체인 구성
```

#### (4) Layered Security by Design

에이전트의 자율성이 높을수록 보안 계층이 세밀해야 한다.

```
적용 방안:
- 도구 정책 → 실행 승인 → 샌드박스의 3계층 보안 적용
- "on-miss" 패턴: 모르는 명령은 인간에게 물어보고 허용 목록에 추가 (점진적 신뢰)
- 서브에이전트의 도구 범위를 부모보다 제한
- 소켓 기반 원격 승인으로 원격 실행 시에도 보안 유지
- 프로바이더별 도구 스키마 차이를 자동 어댑트
```

#### (5) Channel Dock Adapter Pattern

```
적용 방안:
- 필수/선택적 어댑터 분리로 최소 구현으로도 새 채널 추가 가능
- 선택적 역량은 런타임에 존재 여부로 분기 (capabilities 패턴)
- 멀티 계정을 설정 수준에서 지원
- 메시지 정규화 → 라우팅 → 세션 → 에이전트 → 응답의 일관된 파이프라인
```

### 6.2 구현 패턴

#### (1) Session Key Encoding

```typescript
// 세션 키에 모든 라우팅 컨텍스트를 인코딩
sessionKey = `${agentId}:${channel}:${accountId}:${peer}`

// 장점:
// - 별도 매핑 테이블 불필요
// - 키 자체가 라우팅 결과를 기술
// - 디버깅 시 세션의 출처를 즉시 파악 가능
// - identityLinks로 크로스채널 세션 통합 가능
```

#### (2) Attempt-Based Retry with Failover

```typescript
// 에이전트 실행을 attempt 단위로 관리
while (attempts < maxAttempts) {
  try {
    result = await runEmbeddedAttempt(model, tools, session);
    break;
  } catch (e) {
    if (e.isContextOverflow) compact(session);     // 컨텍스트 초과 → 압축
    else if (e.isProviderError) model = fallback;  // 프로바이더 오류 → 폴백
    attempts++;
  }
}
// 토큰 사용량은 attempt 간 누적 추적
```

#### (3) Tool Policy Resolution Chain

```typescript
// 글로벌 → 에이전트 → 그룹 → 서브에이전트 순으로 정책 병합
effectivePolicy = merge(
  globalToolPolicy,
  agentToolPolicy,
  groupToolPolicy,      // 채널/그룹별 제한
  subagentToolPolicy    // 서브에이전트 범위 제한
)
// + 프로바이더별 도구 스키마 어댑테이션 자동 적용
```

#### (4) Hook-Based Lifecycle Interception

```typescript
// 잘 정의된 지점에서 훅 발화
hooks: {
  "message:received":   [/* 메시지 수신 시 */],
  "message:sending":    [/* 응답 전송 전 */],
  "message:sent":       [/* 응답 전송 후 */],
  "before_agent_start": [/* 에이전트 실행 전 (메모리 주입 등) */],
  "agent_end":          [/* 에이전트 종료 후 (자동 캡처 등) */],
  "tool:before_call":   [/* 도구 실행 전 (검증, 로깅) */],
  "tool:after_call":    [/* 도구 실행 후 (결과 가공) */],
  "session:start":      [/* 세션 시작 */],
  "session:end":        [/* 세션 종료 */],
  "compaction:before":  [/* 컨텍스트 압축 전 */],
  "compaction:after":   [/* 컨텍스트 압축 후 */],
}
```

### 6.3 운영 패턴

| 패턴 | 설명 |
|------|------|
| **Gateway as Single Control Plane** | 세션/채널/에이전트/설정을 단일 제어점에서 관리. WebSocket + HTTP 이중 프로토콜. |
| **Config Hot-Reload** | `config.apply` RPC로 무중단 설정 업데이트. 스킬/채널/바인딩 실시간 반영. |
| **Heartbeat + Health Snapshot** | 주기적 헬스 체크. presence version 추적으로 상태 변경 감지. |
| **File-Based Session** | JSON 파일 영속화 + 45초 TTL 캐시. 별도 DB 불필요로 설치 의존성 최소화. |

---

## 7. 개선 가능성 및 한계점

### 7.1 보안 우려

Cisco의 보안 연구팀이 지적한 **"lethal trifecta"** (높은 자율성 + 넓은 시스템 접근 + 인터넷 연결)는 실제 위험이다. 서드파티 스킬의 데이터 유출 및 프롬프트 인젝션이 사용자 인지 없이 발생할 수 있다. ClawHub 스킬 레지스트리의 검증 메커니즘이 아직 성숙하지 않다. v2026.2.6에서 스킬/플러그인 코드 안전 스캐너가 추가되었지만, 아직 초기 단계이다.

### 7.2 확장성 한계

- **파일 기반 세션 저장소**: 대규모 배포 시 병목 가능 (수천 세션)
- **단일 게이트웨이 인스턴스**: 수평 확장 아키텍처 부재
- **인메모리 서브에이전트 레지스트리**: 프로세스 재시작 시 상태 손실 가능
- 엔터프라이즈 수준의 고가용성/클러스터링은 미지원

### 7.3 복잡성

354K+ LOC의 TypeScript 코드베이스는 컨트리뷰션 장벽이 높다. 52개 소스 모듈, 38개 확장, 48개 스킬의 상호작용 이해에 상당한 학습 곡선이 필요하다.

### 7.4 에이전트 코어 의존성

핵심 에이전트 런타임이 `@mariozechner/pi-agent-core`라는 외부 패키지에 의존한다. 이 패키지의 개발 방향이나 유지보수 상태가 프로젝트 전체에 영향을 미칠 수 있다.

---

## 8. 종합 평가

### 8.1 영역별 평가

| 영역 | 평가 | 핵심 교훈 |
|------|------|-----------|
| **멀티채널 통합** | ★★★★★ | 20+ 채널을 하나의 추상화로 통합하는 것은 산업 최고 수준 |
| **플러그인 확장성** | ★★★★★ | 7가지 확장 포인트 + 독립 패키지 격리는 모범적 |
| **설정 기반 라우팅** | ★★★★☆ | 코드 변경 없는 에이전트 토폴로지 변경은 운영 효율 극대화 |
| **스킬 시스템** | ★★★★☆ | Progressive Disclosure + 선언적 의존성은 토큰 효율성에 기여 |
| **보안 모델** | ★★★★☆ | 3계층 보안 + 점진적 신뢰 구축은 실용적 |
| **세션 관리** | ★★★☆☆ | 파일 기반은 간단하지만 대규모 확장에 한계 |
| **수평 확장** | ★★☆☆☆ | 단일 인스턴스 중심 설계로 클러스터링 미지원 |

### 8.2 에이전트 시스템 설계 시 체크리스트

OpenClaw 분석을 통해 도출한, 에이전트 시스템 설계 시 고려할 체크리스트:

- [ ] 에이전트 ↔ 인터페이스 매핑이 설정으로 분리되어 있는가?
- [ ] 멀티에이전트 격리(워크스페이스, 세션, 도구)가 보장되는가?
- [ ] 플러그인 확장 포인트이 충분히 다양한가? (최소 Tool + Hook + Channel)
- [ ] 도구/스킬이 환경에 따라 자동으로 활성화/비활성화되는가?
- [ ] 스킬 로딩이 토큰 효율적인가? (Progressive Disclosure)
- [ ] 보안이 설계 단계부터 계층적으로 내장되어 있는가?
- [ ] 실행 승인에 "점진적 신뢰 구축" 모델이 있는가?
- [ ] 컨텍스트 윈도우 관리(압축, 트렁케이션)가 자동화되어 있는가?
- [ ] 세션 키가 라우팅 컨텍스트를 자체적으로 인코딩하는가?
- [ ] 설정 핫 리로드가 가능한가?
- [ ] 도구 스키마가 여러 LLM 프로바이더에 자동 어댑트되는가?
- [ ] 서브에이전트의 도구 범위가 부모보다 제한되는가?
- [ ] 라우팅 결정 과정이 투명하게 추적 가능한가? (matchedBy 등)

---

## 참고 자료

- [OpenClaw GitHub Repository](https://github.com/openclaw/openclaw)
- [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)
- [OpenClaw Official Site](https://openclaw.ai/)
- [DigitalOcean - What is OpenClaw?](https://www.digitalocean.com/resources/articles/what-is-openclaw)
- [OpenClaw Goes Viral - 145K+ Stars (creati.ai)](https://creati.ai/ai-news/2026-02-11/openclaw-open-source-ai-agent-viral-145k-github-stars/)
- [OpenClaw v2026.2.6 Release Notes (CybersecurityNews)](https://cybersecuritynews.com/openclaw-v2026-2-6-released/)
- [Medium - What is OpenClaw: Setup + Features](https://medium.com/@gemQueenx/what-is-openclaw-open-source-ai-agent-in-2026-setup-features-8e020db20e5e)
