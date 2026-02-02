# OpenClaw 아키텍처 및 시스템 분석

## 1. 프로젝트 개요

**OpenClaw**는 개인용 AI 어시스턴트를 자체 디바이스에서 운영할 수 있게 해주는 오픈소스 프로젝트입니다. 다양한 메시징 채널(WhatsApp, Telegram, Slack, Discord, Signal, iMessage 등)을 통해 사용자와 상호작용하며, 음성 인식/합성, 실시간 캔버스, 멀티 에이전트 라우팅 등 고급 기능을 제공합니다.

### 핵심 특징
- **로컬 우선(Local-first)**: 단일 Gateway가 모든 세션, 채널, 도구, 이벤트를 관리
- **멀티 채널 통합**: 15개 이상의 메시징 플랫폼 지원
- **멀티 에이전트**: 채널/계정/피어별 격리된 에이전트 라우팅
- **확장 가능한 스킬 시스템**: 번들/관리형/워크스페이스 스킬 지원
- **크로스 플랫폼**: macOS, iOS, Android 네이티브 앱 + CLI

### 기술 스택
| 영역 | 기술 |
|------|------|
| **언어** | TypeScript (주력), Swift (macOS/iOS), Kotlin (Android) |
| **런타임** | Node.js 22+, Bun (개발용) |
| **패키지 관리** | pnpm (monorepo) |
| **AI 플랫폼** | Pi Agent Runtime (@mariozechner/pi-*) |
| **LLM 지원** | Anthropic (Claude), OpenAI, Google Gemini, Ollama 등 |
| **메시징** | Baileys (WhatsApp), grammY (Telegram), Bolt (Slack), discord.js |
| **데이터베이스** | SQLite (세션/메모리), sqlite-vec (벡터 검색) |

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           메시징 채널 (Inbound)                               │
│  WhatsApp │ Telegram │ Slack │ Discord │ Signal │ iMessage │ Teams │ Matrix │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Gateway                                         │
│                         (Control Plane)                                      │
│                      ws://127.0.0.1:18789                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   채널      │ │   세션      │ │   라우팅    │ │   인증      │            │
│  │  Registry   │ │  Manager    │ │   Engine    │ │   Handler   │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   도구      │ │   스킬      │ │   메모리    │ │   Cron      │            │
│  │  Executor   │ │  Platform   │ │   Manager   │ │  Scheduler  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pi Agent      │    │   CLI Client    │    │   Native Apps   │
│   (RPC Mode)    │    │   (openclaw)    │    │  macOS/iOS/And  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ • agent         │    │ • Menu Bar      │
│ │ Tool Stream │ │    │ • message send  │    │ • Voice Wake    │
│ │ Block Stream│ │    │ • gateway       │    │ • Talk Mode     │
│ │ Context Mgmt│ │    │ • onboard       │    │ • Canvas        │
│ └─────────────┘ │    │ • doctor        │    │ • Camera Node   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM Providers                                      │
│      Anthropic (Claude) │ OpenAI │ Google Gemini │ Ollama │ Bedrock         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 메시지 수신 (Inbound)                                                    │
│    채널 → Gateway → 라우팅 엔진 → 에이전트 매칭 → 세션 키 생성               │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 에이전트 처리                                                            │
│    세션 로드 → 컨텍스트 구성 → LLM 호출 → 도구 실행 → 응답 생성              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 응답 전송 (Outbound)                                                     │
│    응답 포맷팅 → 청킹 → 채널 어댑터 → 메시지 전송 → 확인 반응                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트 분석

### 3.1 Gateway (컨트롤 플레인)

**위치**: `src/gateway/`

Gateway는 시스템의 중심 허브로, 모든 클라이언트 연결과 채널을 관리합니다.

#### 핵심 기능
- **WebSocket API**: 클라이언트(CLI, 앱, 노드)와의 양방향 통신
- **세션 관리**: 에이전트별 대화 상태 유지
- **이벤트 방출**: presence, tick, 채널 상태 등
- **디바이스 페어링**: 신규 디바이스 인증 및 토큰 발급

#### 연결 수명 주기
```typescript
// 1. 핸드셰이크
{ type: "connect", role: "cli" | "app" | "node", deviceId: string }
// Gateway 응답: 설정 스냅샷

// 2. 이벤트 스트리밍
{ type: "event:presence", ... }
{ type: "event:tick", ... }

// 3. 요청-응답
{ type: "req:agent", id: string, message: string }
{ type: "res", id: string, result: ... }
```

#### 보안 모델
```typescript
// DM 정책 설정
dmPolicy: "pairing" | "open"  // pairing이 기본값
allowFrom: string[]           // 허용된 발신자 목록

// 페어링 코드 기반 인증
openclaw pairing approve <channel> <code>
```

### 3.2 Pi Agent Runtime

**위치**: `src/agents/`

에이전트 실행의 핵심 런타임으로, RPC 모드에서 도구 스트리밍과 블록 스트리밍을 지원합니다.

#### 주요 함수
```typescript
// 에이전트 실행 관리
runEmbeddedPiAgent(options)         // 에이전트 실행 초기화
isEmbeddedPiRunActive()             // 실행 상태 확인
isEmbeddedPiRunStreaming()          // 스트리밍 상태 확인
waitForEmbeddedPiRunEnd()           // 실행 완료 대기

// 세션 및 히스토리 관리
resolveEmbeddedSessionLane()        // 세션 레인 해석
limitHistoryTurns()                 // 히스토리 제한
getDmHistoryLimitFromSessionKey()   // DM 히스토리 한도

// 도구 및 파라미터
splitSdkTools()                     // SDK 도구 분류
applyExtraParamsToAgent()           // 동적 파라미터 적용
buildEmbeddedSandboxInfo()          // 샌드박스 환경 구성
```

#### 도구 정의 및 실행
```typescript
// 도구 생성 (src/agents/pi-tools.ts)
function createOpenClawCodingTools(options) {
  return {
    read: sandboxRoot ? sandboxedRead : localRead,
    write: sandboxRoot ? sandboxedWrite : localWrite,
    edit: sandboxRoot ? sandboxedEdit : localEdit,
    exec: createExecTool(options),
    process: createProcessTool(options),
    applyPatch: openAIModel ? createPatchTool() : null,
  }
}

// 정책 기반 필터링 (계층적 우선순위)
프로필 정책 → 제공자별 → 글로벌 → 에이전트별 → 그룹 → 샌드박스
```

### 3.3 라우팅 엔진

**위치**: `src/routing/`

메시지를 적절한 에이전트로 라우팅하는 핵심 로직입니다.

#### 라우팅 우선순위
```typescript
function resolveAgentRoute(bindings, match): AgentRoute {
  // 1. 피어 직접 매칭 (DM, 그룹, 채널별 바인딩)
  const peerBinding = bindings.find(b => matchesPeer(b, match));
  if (peerBinding) return peerBinding.agent;

  // 2. 부모 피어 매칭 (스레드의 경우)
  if (match.parentPeer) {
    const parentBinding = bindings.find(b => matchesPeer(b, match.parentPeer));
    if (parentBinding) return parentBinding.agent;
  }

  // 3. 길드/팀 매칭
  const guildBinding = bindings.find(b => matchesGuild(b, match));
  if (guildBinding) return guildBinding.agent;

  // 4. 계정 매칭 (와일드카드 포함)
  const accountBinding = bindings.find(b =>
    b.account === match.accountId || b.account === "*"
  );
  if (accountBinding) return accountBinding.agent;

  // 5. 기본값
  return config.defaultAgent;
}
```

#### 세션 키 생성
```typescript
// 직접 메시지 세션 키
sessionKey = `agent:${agentId}:${mainKey}`

// 그룹 채팅 세션 키
sessionKey = `agent:${agentId}:${channel}:group:${groupId}`

// 크론/웹훅 세션 키
sessionKey = `cron:${cronId}` | `hook:${hookUuid}`
```

### 3.4 채널 시스템

**위치**: `src/channels/`, `extensions/`

15개 이상의 메시징 플랫폼을 통합 관리합니다.

#### 채널 레지스트리 구조
```typescript
// 코어 채널 (src/channels/registry.ts)
const CHAT_CHANNEL_ORDER = [
  'whatsapp', 'telegram', 'slack', 'discord',
  'googlechat', 'signal', 'imessage', 'webchat'
];

// 확장 채널 (extensions/)
const EXTENSION_CHANNELS = [
  'msteams', 'matrix', 'bluebubbles', 'zalo',
  'zalouser', 'line', 'mattermost', 'nostr'
];

// 채널 메타데이터
interface ChannelMeta {
  id: ChatChannelId;
  label: string;           // UI 표시명
  docPath: string;         // 문서 경로
  blurb: string;           // 설명
  icon: string;            // 아이콘
}

// 채널 별칭 (호환성)
const CHAT_CHANNEL_ALIASES = {
  'imsg': 'imessage',
  'wa': 'whatsapp',
  'tg': 'telegram',
};
```

#### 채널 플러그인 인터페이스
```typescript
interface ChannelPlugin {
  // 설정
  configSchema: ChannelConfigSchema;

  // 라이프사이클 훅
  setup: ChannelSetupAdapter;
  auth: ChannelAuthAdapter;
  heartbeat: ChannelHeartbeatAdapter;
  status: ChannelStatusAdapter;

  // 메시징
  messaging: ChannelMessagingAdapter;
  outbound: ChannelOutboundAdapter;

  // 명령 및 도구
  command: ChannelCommandAdapter;
  mention: ChannelMentionAdapter;
  tools: ChannelAgentToolFactory;
}
```

### 3.5 스킬 시스템

**위치**: `src/agents/skills.ts`, `skills/`

재사용 가능한 기능 모듈을 관리합니다.

#### 스킬 유형
| 유형 | 설명 | 위치 |
|------|------|------|
| **번들 스킬** | 기본 제공 스킬 | `skills/` |
| **관리형 스킬** | 설치 가능한 외부 스킬 | npm registry |
| **워크스페이스 스킬** | 프로젝트별 커스텀 스킬 | `.openclaw/skills/` |

#### 스킬 로딩 플로우
```typescript
// 1. 스킬 엔트리 로드
loadWorkspaceSkillEntries()

// 2. 필터링 및 검증
filterWorkspaceSkillEntries()

// 3. 스냅샷 생성
buildWorkspaceSkillSnapshot()

// 4. 프롬프트 구성
buildWorkspaceSkillsPrompt()

// 5. 런타임 해석
resolveSkillsPromptForRun()
```

#### 스킬 예시 (50+ 내장 스킬)
```
skills/
├── 1password/          # 비밀번호 관리
├── github/             # GitHub 통합
├── slack/              # Slack 채널 도구
├── discord/            # Discord 봇 도구
├── notion/             # Notion 문서 관리
├── obsidian/           # Obsidian 노트
├── spotify-player/     # 음악 제어
├── weather/            # 날씨 조회
├── voice-call/         # 음성 통화
└── ...
```

### 3.6 메모리 시스템

**위치**: `src/memory/`

장기 기억과 컨텍스트 검색을 위한 하이브리드 검색 시스템입니다.

#### 아키텍처
```
┌─────────────────────────────────────────────────────────────────┐
│                      메모리 매니저                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   파일 동기화    │    │   세션 동기화    │                     │
│  │ (markdown 파일) │    │ (트랜스크립트)   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    청킹 엔진                             │    │
│  │  • 토큰 기반 분할                                        │    │
│  │  • 청크 겹침 (문맥 연속성)                                │    │
│  │  • 중복 제거                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│           ┌──────────────┼──────────────┐                       │
│           ▼              ▼              ▼                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ 임베딩 캐시  │ │  벡터 검색   │ │ FTS 검색    │                │
│  │ (해시 기반) │ │(sqlite-vec)│ │ (키워드)    │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 하이브리드 검색 결합                      │    │
│  │  result = α * vector_score + β * keyword_score           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### 임베딩 파이프라인
```typescript
// 지원 제공자
type EmbeddingProvider = 'openai' | 'gemini' | 'local';

// 배치 처리 (효율성 최적화)
const BATCH_SIZE = 8000; // 토큰 단위

// 재시도 로직 (지수 백오프)
const MAX_RETRIES = 3;

// 타임아웃
const TIMEOUT_LOCAL = 300000;  // 5분 (로컬)
const TIMEOUT_REMOTE = 60000;  // 1분 (원격)
```

### 3.7 세션 관리

**위치**: `src/sessions/`

대화 상태와 히스토리를 관리합니다.

#### 세션 모델
```typescript
// 세션 범위 설정
interface SessionScope {
  dmScope: 'main' | 'per-peer' | 'per-channel-peer';
  groupScope: 'isolated' | 'shared';
}

// 세션 만료 정책
interface SessionExpiry {
  dailyReset: string;      // 예: "04:00" (로컬 시간)
  idleMinutes?: number;    // 유휴 타임아웃
}

// 세션 키 구조
type SessionKey =
  | `agent:${agentId}:${mainKey}`           // DM
  | `agent:${agentId}:${channel}:group:${id}` // 그룹
  | `cron:${cronId}`                         // 크론
  | `hook:${hookUuid}`;                      // 웹훅
```

#### 세션 라이프사이클
```
세션 생성 → 메시지 누적 → 만료 평가 → 리셋
                ↑                    │
                └────────────────────┘
```

---

## 4. 플러그인/확장 시스템

### 4.1 플러그인 SDK

**위치**: `src/plugin-sdk/`

채널 플러그인 개발을 위한 표준 인터페이스를 제공합니다.

#### 플러그인 구조
```typescript
// 플러그인 서비스 인터페이스
interface OpenClawPluginService {
  // 기본 메타데이터
  name: string;
  version: string;

  // 채널 어댑터
  channel: ChannelPlugin;

  // 도구 팩토리
  tools: ChannelAgentToolFactory;

  // HTTP 라우트 (선택적)
  routes?: PluginRouteHandler[];
}

// 라이프사이클 어댑터
interface ChannelSetupAdapter {
  initialize(ctx: PluginContext): Promise<void>;
  validate(config: unknown): ValidationResult;
}

interface ChannelAuthAdapter {
  authenticate(credentials: unknown): Promise<AuthResult>;
  refresh(token: string): Promise<AuthResult>;
}

interface ChannelHeartbeatAdapter {
  ping(): Promise<boolean>;
  getStatus(): ChannelStatus;
}
```

### 4.2 확장 채널 구조

**위치**: `extensions/`

```
extensions/
├── msteams/           # Microsoft Teams
├── matrix/            # Matrix 프로토콜
├── bluebubbles/       # BlueBubbles (iMessage 확장)
├── zalo/              # Zalo 메신저
├── zalouser/          # Zalo Personal
├── signal/            # Signal (확장)
├── slack/             # Slack (확장)
├── telegram/          # Telegram (확장)
├── discord/           # Discord (확장)
├── voice-call/        # 음성 통화
├── memory-core/       # 메모리 코어 확장
├── memory-lancedb/    # LanceDB 벡터 스토어
└── llm-task/          # LLM 태스크 러너
```

---

## 5. 네이티브 앱 아키텍처

### 5.1 macOS 앱

**위치**: `apps/macos/`

```
┌─────────────────────────────────────────────────────────────┐
│                    macOS Menu Bar App                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   SwiftUI View Layer                     │ │
│  │  • Menu Bar Control                                      │ │
│  │  • Settings Panel                                        │ │
│  │  • Debug Tools                                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   Core Services                          │ │
│  │  • Voice Wake (항상 듣기)                                 │ │
│  │  • Talk Mode (PTT/대화)                                  │ │
│  │  • Gateway Control                                       │ │
│  │  • Node Host                                             │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   WebSocket Client                       │ │
│  │  • Gateway 연결                                          │ │
│  │  • 이벤트 수신                                           │ │
│  │  • 명령 전송                                             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 iOS/Android 앱

**위치**: `apps/ios/`, `apps/android/`

#### 공통 기능
- **Canvas**: 에이전트 제어 가능한 UI 렌더링
- **Voice Wake**: 음성 활성화
- **Talk Mode**: 대화형 음성 상호작용
- **Camera Node**: 카메라 스냅/클립
- **Screen Recording**: 화면 녹화
- **Bonjour Pairing**: 로컬 네트워크 디바이스 발견

---

## 6. 도구(Tools) 심층 분석

### 6.1 기본 도구 세트

```typescript
// 파일 시스템 도구
const fileTools = {
  read: {
    description: "파일 읽기",
    parameters: { path: string, encoding?: string },
    sandbox: true,  // 샌드박스 환경에서 격리 실행
  },
  write: {
    description: "파일 작성",
    parameters: { path: string, content: string },
    sandbox: true,
  },
  edit: {
    description: "파일 편집",
    parameters: { path: string, changes: EditChange[] },
    sandbox: true,
  },
};

// 명령 실행 도구
const execTools = {
  exec: {
    description: "쉘 명령 실행",
    parameters: { command: string, cwd?: string },
    approval: true,  // 승인 필요
  },
  process: {
    description: "백그라운드 프로세스 관리",
    parameters: { action: 'start' | 'stop' | 'status', pid?: number },
  },
};

// 브라우저 도구
const browserTools = {
  snapshot: "페이지 스냅샷",
  navigate: "URL 탐색",
  click: "요소 클릭",
  type: "텍스트 입력",
  upload: "파일 업로드",
};

// 캔버스 도구
const canvasTools = {
  push: "A2UI 푸시",
  reset: "캔버스 리셋",
  eval: "JavaScript 실행",
  snapshot: "캔버스 스냅샷",
};

// 노드 도구
const nodeTools = {
  'camera.snap': "카메라 스냅샷",
  'camera.clip': "비디오 클립",
  'screen.record': "화면 녹화",
  'location.get': "위치 조회",
  'notify': "알림 전송",
};
```

### 6.2 도구 정책 시스템

```typescript
// 계층적 정책 적용
interface ToolPolicy {
  // 허용/차단 목록
  allow?: string[];
  deny?: string[];

  // 승인 요구
  requireApproval?: string[];

  // 샌드박스 강제
  forceSandbox?: boolean;
}

// 정책 우선순위 (높은 것이 우선)
const POLICY_PRIORITY = [
  'profile',           // 사용자 프로필
  'provider-profile',  // 제공자별 프로필
  'global',            // 전역 설정
  'provider-global',   // 제공자별 전역
  'agent',             // 에이전트별
  'provider-agent',    // 제공자별 에이전트
  'group',             // 그룹
  'sandbox',           // 샌드박스
  'subagent',          // 서브에이전트
];

// 도구 래퍼 (후처리)
const toolWrappers = [
  wrapToolWithBeforeToolCallHook,  // 호출 전 훅
  wrapToolWithAbortSignal,         // 중단 신호
  wrapToolParamNormalization,      // 스키마 정규화
];
```

---

## 7. 설계 패턴 및 모범 사례

### 7.1 의존성 주입 패턴

```typescript
// 기본 의존성 생성
function createDefaultDeps(config: Config): Dependencies {
  return {
    logger: createLogger(config),
    fs: createFileSystem(config),
    http: createHttpClient(config),
    db: createDatabase(config),
  };
}

// 런타임 환경 주입
interface RuntimeEnv {
  log: (msg: string) => void;
  error: (err: Error) => void;
  config: Config;
  deps: Dependencies;
}

// 함수에 의존성 주입
function runBootOnce(params: {
  cfg: Config;
  deps: Dependencies;
  workspaceDir: string;
  runtimeEnv: RuntimeEnv;
}): BootResult { ... }
```

### 7.2 이벤트 기반 아키텍처

```typescript
// 이벤트 타입 정의
type GatewayEvent =
  | { type: 'presence'; data: PresenceData }
  | { type: 'tick'; data: TickData }
  | { type: 'channel:status'; data: ChannelStatus }
  | { type: 'agent:message'; data: AgentMessage }
  | { type: 'agent:tool'; data: ToolCall };

// 이벤트 핸들러
class GatewayEventEmitter {
  on(type: string, handler: EventHandler): void;
  off(type: string, handler: EventHandler): void;
  emit(event: GatewayEvent): void;
}
```

### 7.3 샌드박스 패턴

```typescript
// 샌드박스 컨텍스트
interface SandboxContext {
  root: string;           // 격리된 루트 디렉토리
  allowedPaths: string[]; // 접근 허용 경로
  env: Record<string, string>; // 환경 변수
}

// 샌드박스 도구 생성
function createSandboxedTool(
  tool: Tool,
  sandbox: SandboxContext
): Tool {
  return {
    ...tool,
    execute: async (params) => {
      // 경로 검증
      validatePath(params.path, sandbox.allowedPaths);
      // 격리된 환경에서 실행
      return tool.execute(params, sandbox);
    },
  };
}
```

### 7.4 페일오버 및 복원력

```typescript
// 모델 페일오버
interface ModelFailover {
  primaryModel: string;
  fallbackModels: string[];
  retryPolicy: RetryPolicy;
}

// 재시도 정책
interface RetryPolicy {
  maxRetries: number;
  backoffType: 'linear' | 'exponential';
  baseDelay: number;
  maxDelay: number;
}

// 인증 프로필 순환
function resolveAuthProfileOrder(profiles: AuthProfile[]): AuthProfile[] {
  return profiles
    .sort((a, b) => {
      // 마지막 성공 > 마지막 사용 > 설정 순서
      if (a.lastGood && !b.lastGood) return -1;
      if (a.lastUsed > b.lastUsed) return -1;
      return a.configOrder - b.configOrder;
    });
}
```

---

## 8. 운영 및 배포

### 8.1 설치 및 온보딩

```bash
# 설치
npm install -g openclaw@latest

# 온보딩 위저드 (권장)
openclaw onboard --install-daemon

# 게이트웨이 시작
openclaw gateway --port 18789 --verbose

# 헬스 체크
openclaw doctor
```

### 8.2 Docker 배포

```yaml
# docker-compose.yml
version: '3.8'
services:
  gateway:
    build: .
    ports:
      - "18789:18789"
    volumes:
      - ~/.openclaw:/root/.openclaw
    environment:
      - OPENCLAW_GATEWAY_TOKEN=${GATEWAY_TOKEN}
```

### 8.3 원격 접근

```bash
# Tailscale Serve/Funnel
openclaw gateway --tailscale-serve

# SSH 터널
ssh -L 18789:localhost:18789 user@gateway-host
```

---

## 9. 에이전트 시스템 설계 인사이트

### 9.1 핵심 설계 원칙

| 원칙 | 설명 | 구현 |
|------|------|------|
| **로컬 우선** | 데이터와 실행을 사용자 디바이스에 유지 | Gateway 로컬 실행 |
| **채널 불가지론** | 메시징 플랫폼에 독립적인 핵심 로직 | 채널 어댑터 패턴 |
| **멀티 에이전트** | 컨텍스트별 격리된 에이전트 | 라우팅 엔진 |
| **확장성** | 플러그인/스킬 기반 기능 확장 | SDK + 레지스트리 |
| **보안 기본값** | 페어링 기반 DM 정책 | 허용 목록 + 토큰 |

### 9.2 벤치마킹 포인트

1. **Gateway 중심 아키텍처**: 단일 컨트롤 플레인이 모든 상태를 관리하여 일관성 보장
2. **세션 키 전략**: 채널/계정/피어 조합으로 고유한 대화 컨텍스트 유지
3. **도구 정책 계층화**: 세밀한 권한 제어로 보안과 유연성 균형
4. **하이브리드 메모리 검색**: 벡터 + 키워드 검색 결합으로 검색 품질 향상
5. **플러그인 SDK**: 표준화된 인터페이스로 확장 개발 간소화

### 9.3 적용 가능한 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│                    에이전트 시스템 설계 참조                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Gateway 패턴: 중앙 집중식 세션/상태 관리                      │
│ 2. 라우팅 엔진: 컨텍스트 기반 에이전트 바인딩                     │
│ 3. 도구 정책: 계층적 권한 모델                                   │
│ 4. 채널 어댑터: 플랫폼 추상화                                    │
│ 5. 스킬 시스템: 모듈식 기능 확장                                 │
│ 6. 메모리 관리: 하이브리드 검색 + 임베딩 캐싱                     │
│ 7. 세션 라이프사이클: 만료 정책 + 수동 리셋                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 결론

OpenClaw는 개인용 AI 어시스턴트 시스템의 참조 구현으로, 다음과 같은 핵심 강점을 가집니다:

### 강점
- **포괄적인 채널 통합**: 15개 이상의 메시징 플랫폼 지원
- **유연한 에이전트 라우팅**: 컨텍스트 기반 멀티 에이전트 지원
- **확장 가능한 아키텍처**: 플러그인/스킬 기반 모듈화
- **크로스 플랫폼**: CLI + 네이티브 앱 (macOS/iOS/Android)
- **보안 기본값**: 페어링 기반 인증, 샌드박스 실행

### 에이전트 시스템 개발자를 위한 시사점
1. **Gateway 중심 설계**는 복잡한 멀티 채널/멀티 에이전트 시나리오에서 상태 일관성을 보장합니다
2. **세션 키 전략**은 대화 컨텍스트 격리와 연속성의 균형을 맞춥니다
3. **도구 정책 계층화**는 보안과 유연성을 동시에 달성합니다
4. **플러그인 SDK**는 생태계 확장을 촉진합니다
5. **하이브리드 메모리 검색**은 장기 컨텍스트 활용을 개선합니다

---

## 참고 자료

- **GitHub**: https://github.com/openclaw/openclaw
- **공식 문서**: https://docs.openclaw.ai
- **DeepWiki 분석**: https://deepwiki.com/openclaw/openclaw
- **Discord 커뮤니티**: https://discord.gg/clawd
