# OpenHarness 분석 — Thronicle 적용 가능성 검토

> **대상**: https://github.com/HKUDS/OpenHarness (HKU Data Science Lab)
> **Stars**: ~7,950 | **언어**: Python 3.10+ | **라이선스**: MIT
> **한 줄 요약**: Claude Code를 Python으로 재구현한 오픈소스. "파이썬 버전 Claude Code."

---

## 1. 프로젝트 정체

OpenHarness는 스스로를 **"Open-source Python port of Claude Code"** 라고 소개합니다.
Claude Code의 핵심 아키텍처를 Python으로 1:1 재현한 것:
- 도구 호출 루프 (에이전트 루프)
- 권한 시스템
- 컨텍스트 compaction
- CLAUDE.md 로딩
- 스킬 시스템
- MCP 통합
- 멀티에이전트 (Swarm)
- React TUI (Ink)

---

## 2. 기술 스택

| 영역 | 사용 기술 |
|------|----------|
| **언어** | Python 3.10+ (백엔드), TypeScript (TUI) |
| **빌드** | hatchling + uv |
| **CLI** | typer + rich |
| **데이터 검증** | pydantic v2 |
| **비동기** | asyncio (네이티브) |
| **TUI (주)** | React + Ink (TypeScript, 번들된 채로 배포) |
| **TUI (대체)** | Textual (Python, Node.js 없을 때 폴백) |
| **DB** | **없음** — 전부 파일 기반 |
| **세션 저장** | `~/.openharness/sessions/` (JSON 파일) |
| **메모리** | `.openharness/memory/*.md` (마크다운 파일) |
| **설정** | `~/.openharness/settings.json` |

### LLM API 호출 방식 — litellm 미사용

**litellm도, Vercel AI SDK도 쓰지 않습니다.** 공식 SDK를 직접 사용합니다:

```
src/openharness/api/
├── client.py          ← anthropic SDK (AsyncAnthropic) 직접 사용
├── openai_client.py   ← openai SDK (AsyncOpenAI) 직접 사용
├── copilot_client.py  ← GitHub Copilot OAuth 전용
├── codex_client.py    ← OpenAI Codex 전용
└── registry.py        ← 20+ 프로바이더 감지/라우팅
```

두 클라이언트 모두 공통 프로토콜 `SupportsStreamingMessages`를 구현하여:
```python
async for event in client.stream_message(**params):
    # ApiTextDeltaEvent | ApiMessageCompleteEvent | ApiRetryEvent
```

이 방식은 **OpenCode가 Vercel AI SDK로 하는 것과 동일한 패턴을 Python으로 직접 구현**한 것입니다.

### 지원 프로바이더 (20+)

| 카테고리 | 프로바이더 |
|---------|-----------|
| **글로벌** | Anthropic, OpenAI, Google (Gemini), Groq, Together AI |
| **게이트웨이** | OpenRouter, AiHubMix, SiliconFlow |
| **중국** | DeepSeek, Moonshot/Kimi, Zhipu/GLM, MiniMax, Baichuan, DashScope (Qwen) |
| **Copilot** | GitHub Models, GitHub Copilot (OAuth) |
| **로컬** | Ollama, vLLM |

---

## 3. 에이전트 루프 — Claude Code의 Python 재현

### 핵심 루프 (`engine/query.py :: run_query()`)

```python
while turn < max_turns:
    # 1. 자동 compaction 체크
    auto_compact_if_needed(messages, context_window)

    # 2. LLM 스트리밍 호출
    async for event in api_client.stream_message(messages, tools, system):
        yield event  # text-delta를 UI로 스트리밍

    # 3. 도구 호출이 없으면 종료
    if stop_reason != "tool_use":
        break

    # 4. 도구 실행 (단일: 순차, 복수: asyncio.gather 병렬)
    for tool_call in tool_calls:
        # 권한 확인 → PreToolUse 훅 → 실행 → PostToolUse 훅
        result = await execute_tool(tool_call)

    # 5. 결과를 messages에 추가, 루프 반복
    messages.append(tool_results_as_user_message)
```

**OpenCode와의 대응:**

| OpenCode (TS) | OpenHarness (Python) |
|--------------|---------------------|
| `SessionProcessor.process()` | `run_query()` |
| `streamText()` (Vercel AI SDK) | `api_client.stream_message()` |
| `Tool.define()` + Zod | `BaseTool` + Pydantic |
| `SessionCompaction.process()` | `auto_compact_if_needed()` |
| max_iterations | max_turns (기본 200) |

---

## 4. 도구 시스템 — 43+ 도구

### 핵심 도구 (OpenCode 대응)

| OpenCode 도구 | OpenHarness 도구 | 비고 |
|--------------|-----------------|------|
| read | `read_file` | 200줄 기본, offset/limit, 2000줄 max |
| write | `write_file` | - |
| edit | `edit_file` | old_string/new_string 치환 |
| glob | `glob` | - |
| grep | `grep` | - |
| bash | `bash` | 600초 타임아웃, **12KB 출력 truncation** |
| webfetch | `web_fetch` | - |
| websearch | `web_search` | - |
| task | `agent`, `task_create/get/list/stop/output/update` | **더 풍부** |
| question | `ask_user_question` | - |
| skill | `skill` | - |
| todowrite | `todo_write` | - |
| lsp | `lsp` | - |
| batch | 없음 | - |

### OpenHarness에만 있는 도구

| 도구 | 설명 |
|------|------|
| `team_create/delete` | 멀티에이전트 팀 생성/삭제 |
| `send_message` | 에이전트 간 메시지 전송 |
| `enter/exit_plan_mode` | 계획 모드 전환 |
| `enter/exit_worktree` | git worktree 격리 |
| `cron_create/list/delete/toggle` | 크론 스케줄링 |
| `remote_trigger` | 원격 에이전트 트리거 |
| `notebook_edit` | Jupyter 노트북 편집 |
| `mcp_tool`, `list_mcp_resources` | MCP 프로토콜 통합 |
| `config` | 런타임 설정 변경 |
| `brief` | 요약 생성 |
| `sleep` | 대기 |
| `tool_search` | 도구 검색 (도구가 많을 때) |

### 도구 정의 패턴

```python
# OpenHarness의 도구 정의 (BaseTool 상속)
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file from the filesystem..."  # LLM에게 보내는 설명
    
    class InputModel(BaseModel):
        file_path: str = Field(description="Absolute path to the file")
        offset: int = Field(default=1, description="Line number to start from")
        limit: int = Field(default=200, description="Max lines to read")
    
    input_model = InputModel
    
    async def execute(self, input: InputModel, context: ToolContext) -> ToolResult:
        ...
```

**Pydantic `BaseModel`이 Zod의 역할**을 합니다. `model_json_schema()`로 자동 JSON Schema 생성 → LLM에게 전달.

---

## 5. 컨텍스트 윈도우 관리 — 3단계 방어

### OpenCode vs OpenHarness 비교

| 단계 | OpenCode | OpenHarness |
|------|----------|-------------|
| **1층: 도구 출력 제한** | read: 2000줄/50KB, bash: 2000줄/50KB, grep: 100건 | read: 200줄 기본(2000 max), bash: **12KB**, grep: 제한 있음 |
| **2층: Truncation** | 전용 서비스 (디스크 저장 + 힌트) | bash에 내장 (12KB 잘림), 범용 서비스 없음 |
| **3층: Pruning (Microcompact)** | 오래된 도구 출력 → `[cleared]`, 최근 40K 토큰 보호 | 오래된 도구 출력 → `[Old tool result content cleared]`, **최근 5개** 도구 결과 보호 |
| **4층: Compaction** | LLM으로 구조화 요약 (Goal/Instructions/Discoveries/Accomplished/Files) | LLM으로 구조화 요약 (`<analysis>` + `<summary>` XML 블록) |
| **Overflow 감지** | `context_window - reserved(20K)` 기준 | `context_window - 20,000 - 13,000` 기준 |

### Microcompact (Pruning) 상세

```python
# OpenHarness의 microcompact 대상 도구
COMPACTABLE_TOOLS = [
    "read_file", "bash", "grep", "glob",
    "web_search", "web_fetch", "edit_file", "write_file"
]

# 최근 5개 도구 결과는 보호
# 그 이전 도구 결과는 "[Old tool result content cleared]"로 교체
```

OpenCode는 **토큰 기반** 보호 (최근 40K 토큰), OpenHarness는 **개수 기반** 보호 (최근 5개). 
OpenCode의 방식이 더 정밀하지만, OpenHarness의 방식이 더 단순합니다.

### Full Compaction 프롬프트

```
요청: older messages → LLM에 요약 요청

응답 형식:
<analysis>
  1. Primary request or task
  2. Key technical concepts/technologies
  3. Files and code sections referenced
  4. Errors, issues, or bugs identified
  5. Pending tasks or steps remaining
  6. What is being worked on right now
</analysis>
<summary>
  [요약본]
</summary>
```

---

## 6. 메모리 시스템

### CLAUDE.md 디스커버리

OpenCode의 AGENTS.md/CLAUDE.md 패턴과 동일:
- 프로젝트 루트에서 위로 올라가며 `CLAUDE.md` 파일 탐색
- 발견된 모든 파일을 시스템 프롬프트에 주입

### 영속 메모리 (.openharness/memory/)

```
.openharness/
├── MEMORY.md            ← 인덱스 파일 (링크 목록)
└── memory/
    ├── user_role.md     ← 사용자 역할/선호도
    ├── feedback_testing.md  ← 피드백 기록
    └── project_auth.md  ← 프로젝트 관련 지식
```

각 메모리 파일:
```markdown
---
name: user_role
description: User is a data scientist focused on observability
type: user
---

사용자는 데이터 사이언티스트이며 현재 로깅 시스템을 조사 중.
```

**검색 시 YAML frontmatter의 description으로 관련성을 판단**하고, 본문 300자 미리보기를 제공합니다.

---

## 7. 멀티에이전트 (Swarm) — OpenCode보다 풍부

```
OpenCode:
├── task 도구로 서브에이전트 생성 (독립 컨텍스트)
└── 결과만 메인에 반환

OpenHarness:
├── agent 도구 (서브에이전트 = OpenCode task와 동일)
├── team_create/delete (팀 생성/삭제)
├── send_message (에이전트 간 메시지)
├── 서브프로세스 기반 워커 (별도 Python 프로세스)
├── git worktree 격리 (병렬 에이전트가 각자의 브랜치에서 작업)
├── 메일박스 (비동기 메시지 큐)
├── 코디네이터 모드 (XML 기반 작업 할당)
└── 크론 스케줄링 (반복 실행)
```

---

## 8. IM 채널 통합 — 독특한 기능

```
src/openharness/channels/
├── slack.py
├── telegram.py
├── discord.py
├── feishu.py
├── dingtalk.py
├── whatsapp.py
├── matrix.py
├── qq.py
├── email.py
└── wechat.py
```

에이전트를 **Slack/Telegram/Discord 봇**으로 배포할 수 있습니다.
이 기능은 OpenCode/Claude Code에는 없습니다.

---

## 9. Thronicle 적용 가능성 평가

### 배경: 내 시스템의 고민과 방향

| 요구사항 | 현재 Thronicle (Agno) | OpenHarness |
|---------|---------------------|-------------|
| 마크다운 보고서 AI 에디터 | ✅ cortex agent + workspace toolkit | ⚠️ 코딩 에이전트 기반, 보고서 특화 아님 |
| 사용자와 대화하면서 파일 편집 | ✅ 프롬프트로 가이드 (모드 A) | ✅ 기본 동작 (도구 호출 인터리빙) |
| 컨텍스트 관리 (pruning/compaction) | ⚠️ CortexToolkit (memo/reflect) + Agno 히스토리 제한 | ✅ **3단계 방어 완전 구현** |
| 멀티 프로바이더 | ⚠️ Agno가 지원하지만 제한적 | ✅ 20+ 프로바이더 |
| 도구 출력 truncation | ✅ workspace_toolkit (2000줄/50KB) | ⚠️ bash만 12KB, 범용 서비스 없음 |
| Obsidian 스타일 노트 링킹 | ✅ workspace_backlinks 구현 | ❌ 없음 |
| 섹션 단위 읽기/편집 | ✅ workspace_read_section, edit_section | ❌ 없음 |
| _index.md 자동 디스커버리 | ✅ 구현됨 | ✅ CLAUDE.md 디스커버리 |
| 세션↔노트 감사 로깅 | ✅ cortex_entries note_op | ❌ 없음 (파일 기반만) |
| S3 백엔드 스토리지 | ✅ | ❌ 로컬 파일 시스템만 |
| PostgreSQL 영속 | ✅ | ❌ 파일 기반만 |
| 금융 도메인 도구 | ✅ 17개 toolkit (DART, FMP 등) | ❌ 코딩 도구만 |
| IM 채널 통합 | ❌ | ✅ 10+ 채널 |
| 멀티에이전트 (swarm) | ❌ (서브에이전트 정도) | ✅ 풍부한 swarm 시스템 |

### "내 입맛에 맞출 수 있는가?"

#### ✅ 가져올 수 있는 것 (높은 가치)

1. **3단계 컨텍스트 관리 엔진** (`services/compact/`)
   - Microcompact + Full Compaction + Overflow 감지
   - Python으로 깔끔하게 구현되어 있어 **Thronicle에 이식 가능**
   - 현재 Thronicle의 `CortexToolkit.context_summarize()`보다 자동화되고 정교함

2. **에이전트 루프 패턴** (`engine/query.py`)
   - 현재 Thronicle은 Agno가 에이전트 루프를 처리하는데, 더 세밀한 제어가 필요하면 OpenHarness의 루프를 참고할 수 있음
   - 특히 **매 turn마다 overflow 체크 → 자동 compaction** 패턴

3. **Swarm/멀티에이전트 시스템** (`swarm/`)
   - Thronicle이 복수 종목 동시 분석 같은 기능을 넣고 싶다면
   - 서브프로세스 기반 워커 + 메일박스 패턴은 훌륭한 참고

4. **IM 채널 통합** (`channels/`)
   - Slack/Telegram 봇으로 cortex agent를 배포하고 싶다면

#### ⚠️ 맞지 않는 부분

1. **스토리지가 로컬 파일 시스템 전용**
   - Thronicle은 S3 + PostgreSQL 기반
   - OpenHarness의 도구를 그대로 쓸 수 없고, 스토리지 레이어 교체 필요

2. **보고서/노트 특화 기능 없음**
   - 마크다운 섹션 읽기/편집, TOC 추출, 백링크 등은 없음
   - 이미 Thronicle에 우리가 구현한 것들이 OpenHarness에는 없음

3. **금융 도메인 도구 없음**
   - DART, FMP, 뉴스 검색 등은 Thronicle 고유

4. **DB 없음**
   - 세션, 메모리, 설정 모두 파일 기반
   - Thronicle의 PostgreSQL 기반 아키텍처와 맞지 않음

5. **Agno 프레임워크와 호환 안 됨**
   - OpenHarness는 자체 에이전트 루프를 가짐
   - Agno 위에서 OpenHarness를 쓸 수 없음 (둘 중 하나를 선택해야 함)

---

## 10. 실용적 적용 전략

### 전략 A: OpenHarness의 컨텍스트 관리 엔진만 이식 (추천)

```
Thronicle (현재)                    가져올 것
├── Agno 에이전트 루프 (유지)         ← 에이전트 루프는 Agno 유지
├── CortexToolkit (유지)             ← working memory 유지
├── WorkspaceToolkit (유지)          ← 노트 도구 유지
└── 컨텍스트 관리 ???               ← OpenHarness의 compact/ 이식
    ├── microcompact()               ← 오래된 도구 출력 정리
    ├── full_compact()               ← LLM 요약 생성
    └── auto_compact_check()         ← 매 turn overflow 체크
```

**이식 난이도: 중**
- `services/compact/` 디렉토리의 코드를 독립 모듈로 추출
- Agno의 `SessionSummaryManager`를 이것으로 교체하거나 보완
- 메시지 포맷 변환 필요 (OpenHarness 형식 → Agno 형식)

### 전략 B: OpenHarness를 기반으로 Thronicle을 재구축

```
OpenHarness 포크
├── 에이전트 루프 (그대로)
├── 컨텍스트 관리 (그대로)
├── 도구 시스템 (확장)
│   ├── 기존 도구 유지
│   ├── + WorkspaceToolkit (S3 백엔드)
│   ├── + 금융 도메인 도구 (DART, FMP 등)
│   └── + 노트 전용 도구 (read_section, edit_section, backlinks)
├── 스토리지 교체
│   ├── 파일 → PostgreSQL (세션)
│   └── 로컬 → S3 (워크스페이스)
└── IM 채널 통합 (활용)
```

**이식 난이도: 높음**
- Agno 완전 제거, OpenHarness 에이전트 루프로 대체
- 스토리지 레이어 전면 교체
- 기존 Thronicle의 금융 도구를 OpenHarness BaseTool로 포팅
- 장점: 가장 정교한 에이전트 아키텍처, 프레임워크 의존성 없음

### 전략 C: 참고만 하고 직접 구현

```
현재 Thronicle + OpenHarness 패턴 학습
    → OpenCode 분석 문서 (UNIFIED_SPEC.md)
    + OpenHarness 코드 참조
    → Thronicle 고유의 컨텍스트 관리 직접 구현
```

**이식 난이도: 중~높음**
- 가장 유연하지만 구현 시간이 길음
- 이미 우리가 해온 방향 (프롬프트 + 도구 확장)의 연장선

### 권장: **전략 A (compact 엔진 이식)** + 필요 시 전략 B로 점진 전환

---

## 11. OpenHarness의 코드 품질 평가

| 항목 | 평가 |
|------|------|
| **코드 구조** | ✅ 깔끔한 모듈 분리, 각 디렉토리가 명확한 역할 |
| **타입 안전** | ✅ Pydantic v2 전면 사용, 타입 힌트 일관됨 |
| **비동기** | ✅ asyncio 네이티브, async/await 일관됨 |
| **테스트** | ⚠️ 테스트 파일은 있지만 커버리지 불확실 |
| **문서화** | ⚠️ README는 상세하지만 코드 내 주석은 보통 |
| **의존성** | ✅ 최소한 (anthropic, openai, pydantic, typer, rich) |
| **확장성** | ✅ BaseTool 상속, ToolRegistry 패턴, 플러그인 시스템 |
| **유지보수** | ⚠️ 학술 그룹(HKU) 프로젝트. 장기 유지보수 불확실 |

---

## 12. 핵심 파일 참조

| 역할 | 파일 경로 |
|------|-----------|
| 에이전트 루프 | `src/openharness/engine/query.py` |
| 엔진 래퍼 | `src/openharness/engine/query_engine.py` |
| Anthropic 클라이언트 | `src/openharness/api/client.py` |
| OpenAI 클라이언트 | `src/openharness/api/openai_client.py` |
| 프로바이더 레지스트리 | `src/openharness/api/registry.py` |
| 도구 베이스 | `src/openharness/tools/base.py` |
| 파일 읽기 도구 | `src/openharness/tools/file_read_tool.py` |
| Bash 도구 | `src/openharness/tools/bash_tool.py` |
| 에이전트 도구 | `src/openharness/tools/agent_tool.py` |
| Microcompact | `src/openharness/services/compact/__init__.py` |
| 토큰 추정 | `src/openharness/services/token_estimation.py` |
| CLAUDE.md 로딩 | `src/openharness/prompts/claudemd.py` |
| 메모리 시스템 | `src/openharness/memory/` |
| 권한 시스템 | `src/openharness/permissions/` |
| Swarm (멀티에이전트) | `src/openharness/swarm/` |
| 코디네이터 | `src/openharness/coordinator/` |
| IM 채널 | `src/openharness/channels/` |
| 시스템 프롬프트 | `src/openharness/prompts/` |
| CLI 진입점 | `src/openharness/cli.py` |

---

## 13. 결론

### OpenHarness는:
- ✅ **Python으로 된 Claude Code** — 에이전트 아키텍처 학습/참조에 최고
- ✅ **litellm 미사용, 프레임워크 미사용** — 공식 SDK 직접 사용 (OpenCode와 동일 철학)
- ✅ **컨텍스트 관리가 잘 구현됨** — 3단계 방어 (도구 제한 → microcompact → full compaction)
- ✅ **MIT 라이선스** — 상업적 사용, 포크, 이식 모두 가능
- ⚠️ **코딩 에이전트 특화** — 보고서/노트 에디터 용도가 아님
- ⚠️ **DB 없음, 파일 기반** — Thronicle의 PostgreSQL/S3 아키텍처와 직접 호환 안 됨
- ⚠️ **학술 프로젝트** — 장기 유지보수 보장 없음

### Thronicle에 대한 가치:
1. **최대 가치**: 컨텍스트 관리 엔진 (`compact/`) 이식
2. **높은 가치**: 에이전트 루프 패턴 참조 (Agno 의존도 낮출 때)
3. **중간 가치**: 멀티에이전트/Swarm 참조 (복수 종목 동시 분석 시)
4. **선택적 가치**: IM 채널 통합 코드 활용

**우리가 지금까지 OpenCode 분석 → Thronicle 적용으로 해온 작업과 완벽하게 보완적입니다.** OpenCode 분석이 "이론/설계"였다면, OpenHarness는 "Python으로 된 참조 구현"입니다.
