# OpenCode 전체 명령어 & 도구 레퍼런스

---

## 1. LLM 도구 (에이전트가 사용)

에이전트가 작업 중 자동으로 호출하는 도구들입니다.

### 파일 조작

| 도구 | 설명 | 핵심 파라미터 |
|------|------|-------------|
| **read** | 파일/디렉토리 읽기 (2000줄/50KB 제한, offset 페이지네이션) | `filePath`, `offset`, `limit` |
| **write** | 파일 생성 또는 덮어쓰기 | `filePath`, `content` |
| **edit** | 문자열 치환 (old→new, 유일해야 함) | `filePath`, `oldString`, `newString`, `replaceAll` |
| **multiedit** | 한 파일에 여러 편집을 순차 적용 | `filePath`, `edits[]` |
| **apply_patch** | unified diff 패치 적용 (GPT 전용) | `patchText` |

### 검색

| 도구 | 설명 | 핵심 파라미터 |
|------|------|-------------|
| **glob** | glob 패턴 파일 검색 (mtime 정렬, 100건) | `pattern`, `path` |
| **grep** | ripgrep 정규식 검색 (100건) | `pattern`, `path`, `include` |

### 실행 & 웹

| 도구 | 설명 | 핵심 파라미터 |
|------|------|-------------|
| **bash** | 셸 명령 실행 (영속 세션, 2분 타임아웃) | `command`, `timeout`, `workdir`, `description` |
| **webfetch** | URL 콘텐츠 가져오기 (markdown/text/html) | `url`, `format`, `timeout` |
| **websearch** | Exa AI 웹 검색 | `query`, `numResults`, `type` |
| **codesearch** | Exa 코드 문서/예제 검색 | `query`, `tokensNum` |

### 에이전트 & 작업 관리

| 도구 | 설명 | 핵심 파라미터 |
|------|------|-------------|
| **task** | 서브에이전트 생성 (독립 컨텍스트) | `description`, `prompt`, `subagent_type` |
| **todowrite** | 작업 목록 생성/관리 | `todos[]` |
| **skill** | 특화 스킬 로드 | `name` |
| **question** | 사용자에게 질문 (조건부) | `questions[]` |

### 코드 인텔리전스 (조건부)

| 도구 | 설명 | 조건 |
|------|------|------|
| **lsp** | 정의 이동, 참조 찾기, hover, 심볼 검색 등 | `OPENCODE_EXPERIMENTAL_LSP_TOOL` |
| **batch** | 최대 25개 도구 병렬 실행 | `experimental.batch_tool` |

---

## 2. 슬래시 명령어 (사용자가 직접 입력)

프롬프트에서 `/`로 시작하는 명령어들입니다.

### 세션 관리

| 명령어 | 별칭 | 설명 |
|--------|------|------|
| `/new` | `/clear` | 새 세션 생성 |
| `/sessions` | `/resume`, `/continue` | 세션 전환 |
| `/rename` | | 세션 이름 변경 |
| `/fork` | | 특정 메시지에서 세션 분기 |
| `/timeline` | | 세션 타임라인 탐색 |
| `/compact` | `/summarize` | 대화 요약 (수동 compaction) |
| `/undo` | | 이전 메시지 취소 |
| `/redo` | | 메시지 복원 |
| `/copy` | | 세션 대화록 복사 |
| `/export` | | 세션 내보내기 |
| `/share` | | 세션 공유 |
| `/unshare` | | 세션 공유 해제 |

### 모델 & 에이전트

| 명령어 | 설명 |
|--------|------|
| `/models` | 모델 전환 |
| `/agents` | 에이전트 전환 |
| `/variants` | 모델 변형 전환 |
| `/mcps` | MCP 서버 토글 |

### 시스템

| 명령어 | 설명 |
|--------|------|
| `/connect` | 프로바이더 연결 |
| `/status` | 시스템 상태 확인 |
| `/themes` | 테마 변경 |
| `/skills` | 사용 가능한 스킬 목록 |
| `/editor` | 외부 에디터로 프롬프트 편집 |
| `/help` | 도움말 |
| `/exit` | `/quit`, `/q` — 종료 |

### 표시 토글

| 명령어 | 설명 |
|--------|------|
| `/thinking` | 추론 블록 표시/숨김 |
| `/timestamps` | 타임스탬프 표시/숨김 |

---

## 3. CLI 명령어 (터미널에서 실행)

`opencode <command>` 형식으로 실행합니다.

### 핵심

| 명령어 | 설명 |
|--------|------|
| `opencode run` | TUI 실행 (기본) |
| `opencode serve` | HTTP 서버 모드 |
| `opencode web` | 웹 서버 모드 |
| `opencode attach` | 기존 세션에 연결 |
| `opencode thread` | 스레드 모드 |

### 세션

| 명령어 | 설명 |
|--------|------|
| `session list` | 세션 목록 |
| `session delete <id>` | 세션 삭제 |
| `export [sessionID]` | 세션 내보내기 |
| `import <file>` | 세션 가져오기 |

### 프로바이더 & 인증

| 명령어 | 설명 |
|--------|------|
| `providers list` | 프로바이더 목록 |
| `providers login <provider>` | 프로바이더 로그인 |
| `providers logout <provider>` | 프로바이더 로그아웃 |
| `login <url>` | 계정 로그인 |
| `logout` | 계정 로그아웃 |
| `switch` | 계정 전환 |

### 에이전트 & 모델

| 명령어 | 설명 |
|--------|------|
| `agent create` | 에이전트 생성 |
| `agent list` | 에이전트 목록 |
| `models` | 모델 목록 |

### MCP

| 명령어 | 설명 |
|--------|------|
| `mcp list` | MCP 서버 목록 |
| `mcp add` | MCP 추가 |
| `mcp auth` | MCP 인증 |
| `mcp debug` | MCP 디버그 |

### GitHub

| 명령어 | 설명 |
|--------|------|
| `github install` | GitHub 통합 설치 |
| `github run` | GitHub 워크플로우 실행 |
| `pr` | PR 관리 |

### 시스템

| 명령어 | 설명 |
|--------|------|
| `db path` | DB 파일 경로 |
| `db migrate` | DB 마이그레이션 |
| `stats` | 사용 통계 |
| `upgrade` | 업그레이드 |
| `uninstall` | 제거 |
| `debug config\|file\|skill\|agent\|lsp\|ripgrep` | 디버깅 |

---

## 4. 키보드 단축키

`ctrl+x`가 리더 키이고, 이후 키를 조합합니다.

### 핵심 단축키

| 단축키 | 기능 |
|--------|------|
| `ctrl+c` / `ctrl+d` | 종료 |
| `ctrl+p` | 명령어 팔레트 (모든 명령어 검색) |
| `escape` | 현재 작업 중단 |

### 세션 (리더 키 + ...)

| 단축키 | 기능 |
|--------|------|
| `<leader>n` | 새 세션 |
| `<leader>l` | 세션 목록 |
| `<leader>b` | 사이드바 토글 |
| `<leader>c` | 세션 요약 (compact) |
| `<leader>g` | 타임라인 |
| `<leader>x` | 세션 내보내기 |
| `<leader>u` | Undo |
| `<leader>r` | Redo |
| `<leader>y` | 메시지 복사 |
| `ctrl+r` | 세션 이름 변경 |

### 모델 & 에이전트

| 단축키 | 기능 |
|--------|------|
| `<leader>m` | 모델 목록 |
| `<leader>a` | 에이전트 목록 |
| `f2` / `shift+f2` | 최근 모델 순환 |
| `tab` / `shift+tab` | 에이전트 순환 |
| `ctrl+t` | 모델 변형 순환 |

### 기타

| 단축키 | 기능 |
|--------|------|
| `<leader>e` | 외부 에디터 열기 |
| `<leader>t` | 테마 목록 |
| `<leader>s` | 상태 보기 |
| `<leader>h` | 코드 블록 접기/펼치기 |
| `ctrl+z` | 터미널 일시 중지 |

### 입력 필드

| 단축키 | 기능 |
|--------|------|
| `return` | 전송 |
| `shift+return` | 줄바꿈 |
| `ctrl+v` | 붙여넣기 |
| `ctrl+k` | 줄 끝까지 삭제 |
| `ctrl+u` | 줄 시작까지 삭제 |
| `ctrl+a` / `ctrl+e` | 줄 시작/끝 이동 |
| `alt+f` / `alt+b` | 단어 앞/뒤 이동 |
| `up` / `down` | 히스토리 탐색 |

---

## 5. 스킬 (Skills)

스킬은 `/skills` 또는 skill 도구로 로드되는 특화 지침입니다.

| 스킬 | 설명 |
|------|------|
| `init` | AGENTS.md 생성/갱신 |
| `review` | 변경사항 리뷰 (commit, branch, PR) |
| 사용자 정의 | `~/.claude/skills/` 또는 `.opencode/skills/`에 마크다운으로 추가 가능 |

---

## 6. 전체 요약

```
OpenCode가 할 수 있는 것
├── LLM 도구 (18+개) — 에이전트가 자동으로 호출
│   ├── 파일: read, write, edit, multiedit, apply_patch
│   ├── 검색: glob, grep
│   ├── 실행: bash
│   ├── 웹: webfetch, websearch, codesearch
│   ├── 에이전트: task, todowrite, skill, question
│   └── 코드 인텔리전스: lsp, batch
│
├── 슬래시 명령어 (25+개) — 사용자가 /로 직접 입력
│   ├── 세션: new, sessions, fork, compact, undo, redo, share...
│   ├── 모델: models, agents, variants, mcps
│   └── 시스템: status, themes, help, exit...
│
├── CLI 명령어 (30+개) — 터미널에서 opencode <cmd>
│   ├── 실행: run, serve, web, attach
│   ├── 관리: session, providers, agent, models, mcp
│   ├── GitHub: github, pr
│   └── 시스템: db, stats, upgrade, debug...
│
├── 키보드 단축키 (70+개) — TUI에서 사용
│   ├── 리더 키 조합: ctrl+x → n/l/m/a/c/...
│   ├── 탐색: pageup/down, ctrl+g, home/end
│   └── 입력: ctrl+a/e/k/u, alt+f/b...
│
└── 스킬 — 확장 가능한 특화 지침
    ├── 내장: init, review
    └── 사용자 정의: ~/.claude/skills/*.md
```
