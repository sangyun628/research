# Claude Code 프롬프트 및 아키텍처 분석

이 디렉토리는 Claude Code 2026-03-31 유출 소스코드에 대한 포괄적인 분석 문서를 포함합니다.

## 분석 문서

### 1. **prompts-catalog.md** (메인 문서) ⭐
- **크기**: 30KB, 1006줄
- **내용**: 
  - 발견된 모든 프롬프트 (8개) 전문
  - 확인된 하지만 발견되지 않은 프롬프트 (8개+)
  - 프롬프트 아키텍처 다이어그램 (Mermaid)
  - 패턴 분석 및 프롬프트 엔지니어링 기법
  - 토큰 사용량 추정
  - 자체 에이전트 구축 시 차용할 패턴

### 2. **ANALYSIS_SUMMARY.txt** (요약본)
- **크기**: 12KB
- **용도**: 빠른 개요 및 검색
- **포함 내용**:
  - 핵심 발견사항 (Key Findings)
  - 8가지 발견된 프롬프트 카테고리
  - 8가지 미발견 프롬프트 (위치 + 추정 내용)
  - 패턴 5가지
  - IMPORTANT 지시문 타입 4가지
  - 프롬프트 엔지니어링 기법 7가지
  - 에이전트 빌더를 위한 핵심 인사이트

### 3. **memory-system-analysis.md**
- **크기**: 31KB
- **주제**: Claude Code의 메모리 시스템 상세 분석
- **내용**: (기존 분석 문서)

## 발견 현황

### 실제 발견된 프롬프트 (8개)

| # | 이름 | 파일 | 토큰 | 카테고리 |
|---|------|------|------|---------|
| 1 | SHUTDOWN_TEAM_PROMPT | cli/print.ts:379 | ~150 | System Message |
| 2 | CRITIQUE_SYSTEM_PROMPT | cli/handlers/autoMode.ts:49 | ~300 | Meta-Prompting |
| 3 | companionIntroText | buddy/prompt.ts:7 | ~80 | UI Message |
| 4 | BRIDGE_LOGIN_INSTRUCTION | bridge/types.ts:5 | ~30 | Error Message |
| 5 | BRIDGE_LOGIN_ERROR | bridge/types.ts:9 | ~40 | Error Message |
| 6-8 | Remote Control Messages | bridge/bridgeMain.ts | ~200 | User Messages |

### 미발견 프롬프트 (알려진 존재하지만 소스 제공 안됨)

| # | 이름 | 추정 토큰 | 중요도 | 비고 |
|---|------|---------|-------|------|
| 1 | Main System Prompt | 2-5K | 🔴 Critical | 시스템 기초 행동 |
| 2 | Auto-mode Classifier | 1-2K | 🔴 Critical | 권한 결정 로직 |
| 3 | Tool Descriptions (18+) | 50-500 each | 🔴 Critical | 도구 사용 지시 |
| 4 | Sub-agent Prompts | 300-500 | 🟠 High | 에이전트별 지시 |
| 5 | Compaction Prompt | 200-500 | 🟠 High | 대화 요약 |
| 6 | Memory Extraction | 300-500 | 🟠 High | 메모리 분류 |
| 7 | Session Title Gen. | 50-100 | 🟡 Medium | 제목 자동 생성 |
| 8 | Skill Prompts | 100-300 | 🟡 Medium | Slash 명령어 |

## 핵심 발견사항

### 프롬프트 아키텍처 5가지 패턴

1. **계층적 구조** (Hierarchical)
   - Level 1: Main System Prompt
   - Level 2: Sub-agent Prompts  
   - Level 3: Tool Descriptions
   - Level 4: System Reminders

2. **동적 리마인더 주입** (Dynamic Injection)
   - `<system-reminder>` 태그로 대화 중간 정보 업데이트
   - 목적: 긴 대화에서 LLM이 초반 지시사항 "잊어버리는" 것 방지

3. **메타 프롬프팅** (Meta-Prompting)
   - 사용자가 작성한 규칙을 LLM이 평가하는 프롬프트
   - 예: auto-mode critique

4. **역할 및 권계 설정** (Role & Boundaries)
   - "You are X" (역할)
   - "You're not Y" (경계)
   - "Don't Z" (금지)

5. **구조화된 기준** (Structured Criteria)
   - 평가 항목을 번호로 명시
   - 일관된 실행 보장

### IMPORTANT 지시문 패턴

```
Type 1: MUST / MUST NOT (절대 요구사항)
  "You MUST shut down your team before preparing your final response"

Type 2: 금지 사항 (Forbidden Actions)
  "Don't explain that you're not ${name}"

Type 3: 역할 배제 (Role Exclusions)
  "You're not ${name} — it's a separate watcher"

Type 4: 경계 제약 (Boundary Constraints)
  "respond in ONE line or less"
```

### 토큰 사용량 (추정)

**매 턴 기본 오버헤드**: ~5-10K 토큰

| 항목 | 토큰 | 빈도 |
|------|------|------|
| 메인 시스템 프롬프트 | 2-5K | 1회 (세션당) |
| 시스템 리마인더 | 500-1K | 매 턴 |
| 도구 설명 | 50-500 | 필요시 |
| 자동모드 분류기 | 1-2K | 권한 확인시 |
| 메모리 | 500-1K | 매 턴 |

**1M 토큰 윈도우에서**: 최대 100-200 턴 가능 (도구 결과 제외)

## 자체 에이전트 구축 시 차용할 패턴

### 패턴 1: System Reminder 사용
```
<system-reminder>
Current date: 2026-04-08
Available tools: [...]
Permission mode: auto-accept
</system-reminder>
```
✅ 긴 대화에서 초반 지시사항 유지

### 패턴 2: 명확한 역할 정의
```
You are an expert in X.
Your job is to Y.
You're not Z.
Don't do W.
```
✅ 역할 충돌 방지, 책임 명확화

### 패턴 3: 구조화된 평가 기준
```
For each item, evaluate:
1. Clarity
2. Completeness  
3. Conflicts
4. Actionability
```
✅ 일관된 의사결정

### 패턴 4: 권한 분리 아키텍처
```
Main Agent (의사결정)
├── Auto-mode Classifier (권한 검증)
└── Sub-agents (특화 작업)
```
✅ 책임 분담, 모듈화

### 패턴 5: 메타 프롬프팅
```
LLM이 사용자가 작성한 프롬프트(규칙)를 평가
└── 사람과 AI의 협력 가능
```
✅ AI 보조 규칙 검증

## 유출 소스의 한계

이 분석은 **소스맵에서 복구된 코드**를 대상으로 하므로:

- ❌ 컴파일된 바이너리 내용 불가 (메인 시스템 프롬프트 등)
- ❌ 외부 모듈 임포트된 함수 구현 불가 (자동모드 분류기 등)
- ❌ 런타임 동적 생성 프롬프트 불가
- ✅ 정적 문자열 상수 발견 가능
- ✅ 함수 시그니처 및 호출 패턴 분석 가능

**발견율**: ~25-30% (추정 전체 프롬프트 대비)

## 파일 구조

```
/research/ai-tools/claude-code/
├── README.md                      # 이 파일
├── prompts-catalog.md            # ⭐ 메인 분석 문서 (1006줄)
├── ANALYSIS_SUMMARY.txt          # 요약본 (빠른 검색용)
└── memory-system-analysis.md     # 메모리 시스템 상세 분석
```

## 사용 방법

1. **빠른 개요**: `ANALYSIS_SUMMARY.txt` 읽기 (5분)
2. **상세 분석**: `prompts-catalog.md` 읽기 (30분)
3. **메모리 시스템**: `memory-system-analysis.md` 참고 (선택)

## 원본 소스

- **유출 소스**: `/Users/sangyun-han/OpenSource/leaked-claude-code/`
- **분석 기반**: Claude Code 2026-03-31 유출 (53 TS 파일, ~512K 줄)
- **분석일**: 2026-04-08

## 주요 통계

| 항목 | 값 |
|------|-----|
| 발견된 프롬프트 | 8개 |
| 미발견 프롬프트 | 8+ 개 |
| 추정 전체 프롬프트 | 25-35개 |
| 발견율 | ~25-30% |
| 분석 문서 규모 | 1006줄 |
| 다루는 패턴 | 7가지 |
| 지시문 타입 | 4가지 |
| 프롬프트 엔지니어링 기법 | 7가지 |

---

**마지막 업데이트**: 2026-04-08
**한국어 문서**: 완전 번역/분석
**코드 인용**: 모두 line reference 포함
