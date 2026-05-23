# CLAUDE.md - 프로젝트 규칙

## 사용자 컨텍스트
- 소프트웨어 엔지니어/개발자 관점에서 기술과 오픈소스를 학습·분석하는 프로젝트
- 테스트 전략, QA 프로세스 등 운영적 세부사항은 관심 대상이 아님

## 오픈소스/기술 분석 규칙

### 소스코드 분석
- 오픈소스 분석 시 해당 프로젝트의 repo를 clone하여 소스코드 단위로 분석한다.
- clone한 프로젝트 폴더는 반드시 `.gitignore`에 추가하여 origin에 업로드되지 않도록 한다.

### 문서화
- 분석 결과는 항상 Markdown(.md) 형식으로 정리하여 문서화한다.
- 각 문서는 적절한 키워드나 기술 단위로 디렉토리를 만들어서 분류한다.
- 아키텍처, 데이터 흐름, 시스템 구조 등 시각적 다이어그램은 반드시 Mermaid 문법으로 작성한다.

### 폴더 구조 및 정리 규칙

**원칙: `category/topic/files` 의 3-depth 를 넘기지 않는다.**

```
research/
├── <category>/        # depth 1 — 큰 분류 (ai-agents · databases · ...)
│   └── <topic>/       # depth 2 — 개별 프로젝트 / 주제
│       └── *.md       # depth 3 — 실제 컨텐츠 (최대 깊이)
```

- **카테고리 (depth 1)** 는 다음 12개 안에서 고른다. 새 카테고리 신설은 *3개 이상의 토픽*이 모일 때만:
  - `ai-agents/` — 에이전트 프레임워크·메모리·구체 구현
  - `ai-coding-tools/` — 코딩 에이전트·IDE (Claude Code, OpenCode, Warp …)
  - `ai-infrastructure/` — RAG·임베딩·MCP·텍스트→SQL 등 AI 인프라 레이어
  - `databases/` — 그래프·벡터·멀티모델 DB
  - `data-platforms/` — ETL·분산 데이터 통합
  - `kubernetes/` — K8s 운영·진단·관측
  - `algorithms/` · `libraries/` · `finance/` · `trends/` · `scripts/` · `_repos/`

- **토픽 (depth 2)** 명명:
  - 프로젝트 이름은 그대로(`agentmemory/`, `lat-md/`, `mirage/`)
  - 횡단 비교는 `<theme>-comparison/` (예: `memory-comparison/`)
  - 단일 카테고리 안에서만 의미 있는 그룹은 `<theme>/` (예: `agent-loops/`, `agentic-rag/`)

- **금지: depth 4 이상**
  - ❌ `ai-agents/frameworks/agno/agno-analysis.md` (4 depth)
  - ✅ `ai-agents/agno/agno-analysis.md` (3 depth)
  - 카테고리 안에 하위 카테고리(`frameworks/`, `memory/`)를 두지 말 것 — 토픽을 바로 둔다

- **단일 파일 토픽**: 토픽에 문서가 1~2개뿐이면 `topic/` 폴더 + 파일 1~2개 형태가 가장 단순. README.md 없이도 OK
- **다문서 토픽**: 3개 이상 문서가 모이면 `topic/README.md`를 1차 진입점으로 만든다

**주기적 정리 체크리스트 (분석 작업 마무리 시 또는 PR 직전 점검)**

1. `find . -maxdepth 5 -type f -name "*.md" -not -path "*/_repos/*"` 로 depth 4 파일이 없는지 확인. 있으면 토픽을 한 단계 끌어올린다.
2. 카테고리 폴더에 *서브카테고리*(`frameworks/`, `memory/` 같은 중간 계층)가 새로 생기지 않았는지 확인 — 발견 즉시 토픽 직접 배치로 평탄화
3. 중복 카테고리(`ai-tools/` vs `ai-coding-tools/`, top-level `opencode/` vs `ai-coding-tools/opencode/`)가 새로 생겼는지 확인 — 발견 시 즉시 병합
4. `README.md` 의 모든 링크가 살아 있는지 확인 (`grep -oE "\(([a-z][^)]+\.md)\)" README.md` 로 추출 후 `test -f`)
5. 1-아이템 카테고리가 3개 이상이면 그중 둘은 인접 카테고리로 흡수 검토
6. 토픽 이름에 공백·대문자 우선 금지 (kebab-case)

**파일 이동 시 절차**
- 반드시 `git mv` 로 이동 (history 보존)
- `README.md` 인덱스 동일 PR에서 갱신
- 옮긴 파일을 *다른 문서가 참조*하면 그 링크도 같은 PR에서 갱신
- 빈 폴더는 `rmdir` 로 정리 (git은 빈 디렉터리 추적 안 함)

**현재 카테고리 매핑 (참조용)**

| 분류 | 위치 |
|---|---|
| 에이전트 메모리 비교 | `ai-agents/memory-comparison/` (8개 시스템 횡단 분석) |
| 에이전트 메모리 개별 구현 | `ai-agents/<project>/` (agentmemory, openchronicle, openviking, supermemory) |
| 코딩 에이전트 | `ai-coding-tools/<tool>/` (claude-code, opencode, warp) |
| 텍스트→SQL | `ai-infrastructure/` (db-gpt, wren-ai) |
| Graph RAG / 온톨로지 | `ai-infrastructure/graph-rag-ontology/` |
| 그래프 DB 비교 | `databases/graphdb/` |
| 관측·모니터링 | `kubernetes/` (Prometheus·VM 등 포함) |

### Mermaid 다이어그램 작성·검증

GitHub의 Mermaid 렌더러는 파싱에 엄격해서 라벨 안의 특수문자가 자주 깨진다. 다음 규칙을 따르고, **푸시 전 반드시 검증**한다.

**작성 규칙 (사고 예방)**

- **노드 라벨은 따옴표로 감싼다** — 한글·공백·특수문자(`()`, `:`, `/`, `?`, `+`) 포함 시 필수.
  - ❌ `MF[MetricFlow (dbt Labs)]` → 괄호가 다른 도형으로 오인되어 파싱 실패
  - ✅ `MF["MetricFlow (dbt Labs)"]` 또는 ✅ `MF["MetricFlow — dbt Labs"]`
- **다이아몬드 `{...}` 안에도 따옴표** — `?`, `+`, `:` 같은 특수문자 포함 시.
  - ❌ `Q1{2개 이상 +1?}` → ✅ `Q1{"2개 이상 +1?"}`
- **dotted edge 라벨도 따옴표** — `A -. "PR · Discussion" .-> B`
- **라벨 안의 `/`, `:` 는 가급적 텍스트로 치환** — `BI/AI` → `BI · AI`, `Phase 1: Eval` → `Phase 1 — Eval`
- **`classDiagram`에 트레일링 `// 주석` 금지** — Mermaid는 인라인 주석을 지원하지 않아 멤버명 일부로 흡수된다.
  - ❌ `+string source  // db.schema.table` → ✅ `+string source`

**검증 프로세스 (푸시 전 필수)**

신규/수정한 mermaid 블록이 있는 문서는 항상 다음을 실행한다. 종료코드 0이면 통과.

```bash
scripts/validate-mermaid.sh path/to/doc.md
# 여러 파일 동시: scripts/validate-mermaid.sh doc1.md doc2.md
```

스크립트는 markdown에서 ```` ```mermaid ```` 블록을 모두 추출해 `mermaid-cli`(npx로 자동 설치)로 SVG 렌더링을 시도하고 블록별 ✅/❌를 출력한다. ❌가 나오면 위 작성 규칙에 따라 수정 후 재검증한다.

### 분석 범위 및 품질
- 웹 검색을 병행하여 정확도를 높이고 최신 정보를 포함한다.
- 분석 시 다음 관점을 반드시 포함한다:
  - 어떤 문제를 해결하고자 하는지 (Problem Statement)
  - 어떤 특징이 있는지 (Key Features)
  - 장단점은 무엇인지 (Pros & Cons)
  - 경쟁/비교 대상은 무엇인지 (Competitors & Comparison)
  - 기반이 되는 기술이 무엇인지 (Underlying Technologies)
  - 생태계 전반적인 지식 (Ecosystem Context)

### 보고서 필수 섹션 (엔지니어 관점)
분석 보고서는 아래 섹션을 중심으로 작성한다. 프로젝트 성격에 따라 취사선택·가감 가능.

1. **프로젝트 개요** - 핵심 정의, 해결하려는 문제, 탄생 배경
2. **핵심 특징 및 차별점** - 주요 기능, 기존 대안 대비 차별화 포인트
3. **아키텍처 분석** - 전체 시스템 구조, 핵심 개념 모델, 데이터 흐름 (다이어그램 포함)
4. **기술 스택** - 언어, 프레임워크, 의존성, 빌드/패키지 시스템
5. **핵심 코드 분석** - 주요 모듈/패키지 구조, 핵심 알고리즘·패턴, 코드 레벨 설계 결정
6. **API 및 인터페이스** - 공개 API 설계, SDK, CLI, 프로토콜
7. **확장성 및 플러그인** - 확장 포인트, 플러그인/커넥터 아키텍처, 커스터마이징 방법
8. **성능 특성** - 벤치마크, 스케일링 전략, 알려진 제약사항
9. **배포 및 운영** - 설치/배포 방식, 인프라 요구사항, 설정 방법
10. **경쟁·비교 분석** - 유사 프로젝트와의 기능/아키텍처/성능 비교표
11. **종합 평가** - 강점, 약점/리스크, 적합·부적합 사례, 엔지니어 관점 인사이트

### 보고서에 포함하지 않는 항목
- 테스트 전략, QA 프로세스, CI/CD 파이프라인 세부사항
- 커뮤니티 통계 (스타 수, 포크 수 등) - 간략히만 언급
- 가격 정책 상세 (있다면 한 줄 요약 정도)
