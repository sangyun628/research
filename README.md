# Research

오픈소스 기술을 **AI 에이전트 인프라 구축 관점**에서 심층 분석한 문서 모음.

각 디렉토리는 하나의 주제 영역을 다루고, 해당 영역에 속하는 오픈소스들을 동일한 깊이/포맷으로 분석한다. 카테고리 개요와 도입 의사결정 가이드는 각 디렉토리의 `README.md`에 정리되어 있다.

---

## 디렉토리 맵

| 디렉토리 | 주제 | 분석 대상 수 |
|---|---|---:|
| [agent-sandbox/](./agent-sandbox/) | **에이전트 코드 실행 샌드박스** (VM/컨테이너/WASM) | 1 |
| [agent-memory/](./agent-memory/) | **에이전트 장기 기억** 시스템 | 7 |
| [agent-harness-analysis/](./agent-harness-analysis/) | **에이전트 하네스/SDK** (LLM 오케스트레이션) | 2 |
| [graphrag-analysis/](./graphrag-analysis/) | **Graph RAG** (지식 그래프 기반 검색) | 1 |

---

## agent-sandbox — 에이전트 코드 실행 샌드박스

LLM 기반 에이전트가 임의의 코드/명령을 실행할 때 쓸 수 있는 **격리 런타임** 후보들.
전체 landscape(마이크로VM, LLM 전용 샌드박스, 사용자공간 커널, 컨테이너, WASM, 개발 환경, 오케스트레이션 7개 카테고리)는 디렉토리 README에 정리.

| 문서 | 대상 | 핵심 |
|---|---|---|
| [README.md](./agent-sandbox/README.md) | 전체 landscape | 7개 카테고리 × 40+ 오픈소스 지도, 시나리오별 추천, 향후 분석 우선순위 |
| [smolvm_analysis.md](./agent-sandbox/smolvm_analysis.md) | **SmolVM** | libkrun 기반 OCI-native 마이크로VM. macOS 1급, <200ms 콜드 스타트, `.smolmachine` 이식 포맷, 도메인 allowlist, SSH agent 포워딩 |

---

## agent-memory — 에이전트 장기 기억

LLM 컨텍스트 윈도우 한계를 넘어, 사용자/세션 정보를 **지속 저장하고 회수**하는 오픈소스 비교.
벡터/지식 그래프/하이브리드 접근법별로 설계 차이를 분석.

| 문서 | 대상 | 핵심 |
|---|---|---|
| [에이전트_메모리_시스템_비교분석.md](./agent-memory/에이전트_메모리_시스템_비교분석.md) | 전체 비교 | 6개 시스템 나란히 비교 + 선택 가이드 |
| [memory_theory_origins.md](./agent-memory/memory_theory_origins.md) | 이론 배경 | 인지과학·HCI 기반 메모리 모델 기원 |
| [mem0_analysis.md](./agent-memory/mem0_analysis.md) | **mem0** | LLM 추출 + 벡터 검색 기반 범용 메모리 |
| [memU_analysis.md](./agent-memory/memU_analysis.md) | **memU** | 다층 메모리 + 에이전트 컨텍스트 통합 |
| [Memori_analysis.md](./agent-memory/Memori_analysis.md) | **Memori** | 관계형 DB 기반 메모리 엔진 |
| [Cognee_analysis.md](./agent-memory/Cognee_analysis.md) | **Cognee** | 지식 그래프 + 온톨로지 기반 메모리 |
| [OpenMemory_analysis.md](./agent-memory/OpenMemory_analysis.md) | **OpenMemory** | MCP 표준 기반 공유 메모리 |
| [SecondMe_analysis.md](./agent-memory/SecondMe_analysis.md) | **SecondMe** | 개인 AI 트윈 지향 메모리 |

---

## agent-harness-analysis — 에이전트 하네스/SDK

LLM tool-use 루프, 서브에이전트 분기, tool 정의 방식 등 **에이전트 하네스**의 설계 패턴.

| 문서 | 대상 | 핵심 |
|---|---|---|
| [openai-agents-python_analysis.md](./agent-harness-analysis/openai-agents-python_analysis.md) | **OpenAI Agents SDK** | OpenAI 공식 Python 에이전트 SDK — agent, handoffs, guardrails, tracing |
| [gemini-cli-subagents_analysis.md](./agent-harness-analysis/gemini-cli-subagents_analysis.md) | **Gemini CLI 서브에이전트** | Google Gemini CLI의 서브에이전트 아키텍처 |

---

## graphrag-analysis — Graph RAG

문서를 벡터가 아닌 **지식 그래프**로 색인해서 검색하는 접근.

| 문서 | 대상 | 핵심 |
|---|---|---|
| [GraphRAG_analysis.md](./graphrag-analysis/GraphRAG_analysis.md) | **Microsoft GraphRAG** | 커뮤니티 감지 + 계층적 요약 기반 글로벌/로컬 질의 응답 |

---

## 공통 분석 포맷

각 기술 분석 문서는 가능한 한 다음 구조를 따른다:

1. **한 눈에 보는 결론** — 요약 표와 도입 적합도
2. **프로젝트 개요 / 포지셔닝** — 무엇을, 왜, 누구를 위해
3. **아키텍처 전체 조감** — 레이어 다이어그램과 책임 분리
4. **핵심 메커니즘 해부** — 격리/검색/기억/라우팅 등 도메인 핵심
5. **통신/프로토콜 / API** — 외부 시스템과의 경계
6. **보안/거버넌스/관측성** — 운영 관점 체크리스트
7. **대안 기술과의 비교** — 유사 범주 오픈소스와의 trade-off
8. **도입 설계 제안** — 우리 하네스/인프라에 올릴 때의 레퍼런스 토폴로지
9. **제약/한계 / 후속 조사** — 아직 확인 못 한 것들
10. **결론** — 도입 의사결정 요약

새 분석을 추가할 때 위 포맷을 참고.

---

## 기여 가이드

1. 새 주제는 소문자+하이픈 디렉토리로 만들고 (`agent-*`, `*-analysis` 네이밍 권장), 그 안에 `README.md`로 카테고리 landscape를 둔다
2. 개별 기술 분석은 `{project}_analysis.md` (소문자 + 언더스코어) 파일명 사용
3. 문서 상단에 **분석일, 대상 버전/커밋, 관점**을 명시
4. 이 루트 README의 **"디렉토리 맵"** 과 해당 섹션 표에 새 항목을 추가
5. 관점(에이전트 샌드박스/에이전트 메모리 등)에 맞춰 **"왜 이 기술을 지금 보는가"** 가 드러나도록 작성

---

## 빠른 찾기: 용도별 문서 인덱스

**에이전트가 임의 코드를 안전하게 실행해야 한다** → [agent-sandbox/](./agent-sandbox/)

**에이전트가 대화/사용자 정보를 기억해야 한다** → [agent-memory/](./agent-memory/)

**에이전트를 LLM tool-use 루프로 오케스트레이션하고 싶다** → [agent-harness-analysis/](./agent-harness-analysis/)

**대규모 문서 집합을 에이전트가 질의해야 한다** → [graphrag-analysis/](./graphrag-analysis/)
