# 조사 근거·소스 스냅샷·읽을 자료

[스터디 인덱스](README.md)

## 1. 조사 방법과 범위

- 조사일: **2026-10-08**, Asia/Seoul.
- 공식 프로젝트 문서·저자 논문·공식 기술 글을 웹에서 확인했다. 검색에 나타난 비공식 요약 글을 기술적 결론의 근거로 사용하지 않았다.
- 아래 8개 저장소를 `.repos/`에 shallow clone하여 핵심 데이터 모델·scorer·환경 evaluator를 읽었다.
- 각 문서의 코드 링크는 **branch 이름이 아닌 commit SHA**에 고정했다. 링크의 파일과 행 번호는 로컬 clone에서 확인했다.
- 구현 분석은 주로 Python이며 TypeScript API나 전체 기능을 동일한 깊이로 분석한 것은 아니다.
- AgentEvals·OpenEvals·Ragas의 12개 결정론적 동작을 실제 Python API로 실행했다. 그 외의 코드 설명은 정적 분석이다.
- LLM judge 품질, 도메인 benchmark 전체, 실제 모델 성능·속도·비용 비교는 실행하지 않았다.
- BFCL·τ-bench 외의 벤치마크는 논문·공식 설명에 기반한 연구 지형 비교다. 해당 구현까지 읽은 것으로 표현하지 않았다.

**코드 사실**, **논문·문서의 설명**, **설계 제안**을 구분했다. 가상의 예약 사례와 공통 trace schema는 이 스터디에서 만든 설명용 예제다.

## 2. Clone한 저장소와 고정 커밋

| 프로젝트 | 원본 | 분석 커밋 SHA | commit 날짜 |
|---|---|---|---|
| AgentEvals | [저장소](https://github.com/langchain-ai/agentevals) | `946ad15b9f2ce9bc9843edd2bb773bd07d78ca7e` | 2026-07-13 |
| OpenEvals | [저장소](https://github.com/langchain-ai/openevals) | `bf12ad47d8520fad85f820f4245bcb0b8c9c4954` | 2026-10-02 |
| DeepEval | [저장소](https://github.com/confident-ai/deepeval) | `0fb05d028b5ded9e4c466140ab3f3af90b801944` | 2026-10-07 |
| Ragas | [저장소](https://github.com/vibrantlabsai/ragas) | `298b68274234c060deacab3cf5fb52aa3a20e885` | 2026-02-24 |
| Inspect AI | [저장소](https://github.com/UKGovernmentBEIS/inspect_ai) | `4acfd207ecca8e1c3b754a147c191f2f0f21ba71` | 2026-10-08 |
| Google ADK | [저장소](https://github.com/google/adk-python) | `7d8c9502f924ed5e7a1ab00dc1f5a603e858abc3` | 2026-10-08 |
| τ-bench | [저장소](https://github.com/sierra-research/tau2-bench) | `4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699` | 2026-10-06 |
| Gorilla · BFCL | [저장소](https://github.com/ShishirPatil/gorilla) | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | 2026-03-23 |

커밋 날짜는 저장소에서 읽은 commit timestamp의 날짜 부분이다. 조사일·논문 발표일·release 발행일과 다르다. 특히 Ragas·Gorilla의 확보된 HEAD 날짜가 조사일보다 앞선다고 해서 이 표만으로 프로젝트의 활동 상태나 최신 release를 단정하지 않는다.

로컬 경로는 각각 `.repos/agentevals`, `.repos/openevals`, `.repos/deepeval`, `.repos/ragas`, `.repos/inspect-ai`, `.repos/adk-python`, `.repos/tau2-bench`, `.repos/gorilla`다. 모두 기존 `.gitignore`의 `/.repos/` 규칙에 포함되며 `git check-ignore`로 확인했다. clone한 소스·가상환경을 연구 저장소의 추적 파일에 추가하지 않았다.

## 3. 재현한 실험

실험 결과와 전체 코드는 [03 — 로컬 재현](03-tool-and-trajectory.md)에 있다. 실행 환경은 Python 3.12.13이며, clone한 AgentEvals·OpenEvals·Ragas를 editable 설치했다.

프로젝트 루트에서 해당 커밋의 clone이 준비되었다면 다음처럼 실행 환경을 만들 수 있다.

```bash
uv venv .context/agent-evals/venv --python 3.12
uv pip install --python .context/agent-evals/venv/bin/python \
  -e .repos/agentevals/python \
  -e .repos/openevals/python \
  -e .repos/ragas \
  'langchain-community==0.3.31'
```

03 문서의 Python 코드를 `.context/agent-evals/metric_probe.py`에 저장한 다음:

```bash
.context/agent-evals/venv/bin/python .context/agent-evals/metric_probe.py
```

실제 실행 시 최신 `langchain-community==0.4.2` 조합에서는 Ragas가 참조하는 `chat_models.vertexai` import가 없어 실패했다. `0.3.31`로 고정한 뒤 12개 사례가 실행됐다. 다른 환경의 보편적 호환성 보증이나 완전한 transitive dependency lock은 아니다. 이 재현은 모델 추론·LangSmith trace 전송·Ragas usage tracking을 사용하지 않는다.

분석한 세 소스를 새로 받을 때는 먼저 아래 저장소를 clone한 뒤 **표의 SHA로 checkout**해야 한다. 예를 들어 AgentEvals는 다음과 같다.

```bash
git clone --depth 1 https://github.com/langchain-ai/agentevals.git .repos/agentevals
git -C .repos/agentevals fetch --depth 1 origin 946ad15b9f2ce9bc9843edd2bb773bd07d78ca7e
git -C .repos/agentevals checkout --detach 946ad15b9f2ce9bc9843edd2bb773bd07d78ca7e

```

OpenEvals와 Ragas도 각각 `langchain-ai/openevals`, `vibrantlabsai/ragas`를 같은 방식으로 받고 위 표의 해당 SHA를 사용한다. shallow clone에서 생성된 Ragas의 package version 문자열은 release tag와 다를 수 있으므로 이 분석의 identity는 version 문자열이 아니라 SHA다.

## 4. 공식 문서·기술 자료

| 자료 | 읽을 이유 |
|---|---|
| [AgentEvals 저장소](https://github.com/langchain-ai/agentevals) | trajectory matcher와 judge API |
| [OpenEvals 저장소](https://github.com/langchain-ai/openevals) | evaluator factory·structured output·prompt 구성 |
| [DeepEval Task Completion](https://deepeval.com/docs/metrics-task-completion) | trace 기반 완료 판단과 eval mode |
| [Ragas Agent Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/) | tool·goal metric 및 collections API 안내 |
| [Inspect Sandboxing](https://inspect.aisi.org.uk/sandboxing.html) | 평가 프로세스와 격리 실행의 경계 |
| [Google ADK 평가 문서](https://github.com/google/adk-docs/blob/main/docs/evaluate/index.md) | invocation·criteria·user simulation |
| [τ-bench Releases](https://github.com/sierra-research/tau2-bench/releases) | τ³ 및 채점 변경의 비교 가능성 |
| [BFCL V4 Web Search](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html) | 함수 호출에서 검색·multi-hop으로 확장 |
| [BFCL V4 Memory](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html) | agent memory 활용 평가 |
| [BFCL V4 Format Sensitivity](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html) | 출력 형식 변화의 영향 |
| [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | transcript·outcome·grader의 개념 구분 |
| [Quantifying infrastructure noise](https://www.anthropic.com/engineering/infrastructure-noise) | 실행 조건을 성능과 구분하는 이유 |
| [Terminal-Bench 2.1](https://www.tbench.ai/news/terminal-bench-2-1) | benchmark task 수정이 점수에 미치는 영향 |

일반 웹 문서는 조회 이후 바뀔 수 있다. 구현 관련 세부 주장은 위 링크만이 아니라 각 장의 commit permalink를 우선 근거로 삼았다.

## 5. 논문 읽기 지도

연도는 최초 공개 연도를 중심으로 표기했다. 수정본·학회 발표 연도는 링크의 version history와 다를 수 있다.

| 우선순위 | 자료 | 관점 |
|---|---|---|
| 입문 | [Survey on Evaluation of LLM-based Agents — 2025](https://arxiv.org/abs/2503.16416) | 능력·도메인·framework의 전체 분류 |
| 입문 | [Evaluation and Benchmarking of LLM Agents — 2025](https://arxiv.org/abs/2507.21504) | 무엇을·어떻게 평가하는지의 두 축 |
| Judge | [G-Eval — 2023](https://arxiv.org/abs/2303.16634) | 구조화된 생성형 평가 |
| Judge | [Judging LLM-as-a-Judge — 2023](https://arxiv.org/abs/2306.05685) | 사람 선호와 일치, 위치·장문·자기 선호 편향 |
| Judge | [Agent-as-a-Judge — 2024](https://arxiv.org/abs/2410.10934) | 능동적인 증거 확인과 중간 과정 평가 |
| 도구·업무 | [τ-bench — 2024](https://arxiv.org/abs/2406.12045) | 정책·도구·사용자·반복 신뢰도 |
| 도구·업무 | [τ²-bench — 2025](https://arxiv.org/abs/2506.07982) | 사용자가 환경을 바꾸는 dual-control |
| 도구·업무 | [ToolSandbox — 2024](https://arxiv.org/abs/2408.04682) | 상태 의존성·대화·milestone |
| 도구·업무 | [StableToolBench — 2024](https://arxiv.org/abs/2403.07714) | API 변화·실패와 평가 안정성 |
| 일반·진전 | [AgentBench — 2023](https://arxiv.org/abs/2308.03688) | 다양한 agent 환경 |
| 일반·진전 | [AgentBoard — 2024](https://arxiv.org/abs/2401.13178) | 최종 성공 외의 progress 분석 |
| 일반·진전 | [GAIA — 2023](https://arxiv.org/abs/2311.12983) | 도구·검색·파일을 조합한 일반 assistant 문제 |
| 실행·산출물 | [WebArena — 2023](https://arxiv.org/abs/2307.13854) | 실제적인 웹 workflow |
| 실행·산출물 | [OSWorld — 2024](https://arxiv.org/abs/2404.07972) | 컴퓨터 앱의 실행 기반 결과 평가 |
| 실행·산출물 | [SWE-bench — 2023](https://arxiv.org/abs/2310.06770) | repository 이슈 해결과 patch |
| 실행·산출물 | [Terminal-Bench — 2026](https://arxiv.org/abs/2601.11868) | terminal의 복합 작업과 agent harness |
| 안전 행동 | [AgentDojo — 2024](https://arxiv.org/abs/2406.13352) | 업무 utility와 공격 결과 |
| 확장 | [τ-Knowledge — 2026](https://arxiv.org/abs/2603.04370) | 비정형 지식 기반 대화형 업무 |
| 확장 | [τ-Voice — 2026](https://arxiv.org/abs/2603.13686) | full-duplex 음성 상호작용 |
| 평가 타당성 | [SABER — 2025](https://arxiv.org/abs/2512.07850) | mutating step의 오류; τ³ task 수정 관련 읽을 자료 |

## 6. 결론의 적용 범위

- 구현상 차이는 해당 커밋의 사실이다. 이후 버전의 변경 여부는 별도로 확인해야 한다.
- 지표의 경계 사례는 평가기의 계약을 이해하기 위한 것이며 전체 제품의 품질 순위가 아니다.
- 모델 judge의 정확도나 오픈소스 간 처리량을 직접 비교한 수치를 제시하지 않았다.
- 학습용 추상화와 추천 조합은 이 조사에서 도출한 의견이다. 특정 제품에서의 효과가 측정됐다는 뜻은 아니다.
- 사용자 승인, 정보 freshness, 중복 쓰기 허용 여부 같은 조건은 평가 라이브러리가 아니라 업무 계약이 결정한다.
