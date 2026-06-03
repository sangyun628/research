# Research Index

오픈소스 및 기술 분석 문서 인덱스. 폴더 구조·정리 규칙은 [CLAUDE.md](CLAUDE.md#폴더-구조-및-정리-규칙) 참조.

> **구조 원칙**: `category/topic/files` (3-depth). 단일 문서는 카테고리 바로 아래, 다문서 프로젝트는 `topic/` 폴더로 묶는다.

---

## AI Agents

에이전트 프레임워크·메모리·스킬·구체적 에이전트 구현.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| Agno | [분석 보고서](ai-agents/agno/agno-analysis-report.md) · [DB 커스터마이징](ai-agents/agno/agno-database-customization-guide.md) · [스토리지·도구](ai-agents/agno/agno-storage-and-tools-guide.md) | Python 에이전트 프레임워크 |
| AG-UI | [심층 분석](ai-agents/ag-ui/AG-UI_심층분석.md) | 에이전트 UI 프로토콜 |
| Agent Loops | [GoClaw 분석](ai-agents/agent-loops/goclaw-analysis.md) · [OpenCode vs ClaudeCode vs OpenHarness](ai-agents/agent-loops/opencode-vs-claudecode-vs-openharness.md) | 에이전트 루프 비교 |
| agentmemory | [분석](ai-agents/agentmemory/README.md) | iii-engine 기반 영속 메모리 (rohitg00) |
| DeerFlow | [분석](ai-agents/deer-flow/deer-flow-analysis.md) · [아키텍처](ai-agents/deer-flow/deer-flow-agent-architecture.md) · [메모리](ai-agents/deer-flow/deer-flow-conversation-memory.md) | 대화형 에이전트 워크플로우 |
| GenericAgent | [심층 분석](ai-agents/generic-agent/GenericAgent_심층분석.md) | 자기진화 LLM 에이전트 |
| Hermes Agent | [심층 분석](ai-agents/hermes-agent/Hermes-Agent_심층분석.md) | NousResearch 에이전트 |
| lat.md | [분석](ai-agents/lat-md/README.md) | 마크다운 코드베이스 지식 그래프 (Yury Selivanov) |
| Memory 비교 | [종합 비교](ai-agents/memory-comparison/에이전트_메모리_시스템_비교분석.md) · [이론 기원](ai-agents/memory-comparison/memory_theory_origins.md) | 에이전트 메모리 시스템 횡단 분석 |
| └ 개별 분석 | [Agno](ai-agents/memory-comparison/Agno_analysis.md) · [Agno 문화](ai-agents/memory-comparison/Agno_culture_deep_dive.md) · [Cognee](ai-agents/memory-comparison/Cognee_analysis.md) · [mem0](ai-agents/memory-comparison/mem0_analysis.md) · [memU](ai-agents/memory-comparison/memU_analysis.md) · [Memori](ai-agents/memory-comparison/Memori_analysis.md) · [OpenMemory](ai-agents/memory-comparison/OpenMemory_analysis.md) · [SecondMe](ai-agents/memory-comparison/SecondMe_analysis.md) | 메모리 시스템 비교 코퍼스 |
| Mastra | [분석](ai-agents/mastra/analysis.md) | TypeScript 풀스택 에이전트 프레임워크 (Apache-2.0 + ee) |
| Mirage | [분석](ai-agents/mirage/README.md) | Strukto.AI 통합 가상 파일시스템 |
| Nanobot | [분석](ai-agents/nanobot/nanobot-analysis.md) | 모듈형 에이전트 프레임워크 |
| Open Agents | (디렉터리) | Open Agents 모음 |
| Open Mythos | (디렉터리) | Open Mythos |
| OpenChronicle | [심층 분석](ai-agents/openchronicle/OpenChronicle_심층분석.md) | 로컬 화면 컨텍스트 메모리 |
| OpenClaw | [분석](ai-agents/openclaw/openclaw-analysis.md) | 자율 AI 에이전트 플랫폼 |
| OpenHarness | [분석](ai-agents/openharness/README.md) · [코드 분석](ai-agents/openharness/OPENHARNESS_ANALYSIS.md) | Claude Code 에이전트 하네스 OSS 구현 |
| OpenSpace | [분석](ai-agents/openspace/README.md) | 자기 진화형 스킬 엔진 |
| OpenViking | [분석](ai-agents/openviking/README.md) | ByteDance 에이전트 컨텍스트 DB |
| pi autoresearch | (디렉터리) | pi 자율 리서치 |
| Supermemory | [분석](ai-agents/supermemory/README.md) | 범용 AI 메모리 레이어 |
| TradingAgents | [분석](ai-agents/tradingagents/README.md) · [에이전트·도구·스킬](ai-agents/tradingagents/agent-tools-skills.md) | 멀티 에이전트 트레이딩 |
| Agent Skills | [아키텍처](ai-agents/skills/agentskills-architecture.md) | Anthropic 에이전트 스킬 스펙 |

---

## AI Coding Tools

코딩 에이전트·IDE 도구.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| Claude Code | [아키텍처 분석](ai-coding-tools/claude-code/ARCHITECTURE_ANALYSIS.md) · [핵심 발견](ai-coding-tools/claude-code/KEY_FINDINGS.md) · [메모리 시스템](ai-coding-tools/claude-code/memory-system-analysis.md) · [프롬프트 카탈로그](ai-coding-tools/claude-code/prompts-catalog.md) · [README](ai-coding-tools/claude-code/README.md) | Anthropic 코딩 CLI |
| OpenCode | [스터디](ai-coding-tools/opencode/STUDY.md) · [분석](ai-coding-tools/opencode/analysis.md) · [도구 시스템](ai-coding-tools/opencode/tools.md) · [메모리 아키텍처](ai-coding-tools/opencode/opencode-memory-architecture.md) · [에이전트 엔진 로드맵](ai-coding-tools/opencode/AGENT_ENGINE_ROADMAP.md) · [Python 구현 spec1](ai-coding-tools/opencode/python-impl-spec1.md) · [spec2](ai-coding-tools/opencode/python-impl-spec2.md) · [통합](ai-coding-tools/opencode/python-impl-total.md) | OSS 코딩 에이전트 |
| Warp | [분석](ai-coding-tools/warp/ANALYSIS.md) | Warp 터미널 LLM·Agent 아키텍처 |

---

## AI Infrastructure

RAG·임베딩·MCP·텍스트→SQL 등 AI 인프라 레이어.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| Agentic RAG | [리서치](ai-infrastructure/agentic-rag/agentic-rag-research.md) · [LlamaIndex 가이드](ai-infrastructure/agentic-rag/llamaindex-developer-guide.md) · [PageIndex](ai-infrastructure/agentic-rag/pageindex.md) | Agentic RAG 연구 |
| Airweave | [분석](ai-infrastructure/airweave/airweave-analysis.md) | AI 컨텍스트 검색 인프라 |
| Chunking | [OSS 가이드](ai-infrastructure/chunking/chunking-oss-guide.md) | 청킹 특화 OSS (Chonkie 등) · Bedrock KB 전략 매핑 |
| DB-GPT | [분석](ai-infrastructure/db-gpt/analysis.md) | DB-GPT (텍스트→SQL) |
| Graph RAG · 온톨로지 | [기술 레퍼런스](ai-infrastructure/graph-rag-ontology/README.md) | Graph 기반 RAG·온톨로지 심화 |
| LangExtract | [분석](ai-infrastructure/langextract/langextract-analysis.md) | Google LLM 텍스트 추출 라이브러리 |
| LightRAG | [분석](ai-infrastructure/lightrag/analysis.md) | 경량 RAG |
| RAG-Anything | [분석](ai-infrastructure/rag-anything/analysis.md) | 멀티모달 RAG |
| RAGFlow | [분석](ai-infrastructure/ragflow/analysis.md) | 심층 문서 이해 기반 RAG·에이전트 엔진 (InfiniFlow) |
| WebMCP | [분석](ai-infrastructure/webmcp/webmcp-analysis.md) | W3C 웹 표준 기반 AI 도구 노출 |
| Wren AI | [분석](ai-infrastructure/wren-ai/analysis.md) | Wren AI 텍스트→SQL |

---

## Databases

데이터베이스(그래프·벡터·멀티모델 포함).

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| Graph DB 모음 | [CozoDB](databases/graphdb/CozoDB_Analysis.md) · [Kuzu](databases/graphdb/Kuzu_Analysis.md) · [FalkorDB](databases/graphdb/FalkorDB_Analysis.md) · [FalkorDBLite](databases/graphdb/FalkorDBLite_Analysis.md) · [Dgraph](databases/graphdb/Dgraph_Analysis.md) · [JanusGraph](databases/graphdb/JanusGraph_Analysis.md) · [HugeGraph](databases/graphdb/HugeGraph_Analysis.md) · [Memgraph](databases/graphdb/Memgraph_Analysis.md) | 그래프 DB 분석 |
| Graph DB 비교 | [임베드 3종 비교](databases/graphdb/CozoDB_vs_Kuzu_vs_FalkorDBLite.md) · [전체 비교](databases/graphdb/OpenSource_GraphDB_Comparison.md) · [K8s 오픈소스](databases/graphdb/K8s_OpenSource_Graph_Technologies.md) | 그래프 DB 비교 |
| kg-gen | [분석](databases/graphdb/kg-gen-analysis-report.md) | 지식 그래프 생성 |
| StarRocks | [Debug Skills 분석](databases/starrocks-debug-skills/starrocks-debug-skills-analysis.md) | StarRocks 디버깅 스킬 모음 |
| SurrealDB | [멀티모델 내부구조](databases/surrealdb/multi-model-internals.md) | Rust 기반 멀티 모델 DB |
| Zvec | [분석](databases/zvec/zvec-analysis.md) · [vs LanceDB](databases/zvec/zvec-vs-lancedb-comparison.md) | Alibaba 벡터 DB |

---

## Data Platforms

ETL·분산 데이터 통합·플랫폼.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| OSI | [적용 참고서](data-platforms/osi/OSI_적용_참고서.md) | OSI 플랫폼 적용 가이드 |
| SeaTunnel | [심층 분석](data-platforms/seatunnel/SeaTunnel_심층분석.md) | Apache SeaTunnel 멀티엔진 분산 데이터 통합 |

---

## Kubernetes · Cloud Native

K8s 운영·관측·진단.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| HolmesGPT | [분석](kubernetes/holmesgpt-analysis.md) · [코드 분석](kubernetes/holmesgpt-code-analysis.md) · [Runbook·Toolset](kubernetes/holmesgpt-runbook-toolset-analysis.md) | AI 기반 K8s 트러블슈팅 |
| K8sGPT | [분석](kubernetes/k8sgpt-analysis.md) | K8s 클러스터 AI 진단 |
| Kagent | [분석](kubernetes/kagent-analysis.md) | K8s 에이전트 |
| Prometheus vs VictoriaMetrics | [심층 비교](kubernetes/prometheus-vs-victoriametrics.md) | 운영자 관점 모니터링 비교 |

---

## Algorithms · Libraries · Domain

소규모/단일 주제.

| 분류 | 문서 | 설명 |
|---|------|------|
| Algorithms | [STATIC 분석](algorithms/STATIC_Analysis.md) | HW 가속 제약 디코딩 알고리즘 |
| Libraries · oban-py | [분석](libraries/oban-py/oban-py-analysis.md) | PostgreSQL 기반 백그라운드 작업 프레임워크 |
| Finance · Fincept Terminal | [분석](finance/fincept-terminal-analysis.md) | 오픈소스 금융 터미널 |

---

## Trends

| 문서 | 설명 |
|------|------|
| [AI 기술 트렌드 2025-2026](trends/ai-technology-trends-2025-2026.md) | RAG 진화, Agentic AI, MCP 표준화 등 |

---

## 폴더 구조 한눈에

```
ai-agents/             # 에이전트 프레임워크·메모리·구체 구현
ai-coding-tools/       # 코딩 에이전트(Claude Code, OpenCode, Warp)
ai-infrastructure/     # RAG·임베딩·MCP·텍스트→SQL
databases/             # 그래프·벡터·멀티모델 DB
data-platforms/        # ETL·분산 데이터 통합
kubernetes/            # K8s 운영·진단·관측 비교
algorithms/            # 알고리즘 단편
libraries/             # 라이브러리 단편
finance/               # 도메인-금융
trends/                # 트렌드 정리
scripts/               # 검증·자동화 스크립트
_repos/                # 분석용 외부 repo 클론 (gitignored)
```
