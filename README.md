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
| AX (Google) | [분석](ai-agents/ax/analysis.md) | 분산 에이전트 런타임 (이벤트로그·재개·K8s Agent Substrate) |
| Agent Loops | [GoClaw 분석](ai-agents/agent-loops/goclaw-analysis.md) · [OpenCode vs ClaudeCode vs OpenHarness](ai-agents/agent-loops/opencode-vs-claudecode-vs-openharness.md) | 에이전트 루프 비교 |
| agentmemory | [분석](ai-agents/agentmemory/README.md) | iii-engine 기반 영속 메모리 (rohitg00) |
| DeerFlow | [분석](ai-agents/deer-flow/deer-flow-analysis.md) · [아키텍처](ai-agents/deer-flow/deer-flow-agent-architecture.md) · [메모리](ai-agents/deer-flow/deer-flow-conversation-memory.md) | 대화형 에이전트 워크플로우 |
| GenericAgent | [심층 분석](ai-agents/generic-agent/GenericAgent_심층분석.md) | 자기진화 LLM 에이전트 |
| Go Micro (v6) | [코드 레벨 분석](ai-agents/go-micro/go-micro-analysis.md) | Go 마이크로서비스 프레임워크의 "에이전트 하네스" 피벗 — 서비스=에이전트=플로우 통합 런타임 (AX와 비교) |
| Hermes Agent | [심층 분석](ai-agents/hermes-agent/Hermes-Agent_심층분석.md) | NousResearch 에이전트 |
| lat.md | [분석](ai-agents/lat-md/README.md) | 마크다운 코드베이스 지식 그래프 (Yury Selivanov) |
| Memory 비교 | [종합 비교](ai-agents/memory-comparison/에이전트_메모리_시스템_비교분석.md) · [이론 기원](ai-agents/memory-comparison/memory_theory_origins.md) · [Hindsight vs Memobase](ai-agents/memory-comparison/hindsight-vs-memobase.md) | 에이전트 메모리 시스템 횡단 분석 |
| Memora | [코드 레벨 분석](ai-agents/memora/memora-code-analysis.md) · [논문 기반 기억 흐름](ai-agents/memora/memora-paper-memory-flow.md) | Microsoft harmonic memory representation 기반 에이전트 장기 메모리 |
| MemPalace | [코드 레벨 분석](ai-agents/mempalace/README.md) | 로컬 우선 원문 보존형 에이전트 메모리 시스템 |
| └ 개별 분석 | [Agno](ai-agents/memory-comparison/Agno_analysis.md) · [Agno 문화](ai-agents/memory-comparison/Agno_culture_deep_dive.md) · [Cognee](ai-agents/memory-comparison/Cognee_analysis.md) · [Hindsight](ai-agents/memory-comparison/Hindsight_analysis.md) · [mem0](ai-agents/memory-comparison/mem0_analysis.md) · [Memobase](ai-agents/memory-comparison/Memobase_analysis.md) · [memU](ai-agents/memory-comparison/memU_analysis.md) · [Memori](ai-agents/memory-comparison/Memori_analysis.md) · [OpenMemory](ai-agents/memory-comparison/OpenMemory_analysis.md) · [SecondMe](ai-agents/memory-comparison/SecondMe_analysis.md) | 메모리 시스템 비교 코퍼스 |
| Mantis (Google) | [사용법·아키텍처·활용 분석](ai-agents/mantis/mantis-analysis.md) | Agent Skills 기반 자율 보안 리뷰 하네스 — 위협 모델링·취약점 재현·패치·재공격 파이프라인 |
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
| Kelos | [분석](ai-coding-tools/kelos/kelos-analysis.md) | Kubernetes 기반 AI 코딩 에이전트 오케스트레이션 |
| OpenCode | [README](ai-coding-tools/opencode/README.md) · [스터디](ai-coding-tools/opencode/STUDY.md) · [분석](ai-coding-tools/opencode/analysis.md) · [도구 시스템](ai-coding-tools/opencode/tools.md) · [SQLite 세션 저장소](ai-coding-tools/opencode/sqlite-session-storage.md) · [메모리 아키텍처](ai-coding-tools/opencode/opencode-memory-architecture.md) · [에이전트 엔진 로드맵](ai-coding-tools/opencode/AGENT_ENGINE_ROADMAP.md) · [Python 구현 spec1](ai-coding-tools/opencode/python-impl-spec1.md) · [spec2](ai-coding-tools/opencode/python-impl-spec2.md) · [통합](ai-coding-tools/opencode/python-impl-total.md) | OSS 코딩 에이전트 |
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
| Docling | [LightRAG 연동 비교](ai-infrastructure/docling/lightrag-docling-comparison.md) · [유즈케이스](ai-infrastructure/docling/use-cases.md) · [LlamaIndex 비교](ai-infrastructure/docling/docling-vs-llamaindex.md) | 문서 파싱·변환 엔진과 RAG 연동 |
| Flint (MS) | [코드 레벨 분석](ai-infrastructure/flint-chart/flint-chart-analysis.md) | AI 에이전트용 시각화 중간 언어(IL)·시맨틱 차트 컴파일러 + MCP 서버 (VL·ECharts·Chart.js) |
| Graph RAG · 온톨로지 | [기술 레퍼런스](ai-infrastructure/graph-rag-ontology/README.md) · [Agentic Ontology](ai-infrastructure/graph-rag-ontology/agentic-ontology.md) | Graph 기반 RAG·온톨로지 심화 |
| Graphiti (Zep) | [README](ai-infrastructure/graphiti/README.md) · [논문·bi-temporal](ai-infrastructure/graphiti/01-paper-and-bitemporal.md) · [아키텍처·모델](ai-infrastructure/graphiti/02-architecture-data-model.md) · [수집](ai-infrastructure/graphiti/03-ingestion-pipeline.md) · [검색](ai-infrastructure/graphiti/04-search-system.md) · [프롬프트](ai-infrastructure/graphiti/05-prompts-reference.md) | Zep Graphiti 코드 레벨 분석 + 논문 (bi-temporal KG·모순 무효화·LightRAG 시간성 약점 보완) |
| GraphRAG 비교 | [README](ai-infrastructure/graphrag-comparison/README.md) · [LightRAG vs GraphRAG-SDK](ai-infrastructure/graphrag-comparison/lightrag-vs-graphrag-sdk.md) · [금융 서비스 적합도](ai-infrastructure/graphrag-comparison/finance-service-fit.md) · [하이브리드 OSS landscape](ai-infrastructure/graphrag-comparison/hybrid-graph-vector-rag-oss-landscape.md) | 청킹·임베딩·그래프+벡터 하이브리드 OSS 비교 |
| GraphRAG-SDK | [README](ai-infrastructure/falkordb-graphrag-sdk/README.md) · [아키텍처·코어](ai-infrastructure/falkordb-graphrag-sdk/01-architecture-core.md) · [로딩·청킹](ai-infrastructure/falkordb-graphrag-sdk/02-loading-chunking.md) · [추출·그래프 구축](ai-infrastructure/falkordb-graphrag-sdk/03-extraction-graph-construction.md) · [검색](ai-infrastructure/falkordb-graphrag-sdk/04-retrieval-pipeline.md) · [온톨로지](ai-infrastructure/falkordb-graphrag-sdk/05-ontology-discovery-evolution.md) · [프롬프트](ai-infrastructure/falkordb-graphrag-sdk/06-prompts-reference.md) | FalkorDB GraphRAG-SDK v1.3 코드 레벨 분석 (수집→검색 파이프라인·청킹·전체 프롬프트) |
| Knowledge Catalog · OKF | [Open Knowledge Format 분석](ai-infrastructure/knowledge-catalog/okf-analysis.md) | Google Cloud Knowledge Catalog와 OKF v0.1 포맷·reference agent·metadata-as-code 도구 분석 |
| LlamaIndex | [유즈케이스](ai-infrastructure/llamaindex/use-cases.md) | RAG·agent 애플리케이션 프레임워크 |
| RAG · GraphRAG 확장 | [2026 landscape](ai-infrastructure/rag-graphrag-expansion/2026-landscape.md) | 최신 RAG·GraphRAG 기술과 OSS 확장 조사 |
| RAPTOR | [단점과 적용 리스크](ai-infrastructure/raptor/limitations.md) | Recursive abstractive tree retrieval |
| LangExtract | [분석](ai-infrastructure/langextract/langextract-analysis.md) | Google LLM 텍스트 추출 라이브러리 |
| LightRAG | [README](ai-infrastructure/lightrag/README.md) · [논문 분석](ai-infrastructure/lightrag/01-paper-analysis.md) · [아키텍처·코어](ai-infrastructure/lightrag/02-architecture-core.md) · [수집·청킹](ai-infrastructure/lightrag/03-ingestion-chunking.md) · [추출·병합](ai-infrastructure/lightrag/04-extraction-merge.md) · [질의](ai-infrastructure/lightrag/05-query-pipeline.md) · [프롬프트](ai-infrastructure/lightrag/06-prompts-reference.md) · [분산 재설계](ai-infrastructure/lightrag/07-distributed-rearchitecture.md) · [약점·보완 맵](ai-infrastructure/lightrag/08-weaknesses-and-complements.md) · [개요(구)](ai-infrastructure/lightrag/analysis.md) | HKUDS LightRAG 코드 레벨 분석 + 원본 논문 (dual-level retrieval·청킹 4종·전체 프롬프트·분산 재설계·약점 보완) |
| LLM MTP | [인터랙티브 보고서(HTML)](ai-infrastructure/llm-mtp/llm-mtp.html) | Multi-Token Prediction 다중 토큰 예측 기술 정리 |
| LLM Observability · LLMOps | [지형도 README](ai-infrastructure/llm-observability/README.md) · [Langfuse 심층](ai-infrastructure/llm-observability/langfuse.md) · [플랫폼 비교](ai-infrastructure/llm-observability/platforms.md) · [표준·생태계](ai-infrastructure/llm-observability/standards-and-ecosystem.md) | 오픈소스 LLM 관측·평가·게이트웨이 지형도 (Langfuse 기준 · 2026 인수 지형 · OTel gen_ai 표준) |
| RAG-Anything | [분석](ai-infrastructure/rag-anything/analysis.md) | 멀티모달 RAG |
| RAGFlow | [분석](ai-infrastructure/ragflow/analysis.md) · [인프라·셋업](ai-infrastructure/ragflow/infrastructure-setup.md) · [LightRAG 비교](ai-infrastructure/ragflow/lightrag-comparison.md) | 심층 문서 이해 기반 RAG·에이전트 엔진 (InfiniFlow) |
| WebMCP | [분석](ai-infrastructure/webmcp/webmcp-analysis.md) | W3C 웹 표준 기반 AI 도구 노출 |
| Wren AI | [분석](ai-infrastructure/wren-ai/analysis.md) | Wren AI 텍스트→SQL |

---

## Databases

데이터베이스(그래프·벡터·멀티모델 포함).

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| ArangoDB | [코드 레벨 분석](databases/arangodb/arangodb-code-analysis.md) | C++ 기반 multi-model graph DB |
| Graph DB 모음 | [CozoDB](databases/graphdb/CozoDB_Analysis.md) · [Kuzu](databases/graphdb/Kuzu_Analysis.md) · [FalkorDB](databases/graphdb/FalkorDB_Analysis.md) · [FalkorDBLite](databases/graphdb/FalkorDBLite_Analysis.md) · [Dgraph](databases/graphdb/Dgraph_Analysis.md) · [JanusGraph](databases/graphdb/JanusGraph_Analysis.md) · [HugeGraph](databases/graphdb/HugeGraph_Analysis.md) · [Memgraph](databases/graphdb/Memgraph_Analysis.md) · [Memgraph 2026 라이선스](databases/graphdb/memgraph-2026-license-analysis.md) | 그래프 DB 분석 |
| Graph DB 비교 | [오픈소스 후보 2026](databases/graphdb/open-source-graphdb-landscape-2026.md) · [임베드 3종 비교](databases/graphdb/CozoDB_vs_Kuzu_vs_FalkorDBLite.md) · [전체 비교](databases/graphdb/OpenSource_GraphDB_Comparison.md) · [K8s 오픈소스](databases/graphdb/K8s_OpenSource_Graph_Technologies.md) · [FalkorDB vs TigerGraph 서비스 채택](databases/graphdb/FalkorDB_vs_TigerGraph_Service_Use.md) | 그래프 DB 비교 |
| kg-gen | [분석](databases/graphdb/kg-gen-analysis-report.md) | 지식 그래프 생성 |
| StarRocks | [Debug Skills 분석](databases/starrocks-debug-skills/starrocks-debug-skills-analysis.md) | StarRocks 디버깅 스킬 모음 |
| SurrealDB | [멀티모델 내부구조](databases/surrealdb/multi-model-internals.md) | Rust 기반 멀티 모델 DB |
| Vector DB 비교 | [Qdrant · ChromaDB · Milvus · Pinecone · Weaviate](databases/vector-db-comparison/qdrant-chroma-milvus-pinecone-weaviate.md) | 주요 벡터 DB 특징·장단점 비교 |
| 웹 SQL 도구 | [웹 기반 SQL 실행·시각화 OSS 조사](databases/web-sql-tools/README.md) | CloudBeaver·Metabase·marimo 등 21종 — DB클라이언트·BI·노트북 3계열 비교 |
| Zvec | [분석](databases/zvec/zvec-analysis.md) · [vs LanceDB](databases/zvec/zvec-vs-lancedb-comparison.md) | Alibaba 벡터 DB |

---

## Data Platforms

ETL·분산 데이터 통합·메타데이터 카탈로그·플랫폼.

| 프로젝트 | 문서 | 설명 |
|-----------|------|------|
| Apache Ossie (구 OSI) | [코드 레벨 분석](data-platforms/ossie/apache-ossie-code-analysis.md) · [적용 참고서](data-platforms/ossie/OSI_적용_참고서.md) | 벤더 중립 시맨틱 메타데이터 교환 스펙 (ASF 인큐베이팅) — 스펙·온톨로지층·컨버터 11종 분석 |
| Gravitino | [심층 분석](data-platforms/gravitino/gravitino-analysis.md) · [vs DataHub vs OpenMetadata](data-platforms/gravitino/gravitino-vs-datahub-openmetadata.md) | Apache Gravitino 연합 메타데이터 레이크 — 카탈로그의 카탈로그, IRC 서버·Fileset/GVFS·모델 카탈로그·Ranger 푸시다운·MCP |
| SeaTunnel | [심층 분석](data-platforms/seatunnel/SeaTunnel_심층분석.md) | Apache SeaTunnel 멀티엔진 분산 데이터 통합 |
| 전사 데이터 스택 | [지형도](data-platforms/enterprise-data-stack/landscape.md) | "접근 포인트 통합"을 5개 레이어로 분해 — 카탈로그·정책엔진·연합쿼리·게이트웨이·시맨틱·디스커버리 OSS/상용 조사 + 대기업 레퍼런스 아키텍처·도입 순서 |

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
.repos/                # 분석용 외부 repo 클론 (gitignored, hidden)
```
