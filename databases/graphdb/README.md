# 그래프 데이터베이스

그래프 DB의 저장·질의 구조와 도입 선택지를 비교한다. 먼저 비교 문서로 범위를 좁힌 뒤 개별 프로젝트 분석을 읽을 수 있다. 라이선스와 기능은 각 문서의 분석 시점을 기준으로 한다.

## 비교와 선택

| 문서 | 내용 |
|---|---|
| [오픈소스 후보 조사](open-source-graphdb-landscape-2026.md) | 2026년 기준 그래프 DB 후보 |
| [전체 비교](OpenSource_GraphDB_Comparison.md) | 주요 프로젝트의 기능과 구조 비교 |
| [임베드형 3종 비교](CozoDB_vs_Kuzu_vs_FalkorDBLite.md) | CozoDB·Kuzu·FalkorDBLite 비교 |
| [Kubernetes 환경의 그래프 기술](K8s_OpenSource_Graph_Technologies.md) | K8s 환경에서 사용하는 그래프 기술 |
| [FalkorDB vs TigerGraph](FalkorDB_vs_TigerGraph_Service_Use.md) | 서비스 그래프 DB 채택 관점의 비교 |

## 개별 구현과 관련 도구

| 프로젝트 | 문서 |
|---|---|
| CozoDB | [분석](CozoDB_Analysis.md) |
| Dgraph | [분석](Dgraph_Analysis.md) |
| FalkorDB | [분석](FalkorDB_Analysis.md) |
| FalkorDBLite | [분석](FalkorDBLite_Analysis.md) |
| HugeGraph | [분석](HugeGraph_Analysis.md) |
| JanusGraph | [분석](JanusGraph_Analysis.md) |
| Kuzu | [분석](Kuzu_Analysis.md) |
| Memgraph | [분석](Memgraph_Analysis.md) · [2026 라이선스·GraphRAG 분석](memgraph-2026-license-analysis.md) |
| kg-gen | [텍스트 기반 지식 그래프 생성 분석](kg-gen-analysis-report.md) |

그래프 DB 위에 검색 파이프라인을 구성하는 방법은 [GraphRAG 비교](../../ai-infrastructure/graphrag-comparison/README.md)를 참고한다.
