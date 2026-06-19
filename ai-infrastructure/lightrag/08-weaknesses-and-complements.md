# 08. LightRAG 약점과 보완 기술·OSS 맵

> 목적: [07 분산 재설계](07-distributed-rearchitecture.md)의 "보충(➕)" 항목을 구체화 — 실데이터로 확인된 LightRAG 약점별로 보완 방법과 "이미 보완한 기술/OSS"를 매핑한다.
> 배경: 코스피 9개 종목 공시(715 md 섹션)를 적재하고 `/query/data`로 "삼성전기의 주요 매출"을 질의해 raw data를 검수한 결과에서 출발.

## 0. 실데이터로 확인된 약점 (요약)

`삼성전기의 주요 매출` 질의의 `/query/data` raw 응답에서 직접 관찰된 것:

- **junk 엔티티**: `매출`(src 83), `매출 발생`(UNKNOWN), `당사`(src 200=상한), `005930`, `031-210-5114`(전화번호), `953억원` 등이 엔티티로 적재됨
- **이름=ID 분열**: `삼성전기 (주)` / `삼성전기` / `삼성전기(주)` / `당사` / `SAMSUNG ELECTRO-MECHANICS...` 가 전부 별도 노드
- **동명 유사기업 혼입**: references 7개 중 삼성전기는 3개뿐, 나머지는 삼성전자·삼성물산·LG에너지솔루션
- **저품질 관계**: `매출 발생 -[기여]- 본사`, `2025년 -[경영 활동]- 삼성전기` 같은 약한 엣지가 고품질 엣지와 공존

## 1. 약점 → 보완 매핑 표

| # | LightRAG 약점 | 보완 방법 | 이미 보완한 기술 / OSS |
|---|---|---|---|
| ① | 엔티티 동일성 = **이름 완전일치** (표기 분열) | Entity Resolution — 임베딩 ANN blocking + 유사도 클러스터링 + LLM 검증 후 canonical 병합 | **GraphRAG-SDK** (4종 resolution: hnswlib ANN + scipy 계층 클러스터링 + LLM 검증) · **Microsoft GraphRAG** (엔티티 summary 병합) · **Cognee** (온톨로지 정규화) |
| ② | **junk 엔티티** (매출·전화번호·금액) | 온톨로지 타입 화이트리스트 + 추출 후 패턴 프루닝 | **GraphRAG-SDK** (`_prune()` — 선언된 (타입,타입) 패턴 밖 폐기) · **GLiNER** 스키마 강제 NER · **MS GraphRAG** (entity_types 제약) |
| ③ | **동명 유사기업 혼입** (삼성전기↔삼성전자) | 메타데이터 사전 필터(회사/종목코드) + 하이브리드 검색 + 리랭킹 | 메타필터: **Qdrant / Weaviate / pgvector** 등 벡터DB 기본 지원 · **RAGFlow** (강한 청킹+메타) · Cohere/BGE 리랭커 |
| ④ | **커뮤니티 요약 부재** (코퍼스 전역 질문) | Leiden 커뮤니티 탐지 + 계층 요약 + global search(map-reduce) | **Microsoft GraphRAG** (핵심 차별점) · **nano-graphrag** (MS 경량 구현) · **RAPTOR** (재귀 요약 트리 — [repo 분석](../raptor/limitations.md)) |
| ⑤ | **단일 프로세스 / 분산 불가** | 워크플로 엔진 + 병합 파티셔닝 ([07 문서](07-distributed-rearchitecture.md)) | 성숙한 분산 GraphRAG OSS **거의 없음** (대부분 단일 노드) → 자체 구축 영역 |
| ⑥ | **시간성 / 수치** (연도별 매출 — 공시의 본질) | bi-temporal KG (fact에 유효 시점) + 정형 데이터 분리 | **Graphiti (Zep)** — temporal knowledge graph, 시점별 fact 추적 · 우리가 이미 한 **dart_financials 정형 분리** |

## 2. 특히 주목할 셋

### GraphRAG-SDK — 약점 ①②의 거울상
우리가 [별도 분석](../falkordb-graphrag-sdk/README.md)한 프로젝트. LightRAG이 "이름=ID, 프롬프트 방어"라면 SDK는 "(이름,타입) 키 + 임베딩 resolution + 온톨로지 프루닝"으로 정확히 ①②를 막는다. 두 프로젝트를 나란히 분석한 이유이며, [07 문서](07-distributed-rearchitecture.md)의 "보충" 맵이 곧 **LightRAG 코어 + SDK식 resolution/프루닝** 조합이다.

### Graphiti (Zep) — 공시 도메인에 가장 관련 높음
공시는 본질적으로 시계열(분기별 실적·임원 변동·정정공시)이다. Graphiti의 **bi-temporal 모델**은 "2024 매출 vs 2025 매출"을 별개 노드가 아니라 **시점 속성(valid-from/valid-to)을 가진 fact**로 다룬다. 앞서 논의한 두 문제 — "연도가 엔티티로 잡히는 문제"와 "정정공시 교체 고민" — 를 구조적으로 해소하는 접근이며, 우리가 재무를 정형으로 분리한 것과 보완적이다.

### nano-graphrag / MS GraphRAG — 약점 ④ 전용
글로벌 질문("이 코퍼스의 주요 테마는?")에 강하나, 비용 구조가 LightRAG과 정반대다 — 커뮤니티 요약이 비싸다(논문에서 본 검색 단계 610K 토큰 문제). **"전역 요약 질문이 실제 요구사항인가"를 먼저 확인하고** 도입할 영역.

## 3. 자체 플랫폼 우선순위 권고

| 순위 | 보완 | 근거 |
|---|---|---|
| **1** | ①② — SDK식 resolution + 타입 프루닝을 LightRAG 추출 뒤 **비동기 잡**으로 | raw data의 junk·분열이 검색 정밀도를 직접 깎음. ROI 최대 |
| **2** | ⑥ — Graphiti식 temporal: fact에 `as_of_date`(보고서 기준일)를 일급 속성으로 | 공시 RAG 정확성의 핵심 — "언제 시점 정보냐" |
| **3** | ③ — 회사/종목코드 메타필터를 검색에 추가 | 벡터DB 필터로 거의 공짜, 단기 즉효 |
| **4** | ④ — nano-graphrag 패턴 차용 | 글로벌 요약 질문이 실제로 들어오는지 확인 **후** |

**핵심 결론**: 단일 OSS가 모든 약점을 보완하지 않는다. 약점마다 강한 기술이 다르므로(① GraphRAG-SDK, ④ nano-graphrag, ⑥ Graphiti), 자체 플랫폼은 **LightRAG 코어를 재사용하되 약점별 보완을 모듈로 조합**하는 것이 답이다. 이는 [07 문서](07-distributed-rearchitecture.md)의 재사용/교체/보충 설계와 일관된다.

## 4. 후속 문서화

LightRAG·GraphRAG-SDK와 같은 코드 레벨 깊이로 분석한/분석할 보완 기술:

- **Graphiti (Zep)** — temporal KG (약점 ⑥, 공시 도메인 직결) → ✅ [분석 완료](../graphiti/README.md) (bi-temporal·모순 무효화·MinHash dedup)
- **nano-graphrag** — 커뮤니티 요약 경량 구현 (약점 ④) — 미분석
- **Cognee** — 온톨로지 정규화 + temporal 종합 (①⑥) — 미분석
