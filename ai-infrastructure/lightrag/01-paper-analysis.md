# 01. 논문 분석 — LightRAG: Simple and Fast Retrieval-Augmented Generation

> arXiv: [2410.05779](https://arxiv.org/abs/2410.05779) (v3, EMNLP 2025) · HKU Data Science Lab
> 이 문서는 논문의 기술 내용을 정리하고, 현재 코드베이스(2026-06)가 논문에서 얼마나 진화했는지 매핑한다.

## 1. 문제 의식

기존 RAG의 두 한계:
1. **평탄한(flat) 데이터 표현** — 청크 단위 벡터 검색은 청크 사이의 관계를 보지 못해, 여러 청크에 걸친 복잡한 상호의존 질문에 파편화된 답을 냄
2. **컨텍스트 인식 부족** — "이 코퍼스에서 전기차가 도시 대기질에 미치는 영향은?" 같은 질문은 EV·대기오염·교통이라는 엔티티들의 **관계망**을 종합해야 답할 수 있음

MS GraphRAG가 그래프로 이를 풀었지만, 커뮤니티 요약 기반 global search의 **검색 비용이 과도**하고 증분 갱신 시 커뮤니티 전체 재구축이 필요하다는 것이 LightRAG의 출발점.

## 2. 그래프 기반 텍스트 인덱싱 — Recog · Prof · Dedupe

문서를 세 함수의 합성으로 인덱싱: `D̂ = Dedupe ∘ Prof ∘ Recog(D)`

| 함수 | 역할 | 코드 대응 (현재) |
|---|---|---|
| **Recog** | LLM이 청크에서 엔티티·관계 식별 | `extract_entities()` — [04 문서](04-extraction-merge.md) |
| **Prof** | 엔티티/관계마다 **Key-Value 쌍** 생성 — Key는 검색용 짧은 단어/구, Value는 생성용 요약 문단 | 엔티티 VDB content = `name\ndescription`, 관계 VDB content = `keywords\tsrc\ntgt\ndescription` |
| **Dedupe** | 청크 간 동일 엔티티/관계 병합 | `merge_nodes_and_edges()` — 이름 단위 병합 |

KV 설계의 핵심 비대칭:
- **엔티티 key = 이름 하나** (구체적 사물 검색용)
- **관계 key = 복수 키워드** — LLM이 관계의 "글로벌 테마"를 키워드로 뽑아 인덱스 키로 사용 (추상 질문 검색용)

이 비대칭이 그대로 dual-level retrieval의 기반이 된다.

## 3. Dual-Level Retrieval

질문을 두 부류로 나눈다:
- **Specific** (구체적): "Pride and Prejudice의 저자는?" → 특정 엔티티 지향
- **Abstract** (추상적): "AI가 현대 교육에 미치는 영향은?" → 테마·개념 지향

검색 3단계:
1. **키워드 추출**: LLM 1콜로 질문에서 local 키워드 `k^(l)`(구체적 엔티티)와 global 키워드 `k^(g)`(테마) 동시 추출
2. **매칭**: `k^(l)` → 엔티티 key 벡터 매칭 (low-level), `k^(g)` → 관계 key 벡터 매칭 (high-level)
3. **고차 연결 확장**: 매칭된 노드/엣지의 **1-hop 이웃**을 합집합으로 추가 — `{vᵢ | vᵢ ∈ 𝒩_v ∪ 𝒩_e}`

> 현재 코드의 `mode` 매핑: low-level만 = `local`, high-level만 = `global`, 둘 다 = `hybrid`, 둘 다 + 청크 벡터검색 = `mix`(기본값). [05 문서](05-query-pipeline.md)

## 4. 증분 갱신 알고리즘

새 문서 D′를 기존과 **동일한 파이프라인**으로 처리해 D̂′를 만들고, 그래프에 합집합으로 결합: `V̂ ∪ V̂′, Ê ∪ Ê′`. 동명 엔티티는 Dedupe 단계에서 설명이 병합된다.

- GraphRAG: 새 데이터 추가 시 커뮤니티 구조 재구축 — 논문 추산 약 1,399 × 2 × 5,000 토큰
- LightRAG: 추출 비용만 발생, 그래프는 union

> 현재 코드는 여기서 더 나아가 **삭제**까지 지원 — 청크별 LLM 추출 캐시를 보존해두고, 문서 삭제 시 남은 청크의 캐시에서 엔티티/관계를 재구성(rebuild)한다. [04 문서 §5](04-extraction-merge.md)

## 5. 비용 분석 — GraphRAG 대비 (논문의 핵심 셀링 포인트)

Legal 데이터셋 검색 단계 비교:

| 지표 | GraphRAG | LightRAG |
|---|---|---|
| 토큰 소비 | 610 커뮤니티 × 1,000토큰 = **610,000 토큰** | **100토큰 미만** (키워드 추출) |
| API 콜 | 수백 회 (610,000 / C_max) | **1회** |

구조적 이유: GraphRAG의 global search는 레벨-2 커뮤니티 리포트를 전부 순회하는 map-reduce인 반면, LightRAG은 키워드 추출 1콜 후 전부 벡터 인덱스 연산.

## 6. 평가

**셋업**: UltraDomain 벤치마크 4개 도메인 (Agriculture 12문서/2.0M토큰, CS 10문서/2.3M토큰, Legal 94문서/5.1M토큰, Mix 61문서/0.6M토큰). 도메인당 125개 질문(가상 사용자 5 × 태스크 5 × 질문 5). GPT-4o-mini를 judge로 한 pairwise 승률. 4개 차원: Comprehensiveness / Diversity / Empowerment / Overall.

**Overall 승률 (LightRAG 기준)**:

| 데이터셋 | vs NaiveRAG | vs GraphRAG |
|---|---|---|
| Agriculture | 67.6% | 54.8% |
| CS | 61.2% | 52.0% |
| Legal | 84.8% | 52.8% |
| Mix | 60.0% | 49.6% |

대형 코퍼스(Legal)일수록 NaiveRAG 대비 격차가 커짐 — 플랫 벡터 검색의 한계가 코퍼스 크기에 비례. GraphRAG 대비로는 동등~소폭 우위인데, **같은 품질을 1/수백 비용으로** 달성한다는 게 논문의 주장.

**Ablation (Legal, vs NaiveRAG Overall)**:

| 변형 | 승률 | 해석 |
|---|---|---|
| -High (low-level만) | 78.0% | 추상 질문 대응력 하락 |
| -Low (high-level만) | 81.2% | comprehensiveness는 유지, 깊이 하락 |
| **Full (dual-level)** | **84.8%** | 두 레벨 결합이 최선 |
| -Origin (원문 청크 제외) | 84.4% | **원문 없이 KG 요약만으로도 거의 동등** — KV 프로파일링(Value=요약문단)이 충분히 정보를 보존한다는 증거 |

`-Origin`이 가장 흥미로운 결과: 엔티티/관계 description이 잘 만들어지면 원문 청크의 기여가 적다. 단, 현재 코드 기본값은 원문 청크를 포함(mix 모드)한다.

**구현 디테일**: GPT-4o-mini 전 구간 사용, 청크 1200토큰 고정, gleaning 1회, nano-vectordb.

## 7. 논문 → 현재 코드 진화 매핑

| 논문 (2024-10) | 현재 코드 (2026-06) |
|---|---|
| 청킹: 1200토큰 고정 | 4종 전략 (F/R/V/P) — 단락 시맨틱·표 행 분할 추가 |
| 인덱싱: 단일 insert | 6상태 문서 상태 머신 + 파서 3계층 + 멀티모달(VLM) |
| dual-level retrieval | + mix 모드(청크 벡터 병행), 리랭킹, 토큰 예산 통합 제어, 인용 references |
| 증분 = union | + 삭제/rebuild, 문서 dedup(파일명·콘텐츠 해시), track_id |
| nano-vectordb | 스토리지 4종 ABC × 12 백엔드 |
| 평가: LLM judge | `lightrag/evaluation/` 모듈로 내장 |

## 8. 자체 구현 관점의 시사점

1. **KV 프로파일링이 본질** — "그래프를 만든다"보다 "검색 가능한 key와 생성 가능한 value를 만든다"가 정확한 프레임. 엔티티 key=이름, 관계 key=테마 키워드의 비대칭 설계를 그대로 채택할 가치
2. **dual-level은 키워드 추출 품질에 의존** — 키워드 LLM 콜이 검색 진입점 전부를 결정. 한국어에서는 이 프롬프트의 출력 품질 검증이 1순위
3. **-Origin ablation의 함의**: description 품질에 투자하면(요약 프롬프트, 병합 정책) 컨텍스트에서 원문 비중을 줄여 토큰 비용을 더 절감할 여지
4. **비용 모델**: 수집 = 청크당 LLM (1 + gleaning)콜 + 병합 요약 콜(조각 8개↑일 때만), 질의 = 2콜 고정. 이 단순한 비용 식이 운영 예측성을 줌
