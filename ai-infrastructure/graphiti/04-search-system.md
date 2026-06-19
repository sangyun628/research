# 04. 검색 시스템

> 소스: `graphiti_core/search/{search,search_config,search_config_recipes,search_filters,search_utils,search_helpers}.py`

## 1. 진입점 — search vs search_

| API | 위치 | 반환 | 용도 |
|---|---|---|---|
| `search(query, center_node_uuid?, num_results=10, search_filter?)` | graphiti.py:1527 | `list[EntityEdge]` (엣지만) | 간단 — 기본 `EDGE_HYBRID_SEARCH_RRF`, center 주면 `EDGE_HYBRID_SEARCH_NODE_DISTANCE` |
| `search_(query, config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER, ...)` | graphiti.py:1603 | `SearchResults`(node+edge+episode+community) | 설정형 — recipe로 차원·리랭킹 조합 |

코어 `search()`(search.py:98)가 4개 차원 검색을 `semaphore_gather()`로 병렬 실행 후 결과 조립.

## 2. 검색 차원 × 방법

| 차원 | BM25(fulltext) | cosine(벡터) | BFS(그래프) |
|---|---|---|---|
| **Node** | ✅ | ✅ | ✅ |
| **Edge** | ✅ | ✅ | ✅ |
| **Episode** | ✅ | — | — |
| **Community** | ✅ | ✅ | — |

- 각 방법은 리랭킹 전 **`2 × limit`** 후보를 가져옴
- cosine 임계 **DEFAULT_MIN_SCORE = 0.6**, BFS 최대 깊이 MAX_SEARCH_DEPTH = 3
- 결과는 uuid로 dedup 병합 후 리랭킹

## 3. 리랭킹 5종

| 전략 | 위치 | 알고리즘 | 핵심 파라미터 |
|---|---|---|---|
| **RRF** (Reciprocal Rank Fusion) | search_utils.py:1780 | 여러 랭킹 리스트를 `Σ 1/(rank + k)`로 융합 | rank_const(k)=1 |
| **MMR** (Maximal Marginal Relevance) | 1901 | `λ·(query·cand) + (1−λ)·max(cand·selected)` — 관련성 vs 다양성 | mmr_lambda=0.5 |
| **Cross-encoder** | search.py:395+ | LLM이 (query, text) 쌍 점수화 | reranker_min_score, limit(2×아님) |
| **Node-distance** | search_utils.py:1798 | center 노드에서 1-hop 이웃 score=1, 도달불가 ~0 | center_node_uuid 필수 |
| **Episode-mentions** | 1860 | RRF seed → MENTIONS 엣지 수로 재정렬 | min_score |

- **RRF**: 예) BM25 rank0 + cosine rank1 = 1/1 + 1/2 = 1.5. 대부분 recipe의 기본
- **MMR**: 후보 임베딩 L2 정규화 → 쌍별 유사도 행렬 → 이미 선택된 것과 유사하면 감점(다양성). λ=0.5는 관련성·다양성 균등
- **Cross-encoder**: 가장 비싸고 정밀. limit를 2×가 아닌 limit로 제한해 비용 억제
- **Node-distance**: center 엔티티 주변 근접성 우선 (특정 엔티티 맥락 질의)
- **Episode-mentions**: 많이 언급된 엔티티 우선 (빈도 = 중요도)

## 4. 검색 레시피 (prebuilt SearchConfig)

`search_config_recipes.py` — 차원·방법·리랭커 조합을 미리 묶음:

- **노드**: `NODE_HYBRID_SEARCH_{RRF, MMR, CROSS_ENCODER, NODE_DISTANCE, EPISODE_MENTIONS}`
- **엣지**: `EDGE_HYBRID_SEARCH_{RRF, MMR, CROSS_ENCODER, NODE_DISTANCE, EPISODE_MENTIONS}`
- **커뮤니티**: `COMMUNITY_HYBRID_SEARCH_{RRF, MMR, CROSS_ENCODER}`
- **통합(전 차원)**: `COMBINED_HYBRID_SEARCH_{RRF, MMR, CROSS_ENCODER}` — `search_()`의 기본은 CROSS_ENCODER

대부분 BM25 + cosine를 기본 방법으로 쓰고, CROSS_ENCODER 계열은 BFS까지 추가(엣지/노드). 리랭커별로 limit가 다름(cross-encoder는 10 또는 3으로 줄여 비용 관리).

## 5. Temporal 필터링 (bi-temporal 활용)

`SearchFilters` (search_filters.py:55-74) — 검색을 시점·구간으로 좁힘:

```python
class SearchFilters(BaseModel):
    valid_at:   list[list[DateFilter]] | None  # event time 시작
    invalid_at: list[list[DateFilter]] | None  # event time 종료
    created_at: list[list[DateFilter]] | None  # transaction time
    expired_at: list[list[DateFilter]] | None
# DateFilter: date + comparison_operator (=, <>, >, <, >=, <=, IS NULL, IS NOT NULL)
# 리스트의 리스트 = 그룹 간 OR, 그룹 내 AND
```

쿼리 생성(`edge_search_filter_query_constructor`, 120-273)이 Cypher WHERE로 변환:
```cypher
(e.valid_at > $v0 AND e.valid_at < $v1) OR ...
(e.invalid_at IS NULL OR e.invalid_at > $date)
```

**Point-in-time 패턴** ("시점 T에 참이던 사실"):
```
valid_at <= T  AND  (invalid_at IS NULL OR invalid_at > T)
```
주의: 무효화된 엣지가 검색에서 **자동 제외되지는 않음** — 필터를 명시해야 한다 (기본 검색은 invalid 엣지도 포함). 이는 "과거에 참이던 사실"도 일부러 질의할 수 있게 한 설계.

필터는 fulltext·cosine·BFS 모든 방법에 전달돼 초기 검색과 리랭킹 후보 양쪽에 적용. OpenSearch 백엔드는 `cypher_to_opensearch_operator()`로 연산자 변환.

## 6. 하이브리드 메커니즘 & 기본값

1. **병렬 디스패치**: edge/node/episode/community 검색을 동시 실행
2. **후보 풀링**: 방법별 `2×limit` → uuid dedup 병합
3. **리랭킹**: 차원·recipe별 전략 적용
4. **출력**: 차원별 top `limit` + 점수

| 상수 | 값 |
|---|---|
| DEFAULT_SEARCH_LIMIT | 10 |
| DEFAULT_MIN_SCORE (cosine) | 0.6 |
| DEFAULT_MMR_LAMBDA | 0.5 |
| MAX_SEARCH_DEPTH (BFS) | 3 |
| RRF rank_const | 1 |

검색 모듈 자체에는 LLM 프롬프트가 없다(cross-encoder는 외부 `CrossEncoderClient`에 위임). `search_helpers.py:27`의 `search_results_to_context_string()`이 결과를 LLM 컨텍스트 문자열로 포맷(각 fact에 valid_at/invalid_at 명시)하나 이는 포맷팅일 뿐.

## 7. 자체 구현에 가져갈 것

1. **temporal 검색 필터를 일급으로** — `valid_at≤T AND (invalid_at IS NULL OR >T)`. 공시 "시점 기준" 질의의 핵심. LightRAG `/query/data`에는 없는 차원
2. **리랭킹 5종 메뉴** — 특히 RRF(저비용 융합)와 node-distance(특정 회사 맥락 질의)는 공시 도메인에 유용. cross-encoder는 정밀하나 비용 고려
3. **무효화 엣지 비자동제외** — "현재 사실"과 "과거 사실" 질의를 필터로 구분. 시계열 분석에 필요
4. **검색 결과 → 컨텍스트 문자열에 시점 명시** — 답변 LLM이 "언제 기준 정보"인지 알게 함
