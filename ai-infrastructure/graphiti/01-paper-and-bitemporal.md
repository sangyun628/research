# 01. 논문 & Bi-Temporal 모델

> arXiv: [2501.13956](https://arxiv.org/abs/2501.13956) — "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (2025)
> 코드 대응: `graphiti_core/edges.py`, `nodes.py`, `utils/maintenance/edge_operations.py`

## 1. 문제 의식

기존 RAG는 **정적 문서 검색**에 묶여 있다. 엔터프라이즈·에이전트 환경은 진행 중인 대화·업무 데이터가 계속 들어오고 **시간에 따라 사실이 바뀐다**(직책 변경, 실적 갱신, 정정). 청크 벡터 검색이나 한 번 구축한 그래프로는 "지금 참인 사실"과 "과거에 참이었던 사실"을 구분하지 못한다. Graphiti는 이를 **bi-temporal 지식 그래프**로 푼다.

## 2. 3계층 아키텍처

```
Episode 계층   — 입력 원본 단위 (메시지/텍스트/JSON/fact triple) + 타임스탬프
   │ MENTIONS
Entity/Edge 계층 — 추출된 엔티티(EntityNode)와 사실(EntityEdge, bi-temporal)
   │ HAS_MEMBER
Community 계층  — label propagation 클러스터 + LLM 요약 (선택)
```

- **Episodic subgraph**(원본 보존)와 **Semantic subgraph**(추출된 의미)가 분리돼 공존 — episode는 불변 기록, entity/edge는 진화. MS GraphRAG가 "추출 후 원본을 버리는" 것과 대조적으로, Graphiti는 episode를 남겨 fact가 어느 입력에서 왔는지 추적(`EntityEdge.episodes`).

## 3. Bi-Temporal 모델 (핵심)

모든 사실(`EntityEdge`)이 **4개의 타임스탬프**를 가진다. 코드 정의 (`edges.py:271-285`):

```python
class EntityEdge(Edge):
    # ── Event time (현실에서 참인 구간) ──
    valid_at:  datetime | None  # 사실이 참이 된 시점
    invalid_at: datetime | None  # 사실이 거짓이 된 시점
    # ── Transaction time (시스템이 알았던 구간) ──
    created_at: datetime         # 그래프에 기록된 시점
    expired_at: datetime | None  # 무효화가 기록된 시점
    reference_time: datetime | None  # 이 엣지를 만든 episode의 타임스탬프
```

두 시간축의 의미:

| 축 | 필드 | 질문 | 예시 |
|---|---|---|---|
| **Event time** | valid_at / invalid_at | "현실에서 언제부터 언제까지 참이었나?" | "김대표의 직책은 2024-01-01부터 2024-12-31까지 CEO" |
| **Transaction time** | created_at / expired_at | "시스템이 언제 알았고 언제 폐기했나?" | "2024-06-01에 알았고, 2025-01-15에 갱신으로 폐기" |

→ 두 축이 독립이라 **백필**(과거 시점 fact를 오늘 입력)과 **point-in-time 질의**("2024-06 시점에 우리가 알던 사실은?")가 모두 가능하다. 단일 timestamp만 있는 LightRAG·SDK와의 근본 차이.

## 4. Edge Invalidation (모순 무효화)

새 사실이 들어와 옛 사실과 모순되면 **삭제가 아니라 무효화**한다 (`resolve_edge_contradictions()`, edge_operations.py:538-573):

```python
# 새 엣지(N)가 옛 엣지(E)를 무효화하는 경우
if E.valid_at < N.valid_at:        # E가 N보다 먼저 참이 됨 = N이 E를 대체
    E.invalid_at = N.valid_at      # E는 N이 참이 되는 시점에 거짓이 됨 (event time)
    E.expired_at = utc_now()       # 무효화를 기록한 시점 (transaction time)
# 단, 시간 구간이 겹치지 않으면(이미 무효였거나 미래 사실) skip
```

흐름:
1. 새 fact 추출 시 두 종류 검색 — 관련 엣지(dedup용, 같은 노드쌍)와 무효화 후보(contradiction용, 더 넓은 집합)
2. **LLM이 모순 판정** — `resolve_edge` 프롬프트가 `duplicate_facts`(중복)와 `contradicted_facts`(모순) 인덱스를 반환 ([05 문서](05-prompts-reference.md))
3. 모순된 옛 엣지에 `invalid_at`/`expired_at` 세팅 → 둘 다 그래프에 남음 (감사 추적)

> 공시 도메인 직결: "삼성전자 CEO가 A→B로 바뀜" 같은 정정/갱신이 옛 fact를 자동 무효화하되, "그때는 A였다"는 이력이 보존됨. 우리가 LightRAG에서 고민한 "정정공시 교체"와 "연도별 값 혼재"가 구조적으로 해결되는 지점.

## 5. Point-in-Time 질의

별도 시간 인덱스는 없고, 타임스탬프가 엣지 속성이라 검색 필터로 구현 ([04 문서](04-search-system.md)):

```
"시점 T에 참이던 사실":  valid_at <= T AND (invalid_at IS NULL OR invalid_at > T)
"시점 T에 시스템이 알던 사실": created_at <= T AND (expired_at IS NULL OR expired_at > T)
```

`SearchFilters`가 valid_at/invalid_at/created_at/expired_at 각각에 대해 비교 연산자(`>`,`<`,`>=`,`<=`,`IS NULL` 등) 필터를 지원.

## 6. 평가 (논문)

| 벤치마크 | Graphiti/Zep | 비교 |
|---|---|---|
| **DMR** (Deep Memory Retrieval) | 94.8% | MemGPT 93.4% |
| **LongMemEval** | 정확도 최대 **+18.5%** | baseline 대비 |
| 지연 | **−90%** | baseline 구현 대비 |

논문은 특히 "교차 세션 정보 종합"과 "장기 컨텍스트 유지"에서 강점을 주장 — 에이전트 메모리 시나리오. (초록 기준 수치이며, GraphRAG와의 직접 비교는 논문 본문에서 다룸.)

## 7. 자체 구현 시사점

1. **bi-temporal 스키마를 fact의 기본형으로** — event(valid/invalid) + transaction(created/expired) 4필드. 공시처럼 시점이 중요한 도메인에서는 단일 timestamp가 사실상 버그
2. **무효화 ≠ 삭제** — 모순 시 invalid_at 마킹 + 이력 보존. 감사·재현·시계열 질의의 토대
3. **episode = 원본 보존 단위** — 추출 결과(semantic)와 원본(episodic)을 분리하면 provenance·재처리가 자연스러움. LightRAG의 `source_id`, SDK의 `MENTIONED_IN`과 같은 계보지만 시간축이 결합됨
