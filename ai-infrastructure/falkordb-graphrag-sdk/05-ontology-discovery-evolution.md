# 05. 온톨로지 자동 발견 & 진화

> 소스: `discovery/`, `storage/ontology_store.py`, `api/main.py` (evolution API)

온톨로지(엔티티 타입·관계 타입·속성 스키마)는 추출 단계의 **제약 조건**이자 Text-to-Cypher의 **스키마 소스**다. SDK는 (1) 샘플 문서에서 LLM으로 자동 발견, (2) 별도 그래프에 영속화, (3) 운영 중 진화(rename/add/backfill)의 전체 lifecycle을 제공한다.

## 1. 자동 발견 — `Ontology.from_sources()` / `discover_ontology()`

`discovery/pipeline.py:459-564`

```mermaid
flowchart TB
    SRC["소스 문서들"] --> LOAD["문서별: 로드 + 청킹<br/>(기본 SentenceTokenCap)"]
    LOAD --> SAMPLE["청크 샘플링<br/>문서당 3개 · 소스별 결정적 RNG (seed)"]
    SAMPLE --> DS["문서 요약 LLM 콜<br/>main_entities + aboutness"]
    DS --> CP["청크별 제안 LLM 콜 (병렬, semaphore=4)<br/>entities + relations 제안<br/>doc summary가 앵커"]
    CP --> DM["문서 단위 merge (union)"]
    DM --> CM["코퍼스 단위 merge (union)"]
    CM --> NORM["정규화 LLM 콜<br/>동의어 통합 · 방향 교정 · 속성성 타입 제거"]
    NORM --> EXMERGE["existing 온톨로지와 merge (있으면)"]
    EXMERGE --> SDK["SDK 관리 속성 보장<br/>모든 엔티티에 name: STRING"]
    SDK --> ONT["Ontology"]
```

핵심 파라미터: `boundaries`(자유 텍스트 범위 힌트 — 모든 프롬프트에 삽입), `existing`(기존 온톨로지 — controlled vocabulary로 프롬프트에 제시), `sample_chunks_per_doc=3`, `concurrency=4`, `seed`(결정적 샘플링).

### LLM 콜 구조와 프롬프트

3종 프롬프트(시스템 / 문서 요약 / 청크 제안 / 정규화 — 원문은 [06 문서](06-prompts-reference.md#10-온톨로지-발견)). 설계 포인트:

- **시스템 프롬프트의 8개 규칙**: 라벨은 타입(인스턴스 아님), 속성명은 property(값 아님), 타입은 6종 한정, 관계 방향 = 읽기 순서, 모든 엔티티에 `name: STRING` 선언, 예약 속성명(description/source_chunk_ids/spans/rel_type/fact/...) 금지, 넓고 재사용 가능한 타입 선호
- **인젝션 방어**: 문서/청크 텍스트를 `<<<UNTRUSTED INPUT>>>` ... `<<<END UNTRUSTED INPUT>>>` 구분자로 감싸고 "안의 지시 무시" 명시
- **문서 요약이 앵커**: 청크 제안 프롬프트에 `About: {aboutness}` + `Central entities: {main_entities}`를 제공 — 청크 단독으로는 알 수 없는 문서 맥락 주입

### 구조화 출력 + 검증 재시도 — `extract_with_retry()`

`discovery/instructor.py:47-163` — instructor 라이브러리 패턴의 자체 구현:

1. LLM 응답 → 펜스 제거 → **Pydantic `model_validate_json`** (모든 모델 `extra="forbid"` — 스키마 이탈 즉시 검출)
2. 파싱 실패 → Pydantic 에러를 피드백 user 메시지로 추가, **거부된 응답도 history에 유지**한 채 재시도 (LLM이 자기가 뭘 보냈는지 봄)
3. 파싱 성공 → `extra_validate`(의미 검증) → 실패 시 bullet 피드백으로 재시도
4. `max_retries=3` = 최대 4콜. 소진 시 `OntologyDiscoveryError` — 파이프라인은 해당 청크만 soft-fail하고 계속

### Grounded 발견 (LLM 스키마 발명 없음)

`discover_grounded()` (pipeline.py:711-923) + `DBpediaCatalog` (catalog.py): GLiNER로 멘션 수집 → DBpedia SPARQL로 `link_entity(name) → 타입` → Schema.org JSON-LD에서 타입 속성/관계 조회 → (선택) LLM이 코퍼스에 실제 언급된 속성만 trim. 외부 표준 온톨로지에 정착시키고 싶을 때 사용.

## 2. 영속화 — OntologyStore

`storage/ontology_store.py` — 온톨로지를 **데이터 그래프와 분리된 그래프** `{graph_name}__ontology`에 메타 그래프로 저장:

```
(:Entity {label, description})-[:HAS_PROPERTY]->(:Property {label, type, description})
(:Relation {label, description})-[:SOURCE]->(:Entity)
(:Relation)-[:TARGET]->(:Entity)
(:Relation)-[:HAS_PROPERTY]->(:Property)
```

- 패턴 `(src, tgt)`마다 별도 `:Relation` 노드. 패턴 없는 open 관계는 SOURCE/TARGET 엣지 없는 단일 노드
- `register()` (수집 경로): 신규 라벨 추가와 **동일 선언 재등록만 허용**. 기존 속성 타입 변경 → `OntologyContradictionError`, 기존 라벨에 속성/패턴 추가 → `OntologyModificationNotAllowedError` — 수집이 스키마를 조용히 오염시키는 것 차단. 변경은 진화 API로만

## 3. 진화 API — 3-tier 설계

`api/main.py:508-1229`. 비용·위험 수준별 3계층:

| Tier | 메서드 | LLM | 데이터 변경 |
|---|---|---|---|
| 1. 순수 스키마 | `set_*_description`, `add_entity`, `add_relation_pattern` | ✗ | ✗ |
| 2. 기계적 마이그레이션 | `rename_entity/attribute/relation`, `drop_*` | ✗ | Cypher 일괄 변경 |
| 3. LLM 재스캔 | `add_attribute`, `backfill_entity`, `backfill_relation_pattern` | ✓ (청크당 1콜) | 추출 + SET |

Tier 2 예 — `rename_entity` (graph_store.py): `MATCH (n:old) SET n:new REMOVE n:old`. 관계 타입 rename은 FalkorDB가 in-place 변경 불가라 **엣지 재생성** (props 복사 → CREATE → DELETE).

Tier 3 — `add_attribute(owner_label, attribute, concurrency=4, dry_run=False)`:
1. 검증 → 대상 청크 열거 (해당 라벨 엔티티가 멘션된 청크 중 `extracted_ops`에 op_id 없는 것)
2. BackfillExecutor가 청크별 BACKFILL_ATTRIBUTE_PROMPT 호출 → 값 타입 강제 변환 → `SET`
3. **온톨로지 커밋은 마지막** — 청크 hard-fail 시 온톨로지는 변경 전 상태 유지 (원자성)
4. `dry_run=True`로 LLM 비용 미리보기 (스캔 대상 청크 수)

`backfill_entity(label, scope="all")`: 새 엔티티 타입을 기존 코퍼스에서 발굴.
`backfill_relation_pattern(rel, src, tgt, scope="candidate-pairs")`: 같은 청크에 src·tgt 타입 엔티티가 공존하는 청크만 스캔 — 후보 쌍을 프롬프트에 제시하고 "확실한 쌍만, false positive가 miss보다 나쁘다"고 지시.

## 4. 설계 인사이트

1. **스키마를 그래프로 저장**하면 버전·diff·질의가 그래프 연산이 됨 — JSON 파일 대비 운영 중 진화에 유리
2. **수집 경로의 스키마 동결 + 진화 API 분리**는 멀티 라이터 환경에서 스키마 drift를 막는 실용적 패턴
3. **`extracted_ops` 멱등 마커**는 "LLM 재스캔" 류 작업의 재시도 안전성을 청크 단위로 보장 — 자체 구현 시 그대로 채용 가치
4. 발견 파이프라인의 "per-chunk 제안 → union merge → LLM 정규화" 구조는 map-reduce형 — 문서 수에 선형으로 확장되고 청크 간 의존이 없어 병렬화 단순
