# 05. 프롬프트 레퍼런스 — Graphiti 전체 프롬프트

> 소스: `graphiti_core/prompts/`. `{placeholder}`는 런타임 치환. 모든 프롬프트는 Pydantic 구조화 출력(response model)을 강제.

## 0. 프롬프트 조립 구조

- `models.py`: `Message`(role + content), `PromptVersion`/`PromptFunction`(context dict → `list[Message]`)
- `prompt_helpers.py`: `DO_NOT_ESCAPE_UNICODE`(시스템 메시지에 append), `to_prompt_json()`(ensure_ascii=False — 비ASCII 보존)
- `lib.py`: `PromptLibrary` 레지스트리 — extract_nodes / dedupe_nodes / extract_edges / extract_nodes_and_edges / dedupe_edges / summarize_nodes / summarize_sagas / eval

각 프롬프트는 system + user 메시지 쌍 + response model(Pydantic)로 구성. **출력 계약을 스키마로 강제**하므로 LightRAG/SDK의 "Return ONLY JSON" 텍스트 지시보다 견고.

| 키 | 용도 | response model |
|---|---|---|
| extract_nodes (message/text/json) | 엔티티 추출 | `ExtractedEntities` |
| extract_edges `edge` | fact 추출 + temporal 인라인 | `ExtractedEdges` (valid_at/invalid_at 포함) |
| extract_edges `extract_timestamps` | temporal 보충 | valid_at/invalid_at |
| dedupe_nodes `node`/`nodes` | 엔티티 중복 판정 | `duplicate_candidate_id` |
| **dedupe_edges `resolve_edge`** | **중복 + 모순 판정** | `EdgeDuplicate(duplicate_facts, contradicted_facts)` |
| summarize_nodes `summarize_pair`/`summarize_context` | 요약 | `Summary` |

---

## 1. 엔티티 추출 (대화 메시지)

`extract_nodes.py` `extract_message` — **system**:
```
You are an entity extraction specialist for conversational messages. NEVER extract abstract concepts, feelings, or generic words.
```

**user** (핵심 발췌 — 전문은 repo): 추출 금지 목록(대명사·추상개념·일반명사·맨 관계어/동물어)과 추출 규칙을 상세 예시와 함께 제공. 핵심 규칙:
- 화자(speaker, `:` 앞부분)를 첫 엔티티로 추출
- "Wikipedia 항목이 될 만큼 구체적인가?" 기준 — 맨 명사(car/coat) 금지, 한정된 것(Gamecube, wool coat)만
- 맨 관계어는 소유자로 한정: `dad`→`Nisha's dad`
- **날짜·시간은 추출 안 함** (별도 처리)
- response model: `ExtractedEntity(name, entity_type_id, episode_indices)`

`extract_text`(문서)·`extract_json`(구조화 데이터)도 유사하되 도메인 맞춤. text 프롬프트도 "temporal 정보는 노드로 만들지 않음"(line 283) 명시.

---

## 2. 엣지(fact) 추출 + Temporal — 핵심

`extract_edges.py` `edge` — **system**:
```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
```

**user** (DATETIME RULES 부분 — temporal 추출의 실체, 전문 발췌):
```
# DATETIME RULES
- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
```
추출 규칙 요지: source/target는 ENTITIES 리스트의 name만 사용(아니면 reject), 두 distinct 엔티티 필수, 구체 디테일(브랜드·수량·색·모델) 일반화 금지(`Gamecube`→`gaming console` 금지), reference_time으로 상대 표현 해소, **시간 추론·환각 금지**.

response model: `Edge(source_entity_name, target_entity_name, relation_type, fact, valid_at, invalid_at, episode_indices)`

### Temporal 보충 — `extract_timestamps`
```
[system] You extract temporal bounds from facts. NEVER hallucinate dates.
[user] Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).
Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.
<FACT>{context['fact']}</FACT>
<REFERENCE TIME>{context['reference_time']}</REFERENCE TIME>
```
(배치판 `extract_timestamps_batch`도 동일 규칙, facts 리스트 입력.)

---

## 3. 엔티티 Dedup

`dedupe_nodes.py` `node` — **system**:
```
You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.
```
**user** 요지: NEW ENTITY를 EXISTING ENTITIES(candidate_id)와 비교, **같은 real-world 객체/개념일 때만** duplicate. 관련은 있으나 별개면 금지. 매치 없거나 불확실하면 `duplicate_candidate_id = -1`. 예시로 `NYC↔New York City`(=0, 약어), `Java(언어)↔Java(섬)`(=-1, 동명이의), `Marco's car↔Marco's vehicle`(=0, 동의어) 제시. (배치판 `nodes`, 그룹핑판 `node_list` 존재.)

response model: `NodeDuplicate(duplicate_candidate_id: int)` (-1 = 신규)

---

## 4. 엣지 Dedup + 모순 탐지 (★ 핵심)

`dedupe_edges.py` `resolve_edge` — temporal 무효화의 트리거. **전문 verbatim**:

**system**:
```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
```

**user**:
```
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
```

response model: `EdgeDuplicate(duplicate_facts: list[int], contradicted_facts: list[int])`

> 이 프롬프트가 bi-temporal의 두뇌다. `contradicted_facts`로 지목된 옛 엣지에 [03 문서 §3](03-ingestion-pipeline.md)의 무효화 로직(invalid_at=새 valid_at, expired_at=now)이 적용된다. 두 번째 예시("software engineer"→"senior engineer"가 duplicate가 아니라 contradiction)가 핵심 — **같은 관계의 값 갱신을 중복이 아닌 모순(=시점 전환)으로 처리**하는 것이 시계열 정확성의 토대.

---

## 5. 요약

`summarize_nodes.py` `summarize_pair` (커뮤니티 빌드의 pairwise 병합):
```
[system] You are a helpful assistant that combines summaries into a single dense factual summary.
[user] Synthesize the information from the following two summaries into a single information-dense summary.
IMPORTANT:
- Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
- Prefer compact factual sentences over vague thematic phrasing.
- ... (filler 동사 "mentioned/stated/noted" 회피)
- SUMMARIES MUST BE LESS THAN {MAX_SUMMARY_CHARS} CHARACTERS.
Summaries: {to_prompt_json(context['node_summaries'])}
```
`summarize_context`: episode 메시지에서 엔티티 요약 + 커스텀 속성 추출 (없으면 None).

---

## 6. 프롬프트 설계 관찰 — LightRAG/SDK와 비교

1. **Pydantic 구조화 출력 강제** — 모든 프롬프트가 response model을 가져 출력 계약이 스키마 레벨. LightRAG/SDK의 텍스트 "Return ONLY JSON"보다 견고 (파싱 실패율↓)
2. **모순을 중복과 명시적 분리** — `resolve_edge`가 `duplicate_facts`/`contradicted_facts`를 나눠 받는 게 핵심. "값 갱신 = 모순 = 시점 전환"을 LLM에 학습시켜 bi-temporal을 작동시킴. 두 프로젝트에 없는 개념
3. **temporal 추출 전용 프롬프트** — "NEVER hallucinate dates", "ongoing이면 reference_time", "연도만 있으면 1월 1일 00:00" 등 시간 해소 규칙이 정밀. 공시 날짜 처리에 직접 차용 가능
4. **엔티티 추출의 구체성 강제** — "맨 명사 금지, 한정어 보존"(Gamecube를 console로 일반화 금지)이 매우 상세. LightRAG에서 본 junk 엔티티(매출/숫자) 방지에 참고할 만한 패턴
5. **reference_time 기반 상대 표현 해소** — "지난주"를 episode 시점 기준으로 절대 시각화. 공시의 "당분기", "전년 동기" 같은 상대 표현 처리에 직결
