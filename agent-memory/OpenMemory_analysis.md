# OpenMemory (CaviraOSS) 분석 보고서

> 저장소: https://github.com/CaviraOSS/OpenMemory
> 분석일: 2026-01-21

---

## 1. 개요

OpenMemory는 AI 에이전트 및 애플리케이션을 위한 오픈소스 메모리 레이어입니다. 시간 기반 지식 그래프, 감쇠 메커니즘, 연상 메모리 링크(Waypoints)를 갖춘 정교한 멀티 섹터 메모리 시스템을 제공합니다.

### 핵심 기능

- **멀티 섹터 메모리**: 5개의 구분된 인지 섹터 (일화, 의미, 절차, 감정, 성찰)
- **시간 지식 그래프**: 유효 기간과 신뢰도 감쇠가 있는 사실들
- **계층적 의미 그래프 (HSG)**: Waypoint 기반 연상을 가진 핵심 메모리 구조
- **Decay 엔진**: 벡터 압축이 있는 3계층 감쇠 시스템
- **Reflection 시스템**: 자동 메모리 통합 및 패턴 감지
- **MCP 서버**: Claude, Cursor, Windsurf와의 네이티브 통합

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenMemory                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   MCP       │   │   REST      │   │  LangChain  │           │
│  │   서버      │   │   API       │   │   커넥터    │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                  │                  │
│         └─────────────────┼──────────────────┘                  │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Memory 클래스                         │   │
│  │    add() / search() / get() / delete() / history()      │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│         ┌─────────────────────┼─────────────────────┐          │
│         ▼                     ▼                     ▼          │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐      │
│  │    HSG     │       │  Temporal  │       │   Decay    │      │
│  │  (메모리)  │◄─────►│   Graph    │◄─────►│   엔진    │      │
│  └─────┬──────┘       └─────┬──────┘       └─────┬──────┘      │
│        │                    │                    │              │
│        ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    벡터 저장소                            │   │
│  │         (PostgreSQL/pgvector, Valkey, SQLite)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 메모리 섹터 (5가지 타입)

OpenMemory는 Tulving의 인지 메모리 이론을 5개의 구분된 섹터로 구현합니다.

### 3.1 섹터 설정

```python
# /core/constants.py

SECTOR_CONFIGS: Dict[str, SectorCfg] = {
    "episodic": {
        "model": "episodic-optimized",
        "decay_lambda": 0.015,      # 빠른 감쇠
        "weight": 1.2,
        "patterns": [...]
    },
    "semantic": {
        "model": "semantic-optimized",
        "decay_lambda": 0.005,      # 느린 감쇠
        "weight": 1.0,
        "patterns": [...]
    },
    "procedural": {
        "model": "procedural-optimized",
        "decay_lambda": 0.008,
        "weight": 1.1,
        "patterns": [...]
    },
    "emotional": {
        "model": "emotional-optimized",
        "decay_lambda": 0.02,       # 가장 빠른 감쇠
        "weight": 1.3,              # 가장 높은 가중치
        "patterns": [...]
    },
    "reflective": {
        "model": "reflective-optimized",
        "decay_lambda": 0.001,      # 가장 느린 감쇠
        "weight": 0.8,
        "patterns": [...]
    },
}
```

### 3.2 섹터 특성

| 섹터 | Decay Lambda | 가중치 | 설명 |
|------|-------------|--------|------|
| **Episodic (일화)** | 0.015 | 1.2 | 개인 이벤트, 시간적 맥락이 있는 경험 |
| **Semantic (의미)** | 0.005 | 1.0 | 사실, 개념, 일반 지식 |
| **Procedural (절차)** | 0.008 | 1.1 | 방법론, 기술, 프로세스 |
| **Emotional (감정)** | 0.02 | 1.3 | 감정, 기분, 정서 상태 |
| **Reflective (성찰)** | 0.001 | 0.8 | 통찰, 패턴, 메타인지 |

### 3.3 섹터 감지 패턴

```python
# 일화 패턴
re.compile(r"\b(today|yesterday|tomorrow|last\s+(week|month|year))\b", re.I)
re.compile(r"\b(remember\s+when|recall|that\s+time|when\s+I)\b", re.I)
re.compile(r"\b(went|saw|met|felt|heard|visited|attended)\b", re.I)

# 의미 패턴
re.compile(r"\b(is\s+a|represents|means|defined\s+as)\b", re.I)
re.compile(r"\b(concept|theory|principle|law|hypothesis)\b", re.I)
re.compile(r"\b(fact|statistic|data|evidence|proof)\b", re.I)

# 절차 패턴
re.compile(r"\b(how\s+to|step\s+by\s+step|guide|tutorial)\b", re.I)
re.compile(r"\b(first|second|then|next|finally)\b", re.I)
re.compile(r"\b(install|run|execute|compile|build|deploy)\b", re.I)

# 감정 패턴
re.compile(r"\b(feel|feeling|felt|emotions?|mood)\b", re.I)
re.compile(r"\b(happy|sad|angry|excited|scared|anxious)\b", re.I)
re.compile(r"[!]{2,}", re.I)  # 여러 느낌표

# 성찰 패턴
re.compile(r"\b(realize|realized|realization|insight|epiphany)\b", re.I)
re.compile(r"\b(pattern|trend|connection|link|relationship)\b", re.I)
re.compile(r"\b(lesson|moral|takeaway|conclusion)\b", re.I)
```

### 3.4 섹터 간 관계

```python
# /memory/hsg.py

SECTOR_RELATIONSHIPS = {
    "semantic":   {"procedural": 0.8, "episodic": 0.6, "reflective": 0.7, "emotional": 0.4},
    "procedural": {"semantic": 0.8, "episodic": 0.6, "reflective": 0.6, "emotional": 0.3},
    "episodic":   {"reflective": 0.8, "semantic": 0.6, "procedural": 0.6, "emotional": 0.7},
    "reflective": {"episodic": 0.8, "semantic": 0.7, "procedural": 0.6, "emotional": 0.6},
    "emotional":  {"episodic": 0.7, "reflective": 0.6, "semantic": 0.4, "procedural": 0.3},
}
```

---

## 4. 계층적 의미 그래프 (HSG)

HSG는 메모리의 저장, 검색, 연상을 처리하는 핵심 메모리 구조입니다.

### 4.1 메모리 저장 스키마

```python
# /core/db.py - memories 테이블

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    segment INTEGER,              # 세그먼트 기반 저장
    content TEXT,                 # 저장된 콘텐츠 (추출된 핵심)
    simhash TEXT,                 # 중복 제거용
    primary_sector TEXT,          # 주요 섹터 분류
    tags TEXT,                    # JSON 태그 배열
    meta TEXT,                    # JSON 메타데이터
    created_at INTEGER,
    updated_at INTEGER,
    last_seen_at INTEGER,         # 최근성 스코어링용
    salience REAL,                # 중요도 점수 (0-1)
    decay_lambda REAL,            # 섹터별 감쇠율
    version INTEGER,
    mean_dim INTEGER,             # 평균 벡터 차원
    mean_vec BLOB,                # 섹터 전체 평균 임베딩
    compressed_vec BLOB,          # 콜드 메모리용 압축 벡터
    feedback_score INTEGER        # 공활성화 횟수
)
```

### 4.2 메모리 추가 흐름

```python
# /memory/hsg.py

async def add_hsg_memory(content, tags=None, metadata=None, user_id=None):
    # 1. 중복 제거를 위한 SimHash 계산
    simhash = compute_simhash(content)
    existing = db.fetchone("SELECT * FROM memories WHERE simhash=?", (simhash,))

    if existing and hamming_dist(simhash, existing["simhash"]) <= 3:
        # 중복 발견 - 새로 생성하지 않고 salience 부스트
        boost = min(1.0, existing["salience"] + 0.15)
        db.execute("UPDATE memories SET salience=? WHERE id=?", (boost, existing["id"]))
        return {"id": existing["id"], "deduplicated": True}

    # 2. 콘텐츠를 섹터로 분류
    cls = classify_content(content, metadata)
    all_secs = [cls["primary"]] + cls["additional"]

    # 3. 핵심 추출 (필요시 요약)
    stored = extract_essence(content, cls["primary"], max_length)

    # 4. 초기 salience 계산
    init_sal = max(0.0, min(1.0, 0.4 + 0.1 * len(cls["additional"])))

    # 5. 메모리 레코드 삽입
    q.ins_mem(id=mid, content=stored, primary_sector=cls["primary"], ...)

    # 6. 멀티 섹터 임베딩 생성
    emb_res = await embed_multi_sector(mid, content, all_secs)
    for r in emb_res:
        await store.storeVector(mid, r["sector"], r["vector"], r["dim"], user_id)

    # 7. 평균 벡터 계산
    mean_vec = calc_mean_vec(emb_res, all_secs)

    # 8. Waypoint (연상 링크) 생성
    await create_single_waypoint(mid, mean_vec, now, user_id)

    return {"id": mid, "primary_sector": cls["primary"], "sectors": all_secs}
```

### 4.3 SimHash 중복 제거

```python
def compute_simhash(text: str) -> str:
    tokens = canonical_token_set(text)

    # 각 토큰 해시
    hashes = []
    for t in tokens:
        h = 0
        for c in t:
            h = (h << 5) - h + ord(c)
            h = h & 0xffffffff
        hashes.append(h)

    # 64비트 핑거프린트 구축
    vec = [0] * 64
    for h in hashes:
        for i in range(64):
            bit = 1 << (i % 32)
            if h & bit:
                vec[i] += 1
            else:
                vec[i] -= 1

    # 16진수 문자열로 변환
    res_hash = ""
    for i in range(0, 64, 4):
        nibble = sum(8 >> j if vec[i+j] > 0 else 0 for j in range(4))
        res_hash += format(nibble, 'x')

    return res_hash

def hamming_dist(h1: str, h2: str) -> int:
    dist = 0
    for i in range(len(h1)):
        x = int(h1[i], 16) ^ int(h2[i], 16)
        dist += bin(x).count('1')
    return dist
```

---

## 5. Waypoint 그래프 (연상 메모리)

Waypoint는 메모리 간 연상 링크를 생성하여 그래프 기반 검색 확장을 가능하게 합니다.

### 5.1 Waypoint 스키마

```sql
CREATE TABLE waypoints (
    src_id TEXT,
    dst_id TEXT,
    user_id TEXT,
    weight REAL,          # 링크 강도 (0-1)
    created_at INTEGER,
    updated_at INTEGER,
    PRIMARY KEY (src_id, dst_id)
)
```

### 5.2 Waypoint 생성

```python
# 단일 Waypoint - 가장 유사한 기존 메모리에 링크
async def create_single_waypoint(new_id, new_mean, ts, user_id):
    mems = q.all_mem_by_user(user_id, 1000, 0)
    best = None
    best_sim = -1.0

    nm = np.array(new_mean, dtype=np.float32)

    for mem in mems:
        if mem["id"] == new_id or not mem["mean_vec"]:
            continue
        ex_mean = np.array(buf_to_vec(mem["mean_vec"]), dtype=np.float32)
        sim = cos_sim(nm, ex_mean)
        if sim > best_sim:
            best_sim = sim
            best = mem["id"]

    if best:
        db.execute("INSERT OR REPLACE INTO waypoints VALUES (?,?,?,?,?,?)",
                   (new_id, best, user_id, float(best_sim), ts, ts))
```

### 5.3 검색 시 Waypoint 확장

```python
async def expand_via_waypoints(ids: List[str], max_exp: int = 10):
    exp = []
    vis = set(ids)
    q_arr = [{"id": i, "weight": 1.0, "path": [i]} for i in ids]
    cnt = 0

    while q_arr and cnt < max_exp:
        cur = q_arr.pop(0)
        neighs = db.fetchall(
            "SELECT dst_id, weight FROM waypoints WHERE src_id=? ORDER BY weight DESC",
            (cur["id"],)
        )

        for n in neighs:
            dst = n["dst_id"]
            if dst in vis:
                continue

            wt = float(n["weight"])
            exp_wt = cur["weight"] * wt * 0.8  # 감쇠 계수

            if exp_wt < 0.1:  # 프루닝 임계값
                continue

            item = {"id": dst, "weight": exp_wt, "path": cur["path"] + [dst]}
            exp.append(item)
            vis.add(dst)
            q_arr.append(item)
            cnt += 1

    return exp
```

### 5.4 Waypoint 강화

```python
REINFORCEMENT = {
    "salience_boost": 0.1,
    "waypoint_boost": 0.05,
    "max_salience": 1.0,
    "max_waypoint_weight": 1.0,
    "prune_threshold": 0.05,
}

async def reinforce_waypoints(trav_path: List[str]):
    """탐색 경로를 따라 Waypoint 강화"""
    now = int(time.time() * 1000)

    for i in range(len(trav_path) - 1):
        src_id = trav_path[i]
        dst_id = trav_path[i + 1]

        wp = db.fetchone("SELECT * FROM waypoints WHERE src_id=? AND dst_id=?",
                         (src_id, dst_id))
        if wp:
            new_wt = min(
                REINFORCEMENT["max_waypoint_weight"],
                float(wp["weight"]) + REINFORCEMENT["waypoint_boost"]
            )
            db.execute("UPDATE waypoints SET weight=?, updated_at=? WHERE ...",
                       (new_wt, now, src_id, dst_id))
```

---

## 6. 스코어링 시스템

### 6.1 하이브리드 스코어링 가중치

```python
# /memory/hsg.py

SCORING_WEIGHTS = {
    "similarity": 0.35,     # 벡터 유사도
    "overlap": 0.20,        # 토큰 중복
    "waypoint": 0.15,       # Waypoint 가중치
    "recency": 0.10,        # 시간 기반 최근성
    "tag_match": 0.20,      # 태그 매칭
}

HYBRID_PARAMS = {
    "tau": 3.0,             # 유사도 부스트 계수
    "beta": 2.0,
    "eta": 0.1,
    "gamma": 0.2,           # 컨텍스트 부스트 계수
    "alpha_reinforce": 0.08,
    "t_days": 7.0,          # 최근성 반감기
    "t_max_days": 60.0,     # 최대 최근성 윈도우
    "tau_hours": 1.0,
    "epsilon": 1e-8,
}
```

### 6.2 하이브리드 스코어 계산

```python
def compute_hybrid_score(sim, tok_ov, wp_wt, rec_sc, kw_score=0, tag_match=0):
    # 시그모이드 유사 변환으로 유사도 부스트
    s_p = boosted_sim(sim)  # 1 - exp(-tau * sim)

    raw = (SCORING_WEIGHTS["similarity"] * s_p +
           SCORING_WEIGHTS["overlap"] * tok_ov +
           SCORING_WEIGHTS["waypoint"] * wp_wt +
           SCORING_WEIGHTS["recency"] * rec_sc +
           SCORING_WEIGHTS["tag_match"] * tag_match +
           kw_score)

    return sigmoid(raw)

def boosted_sim(s: float) -> float:
    return 1 - math.exp(-HYBRID_PARAMS["tau"] * s)

def calc_recency_score(last_seen: int) -> float:
    days = (time.time() * 1000 - last_seen) / 86400000.0
    t = HYBRID_PARAMS["t_days"]       # 7일
    tmax = HYBRID_PARAMS["t_max_days"]  # 60일
    return math.exp(-days / t) * (1 - days / tmax)
```

### 6.3 섹터 간 공명

```python
# /ops/dynamics.py

SECTORAL_INTERDEPENDENCE_MATRIX = [
    #   epi   sem   pro   emo   ref
    [1.0, 0.7, 0.3, 0.6, 0.6],  # 일화
    [0.7, 1.0, 0.4, 0.7, 0.8],  # 의미
    [0.3, 0.4, 1.0, 0.5, 0.2],  # 절차
    [0.6, 0.7, 0.5, 1.0, 0.8],  # 감정
    [0.6, 0.8, 0.2, 0.8, 1.0],  # 성찰
]

async def calculateCrossSectorResonanceScore(ms, qs, bs):
    """섹터 관계에 기반한 섹터 간 페널티/부스트 적용"""
    si = SECTOR_INDEX.get(ms, 1)
    ti = SECTOR_INDEX.get(qs, 1)
    return bs * SECTORAL_INTERDEPENDENCE_MATRIX[si][ti]
```

---

## 7. Decay 엔진

### 7.1 Decay 설정

```python
# /memory/decay.py

class DecayCfg:
    def __init__(self):
        self.threads = 3
        self.cold_threshold = 0.25
        self.reinforce_on_query = True
        self.regeneration_enabled = True
        self.max_vec_dim = 1536
        self.min_vec_dim = 64
        self.summary_layers = 3
        self.lambda_hot = 0.005       # Hot 메모리 느린 감쇠
        self.lambda_warm = 0.02       # 중간 감쇠
        self.lambda_cold = 0.05       # Cold 메모리 빠른 감쇠
        self.time_unit_ms = 86_400_000  # 1일
```

### 7.2 3계층 시스템

```python
def pick_tier(m: Dict, now_ts: int) -> str:
    """메모리를 hot/warm/cold 계층으로 분류"""
    dt = max(0, now_ts - (m["last_seen_at"] or m["updated_at"] or now_ts))
    recent = dt < 6 * 86_400_000  # 6일 이내

    high = (m.get("coactivations") or 0) > 5 or (m["salience"] or 0) > 0.7

    if recent and high:
        return "hot"    # 최근 접근 + 높은 중요도
    if recent or (m["salience"] or 0) > 0.4:
        return "warm"   # 최근이거나 중간 중요도
    return "cold"       # 오래되고 낮은 중요도
```

### 7.3 벡터 압축

```python
def compress_vector(vec: List[float], f: float, min_dim=64, max_dim=1536):
    """감쇠 계수에 따라 벡터 압축"""
    src = vec if vec else [1.0]

    # 감쇠 계수 기반 목표 차원
    tgt_dim = max(min_dim, min(max_dim, math.floor(len(src) * f)))
    dim = max(min_dim, min(len(src), tgt_dim))

    if dim >= len(src):
        return list(src)

    # 평균 풀링 압축
    pooled = []
    bucket = math.ceil(len(src) / dim)
    for i in range(0, len(src), bucket):
        sub = src[i:i + bucket]
        pooled.append(sum(sub) / len(sub))

    # 정규화
    normalize(pooled)
    return pooled
```

---

## 8. 시간 지식 그래프

### 8.1 시간 사실 스키마

```sql
CREATE TABLE temporal_facts (
    id TEXT PRIMARY KEY,
    subject TEXT,           # 엔티티 주어
    predicate TEXT,         # 관계 타입
    object TEXT,            # 엔티티 목적어
    valid_from INTEGER,     # 시작 타임스탬프 (ms)
    valid_to INTEGER,       # 종료 타임스탬프 (ms), 현재면 NULL
    confidence REAL,        # 0-1 신뢰도 점수
    last_updated INTEGER,
    metadata TEXT           # JSON
)
```

### 8.2 시점 쿼리

```python
# /temporal_graph/query.py

async def query_facts_at_time(subject=None, predicate=None, subject_object=None,
                               at=None, min_confidence=0.1):
    """특정 시점에 유효한 사실 쿼리"""
    ts = at if at else int(time.time() * 1000)

    conds = ["(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))"]
    params = [ts, ts]

    if subject:
        conds.append("subject = ?")
        params.append(subject)
    if predicate:
        conds.append("predicate = ?")
        params.append(predicate)

    sql = f"""
        SELECT * FROM temporal_facts
        WHERE {' AND '.join(conds)}
        ORDER BY confidence DESC, valid_from DESC
    """

    return db.fetchall(sql, tuple(params))
```

---

## 9. Reflection 시스템

### 9.1 Reflection 프로세스

```python
# /memory/reflect.py

async def run_reflection():
    """자동 Reflection 작업 - 유사한 메모리를 통합"""
    print("[REFLECT] Reflection 작업 시작...")

    min_mems = env.reflect_min or 20
    mems = q.all_mem(100, 0)

    if len(mems) < min_mems:
        return {"created": 0, "reason": "low"}

    # 유사한 메모리 클러스터링
    cls = cluster(mems)

    n = 0
    for c in cls:
        # 클러스터에 대한 요약 생성
        txt = summ(c)
        s = calc_sal(c)
        src = [m["id"] for m in c["mem"]]

        meta = {
            "type": "auto_reflect",
            "sources": src,
            "freq": c["n"],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        # 성찰 메모리 생성
        await add_hsg_memory(txt, json.dumps(["reflect:auto"]), meta)

        # 소스 메모리를 통합됨으로 표시
        await mark_consolidated(src)

        # 소스 메모리의 salience 부스트
        await boost(src)
        n += 1

    return {"created": n, "clusters": len(cls)}
```

---

## 10. MCP 서버 통합

### 10.1 사용 가능한 도구

```python
# /ai/mcp.py

TOOLS = [
    Tool(
        name="openmemory_query",
        description="OpenMemory에 대해 시맨틱 검색 실행",
        inputSchema={
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "user_id": {"type": "string"},
                "sector": {"type": "string"}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="openmemory_store",
        description="OpenMemory에 새 콘텐츠 저장",
        inputSchema={...}
    ),
    Tool(
        name="openmemory_get",
        description="ID로 단일 메모리 가져오기"
    ),
    Tool(
        name="openmemory_delete",
        description="ID로 메모리 삭제"
    ),
    Tool(
        name="openmemory_list",
        description="최근 메모리 목록"
    )
]
```

---

## 11. 다른 메모리 시스템과의 비교

| 기능 | OpenMemory | Mem0 | MemU | Memori |
|------|------------|------|------|--------|
| **메모리 섹터** | 5개 (Tulving 기반) | 3개 (User/Agent/Procedural) | 5타입 + 10카테고리 | 시맨틱 트리플 |
| **시간 모델링** | 전체 Temporal KG | 기본 타임스탬프 | 이벤트 메모리 | valid_from/valid_to |
| **Decay 시스템** | 3계층 + 압축 | 없음 | 없음 | 없음 |
| **연상 링크** | Waypoint 그래프 | Graph Memory (Neo4j) | 없음 | 지식 그래프 |
| **중복 제거** | SimHash | 예 (기본) | 없음 | 컨텍스트 인식 |
| **Reflection** | 자동 클러스터링 | 없음 | 카테고리 요약 | 없음 |
| **벡터 저장소** | SQLite/PostgreSQL/Valkey | 20+ 옵션 | 파일 기반 | 인메모리 |
| **MCP 통합** | 네이티브 | 예 | 없음 | 없음 |

---

## 12. 강점과 약점

### 강점

1. **포괄적 인지 모델**: 5섹터로 Tulving 메모리 이론 완전 구현
2. **정교한 Decay**: 3계층 시스템 + 벡터 압축으로 무한 성장 방지
3. **시간 지식 그래프**: 시간 쿼리와 사실 유효성에 대한 일급 지원
4. **Waypoint 기반 연상**: 그래프 기반 메모리 확장으로 창의적 검색 가능
5. **SimHash 중복 제거**: 효율적인 근접 중복 감지
6. **자동 Reflection**: 인간 개입 없는 자가 통합
7. **멀티모달 지원**: 네이티브 오디오/비디오/문서 수집
8. **MCP 네이티브**: 현대 AI 도구와의 깊은 통합

### 약점

1. **SQLite 기본**: 프로덕션 워크로드에 확장성 제한
2. **클라우드 서비스 없음**: 자체 호스팅만 가능
3. **복잡한 설정**: 튜닝할 파라미터가 많음
4. **제한된 그래프 연산**: 명시적 그래프 알고리즘 (PageRank 등) 없음
5. **LLM 기반 추출 없음**: 섹터 분류에 패턴 매칭만 사용

---

## 13. 사용 예시

```python
from openmemory import Memory

# 초기화
mem = Memory()

# 메모리 저장
result = await mem.add(
    "어제 웹 스크래퍼를 만들면서 Python 데코레이터를 배웠어",
    user_id="user123",
    tags=["python", "learning"],
    metadata={"source": "study_notes"}
)
# 출력: primary_sector="procedural", sectors=["procedural", "episodic"]

# 쿼리
results = await mem.search(
    "Python에 대해 뭘 배웠지?",
    user_id="user123",
    limit=5
)
# 하이브리드 스코어로 정렬된 메모리 반환

# 메모리 가져오기
mem_obj = mem.get(result["id"])

# 삭제
await mem.delete(result["id"])
```

---

## 14. 결론

OpenMemory는 AI 에이전트를 위한 인지 메모리 이론의 가장 포괄적인 오픈소스 구현 중 하나입니다. 멀티 섹터 아키텍처, 시간 지식 그래프, 정교한 감쇠/강화 메커니즘은 인간과 같은 메모리 능력을 가진 에이전트를 구축하기 위한 강력한 기반을 제공합니다.

**적합한 사용 사례:**
- 장기 메모리가 필요한 개인 AI 어시스턴트
- 시간에 따른 기억과 추론이 필요한 연구 에이전트
- 컨텍스트 지속성이 필요한 멀티 세션 챗봇
- 시간 요구사항이 있는 지식 관리 시스템

---

*분석 완료일: 2026-01-21*
