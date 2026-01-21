# OpenMemory (CaviraOSS) Analysis Report

> Repository: https://github.com/CaviraOSS/OpenMemory
> Analysis Date: 2026-01-21

---

## 1. Overview

OpenMemory is an open-source memory layer for AI agents and applications. It provides a sophisticated multi-sector memory system with temporal knowledge graphs, decay mechanisms, and associative memory links (waypoints).

### Key Features

- **Multi-Sector Memory**: 5 distinct cognitive sectors (Episodic, Semantic, Procedural, Emotional, Reflective)
- **Temporal Knowledge Graph**: Facts with validity periods and confidence decay
- **Hierarchical Semantic Graph (HSG)**: Core memory structure with waypoint-based associations
- **Decay Engine**: 3-tier decay system with vector compression
- **Reflection System**: Automatic memory consolidation and pattern detection
- **MCP Server**: Native integration with Claude, Cursor, Windsurf

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenMemory                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   MCP       │   │   REST      │   │  LangChain  │           │
│  │   Server    │   │   API       │   │  Connector  │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                  │                  │
│         └─────────────────┼──────────────────┘                  │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Memory Class                          │   │
│  │    add() / search() / get() / delete() / history()      │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│         ┌─────────────────────┼─────────────────────┐          │
│         ▼                     ▼                     ▼          │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐      │
│  │    HSG     │       │  Temporal  │       │   Decay    │      │
│  │  (Memory)  │◄─────►│   Graph    │◄─────►│   Engine   │      │
│  └─────┬──────┘       └─────┬──────┘       └─────┬──────┘      │
│        │                    │                    │              │
│        ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Vector Store                          │   │
│  │         (PostgreSQL/pgvector, Valkey, SQLite)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Sectors (5 Types)

OpenMemory implements Tulving's cognitive memory theory with 5 distinct sectors, each with specialized characteristics.

### 3.1 Sector Configuration

```python
# /core/constants.py

SECTOR_CONFIGS: Dict[str, SectorCfg] = {
    "episodic": {
        "model": "episodic-optimized",
        "decay_lambda": 0.015,      # Fast decay
        "weight": 1.2,
        "patterns": [...]
    },
    "semantic": {
        "model": "semantic-optimized",
        "decay_lambda": 0.005,      # Slow decay
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
        "decay_lambda": 0.02,       # Fastest decay
        "weight": 1.3,              # Highest weight
        "patterns": [...]
    },
    "reflective": {
        "model": "reflective-optimized",
        "decay_lambda": 0.001,      # Slowest decay
        "weight": 0.8,
        "patterns": [...]
    },
}
```

### 3.2 Sector Characteristics

| Sector | Decay Lambda | Weight | Description |
|--------|-------------|--------|-------------|
| **Episodic** | 0.015 | 1.2 | Personal events, experiences with temporal context |
| **Semantic** | 0.005 | 1.0 | Facts, concepts, general knowledge |
| **Procedural** | 0.008 | 1.1 | How-to knowledge, skills, processes |
| **Emotional** | 0.02 | 1.3 | Feelings, moods, affective states |
| **Reflective** | 0.001 | 0.8 | Insights, patterns, meta-cognition |

### 3.3 Sector Detection Patterns

```python
# Episodic patterns
re.compile(r"\b(today|yesterday|tomorrow|last\s+(week|month|year))\b", re.I)
re.compile(r"\b(remember\s+when|recall|that\s+time|when\s+I)\b", re.I)
re.compile(r"\b(went|saw|met|felt|heard|visited|attended)\b", re.I)

# Semantic patterns
re.compile(r"\b(is\s+a|represents|means|defined\s+as)\b", re.I)
re.compile(r"\b(concept|theory|principle|law|hypothesis)\b", re.I)
re.compile(r"\b(fact|statistic|data|evidence|proof)\b", re.I)

# Procedural patterns
re.compile(r"\b(how\s+to|step\s+by\s+step|guide|tutorial)\b", re.I)
re.compile(r"\b(first|second|then|next|finally)\b", re.I)
re.compile(r"\b(install|run|execute|compile|build|deploy)\b", re.I)

# Emotional patterns
re.compile(r"\b(feel|feeling|felt|emotions?|mood)\b", re.I)
re.compile(r"\b(happy|sad|angry|excited|scared|anxious)\b", re.I)
re.compile(r"[!]{2,}", re.I)  # Multiple exclamation marks

# Reflective patterns
re.compile(r"\b(realize|realized|realization|insight|epiphany)\b", re.I)
re.compile(r"\b(pattern|trend|connection|link|relationship)\b", re.I)
re.compile(r"\b(lesson|moral|takeaway|conclusion)\b", re.I)
```

### 3.4 Cross-Sector Relationships

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

## 4. Hierarchical Semantic Graph (HSG)

The HSG is the core memory structure that handles storage, retrieval, and association of memories.

### 4.1 Memory Storage Schema

```python
# /core/db.py - memories table

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    segment INTEGER,              # Segment-based storage
    content TEXT,                 # Stored content (essence extracted)
    simhash TEXT,                 # For deduplication
    primary_sector TEXT,          # Main sector classification
    tags TEXT,                    # JSON array of tags
    meta TEXT,                    # JSON metadata
    created_at INTEGER,
    updated_at INTEGER,
    last_seen_at INTEGER,         # For recency scoring
    salience REAL,                # Importance score (0-1)
    decay_lambda REAL,            # Sector-specific decay rate
    version INTEGER,
    mean_dim INTEGER,             # Mean vector dimension
    mean_vec BLOB,                # Mean embedding across sectors
    compressed_vec BLOB,          # Compressed vector for cold memories
    feedback_score INTEGER        # Coactivation count
)
```

### 4.2 Memory Addition Flow

```python
# /memory/hsg.py

async def add_hsg_memory(content, tags=None, metadata=None, user_id=None):
    # 1. Compute SimHash for deduplication
    simhash = compute_simhash(content)
    existing = db.fetchone("SELECT * FROM memories WHERE simhash=?", (simhash,))

    if existing and hamming_dist(simhash, existing["simhash"]) <= 3:
        # Duplicate found - boost salience instead of creating new
        boost = min(1.0, existing["salience"] + 0.15)
        db.execute("UPDATE memories SET salience=? WHERE id=?", (boost, existing["id"]))
        return {"id": existing["id"], "deduplicated": True}

    # 2. Classify content into sectors
    cls = classify_content(content, metadata)
    all_secs = [cls["primary"]] + cls["additional"]

    # 3. Extract essence (summarize if needed)
    stored = extract_essence(content, cls["primary"], max_length)

    # 4. Calculate initial salience
    init_sal = max(0.0, min(1.0, 0.4 + 0.1 * len(cls["additional"])))

    # 5. Insert memory record
    q.ins_mem(id=mid, content=stored, primary_sector=cls["primary"], ...)

    # 6. Generate multi-sector embeddings
    emb_res = await embed_multi_sector(mid, content, all_secs)
    for r in emb_res:
        await store.storeVector(mid, r["sector"], r["vector"], r["dim"], user_id)

    # 7. Calculate mean vector
    mean_vec = calc_mean_vec(emb_res, all_secs)

    # 8. Create waypoint (associative link)
    await create_single_waypoint(mid, mean_vec, now, user_id)

    return {"id": mid, "primary_sector": cls["primary"], "sectors": all_secs}
```

### 4.3 Content Classification

```python
def classify_content(content, metadata=None):
    # Check if sector is explicitly specified in metadata
    if metadata and metadata.get("sector"):
        return {"primary": metadata["sector"], "additional": [], "confidence": 1.0}

    # Score each sector based on pattern matches
    scores = {k: 0.0 for k in SECTOR_CONFIGS}
    for sec, cfg in SECTOR_CONFIGS.items():
        score = 0
        for pat in cfg["patterns"]:
            matches = pat.findall(content)
            if matches:
                score += len(matches) * cfg["weight"]
        scores[sec] = score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary, p_score = sorted_scores[0]

    # Secondary sectors above threshold
    thresh = max(1.0, p_score * 0.3)
    additional = [s for s, sc in sorted_scores[1:] if sc >= thresh]

    # Confidence based on score gap
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    confidence = p_score / (p_score + second_score + 1) if p_score > 0 else 0.2

    return {
        "primary": primary if p_score > 0 else "semantic",
        "additional": additional,
        "confidence": confidence
    }
```

### 4.4 SimHash Deduplication

```python
def compute_simhash(text: str) -> str:
    tokens = canonical_token_set(text)

    # Hash each token
    hashes = []
    for t in tokens:
        h = 0
        for c in t:
            h = (h << 5) - h + ord(c)
            h = h & 0xffffffff
        hashes.append(h)

    # Build 64-bit fingerprint
    vec = [0] * 64
    for h in hashes:
        for i in range(64):
            bit = 1 << (i % 32)
            if h & bit:
                vec[i] += 1
            else:
                vec[i] -= 1

    # Convert to hex string
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

## 5. Waypoint Graph (Associative Memory)

Waypoints create associative links between memories, enabling graph-based retrieval expansion.

### 5.1 Waypoint Schema

```sql
CREATE TABLE waypoints (
    src_id TEXT,
    dst_id TEXT,
    user_id TEXT,
    weight REAL,          # Link strength (0-1)
    created_at INTEGER,
    updated_at INTEGER,
    PRIMARY KEY (src_id, dst_id)
)
```

### 5.2 Waypoint Creation

```python
# Single waypoint - link to most similar existing memory
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

# Inter-memory waypoints - link memories above similarity threshold
async def create_inter_mem_waypoints(new_id, prim_sec, new_vec, ts, user_id):
    thresh = 0.75  # Similarity threshold
    vecs = await store.getVectorsBySector(prim_sec)

    nm = np.array(new_vec, dtype=np.float32)

    for vr in vecs:
        if vr["id"] == new_id:
            continue
        ex_vec = np.array(vr["vector"], dtype=np.float32)
        sim = cos_sim(nm, ex_vec)

        if sim >= thresh:
            # Bidirectional link
            db.execute("INSERT OR REPLACE INTO waypoints VALUES (...)")
```

### 5.3 Waypoint Expansion During Retrieval

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
            exp_wt = cur["weight"] * wt * 0.8  # Decay factor

            if exp_wt < 0.1:  # Pruning threshold
                continue

            item = {"id": dst, "weight": exp_wt, "path": cur["path"] + [dst]}
            exp.append(item)
            vis.add(dst)
            q_arr.append(item)
            cnt += 1

    return exp
```

### 5.4 Waypoint Reinforcement

```python
REINFORCEMENT = {
    "salience_boost": 0.1,
    "waypoint_boost": 0.05,
    "max_salience": 1.0,
    "max_waypoint_weight": 1.0,
    "prune_threshold": 0.05,
}

async def reinforce_waypoints(trav_path: List[str]):
    """Reinforce waypoints along a traversal path"""
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

## 6. Scoring System

### 6.1 Hybrid Scoring Weights

```python
# /memory/hsg.py

SCORING_WEIGHTS = {
    "similarity": 0.35,     # Vector similarity
    "overlap": 0.20,        # Token overlap
    "waypoint": 0.15,       # Waypoint weight
    "recency": 0.10,        # Time-based recency
    "tag_match": 0.20,      # Tag matching
}

HYBRID_PARAMS = {
    "tau": 3.0,             # Similarity boost factor
    "beta": 2.0,
    "eta": 0.1,
    "gamma": 0.2,           # Context boost coefficient
    "alpha_reinforce": 0.08,
    "t_days": 7.0,          # Recency half-life
    "t_max_days": 60.0,     # Max recency window
    "tau_hours": 1.0,
    "epsilon": 1e-8,
}
```

### 6.2 Hybrid Score Calculation

```python
def compute_hybrid_score(sim, tok_ov, wp_wt, rec_sc, kw_score=0, tag_match=0):
    # Boost similarity with sigmoid-like transform
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
    t = HYBRID_PARAMS["t_days"]       # 7 days
    tmax = HYBRID_PARAMS["t_max_days"]  # 60 days
    return math.exp(-days / t) * (1 - days / tmax)
```

### 6.3 Multi-Vector Fusion Score

```python
async def calc_multi_vec_fusion_score(mid, qe, w):
    """Calculate score across multiple sector vectors"""
    vecs = await store.getVectorsById(mid)
    s = 0.0
    tot = 0.0

    # Weight mapping from query classification
    wm = {
        "semantic": w.get("semantic_dimension_weight", 0),
        "emotional": w.get("emotional_dimension_weight", 0),
        "procedural": w.get("procedural_dimension_weight", 0),
        "episodic": w.get("temporal_dimension_weight", 0),
        "reflective": w.get("reflective_dimension_weight", 0),
    }

    for v in vecs:
        qv = qe.get(v.sector)
        if not qv:
            continue
        sim = cos_sim(v.vector, qv)
        wgt = wm.get(v.sector, 0.5)
        s += sim * wgt
        tot += wgt

    return s / tot if tot > 0 else 0.0
```

### 6.4 Cross-Sector Resonance

```python
# /ops/dynamics.py

SECTORAL_INTERDEPENDENCE_MATRIX = [
    #   epi   sem   pro   emo   ref
    [1.0, 0.7, 0.3, 0.6, 0.6],  # episodic
    [0.7, 1.0, 0.4, 0.7, 0.8],  # semantic
    [0.3, 0.4, 1.0, 0.5, 0.2],  # procedural
    [0.6, 0.7, 0.5, 1.0, 0.8],  # emotional
    [0.6, 0.8, 0.2, 0.8, 1.0],  # reflective
]

SECTOR_INDEX = {
    "episodic": 0,
    "semantic": 1,
    "procedural": 2,
    "emotional": 3,
    "reflective": 4,
}

async def calculateCrossSectorResonanceScore(ms, qs, bs):
    """Apply cross-sector penalty/boost based on sector relationship"""
    si = SECTOR_INDEX.get(ms, 1)
    ti = SECTOR_INDEX.get(qs, 1)
    return bs * SECTORAL_INTERDEPENDENCE_MATRIX[si][ti]
```

---

## 7. Decay Engine

### 7.1 Decay Configuration

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
        self.lambda_hot = 0.005       # Slow decay for hot memories
        self.lambda_warm = 0.02       # Medium decay
        self.lambda_cold = 0.05       # Fast decay for cold memories
        self.time_unit_ms = 86_400_000  # 1 day
```

### 7.2 Three-Tier System

```python
def pick_tier(m: Dict, now_ts: int) -> str:
    """Classify memory into hot/warm/cold tier"""
    dt = max(0, now_ts - (m["last_seen_at"] or m["updated_at"] or now_ts))
    recent = dt < 6 * 86_400_000  # Within 6 days

    high = (m.get("coactivations") or 0) > 5 or (m["salience"] or 0) > 0.7

    if recent and high:
        return "hot"    # Recently accessed + high importance
    if recent or (m["salience"] or 0) > 0.4:
        return "warm"   # Either recent or moderately important
    return "cold"       # Old and low importance
```

### 7.3 Decay Application

```python
async def apply_decay():
    """Main decay loop - runs periodically"""
    now_ts = int(time.time() * 1000)
    segments = db.fetchall("SELECT DISTINCT segment FROM memories")

    for seg in segments:
        rows = db.fetchall("SELECT * FROM memories WHERE segment=?", (seg,))

        # Sample batch for efficiency
        decay_ratio = 0.03
        batch_sz = max(1, int(len(rows) * decay_ratio))
        batch = random.sample(rows, batch_sz)

        for m in batch:
            tier = pick_tier(m, now_ts)

            # Select decay rate based on tier
            lam = {
                "hot": cfg.lambda_hot,    # 0.005
                "warm": cfg.lambda_warm,  # 0.02
                "cold": cfg.lambda_cold   # 0.05
            }[tier]

            # Time since last access
            dt = (now_ts - m["last_seen_at"]) / cfg.time_unit_ms

            # Activity boost
            act = max(0, m.get("coactivations") or 0)
            sal = max(0.0, min(1.0, m["salience"] * (1 + math.log1p(act))))

            # Exponential decay with activity adjustment
            f = math.exp(-lam * (dt / (sal + 0.1)))
            new_sal = max(0.0, min(1.0, sal * f))

            # Vector compression for highly decayed memories
            if f < 0.7:
                vec = await store.getVector(m["id"], m["primary_sector"])
                new_vec = compress_vector(vec, f, min_dim=64, max_dim=1536)
                await store.storeVector(m["id"], m["primary_sector"], new_vec)

            # Fingerprinting for very cold memories
            if f < cfg.cold_threshold:  # < 0.25
                fp = fingerprint_mem(m)
                await store.storeVector(m["id"], sector, fp["vector"])
                db.execute("UPDATE memories SET summary=? WHERE id=?",
                          (fp["summary"], m["id"]))

            db.execute("UPDATE memories SET salience=? WHERE id=?",
                      (new_sal, m["id"]))
```

### 7.4 Vector Compression

```python
def compress_vector(vec: List[float], f: float, min_dim=64, max_dim=1536):
    """Compress vector based on decay factor"""
    src = vec if vec else [1.0]

    # Target dimension based on decay factor
    tgt_dim = max(min_dim, min(max_dim, math.floor(len(src) * f)))
    dim = max(min_dim, min(len(src), tgt_dim))

    if dim >= len(src):
        return list(src)

    # Average pooling compression
    pooled = []
    bucket = math.ceil(len(src) / dim)
    for i in range(0, len(src), bucket):
        sub = src[i:i + bucket]
        pooled.append(sum(sub) / len(sub))

    # Normalize
    normalize(pooled)
    return pooled
```

### 7.5 Reinforcement on Query

```python
async def on_query_hit(mem_id: str, sector: str, reembed_fn=None):
    """Called when a memory is retrieved"""
    m = q.get_mem(mem_id)
    if not m:
        return

    # Regenerate compressed vectors if needed
    if cfg.regeneration_enabled and reembed_fn:
        vec_row = await store.getVector(mem_id, sector)
        if vec_row and len(vec_row.vector) <= 64:  # Very compressed
            base = m["summary"] or m["content"] or ""
            new_vec = await reembed_fn(base)
            await store.storeVector(mem_id, sector, new_vec)

    # Reinforce salience
    if cfg.reinforce_on_query:
        new_sal = min(1.0, (m["salience"] or 0.5) + 0.5)
        db.execute("UPDATE memories SET salience=?, last_seen_at=? WHERE id=?",
                  (new_sal, now, mem_id))
```

---

## 8. Temporal Knowledge Graph

### 8.1 Temporal Facts Schema

```sql
CREATE TABLE temporal_facts (
    id TEXT PRIMARY KEY,
    subject TEXT,           # Entity subject
    predicate TEXT,         # Relationship type
    object TEXT,            # Entity object
    valid_from INTEGER,     # Start timestamp (ms)
    valid_to INTEGER,       # End timestamp (ms), NULL if current
    confidence REAL,        # 0-1 confidence score
    last_updated INTEGER,
    metadata TEXT           # JSON
)

CREATE TABLE temporal_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT,         # Source fact ID
    target_id TEXT,         # Target fact ID
    relation_type TEXT,     # Edge type
    valid_from INTEGER,
    valid_to INTEGER,
    weight REAL,
    metadata TEXT
)
```

### 8.2 Fact Insertion with Temporal Logic

```python
# /temporal_graph/store.py

async def insert_fact(subject, predicate, subject_object, valid_from=None,
                      confidence=1.0, metadata=None, user_id=None):
    fact_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    valid_from_ts = valid_from if valid_from else now

    # Close existing facts with same subject-predicate
    existing = db.fetchall(
        "SELECT id, valid_from FROM temporal_facts "
        "WHERE subject=? AND predicate=? AND valid_to IS NULL",
        (subject, predicate)
    )

    for old in existing:
        if old["valid_from"] < valid_from_ts:
            # Close old fact just before new one starts
            db.execute("UPDATE temporal_facts SET valid_to=? WHERE id=?",
                       (valid_from_ts - 1, old["id"]))

    # Insert new fact
    db.execute(
        "INSERT INTO temporal_facts VALUES (?,?,?,?,?,NULL,?,?,?)",
        (fact_id, subject, predicate, subject_object, valid_from_ts,
         confidence, now, json.dumps(metadata))
    )

    return fact_id
```

### 8.3 Point-in-Time Queries

```python
# /temporal_graph/query.py

async def query_facts_at_time(subject=None, predicate=None, subject_object=None,
                               at=None, min_confidence=0.1):
    """Query facts valid at a specific point in time"""
    ts = at if at else int(time.time() * 1000)

    conds = ["(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))"]
    params = [ts, ts]

    if subject:
        conds.append("subject = ?")
        params.append(subject)
    if predicate:
        conds.append("predicate = ?")
        params.append(predicate)
    if min_confidence > 0:
        conds.append("confidence >= ?")
        params.append(min_confidence)

    sql = f"""
        SELECT * FROM temporal_facts
        WHERE {' AND '.join(conds)}
        ORDER BY confidence DESC, valid_from DESC
    """

    return db.fetchall(sql, tuple(params))

async def get_current_fact(subject: str, predicate: str):
    """Get the currently valid fact"""
    sql = """
        SELECT * FROM temporal_facts
        WHERE subject = ? AND predicate = ? AND valid_to IS NULL
        ORDER BY valid_from DESC LIMIT 1
    """
    return db.fetchone(sql, (subject, predicate))
```

### 8.4 Temporal Range Queries

```python
async def query_facts_in_range(subject=None, predicate=None,
                                start=None, end=None, min_confidence=0.1):
    """Query facts overlapping with a time range"""
    conds = []
    params = []

    if start and end:
        # Facts that overlap with [start, end]
        conds.append(
            "((valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)) "
            "OR (valid_from >= ? AND valid_from <= ?))"
        )
        params.extend([end, start, start, end])

    # ... additional filters

    return db.fetchall(sql, tuple(params))
```

### 8.5 Confidence Decay for Facts

```python
async def apply_confidence_decay(decay_rate: float = 0.01):
    """Apply time-based confidence decay to facts"""
    now = int(time.time() * 1000)
    one_day = 86400000

    sql = """
        UPDATE temporal_facts
        SET confidence = MAX(0.1, confidence * (1 - ? * ((? - valid_from) / ?)))
        WHERE valid_to IS NULL AND confidence > 0.1
    """
    db.execute(sql, (decay_rate, now, one_day))
```

---

## 9. Reflection System

### 9.1 Reflection Process

```python
# /memory/reflect.py

async def run_reflection():
    """Automatic reflection job - consolidates similar memories"""
    print("[REFLECT] Starting reflection job...")

    min_mems = env.reflect_min or 20
    mems = q.all_mem(100, 0)

    if len(mems) < min_mems:
        return {"created": 0, "reason": "low"}

    # Cluster similar memories
    cls = cluster(mems)

    n = 0
    for c in cls:
        # Generate summary for cluster
        txt = summ(c)
        s = calc_sal(c)
        src = [m["id"] for m in c["mem"]]

        meta = {
            "type": "auto_reflect",
            "sources": src,
            "freq": c["n"],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        # Create reflective memory
        await add_hsg_memory(txt, json.dumps(["reflect:auto"]), meta)

        # Mark source memories as consolidated
        await mark_consolidated(src)

        # Boost salience of source memories
        await boost(src)
        n += 1

    return {"created": n, "clusters": len(cls)}
```

### 9.2 Memory Clustering

```python
def cluster(mems: List[Dict]) -> List[Dict]:
    """Cluster similar memories using Jaccard similarity"""
    cls = []
    used = set()

    for m in mems:
        if m["id"] in used:
            continue
        if m["primary_sector"] == "reflective":  # Skip existing reflections
            continue
        if m.get("meta") and "consolidated" in str(m["meta"]):
            continue

        c = {"mem": [m], "n": 1}
        used.add(m["id"])

        for o in mems:
            if o["id"] in used:
                continue
            if m["primary_sector"] != o["primary_sector"]:
                continue

            # Jaccard similarity on tokens
            if sim_txt(m["content"], o["content"]) > 0.8:
                c["mem"].append(o)
                c["n"] += 1
                used.add(o["id"])

        if c["n"] >= 2:  # Minimum cluster size
            cls.append(c)

    return cls

def sim_txt(t1: str, t2: str) -> float:
    """Jaccard similarity between texts"""
    s1 = set(t1.lower().split())
    s2 = set(t2.lower().split())
    if not s1 or not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return inter / union if union > 0 else 0.0
```

### 9.3 Reflection Scheduling

```python
_timer_task = None

async def reflection_loop():
    interval = (env.reflect_interval or 10) * 60  # Default: 10 minutes
    while True:
        try:
            await run_reflection()
        except Exception as e:
            print(f"[REFLECT] Error: {e}")
        await asyncio.sleep(interval)

def start_reflection():
    global _timer_task
    if not env.get("auto_reflect", True) or _timer_task:
        return
    _timer_task = asyncio.create_task(reflection_loop())
    print(f"[REFLECT] Started: every {env.reflect_interval or 10}m")
```

---

## 10. Query System

### 10.1 HSG Query Flow

```python
# /memory/hsg.py

async def hsg_query(qt: str, k: int = 10, f: Dict = None):
    """Main query endpoint"""
    inc_q()  # Track active queries for decay coordination

    try:
        # 1. Check cache
        cache_key = f"{qt}:{k}:{json.dumps(f)}"
        if cache_key in cache:
            entry = cache[cache_key]
            if time.time() * 1000 - entry["t"] < TTL:
                return entry["r"]

        # 2. Classify query
        qc = classify_content(qt)
        qtk = canonical_token_set(qt)

        # 3. Embed query for all sectors
        ss = f.get("sectors") or list(SECTOR_CONFIGS.keys())
        qe = await embed_query_for_all_sectors(qt, ss)

        # 4. Dynamic weight based on query classification
        w = {
            "semantic_dimension_weight": 1.2 if qc["primary"] == "semantic" else 0.8,
            "emotional_dimension_weight": 1.5 if qc["primary"] == "emotional" else 0.6,
            "procedural_dimension_weight": 1.3 if qc["primary"] == "procedural" else 0.7,
            "temporal_dimension_weight": 1.4 if qc["primary"] == "episodic" else 0.7,
            "reflective_dimension_weight": 1.1 if qc["primary"] == "reflective" else 0.5,
        }

        # 5. Vector search in each sector
        sr = {}
        for s in ss:
            qv = qe[s]
            res = await store.search(qv, s, k * 3, {"user_id": f.get("user_id")})
            sr[s] = res

        # 6. Adaptive waypoint expansion
        all_sims = [r["similarity"] for res in sr.values() for r in res]
        avg_top = sum(all_sims) / len(all_sims) if all_sims else 0
        high_conf = avg_top >= 0.55

        ids = set(r["id"] for res in sr.values() for r in res)

        # Expand via waypoints if low confidence
        exp = []
        if not high_conf:
            exp = await expand_via_waypoints(list(ids), k * 2)
            for e in exp:
                ids.add(e["id"])

        # 7. Score and rank
        res_list = []
        for mid in ids:
            m = q.get_mem(mid)
            if not m:
                continue

            # Multi-vector fusion
            mvf = await calc_multi_vec_fusion_score(mid, qe, w)

            # Cross-sector resonance
            csr = await calculateCrossSectorResonanceScore(
                m["primary_sector"], qc["primary"], mvf
            )

            # Sector penalty
            penalty = SECTOR_RELATIONSHIPS.get(qc["primary"], {}).get(
                m["primary_sector"], 0.3
            )
            adj = best_sim * penalty

            # Compute hybrid score
            fs = compute_hybrid_score(adj, tok_ov, ww, rec_sc, kw_score, tag_match)

            res_list.append({
                "id": mid,
                "content": m["content"],
                "score": fs,
                "primary_sector": m["primary_sector"],
                "path": em["path"] if em else [mid],
                "salience": sal,
                "tags": json.loads(m["tags"] or "[]"),
                "metadata": json.loads(m["meta"] or "{}")
            })

        # 8. Sort and reinforce top results
        res_list.sort(key=lambda x: x["score"], reverse=True)
        top = res_list[:k]

        for r in top:
            # Apply retrieval trace reinforcement
            rsal = await applyRetrievalTraceReinforcementToMemory(r["id"], r["salience"])
            db.execute("UPDATE memories SET salience=?, last_seen_at=? WHERE id=?",
                      (rsal, now, r["id"]))

            # Propagate to linked nodes
            if len(r["path"]) > 1:
                wps = db.fetchall("SELECT dst_id, weight FROM waypoints WHERE src_id=?",
                                 (r["id"],))
                pru = await propagateAssociativeReinforcementToLinkedNodes(
                    r["id"], rsal, wps
                )
                for u in pru:
                    # Context boost for linked memories
                    linked_mem = q.get_mem(u["node_id"])
                    if linked_mem:
                        ctx_boost = HYBRID_PARAMS["gamma"] * (rsal - linked_mem["salience"])
                        new_sal = min(1.0, linked_mem["salience"] + ctx_boost)
                        db.execute("UPDATE memories SET salience=? WHERE id=?",
                                  (new_sal, u["node_id"]))

            # Trigger regeneration if needed
            await on_query_hit(r["id"], r["primary_sector"], ...)

        # 9. Cache and return
        cache[cache_key] = {"r": top, "t": time.time() * 1000}
        return top

    finally:
        dec_q()
```

---

## 11. MCP Server Integration

### 11.1 Available Tools

```python
# /ai/mcp.py

TOOLS = [
    Tool(
        name="openmemory_query",
        description="Run a semantic retrieval against OpenMemory",
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
        description="Persist new content into OpenMemory",
        inputSchema={
            "properties": {
                "content": {"type": "string"},
                "user_id": {"type": "string"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"}
            },
            "required": ["content"]
        }
    ),
    Tool(
        name="openmemory_get",
        description="Fetch a single memory by ID"
    ),
    Tool(
        name="openmemory_delete",
        description="Delete a memory by ID"
    ),
    Tool(
        name="openmemory_list",
        description="List recent memories"
    )
]
```

### 11.2 MCP Server Setup

```python
async def run_mcp_server():
    server = Server("openmemory-mcp")

    @server.list_tools()
    async def handle_list_tools():
        return TOOLS

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        if name == "openmemory_query":
            results = await mem.search(
                arguments["query"],
                user_id=arguments.get("user_id"),
                limit=arguments.get("k", 10)
            )
            return [TextContent(type="text", text=json.dumps(results))]

        elif name == "openmemory_store":
            res = await mem.add(
                arguments["content"],
                user_id=arguments.get("user_id"),
                meta=arguments.get("metadata", {})
            )
            return [TextContent(type="text", text=f"Stored: {res['id']}")]

        # ... other tools

    async with stdio_server() as (read, write):
        await server.run(read, write, NotificationOptions())
```

---

## 12. Multi-Modal Content Extraction

### 12.1 Supported Formats

```python
# /ops/extract.py

async def extract_text(content_type: str, data):
    ctype = content_type.lower()

    # Audio (Whisper transcription)
    if any(x in ctype for x in ["audio", "mp3", "wav", "m4a"]):
        return await extract_audio(data, ctype)

    # Video (FFmpeg + Whisper)
    if any(x in ctype for x in ["video", "mp4", "avi", "mov"]):
        return await extract_video(data)

    # PDF (pypdf)
    if "pdf" in ctype:
        return await extract_pdf(data)

    # DOCX (mammoth)
    if "docx" in ctype or "msword" in ctype:
        return await extract_docx(data)

    # HTML (markdownify)
    if "html" in ctype:
        return await extract_html(data)

    # Text/Markdown (passthrough)
    if any(x in ctype for x in ["markdown", "txt", "text"]):
        return {"text": data, "metadata": {...}}
```

### 12.2 Audio Transcription

```python
async def extract_audio(data: bytes, mime_type: str):
    if len(data) > 25 * 1024 * 1024:  # 25MB limit
        raise ValueError("Audio file too large")

    client = AsyncOpenAI(api_key=env.openai_api_key)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcription = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json"
        )

    return {
        "text": transcription.text,
        "metadata": {
            "content_type": "audio",
            "extraction_method": "whisper",
            "duration_seconds": transcription.duration,
            "language": transcription.language
        }
    }
```

---

## 13. Connectors

OpenMemory supports multiple data source connectors:

### 13.1 Available Connectors

| Connector | Description |
|-----------|-------------|
| `github` | GitHub repositories, issues, PRs |
| `google_drive` | Google Drive documents |
| `google_sheets` | Google Sheets data |
| `google_slides` | Google Slides content |
| `notion` | Notion pages and databases |
| `onedrive` | Microsoft OneDrive files |
| `web_crawler` | Web page crawling |
| `langchain` | LangChain integration |
| `agents` | Multi-agent coordination |

---

## 14. Comparison with Other Memory Systems

| Feature | OpenMemory | Mem0 | MemU | Memori |
|---------|------------|------|------|--------|
| **Memory Sectors** | 5 (Tulving-based) | 3 (User/Agent/Procedural) | 5 types + 10 categories | Semantic Triples |
| **Temporal Modeling** | Full Temporal KG | Basic timestamps | Event memory | valid_from/valid_to |
| **Decay System** | 3-tier with compression | No | No | No |
| **Associative Links** | Waypoint graph | Graph Memory (Neo4j) | No | Knowledge Graph |
| **Deduplication** | SimHash | Yes (basic) | No | Context-aware |
| **Reflection** | Auto-clustering | No | Category summaries | No |
| **Vector Store** | SQLite/PostgreSQL/Valkey | 20+ options | File-based | In-memory |
| **MCP Integration** | Native | Yes | No | No |

---

## 15. Key Algorithms Summary

### 15.1 Memory Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Lifecycle                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input ──► SimHash ──► Classify ──► Extract ──► Embed ──►    │
│                  │          │            │          │          │
│              Dedup?     Sectors      Essence    Multi-Vec      │
│                  │          │            │          │          │
│                  ▼          ▼            ▼          ▼          │
│            ┌─────────────────────────────────────────┐         │
│            │              Memory Store               │         │
│            └────────────────────┬────────────────────┘         │
│                                 │                              │
│         ┌───────────────────────┼───────────────────────┐      │
│         ▼                       ▼                       ▼      │
│   ┌───────────┐           ┌───────────┐           ┌──────────┐ │
│   │ Waypoints │           │   Decay   │           │ Reflect  │ │
│   │  (Links)  │           │  Engine   │           │  System  │ │
│   └───────────┘           └───────────┘           └──────────┘ │
│         │                       │                       │      │
│         ▼                       ▼                       ▼      │
│   Association           Tier Classification       Clustering   │
│   Creation              Vector Compression        Consolidation│
│   Reinforcement         Fingerprinting            Boosting     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 15.2 Query Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     Query Lifecycle                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ──► Classify ──► Multi-Embed ──► Vector Search ──►     │
│                 │              │               │                │
│              Primary       Per-Sector      Per-Sector          │
│              Sector        Embedding        Search             │
│                 │              │               │                │
│                 ▼              ▼               ▼                │
│            ┌─────────────────────────────────────────┐         │
│            │            Initial Results              │         │
│            └────────────────────┬────────────────────┘         │
│                                 │                              │
│                    ┌────────────┴────────────┐                 │
│                    │   Confidence Check      │                 │
│                    │   (avg_sim >= 0.55?)    │                 │
│                    └────────────┬────────────┘                 │
│                          │             │                       │
│                        High          Low                       │
│                          │             │                       │
│                          │      ┌──────┴──────┐                │
│                          │      │  Waypoint   │                │
│                          │      │  Expansion  │                │
│                          │      └──────┬──────┘                │
│                          │             │                       │
│                          └──────┬──────┘                       │
│                                 │                              │
│                    ┌────────────┴────────────┐                 │
│                    │    Hybrid Scoring       │                 │
│                    │  (sim + overlap + wp    │                 │
│                    │   + recency + tags)     │                 │
│                    └────────────┬────────────┘                 │
│                                 │                              │
│                    ┌────────────┴────────────┐                 │
│                    │    Reinforcement        │                 │
│                    │  (salience + waypoint   │                 │
│                    │   + linked nodes)       │                 │
│                    └────────────┬────────────┘                 │
│                                 │                              │
│                                 ▼                              │
│                           Top K Results                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 16. Strengths and Weaknesses

### Strengths

1. **Comprehensive Cognitive Model**: Full implementation of Tulving's memory theory with 5 sectors
2. **Sophisticated Decay**: 3-tier system with vector compression prevents unbounded growth
3. **Temporal Knowledge Graph**: First-class support for temporal queries and fact validity
4. **Waypoint-based Association**: Graph-based memory expansion enables creative retrieval
5. **SimHash Deduplication**: Efficient near-duplicate detection
6. **Automatic Reflection**: Self-consolidation without human intervention
7. **Multi-modal Support**: Native audio/video/document ingestion
8. **MCP Native**: Deep integration with modern AI tools

### Weaknesses

1. **SQLite Default**: May not scale for production workloads
2. **No Cloud Service**: Self-hosted only
3. **Complex Configuration**: Many parameters to tune
4. **Limited Graph Operations**: No explicit graph algorithms (PageRank, etc.)
5. **No LLM-based Extraction**: Pattern-matching only for sector classification

---

## 17. Usage Example

```python
from openmemory import Memory

# Initialize
mem = Memory()

# Store memory
result = await mem.add(
    "Yesterday I learned Python decorators while building a web scraper",
    user_id="user123",
    tags=["python", "learning"],
    metadata={"source": "study_notes"}
)
# Output: primary_sector="procedural", sectors=["procedural", "episodic"]

# Query
results = await mem.search(
    "What did I learn about Python?",
    user_id="user123",
    limit=5
)
# Returns memories sorted by hybrid score

# Get memory
mem_obj = mem.get(result["id"])

# Delete
await mem.delete(result["id"])
```

---

## 18. Conclusion

OpenMemory represents one of the most comprehensive open-source implementations of cognitive memory theory for AI agents. Its multi-sector architecture, temporal knowledge graph, and sophisticated decay/reinforcement mechanisms provide a strong foundation for building agents with human-like memory capabilities.

The system excels at:
- Long-term memory management with automatic consolidation
- Temporal reasoning ("What did I know in March?")
- Associative retrieval through waypoint graphs
- Efficient deduplication and compression

It's particularly well-suited for:
- Personal AI assistants with long-term memory
- Research agents that need to remember and reason over time
- Multi-session chatbots requiring context persistence
- Knowledge management systems with temporal requirements

---

*Analysis completed on 2026-01-21*
