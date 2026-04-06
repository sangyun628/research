# kg-gen: 텍스트 기반 지식 그래프 생성 라이브러리 분석 보고서

## 1. 프로젝트 개요

### 1.1 기본 정보
- **프로젝트명**: kg-gen (Knowledge Graph Generation)
- **버전**: 0.4.0
- **저자**: Belinda Mo (STAIR Lab)
- **라이선스**: MIT
- **GitHub**: https://github.com/stair-lab/kg-gen
- **논문**: [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://arxiv.org/abs/2502.09956)

### 1.2 목적
kg-gen은 **일반 텍스트에서 AI를 사용하여 지식 그래프(Knowledge Graph)를 자동으로 추출**하는 Python 라이브러리입니다. 주요 사용 사례:

- RAG(Retrieval-Augmented Generation)를 위한 그래프 생성
- 모델 훈련/테스트를 위한 그래프 합성 데이터 생성
- 텍스트 구조화
- 개념 간 관계 분석

### 1.3 핵심 기능
- 텍스트에서 엔티티(Entity) 추출
- 엔티티 간 관계(Relation) 추출 (Subject-Predicate-Object 트리플)
- 대용량 텍스트 청킹(Chunking) 처리
- 유사 엔티티/관계 클러스터링 및 중복 제거
- 대화(Messages) 형식 지원
- 시각화 도구 (인터랙티브 HTML)
- MCP(Model Context Protocol) 서버 지원 (AI Agent 메모리용)

---

## 2. 아키텍처 구조

### 2.1 프로젝트 디렉토리 구조
```
kg-gen/
├── src/kg_gen/
│   ├── __init__.py
│   ├── kg_gen.py           # 메인 KGGen 클래스
│   ├── models.py           # 데이터 모델 (Graph)
│   ├── cli.py              # CLI 인터페이스
│   ├── steps/              # 파이프라인 단계별 모듈
│   │   ├── _1_get_entities.py    # 엔티티 추출
│   │   ├── _2_get_relations.py   # 관계 추출
│   │   └── _3_deduplicate.py     # 중복 제거
│   ├── prompts/            # LLM 프롬프트 템플릿
│   │   ├── entities.txt
│   │   └── relations.txt
│   └── utils/              # 유틸리티 모듈
│       ├── chunk_text.py       # 텍스트 청킹
│       ├── deduplicate.py      # SemHash 중복제거
│       ├── llm_deduplicate.py  # LLM 기반 중복제거
│       ├── visualize_kg.py     # 시각화
│       └── neo4j_integration.py # Neo4j 연동
├── mcp/
│   └── server.py           # MCP 서버 (AI Agent 메모리)
├── experiments/            # 벤치마크 실험
├── tests/                  # 테스트 코드
└── pyproject.toml          # 패키지 설정
```

### 2.2 처리 파이프라인

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Input Text │───▶│   Chunking  │───▶│  Entity     │───▶│  Relation   │
│  (or Chat)  │    │ (Optional)  │    │  Extraction │    │  Extraction │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐
│   Output    │◀───│  Aggregate  │◀───│    Deduplication        │
│   Graph     │    │  (Optional) │    │  (SemHash + LLM-based)  │
└─────────────┘    └─────────────┘    └─────────────────────────┘
```

---

## 3. 핵심 컴포넌트 상세 분석

### 3.1 데이터 모델 (`models.py`)

```python
class Graph(BaseModel):
    entities: set[str]                          # 모든 엔티티 집합
    edges: set[str]                             # 모든 관계 유형(predicate) 집합
    relations: set[Tuple[str, str, str]]        # (subject, predicate, object) 트리플 집합
    entity_clusters: Optional[dict[str, set[str]]]  # 엔티티 클러스터링 결과
    edge_clusters: Optional[dict[str, set[str]]]    # 관계 클러스터링 결과
    entity_metadata: dict[str, set[str]] | None     # 엔티티 메타데이터
```

**핵심 특징**:
- Pydantic `BaseModel` 기반으로 데이터 검증
- 모든 컬렉션이 `set` 자료형 → 자동 중복 방지
- JSON 직렬화/역직렬화 지원 (`from_file`, `to_file`)

### 3.2 메인 클래스 (`kg_gen.py`)

#### KGGen 클래스 초기화
```python
class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",        # LiteLLM 형식
        max_tokens: int = 16000,
        temperature: float = 0.0,
        reasoning_effort: str = None,         # GPT 추론 강도
        api_key: str = None,
        api_base: str = None,                 # 커스텀 API 엔드포인트
        retrieval_model: Optional[str] = None, # 임베딩 모델
        disable_cache: bool = False,
    )
```

**지원 모델** (LiteLLM 라우팅):
- `openai/gpt-4o`, `openai/gpt-5`
- `gemini/gemini-2.5-flash`
- `ollama_chat/deepseek-r1:14b`
- `anthropic/claude-3-...` 등

#### generate() 메서드 핵심 로직
```python
def generate(self, input_data, chunk_size=None, deduplication_method=...) -> Graph:
    # 1. 입력 처리 (문자열 or 대화 메시지 배열)
    if isinstance(input_data, list):  # 대화 형식
        processed_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_data])
    else:
        processed_input = input_data

    # 2. 청킹 (대용량 텍스트)
    if chunk_size:
        chunks = chunk_text(processed_input, chunk_size)
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor() as executor:
            for chunk in chunks:
                entities, relations = _process(chunk)
                all_entities.update(entities)
                all_relations.update(relations)

    # 3. 엔티티/관계 추출
    entities = get_entities(content, is_conversation)
    relations = get_relations(content, entities, is_conversation)

    # 4. 중복 제거
    if deduplication_method:
        graph = self.deduplicate(graph, method=deduplication_method)

    return Graph(entities=entities, relations=relations, edges={r[1] for r in relations})
```

---

## 4. 내부 구현 상세

### 4.1 엔티티 추출 (`_1_get_entities.py`)

#### DSPy Signature 기반 구현
```python
class TextEntities(dspy.Signature):
    """Extract key entities from the source text."""
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

def get_entities(input_data: str, is_conversation: bool = False) -> List[str]:
    extract = dspy.Predict(TextEntities if not is_conversation else ConversationEntities)
    result = extract(source_text=input_data)
    return result.entities
```

#### LiteLLM 직접 호출 방식 (no_dspy 옵션)
```python
def _get_entities_litellm(input_data: str, model: str, ...) -> List[str]:
    # 프롬프트 템플릿 로드
    prompt_template = _load_entities_prompt()  # prompts/entities.txt

    # Structured Output (JSON Schema)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    response = litellm.responses(
        model=model,
        input=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"<article>{input_data}</article>"}
        ],
        text={"format": {"type": "json_schema", "name": "entities_response", "schema": schema, "strict": True}}
    )
    return EntitiesResponse.model_validate_json(response.output[-1].content[0].text).entities
```

#### 엔티티 추출 프롬프트 (`prompts/entities.txt`)
- **엔티티 카테고리**: 인물, 조직, 장소, 날짜, 이벤트, 창작물, 개념, 제품
- **추출 가이드라인**:
  - 실질적인 언급에 집중 (단순 언급 제외)
  - 관계 가능성이 있는 엔티티 우선
  - 공식/일반 명칭 사용
  - 너무 일반적인 용어 제외
  - 전체 이름 사용 (예: "Marie Curie")
  - 정규화 (예: "USA" → "United States")

### 4.2 관계 추출 (`_2_get_relations.py`)

#### 동적 Pydantic 모델 생성
```python
def _create_relations_model(entities: List[str]):
    # 추출된 엔티티로 Literal 타입 생성 → 잘못된 엔티티 참조 방지
    EntityLiteral = Literal[tuple(entities)]

    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),   # 엔티티 리스트 내의 값만 허용
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )

    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )
    return RelationItem, RelationsResponse
```

#### 폴백 메커니즘
```python
def get_relations(input_data, entities, is_conversation=False) -> List[Tuple[str, str, str]]:
    try:
        # 1차 시도: 엄격한 타입의 엔티티 제약
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        return [(r.subject, r.predicate, r.object) for r in result.relations]
    except Exception:
        # 폴백: 느슨한 타입 + ChainOfThought로 수정
        Relation, ExtractRelations = fallback_extraction_sig(entities, is_conversation)
        result = dspy.Predict(ExtractRelations)(source_text=input_data, entities=entities)

        # FixedRelations: 잘못된 subject/object를 수정
        fix = dspy.ChainOfThought(FixedRelations)
        fix_res = fix(source_text=input_data, entities=entities, relations=result.relations)

        return [(r.subject, r.predicate, r.object)
                for r in fix_res.fixed_relations
                if r.subject in entities and r.object in entities]
```

#### 관계 추출 프롬프트 (`prompts/relations.txt`)
- **형식**: (Subject | Predicate | Object) 트리플
- **가이드라인**:
  - 명시적/암시적 관계만 추출
  - 구체적인 predicate 사용 ("founded_by", "located_in" 등)
  - 올바른 방향성 유지
  - 철저함 (고립된 엔티티 최소화)
- **분석 프로세스**:
  1. 엔티티를 텍스트에 매핑
  2. 체계적으로 관계 식별
  3. 초기 트리플 초안 작성
  4. 고립된 엔티티 확인
  5. 최종 트리플 확정

### 4.3 텍스트 청킹 (`chunk_text.py`)

```python
def chunk_text(text: str, max_chunk_size=500) -> list[str]:
    # NLTK 문장 토크나이저 사용
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 문장이 너무 긴 경우 단어 단위로 분할
            if len(sentence) > max_chunk_size:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                        temp_chunk += word + " "
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                current_chunk = sentence + " "

    return chunks
```

**핵심 특징**:
- NLTK `sent_tokenize`로 문장 경계 유지
- 문장이 `max_chunk_size`보다 큰 경우 단어 단위 폴백
- 병렬 처리를 위한 독립적 청크 생성

### 4.4 중복 제거 시스템 (`_3_deduplicate.py`)

#### 중복 제거 방법 (DeduplicateMethod)
```python
class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"   # 결정적 규칙 + 시맨틱 해싱
    LM_BASED = "lm_based" # KNN 클러스터링 + 클러스터 내 LLM 중복제거
    FULL = "full"         # SEMHASH + LM_BASED 조합
```

#### (1) SemHash 기반 중복제거 (`deduplicate.py`)
```python
class DeduplicateList:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.inflect_engine = inflect.engine()  # 복수형/단수형 변환

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)  # 유니코드 정규화

    def singularize(self, text: str) -> str:
        # 각 토큰을 단수형으로 변환
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) else tok)
        return " ".join(tokens)

    def deduplicate(self, items: list[str]) -> list[str]:
        # 1. 정규화 + 단수화
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            normalized_items.add(singular)

        # 2. SemHash 라이브러리로 시맨틱 중복 제거
        semhash = SemHash.from_records(records=list(normalized_items))
        result = semhash.self_deduplicate(threshold=self.threshold)

        # 3. 원본 문자열로 매핑
        return [self.items_map[item] for item in result.selected]
```

**SemHash 작동 원리**:
- MinHash + Locality Sensitive Hashing (LSH) 기반
- 코사인 유사도 임계값(기본 0.95)으로 중복 판단
- O(n log n) 복잡도로 대규모 데이터 처리 가능

#### (2) LLM 기반 중복제거 (`llm_deduplicate.py`)
```python
class LLMDeduplicate:
    def __init__(self, retrieval_model: SentenceTransformer, lm: dspy.LM, graph: Graph):
        self.retrieval_model = retrieval_model
        self.lm = lm

        # 임베딩 생성
        self.node_embeddings = retrieval_model.encode(list(graph.entities))
        self.edge_embeddings = retrieval_model.encode(list(graph.edges))

        # BM25 인덱스 구축
        self.node_bm25 = BM25Okapi([text.lower().split() for text in self.nodes])

    def get_relevant_items(self, query: str, top_k: int = 50, type: str = "node"):
        """BM25 + 임베딩 Rank Fusion"""
        # BM25 점수
        bm25_scores = self.node_bm25.get_scores(query.lower().split())

        # 임베딩 유사도
        query_embedding = self.retrieval_model.encode([query])
        embedding_scores = cosine_similarity(query_embedding, self.node_embeddings).flatten()

        # 점수 융합 (50:50 가중치)
        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [self.nodes[i] for i in top_indices]

    def cluster(self):
        """KMeans 클러스터링"""
        cluster_size = 128
        n_samples = len(self.node_embeddings)
        num_clusters = max(1, n_samples // cluster_size)

        kmeans = KMeans(n_clusters=num_clusters, init="random", n_init=1, max_iter=20)
        kmeans.fit(self.node_embeddings.astype(np.float32))

        # 클러스터 크기 제한 (최대 128개)
        # ...

    def deduplicate_cluster(self, cluster: list[str], type: str = "node"):
        """클러스터 내 LLM 기반 중복제거"""
        items = set()
        item_clusters = {}

        while len(cluster) > 0:
            item = cluster.pop()
            relevant_items = self.get_relevant_items(item, 16, type)

            # DSPy로 중복 찾기
            class Deduplicate(dspy.Signature):
                """Find duplicate entities for the item..."""
                item: str = dspy.InputField()
                set: list[str] = dspy.InputField()
                duplicates: list[str] = dspy.OutputField()
                alias: str = dspy.OutputField()  # 대표 이름

            result = dspy.Predict(Deduplicate)(item=item, set=relevant_items)
            items.add(result.alias)

            for dup in result.duplicates:
                if dup in cluster:
                    cluster.remove(dup)
                    item_clusters[result.alias].add(dup)

        return items, item_clusters
```

**LLM 중복제거 파이프라인**:
1. **KMeans 클러스터링**: 임베딩 기반으로 유사 항목 그룹화
2. **클러스터 내 검색**: BM25 + 임베딩 Rank Fusion으로 관련 항목 검색
3. **LLM 판단**: DSPy Signature로 중복 여부 및 대표 별칭 결정
4. **병렬 처리**: `ThreadPoolExecutor(max_workers=64)`로 클러스터별 병렬 처리

---

## 5. 부가 기능

### 5.1 그래프 시각화 (`visualize_kg.py`)

```python
def visualize(graph: Graph, output_path: str, open_in_browser: bool = False):
    # 뷰 모델 구축
    view_model = _build_view_model(graph)

    # 노드/엣지 통계 계산
    stats = {
        "entities": len(entities),
        "relations": len(edges_view),
        "relationTypes": len(predicate_counts),
        "entityClusters": len(cluster_view),
        "isolatedEntities": len(isolated_entities),
        "components": len(components),  # 연결 컴포넌트
        "averageDegree": ...,
        "density": ...,
    }

    # HTML 템플릿에 JSON 데이터 삽입
    html = HTML_TEMPLATE.replace("<!--DATA-->", json.dumps(view_model))

    destination.write_text(html, encoding="utf-8")
```

**시각화 특징**:
- 인터랙티브 HTML 대시보드
- Force-directed 그래프 레이아웃 (추정)
- 노드 색상: 해시 기반 결정적 색상 생성
- 클러스터 시각화, Top 엔티티/관계 테이블
- 연결 컴포넌트 분석

### 5.2 MCP 서버 (`mcp/server.py`)

AI Agent를 위한 **영구 메모리 시스템**:

```python
mcp = FastMCP(name="KGGen")

@mcp.tool
def add_memories(text: str) -> str:
    """텍스트에서 메모리 추출 및 저장"""
    new_graph = kg_gen_instance.generate(input_data=text)
    memory_graph = kg_gen_instance.aggregate([memory_graph, new_graph])
    save_memory_graph()  # JSON 파일로 영속화

@mcp.tool
def retrieve_relevant_memories(query: str) -> str:
    """쿼리와 관련된 메모리 검색"""
    # 키워드 기반 검색
    relevant_entities = [e for e in memory_graph.entities if query.lower() in e.lower()]
    relevant_relations = [r for r in memory_graph.relations if any(query.lower() in str(part).lower() for part in r)]

@mcp.tool
def visualize_memories(output_filename: str = "memory_graph.html") -> str:
    """메모리 그래프 시각화"""
    KGGen.visualize(memory_graph, output_path)

@mcp.tool
def get_memory_stats() -> str:
    """메모리 통계 조회"""
```

**MCP 설정**:
```bash
# 환경 변수
KG_MODEL=openai/gpt-4o
KG_API_KEY=your_api_key
KG_STORAGE_PATH=./kg_memory.json
KG_CLEAR_MEMORY=true

# 실행
kggen mcp  # 기본값: 메모리 초기화
kggen mcp --keep-memory  # 기존 메모리 유지
```

### 5.3 그래프 검색 (RAG 지원)

```python
def retrieve(self, query: str, node_embeddings: dict, graph: nx.DiGraph, k: int = 8):
    """쿼리에 관련된 노드와 컨텍스트 검색"""
    # 1. 임베딩 유사도로 Top-k 노드 검색
    top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)

    # 2. 각 노드의 이웃 관계 수집 (depth=2)
    context = set()
    for node, _ in top_nodes:
        node_context = self.retrieve_context(node, graph, depth=2)
        context.update(node_context)

    return top_nodes, context, " ".join(context)

def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2):
    """노드 주변 관계 수집"""
    context = set()
    def explore_neighbors(current_node, current_depth):
        if current_depth > depth:
            return
        for neighbor in graph.neighbors(current_node):
            rel = graph[current_node][neighbor]["relation"]
            context.add(f"{current_node} {rel} {neighbor}.")
            explore_neighbors(neighbor, current_depth + 1)
        for neighbor in graph.predecessors(current_node):
            rel = graph[neighbor][current_node]["relation"]
            context.add(f"{neighbor} {rel} {current_node}.")
    explore_neighbors(node, 1)
    return list(context)
```

---

## 6. 기술 스택

### 6.1 핵심 의존성
| 라이브러리 | 용도 | 버전 |
|-----------|------|------|
| **dspy-ai** | LLM 프레임워크 (Structured Output) | ≥3.0.4 |
| **LiteLLM** | 다중 LLM 프로바이더 라우팅 | (dspy 내장) |
| **pydantic** | 데이터 검증 및 모델링 | ≥2.0.0 |
| **sentence-transformers** | 텍스트 임베딩 | ≥5.1.0 |
| **semhash** | 시맨틱 해싱 기반 중복제거 | ≥0.3.2 |
| **nltk** | 자연어 처리 (문장 분리) | - |
| **scikit-learn** | KMeans 클러스터링, 코사인 유사도 | ≥1.7.2 |
| **rank-bm25** | BM25 검색 알고리즘 | ≥0.2.2 |
| **networkx** | 그래프 자료구조 | ≥3.0 |
| **inflect** | 단수/복수 변환 | ≥7.5.0 |
| **fastmcp** | MCP 서버 프레임워크 | ≥2.10.6 |
| **neo4j** | Neo4j 데이터베이스 연동 | ≥5.0.0 |

### 6.2 DSPy 활용 패턴
```python
# Signature 정의
class TextEntities(dspy.Signature):
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="...")

# Predict (단순 호출)
extract = dspy.Predict(TextEntities)
result = extract(source_text=input_data)

# ChainOfThought (추론 과정 포함)
fix = dspy.ChainOfThought(FixedRelations)
result = fix(source_text=input_data, entities=entities, relations=relations)

# Context 관리
with dspy.context(lm=self.lm):
    result = dspy.Predict(...)(...)
```

---

## 7. 사용 예시

### 7.1 기본 사용법
```python
from kg_gen import KGGen

kg = KGGen(model="openai/gpt-4o", api_key="YOUR_KEY")

# 단순 텍스트
text = "Linda is Josh's mother. Ben is Josh's brother."
graph = kg.generate(input_data=text, context="Family relationships")

print(graph.entities)   # {'Linda', 'Ben', 'Josh'}
print(graph.relations)  # {('Linda', 'is mother of', 'Josh'), ...}
```

### 7.2 대용량 텍스트 처리
```python
with open('large_document.txt') as f:
    large_text = f.read()

graph = kg.generate(
    input_data=large_text,
    chunk_size=5000,            # 5000자 단위 청킹
    deduplication_method=DeduplicateMethod.FULL  # 완전 중복제거
)
```

### 7.3 대화 처리
```python
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."}
]
graph = kg.generate(input_data=messages)
# entities: {'user', 'assistant', 'Paris', 'France'}
# relations: {('France', 'has capital', 'Paris'), ...}
```

### 7.4 그래프 병합 및 시각화
```python
graph1 = kg.generate(input_data=text1)
graph2 = kg.generate(input_data=text2)

combined = kg.aggregate([graph1, graph2])
clustered = kg.deduplicate(combined, method=DeduplicateMethod.FULL)

KGGen.visualize(clustered, "output.html", open_in_browser=True)
```

---

## 8. 설계 특징 및 장점

### 8.1 강점
1. **다중 LLM 지원**: LiteLLM을 통해 OpenAI, Anthropic, Google, Ollama 등 다양한 모델 활용
2. **Structured Output**: DSPy + Pydantic으로 안정적인 JSON 출력 보장
3. **스케일러블**: 청킹 + 병렬 처리로 대용량 텍스트 처리
4. **지능적 중복제거**: SemHash(빠름) + LLM(정확함) 하이브리드 방식
5. **유연한 입력**: 일반 텍스트와 대화 형식 모두 지원
6. **통합 시각화**: 별도 도구 없이 인터랙티브 HTML 생성
7. **MCP 지원**: AI Agent 메모리 시스템으로 활용 가능

### 8.2 아키텍처 결정
- **파이프라인 분리**: 엔티티 추출 → 관계 추출 → 중복제거 순차 처리
- **폴백 메커니즘**: 관계 추출 실패 시 ChainOfThought로 보정
- **동적 타입 생성**: `Literal[tuple(entities)]`로 추출된 엔티티만 참조 가능하도록 제약
- **캐싱**: DSPy LM 레벨 캐싱으로 반복 호출 최적화

### 8.3 개선 가능 영역
- 메모리 검색이 단순 키워드 매칭 (임베딩 기반 검색으로 개선 가능)
- 관계 방향성 검증 로직 부재
- 실시간 스트리밍 미지원

---

## 9. 결론

kg-gen은 **텍스트에서 지식 그래프를 자동 추출**하는 실용적인 도구입니다. DSPy와 LiteLLM을 활용하여 다양한 LLM을 통합하고, SemHash와 LLM 기반 하이브리드 중복제거로 품질 높은 그래프를 생성합니다. MCP 서버 지원으로 AI Agent의 메모리 시스템으로도 활용 가능하며, 연구 및 프로덕션 환경 모두에서 사용할 수 있는 완성도 높은 라이브러리입니다.

---

**작성일**: 2026-01-23
**분석 대상 버전**: kg-gen 0.4.0
**분석자**: Claude (Anthropic)
