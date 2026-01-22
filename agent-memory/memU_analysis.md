# MemU 에이전트 메모리 시스템 분석 보고서

## 1. 개요

**MemU**는 LLM 및 AI 에이전트 백엔드를 위한 **Agentic Memory Framework**입니다. 멀티모달 입력(대화, 문서, 이미지, 비디오, 오디오)을 받아 구조화된 메모리로 추출하고, **계층적 파일 시스템** 형태로 구성하여 **임베딩 기반(RAG)** 및 **비임베딩(LLM)** 검색을 모두 지원합니다.

- **GitHub**: https://github.com/NevaMind-AI/memU
- **라이선스**: Apache 2.0
- **Python 버전**: 3.13+
- **현재 버전**: 1.2.0

### 1.1 핵심 특징

| 기능 | 설명 |
|------|------|
| **계층적 파일 시스템** | Resource → Item → Category 3계층 아키텍처, 완전한 추적성 |
| **이중 검색 방식** | RAG(임베딩 기반) + LLM(시맨틱 이해) |
| **멀티모달 지원** | 대화, 문서, 이미지, 오디오, 비디오 처리 |
| **자기 진화 메모리** | 사용 패턴에 따라 메모리 구조 적응 및 개선 |

---

## 2. 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         MemoryService                            │
│  ┌─────────────────┬──────────────────┬───────────────────┐     │
│  │  MemorizeMixin  │  RetrieveMixin   │    CRUDMixin      │     │
│  └────────┬────────┴────────┬─────────┴─────────┬─────────┘     │
│           │                 │                   │                │
│  ┌────────▼─────────────────▼───────────────────▼────────┐      │
│  │                   Workflow Pipeline                    │      │
│  │   ┌─────────────────────────────────────────────┐     │      │
│  │   │  WorkflowStep → WorkflowStep → WorkflowStep │     │      │
│  │   └─────────────────────────────────────────────┘     │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                      Database Layer                    │      │
│  │   ┌────────────┬────────────────┬───────────────┐     │      │
│  │   │  InMemory  │   PostgreSQL   │    SQLite     │     │      │
│  │   └────────────┴────────────────┴───────────────┘     │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │                     LLM Layer                          │      │
│  │   ┌──────────────┬──────────────┬──────────────┐      │      │
│  │   │ OpenAI SDK   │   HTTP/httpx │  OpenRouter  │      │      │
│  │   └──────────────┴──────────────┴──────────────┘      │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 3계층 메모리 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     Resource (원시 데이터)                    │
│  • URL, 모달리티, 로컬 경로, 캡션, 임베딩 저장                  │
│  • 예: JSON 대화 로그, 텍스트 문서, 이미지, 비디오              │
└────────────────────────────┬────────────────────────────────┘
                             │ 추출
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemoryItem (메모리 단위)                   │
│  • 개별 선호도, 스킬, 의견, 습관 등                           │
│  • 메모리 타입: profile, event, knowledge, behavior, skill   │
│  • 요약 텍스트 + 임베딩 벡터 저장                             │
└────────────────────────────┬────────────────────────────────┘
                             │ 분류
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 MemoryCategory (집계된 메모리)                │
│  • 텍스트 요약이 포함된 집계 메모리                           │
│  • 예: preferences.md, work_life.md, relationships.md        │
│  • 자동 요약 생성 및 업데이트                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트 상세 분석

### 3.1 MemoryService (src/memu/app/service.py)

메인 서비스 클래스로, 모든 메모리 작업을 조정합니다.

```python
class MemoryService(MemorizeMixin, RetrieveMixin, CRUDMixin):
    def __init__(
        self,
        llm_profiles: LLMProfilesConfig,      # LLM 설정 프로필
        blob_config: BlobConfig,              # 파일 저장소 설정
        database_config: DatabaseConfig,      # 데이터베이스 설정
        memorize_config: MemorizeConfig,      # 메모리화 설정
        retrieve_config: RetrieveConfig,      # 검색 설정
        workflow_runner: WorkflowRunner,      # 워크플로우 실행기
        user_config: UserConfig,              # 사용자 모델 설정
    )
```

**주요 기능:**
- **LLM 클라이언트 관리**: 프로필별 지연 초기화, 캐싱
- **파이프라인 관리**: memorize, retrieve_rag, retrieve_llm 등 등록
- **인터셉터 시스템**: LLM 호출 전/후, 워크플로우 스텝 전/후 훅
- **파이프라인 커스터마이징**: 스텝 삽입, 교체, 제거 API

### 3.2 데이터 모델 (src/memu/database/models.py)

```python
# 기본 레코드 인터페이스
class BaseRecord(BaseModel):
    id: str           # UUID
    created_at: datetime
    updated_at: datetime

# 원시 리소스
class Resource(BaseRecord):
    url: str
    modality: str           # conversation, document, image, video, audio
    local_path: str
    caption: str | None
    embedding: list[float] | None

# 메모리 아이템
class MemoryItem(BaseRecord):
    resource_id: str | None
    memory_type: MemoryType  # profile, event, knowledge, behavior, skill
    summary: str
    embedding: list[float] | None
    happened_at: datetime | None
    extra: dict[str, Any]

# 메모리 카테고리
class MemoryCategory(BaseRecord):
    name: str
    description: str
    embedding: list[float] | None
    summary: str | None       # LLM이 생성한 요약

# 카테고리-아이템 관계
class CategoryItem(BaseRecord):
    item_id: str
    category_id: str
```

**스코프 모델 병합:**
```python
def build_scoped_models(user_model: type[BaseModel]) -> tuple:
    """사용자 스코프 모델과 기본 레코드를 병합하여 멀티테넌시 지원"""
```

### 3.3 워크플로우 시스템

#### 3.3.1 WorkflowStep (src/memu/workflow/step.py)

```python
@dataclass
class WorkflowStep:
    step_id: str                    # 고유 식별자
    role: str                       # 역할 (ingest, preprocess, extract, ...)
    handler: WorkflowHandler        # 실행 함수
    description: str = ""
    requires: set[str]              # 필요한 상태 키
    produces: set[str]              # 생성하는 상태 키
    capabilities: set[str]          # 필요 기능 (llm, vector, db, io, vision)
    config: dict[str, Any]          # 설정 (llm_profile 등)
```

#### 3.3.2 PipelineManager (src/memu/workflow/pipeline.py)

```python
class PipelineManager:
    """워크플로우 파이프라인 관리 및 검증"""

    def register(name, steps, initial_state_keys)  # 파이프라인 등록
    def build(name) -> list[WorkflowStep]          # 실행용 복사본 생성
    def config_step(name, step_id, configs)        # 스텝 설정 변경
    def insert_after(name, target, new_step)       # 스텝 삽입
    def insert_before(name, target, new_step)
    def replace_step(name, target, new_step)       # 스텝 교체
    def remove_step(name, target)                  # 스텝 제거
```

**리비전 관리**: 모든 변경은 새 리비전으로 저장되어 추적 가능

### 3.4 Memorize 파이프라인 (src/memu/app/memorize.py)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memorize Workflow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ingest_resource      리소스 가져오기 (파일 다운로드/읽기)      │
│         ↓                                                        │
│  2. preprocess_multimodal 멀티모달 전처리 (이미지/비디오/오디오)    │
│         ↓                                                        │
│  3. extract_items        메모리 아이템 추출 (LLM)                 │
│         ↓                                                        │
│  4. dedupe_merge         중복 제거 및 병합                        │
│         ↓                                                        │
│  5. categorize_items     카테고리 분류 + 임베딩 생성               │
│         ↓                                                        │
│  6. persist_index        DB 저장 + 카테고리 요약 업데이트          │
│         ↓                                                        │
│  7. build_response       응답 구성                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**멀티모달 전처리:**
```python
async def _preprocess_resource_url(local_path, text, modality, llm_client):
    """모달리티별 전처리"""
    # conversation: 대화 세그먼트 분할 + 요약
    # document: 텍스트 압축 + 캡션 추출
    # image: Vision API로 설명 추출
    # video: 프레임 추출 + Vision 분석
    # audio: 음성 인식(Whisper) + 텍스트 처리
```

**메모리 타입별 프롬프트:**
- `profile`: 사용자 프로필 정보 (직업, 나이, 취미 등)
- `event`: 이벤트/사건 정보
- `knowledge`: 지식/사실 정보
- `behavior`: 행동 패턴
- `skill`: 스킬/능력

### 3.5 Retrieve 파이프라인 (src/memu/app/retrieve.py)

#### RAG 방식 (method="rag")
```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Retrieve Workflow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. route_intention        검색 필요성 판단 + 쿼리 재작성         │
│         ↓                                                        │
│  2. route_category         카테고리 검색 (코사인 유사도)           │
│         ↓                                                        │
│  3. sufficiency_after_cat  충분성 체크 (더 검색 필요?)            │
│         ↓                                                        │
│  4. recall_items           아이템 검색 (벡터 검색)                │
│         ↓                                                        │
│  5. sufficiency_after_item 충분성 체크                           │
│         ↓                                                        │
│  6. recall_resources       리소스 검색                           │
│         ↓                                                        │
│  7. build_context          결과 조합                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### LLM 방식 (method="llm")
```
동일 구조, 벡터 검색 대신 LLM이 직접 랭킹 수행
- 더 깊은 시맨틱 이해
- 쿼리 재작성 지원
- 비용/지연 시간 상승
```

**Sufficiency Check (충분성 검사):**
```python
# 검색 결과가 충분한지 LLM이 판단
# 불충분하면 쿼리를 재작성하여 다음 계층 검색 진행
async def _decide_if_retrieval_needed(query, context, retrieved_content):
    # RETRIEVE: 더 검색 필요
    # NO_RETRIEVE: 검색 불필요 (일상 대화, 인사 등)
```

### 3.6 벡터 검색 (src/memu/database/inmemory/vector.py)

```python
def cosine_topk(query_vec, corpus, k=5):
    """
    최적화된 코사인 유사도 기반 Top-K 검색

    - NumPy 벡터화 연산 사용
    - O(n) argpartition으로 Top-K 선택 (전체 정렬 O(n log n) 회피)
    """
    q = np.array(query_vec)
    matrix = np.array(vecs)

    # 벡터화된 코사인 유사도 계산
    scores = matrix @ q / (vec_norms * q_norm + 1e-9)

    # 효율적인 Top-K 선택
    topk_indices = np.argpartition(scores, -k)[-k:]
    return [(ids[i], scores[i]) for i in topk_indices]
```

### 3.7 LLM 클라이언트 래퍼 (src/memu/llm/wrapper.py)

```python
class LLMClientWrapper:
    """LLM 클라이언트 래퍼 - 인터셉터, 사용량 추적, 오류 처리"""

    async def summarize(text, max_tokens, system_prompt)
    async def vision(prompt, image_path, ...)
    async def embed(inputs: list[str])
    async def transcribe(audio_path, ...)
```

**인터셉터 시스템:**
```python
class LLMInterceptorRegistry:
    """LLM 호출 전/후/오류 인터셉터 등록"""

    def register_before(fn, name, priority, where)   # 호출 전
    def register_after(fn, name, priority, where)    # 호출 후
    def register_on_error(fn, name, priority, where) # 오류 시

# 사용 예:
service.intercept_after_llm_call(
    lambda ctx, req, resp, usage: log_usage(usage),
    where={"operation": "memorize"}
)
```

**사용량 추적:**
```python
@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None   # 캐시된 토큰
    reasoning_tokens: int | None      # 추론 토큰
    latency_ms: float | None
    finish_reason: str | None
```

---

## 4. 설정 시스템 (src/memu/app/settings.py)

### 4.1 LLM 프로필 설정

```python
class LLMConfig(BaseModel):
    provider: str = "openai"           # openai, grok, openrouter
    base_url: str = "https://api.openai.com/v1"
    api_key: str
    chat_model: str = "gpt-4o-mini"
    client_backend: str = "sdk"        # sdk 또는 httpx
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 1

class LLMProfilesConfig:
    """여러 LLM 프로필 관리 - default, embedding 등"""
```

### 4.2 데이터베이스 설정

```python
class DatabaseConfig(BaseModel):
    metadata_store: MetadataStoreConfig  # inmemory, postgres, sqlite
    vector_index: VectorIndexConfig      # bruteforce, pgvector, none

class MetadataStoreConfig(BaseModel):
    provider: Literal["inmemory", "postgres", "sqlite"] = "inmemory"
    ddl_mode: Literal["create", "validate"] = "create"
    dsn: str | None = None

class VectorIndexConfig(BaseModel):
    provider: Literal["bruteforce", "pgvector", "none"] = "bruteforce"
```

### 4.3 검색 설정

```python
class RetrieveConfig(BaseModel):
    method: Literal["rag", "llm"] = "rag"
    route_intention: bool = True          # 의도 라우팅 활성화
    sufficiency_check: bool = True        # 충분성 검사 활성화

    category: RetrieveCategoryConfig      # top_k=5
    item: RetrieveItemConfig              # top_k=5
    resource: RetrieveResourceConfig      # top_k=5
```

### 4.4 메모리 카테고리 기본값

```python
DEFAULT_CATEGORIES = [
    {"name": "personal_info", "description": "Personal information"},
    {"name": "preferences", "description": "User preferences"},
    {"name": "relationships", "description": "Relationships with others"},
    {"name": "activities", "description": "Activities, hobbies"},
    {"name": "goals", "description": "Goals, aspirations"},
    {"name": "experiences", "description": "Past experiences"},
    {"name": "knowledge", "description": "Facts, learned info"},
    {"name": "opinions", "description": "Viewpoints, perspectives"},
    {"name": "habits", "description": "Routines, patterns"},
    {"name": "work_life", "description": "Work-related info"},
]
```

---

## 5. 메모리 그룹화 및 계층화 상세

### 5.1 메모리 그룹화 단위

MemU는 메모리를 **3가지 차원**으로 그룹화합니다:

#### 5.1.1 메모리 타입 (Memory Type) - 의미적 분류

| 타입 | 설명 | 추출 대상 | 제외 대상 |
|------|------|----------|----------|
| **profile** | 사용자 프로필 정보 | 직업, 나이, 취미, 성격 등 장기 안정적 특성 | 이벤트, 임시 상태, 일회성 행동 |
| **event** | 특정 시점의 사건/경험 | 시간/장소가 특정된 활동, 계획, 경험 | 습관, 선호도, 일반 지식 |
| **knowledge** | 학습된 지식/사실 | 개념, 정의, 객관적 사실 | 개인 의견, 경험, 선호도 |
| **behavior** | 행동 패턴/루틴 | 반복되는 행동, 문제 해결 방식 | 일회성 행동, 프로필 정보 |
| **skill** | 스킬/능력 | 기술, 역량, 학습된 능력 | 일반 지식, 이벤트 |

```
대화 예시:
"퇴근 후에 요리하는 걸 좋아해요. 30살이고, 다음 주에 여행 갈 예정이에요."

추출 결과:
├── profile: "사용자는 30살이다", "사용자는 퇴근 후 요리를 좋아한다"
├── event: "사용자가 다음 주에 여행할 예정이다"
└── behavior: "사용자는 퇴근 후 직접 요리한다"
```

#### 5.1.2 메모리 카테고리 (Memory Category) - 주제별 분류

카테고리는 메모리 아이템을 **주제별로 그룹화**하여 요약을 생성합니다:

```python
DEFAULT_CATEGORIES = [
    "personal_info",    # 개인 정보 (이름, 나이, 직업 등)
    "preferences",      # 선호도 (좋아하는 것, 싫어하는 것)
    "relationships",    # 관계 (가족, 친구, 동료)
    "activities",       # 활동 (취미, 여가)
    "goals",           # 목표 (계획, 포부)
    "experiences",     # 경험 (과거 사건, 여행)
    "knowledge",       # 지식 (학습한 내용)
    "opinions",        # 의견 (관점, 견해)
    "habits",          # 습관 (루틴, 패턴)
    "work_life",       # 직장 생활 (업무, 커리어)
]
```

**카테고리 → 아이템 매핑:**
- 하나의 아이템이 **여러 카테고리에 속할 수 있음**
- 예: "사용자가 가족과 등산을 좋아한다" → `relationships`, `activities`

#### 5.1.3 리소스 (Resource) - 원본 데이터 그룹

```
Resource (conv1.json)
├── MemoryItem 1 (profile) ──┬── Category: personal_info
├── MemoryItem 2 (profile) ──┼── Category: preferences
├── MemoryItem 3 (event) ────┼── Category: experiences
└── MemoryItem 4 (behavior) ─┴── Category: habits
```

### 5.2 계층화 전략

#### 5.2.1 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                         원시 입력                                    │
│  (대화 JSON, 문서, 이미지, 비디오, 오디오)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼ [전처리]
┌─────────────────────────────────────────────────────────────────────┐
│                      전처리된 텍스트                                  │
│  • 대화: 세그먼트 분할 (20+ 메시지 단위)                              │
│  • 이미지/비디오: Vision API 설명 추출                               │
│  • 오디오: Whisper 음성인식 → 텍스트                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼ [메모리 타입별 추출]
┌─────────────────────────────────────────────────────────────────────┐
│                    MemoryItem (개별 메모리)                          │
│  • 각 메모리 타입별 LLM 프롬프트로 병렬 추출                          │
│  • 임베딩 벡터 생성                                                  │
│  • 카테고리 자동 분류                                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼ [카테고리별 집계]
┌─────────────────────────────────────────────────────────────────────┐
│                 MemoryCategory (집계 요약)                           │
│  • 각 카테고리별 Markdown 요약 자동 생성                              │
│  • 새 아이템 추가 시 요약 점진적 업데이트                             │
│  • 목표 길이(기본 400토큰) 내로 압축                                  │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 카테고리 요약 업데이트 로직

```python
# 카테고리 요약 업데이트 흐름
async def _update_category_summaries(updates, ctx, store, llm_client):
    for category_id, new_memories in updates.items():
        category = store.get_category(category_id)

        # 기존 요약 + 새 메모리 → LLM으로 병합
        prompt = build_summary_prompt(
            category=category.name,
            original_content=category.summary,  # 기존 요약
            new_items=new_memories,             # 새로 추가된 메모리들
            target_length=400                   # 목표 토큰 수
        )

        # 업데이트된 요약 생성
        new_summary = await llm_client.summarize(prompt)
        store.update_category(category_id, summary=new_summary)
```

#### 5.2.3 대화 세그먼트 분할

긴 대화는 자동으로 **의미 단위로 세그먼트 분할**됩니다:

```
입력: 100개 메시지의 대화

분할 기준:
├── 주제 변경 (Topic Change)
├── 시간 간격 (Time Gap)
├── 대화 자연스러운 종료
└── 톤/의미 초점 변화

결과:
├── Segment 1: 메시지 0-25 (주제: 일상 대화)
├── Segment 2: 메시지 26-55 (주제: 여행 계획)
└── Segment 3: 메시지 56-100 (주제: 업무 논의)

각 세그먼트는 최소 20개 메시지 포함
```

### 5.3 검색 시 계층 활용

```
쿼리: "사용자의 여행 계획이 뭐야?"

검색 흐름:
1️⃣ Category 검색 (요약 기반)
   └── experiences 카테고리 매칭 (score: 0.85)

2️⃣ 충분성 검사 → "더 상세한 정보 필요"

3️⃣ Item 검색 (임베딩 기반)
   ├── "사용자가 다음 주에 제주도 여행 예정" (score: 0.92)
   └── "사용자가 여행 짐을 아직 안 쌌다" (score: 0.78)

4️⃣ 충분성 검사 → "충분함"

5️⃣ 결과 반환 (Resource 검색 생략)
```

---

## 6. 주요 프롬프트 상세

### 6.1 메모리 추출 프롬프트

#### 6.1.1 Profile 추출 프롬프트

**목적**: 사용자의 장기적/안정적 특성 추출 (직업, 나이, 취미, 성격 등)

```
# Task Objective
You are a professional User Memory Extractor. Your core task is to extract
independent user memory items about the user (e.g., basic info, preferences,
habits, other long-term stable traits).

# Workflow
1. Read the full conversation to understand topics and meanings.
2. Extract memories: Select turns that contain valuable User Information.
3. Review & validate: Merge semantically similar items, resolve contradictions.
4. Final output: Output User Information.

# Rules
## General requirements
- Use "user" to refer to the user consistently.
- Each memory item must be complete and self-contained.
- Each memory item must be < 30 words.
- A single memory item must NOT contain timestamps.

## Special rules for User Information
- Any event-related item is forbidden in User Information.
- Do not extract content from assistant's follow-up questions.

## Forbidden content
- Knowledge Q&A without a clear user fact.
- Trivial updates (e.g., "full → too full").
- Turns where only the assistant spoke.
- Illegal/harmful sensitive topics.
- Private financial accounts, IDs, addresses.

# Output Format (XML)
<profile>
    <memory>
        <content>User memory item content</content>
        <categories>
            <category>Category Name</category>
        </categories>
    </memory>
</profile>
```

**예시 입/출력:**
```
입력:
user: 퇴근하고 장 보러 가는 중이에요.
assistant: 직접 요리하세요?
user: 네, 건강에 좋잖아요. 인터넷 회사에서 PM으로 일하고 있어요.
     올해 30살이에요. 퇴근 후에 요리 실험하는 걸 좋아해요.

출력:
<profile>
    <memory>
        <content>The user works as a PM at an internet company</content>
        <categories><category>Basic Information</category></categories>
    </memory>
    <memory>
        <content>The user is 30 years old</content>
        <categories><category>Basic Information</category></categories>
    </memory>
    <memory>
        <content>The user likes experimenting with cooking after work</content>
        <categories><category>Basic Information</category></categories>
    </memory>
</profile>

제외된 것: "다음 주 여행 계획" (이벤트), "장 보러 가는 중" (일시적 상태)
```

#### 6.1.2 Event 추출 프롬프트

**목적**: 특정 시점의 사건, 경험, 활동 추출

```
# Task Objective
Extract specific events and experiences that happened to or involved the user.

# Special rules for Event Information
- Behavioral patterns, habits, preferences are forbidden in Event Information.
- Focus on concrete happenings with time/place references.
- Include relevant details: time, location, participants.

# Rules
- Each memory item must be < 50 words.
- Focus on specific events at a particular time or period.
```

**예시:**
```
입력: "다음 주에 여행 가요. 아직 짐도 안 쌌어요."

출력:
<events>
    <memory>
        <content>The user is planning a trip next weekend and hasn't packed yet</content>
        <categories><category>Travel</category></categories>
    </memory>
</events>
```

#### 6.1.3 Knowledge 추출 프롬프트

**목적**: 학습된 지식, 개념, 사실 정보 추출

```
# Task Objective
Extract factual knowledge, concepts, definitions, and information
that the user learned or discussed.

# Special rules for Knowledge Information
- Personal opinions, preferences are forbidden.
- Focus on objective facts, concepts, explanations.
- User-specific traits, events are not knowledge items.
```

**예시:**
```
입력:
user: Python 데코레이터가 뭐예요?
assistant: 다른 함수를 감싸서 기능을 확장하는 함수예요.
user: 아, @ 기호가 문법적 설탕이군요.

출력:
<knowledge>
    <memory>
        <content>In Python, a decorator extends a function's behavior without modifying it</content>
        <categories><category>Programming</category></categories>
    </memory>
    <memory>
        <content>The @ symbol in Python is syntactic sugar for applying decorators</content>
        <categories><category>Programming</category></categories>
    </memory>
</knowledge>
```

#### 6.1.4 Behavior 추출 프롬프트

**목적**: 반복되는 행동 패턴, 루틴, 문제 해결 방식 추출

```
# Task Objective
Extract behavioral patterns, routines, and solutions that characterize
how the user acts or behaves.

# Special rules for Behavior Information
- One-time actions are forbidden unless they demonstrate a pattern.
- Focus on recurring patterns, typical approaches, established routines.
```

**예시:**
```
입력: "퇴근 후에 요리해요. 배달보다 낫죠. 자주 새 요리 실험해요."

출력:
<behaviors>
    <memory>
        <content>The user typically cooks after work instead of ordering takeout</content>
        <categories><category>Daily Routine</category></categories>
    </memory>
    <memory>
        <content>The user often experiments with cooking new dishes</content>
        <categories><category>Daily Routine</category></categories>
    </memory>
</behaviors>
```

### 6.2 카테고리 요약 프롬프트

**목적**: 기존 카테고리 요약에 새 메모리 아이템을 병합하여 업데이트

```
# Task Objective
You are a professional User Profile Synchronization Specialist.
Merge newly extracted user information items into the user's profile
using only two operations: add and update.

# Workflow
## Step 1: Preprocessing
- Parse initial profile and new items.
- Mark each new item as Add or Update.
- Remove invalid items: vague, one-off events.

## Step 2: Core Operations
A. Update
- Conflict detection: semantic overlap check.
- Validity priority: keep more specific, clearer information.
- Overwrite outdated entries.

B. Add
- Deduplication check.
- Category matching.
- Insert following original style.

## Step 3: Merge & Formatting
- Structured ordering by category.
- Markdown format (# title, ## category).
- No contradictions or duplicates.

## Step 4: Summarize
- Target length: {target_length} tokens.
- Cluster items and update category names.

# Output Format (Markdown)
# {category}
## <category name>
- User information item
- User information item
```

**예시:**
```
기존 요약:
# Personal Basic Information
## Basic Information
- The user is 28 years old
- The user lives in Beijing

새 메모리:
- The user is 30 years old
- The user lives in Shanghai
- The user ate Malatang today  ← 제외됨 (일회성)

결과:
# Personal Basic Information
## Basic Information
- The user is 30 years old    ← 업데이트
- The user lives in Shanghai  ← 업데이트
```

### 6.3 검색 관련 프롬프트

#### 6.3.1 검색 의도 판단 (Pre-Retrieval Decision)

**목적**: 쿼리가 메모리 검색이 필요한지 판단하고, 필요시 쿼리 재작성

```
# Task Objective
Determine whether the current query requires retrieving information
from memory or can be answered directly without retrieval.

# Rules
- NO_RETRIEVE for:
  - Greetings, casual chat
  - Questions about only the current conversation
  - General knowledge questions
  - Requests for clarification

- RETRIEVE for:
  - Questions about past events, conversations
  - Queries about user preferences, habits
  - Requests to recall specific information

# Output Format
<decision>RETRIEVE or NO_RETRIEVE</decision>
<rewritten_query>Context-aware rewritten query</rewritten_query>
```

#### 6.3.2 카테고리 랭킹 (LLM Category Ranker)

```
# Task Objective
Search through categories and identify the most relevant ones,
then rank them by relevance.

# Rules
- Only include actually relevant categories.
- Include at most {top_k} categories.
- Ranking matters: first = most relevant.
- Do not invent category IDs.

# Output Format
{
  "analysis": "reasoning process",
  "categories": ["category_id_1", "category_id_2"]
}
```

#### 6.3.3 아이템 랭킹 (LLM Item Ranker)

```
# Task Objective
Search through memory items within relevant categories
and identify the most relevant ones.

# Rules
- Only consider items in provided relevant categories.
- Include at most {top_k} items.
- Order matters: first = most relevant.

# Output Format
{
  "analysis": "reasoning process",
  "items": ["item_id_1", "item_id_2"]
}
```

### 6.4 멀티모달 전처리 프롬프트

#### 6.4.1 대화 세그먼트 분할

```
# Task Objective
Divide conversation into meaningful segments based on:
- Topic changes
- Time gaps or pauses
- Natural conclusions
- Shifts in tone or semantic focus

# Rules
- Each segment must contain ≥ 20 messages.
- Maintain coherent theme per segment.
- Clear boundary from adjacent segments.
- No overlapping segments.

# Output Format
{
    "segments": [
        {"start": 0, "end": 25},
        {"start": 26, "end": 55}
    ]
}
```

#### 6.4.2 이미지 분석

```
# Task Objective
Analyze the image and produce:
1. Detailed Description - thorough explanation
2. Caption - one-sentence summary

# Workflow
1. Identify main subjects and objects.
2. Describe actions and activities.
3. Analyze setting and environment.
4. Note visible text, signs, labels.
5. Describe colors, lighting, composition.
6. Infer mood and atmosphere.

# Output Format
<detailed_description>...</detailed_description>
<caption>One sentence summary</caption>
```

#### 6.4.3 비디오 분석

```
# Task Objective
Analyze video and produce:
1. Detailed Description - comprehensive explanation
2. Caption - one-sentence summary

# Workflow
1. Watch video from start to finish.
2. Identify main actions over time.
3. Describe key objects and people.
4. Analyze scene and setting.
5. Note audio elements (dialogue, music).
6. Highlight important events.
7. Describe temporal flow.

# Output Format
<detailed_description>...</detailed_description>
<caption>One sentence summary</caption>
```

#### 6.4.4 문서 압축

```
# Task Objective
Produce:
1. Condensed version - preserve key info, remove verbosity
2. Caption - one-sentence summary

# Rules
- Preserve all key information and conclusions.
- Do not introduce new information.
- Caption must be exactly one sentence.
```

---

## 7. 통합 및 확장

### 7.1 LangGraph 통합

```python
from memu.integrations.langgraph import MemULangGraphTools

memu_tools = MemULangGraphTools(memory_service)
tools = memu_tools.tools()

# save_memory: 정보 저장
# search_memory: 메모리 검색
```

### 7.2 파이프라인 커스터마이징

```python
# 새 스텝 삽입
service.insert_after(
    target_step_id="extract_items",
    new_step=WorkflowStep(
        step_id="custom_validation",
        handler=my_validation_fn,
        requires={"resource_plans"},
        produces={"validated_plans"},
    ),
    pipeline="memorize"
)

# 스텝 설정 변경
service.configure_pipeline(
    step_id="preprocess_multimodal",
    configs={"chat_llm_profile": "gpt-4"},
    pipeline="memorize"
)
```

### 7.3 지원 LLM 프로바이더

| 프로바이더 | 설명 |
|-----------|------|
| OpenAI | 기본 프로바이더 |
| Grok (X.AI) | 자동 설정 변경 |
| OpenRouter | 다중 프로바이더 라우팅 |
| Doubao | 바이트댄스 모델 |
| 커스텀 | base_url 지정으로 호환 API |

---

## 8. 성능 최적화 기법

### 8.1 지연 초기화
```python
# LLM 클라이언트 첫 사용 시에만 초기화
def _get_llm_base_client(profile):
    if name in self._llm_clients:
        return self._llm_clients[name]
    client = self._init_llm_client(cfg)
    self._llm_clients[name] = client
    return client
```

### 8.2 배치 임베딩
```python
# 여러 텍스트를 한 번에 임베딩
item_embeddings = await client.embed(summary_payloads)
```

### 8.3 병렬 LLM 호출
```python
# 메모리 타입별 병렬 추출
tasks = [client.summarize(prompt) for prompt in prompts]
responses = await asyncio.gather(*tasks)
```

### 8.4 효율적인 Top-K
```python
# O(n log n) 정렬 대신 O(n) argpartition
topk_indices = np.argpartition(scores, -k)[-k:]
```

---

## 9. 벤치마크 결과

MemU는 Locomo 벤치마크에서 **평균 92.09% 정확도**를 달성했습니다.

---

## 10. 에이전트 시스템 적용 시사점

### 10.1 컨텍스트 관리 전략

1. **계층적 구조**: Resource → Item → Category로 세분화하여 효율적 검색
2. **점진적 요약**: 각 레이어가 점차 추상화된 뷰 제공
3. **쿼리 재작성**: 대화 컨텍스트를 반영한 동적 쿼리 변환

### 10.2 메모리 추출 패턴

1. **선별적 추출**: 사용자가 직접 언급한 정보만 추출
2. **중복 제거**: 유사한 아이템 자동 병합
3. **카테고리 기반 분류**: 미리 정의된 카테고리로 자동 분류

### 10.3 검색 최적화

1. **의도 라우팅**: 검색이 필요한지 먼저 판단
2. **충분성 검사**: 각 계층에서 충분한 정보를 얻었는지 확인
3. **조기 종료**: 충분한 정보가 있으면 더 깊은 검색 생략

### 10.4 확장성 고려사항

1. **멀티 프로바이더**: 작업별 다른 LLM 사용 가능
2. **파이프라인 커스터마이징**: 도메인별 스텝 추가/교체
3. **인터셉터**: 모니터링, 로깅, 커스텀 처리 삽입

---

## 11. 주요 의존성

```toml
[dependencies]
defusedxml = ">=0.7.1"      # XML 보안 파싱
httpx = ">=0.28.1"           # HTTP 클라이언트
numpy = ">=2.3.4"            # 벡터 연산
openai = ">=2.8.0"           # OpenAI SDK
pydantic = ">=2.12.4"        # 데이터 검증
sqlmodel = ">=0.0.27"        # ORM
alembic = ">=1.14.0"         # DB 마이그레이션
pendulum = ">=3.1.0"         # 날짜/시간
langchain-core = ">=1.2.7"   # LangChain 코어

[optional]
postgres = ["pgvector", "sqlalchemy[postgresql]"]
langgraph = ["langgraph", "langchain-core"]
```

---

## 12. 결론

MemU는 에이전트 시스템을 위한 포괄적인 메모리 솔루션을 제공합니다:

- **3계층 아키텍처**로 원시 데이터부터 요약까지 완전한 추적성
- **RAG + LLM 이중 검색**으로 속도와 이해도 균형
- **멀티모달 지원**으로 다양한 입력 처리
- **파이프라인 기반 아키텍처**로 높은 확장성
- **인터셉터 시스템**으로 유연한 커스터마이징

에이전트 시스템 구현 시 장기 메모리, 컨텍스트 관리, 사용자 프로파일링 등의 요구사항에 참조할 수 있는 우수한 설계 패턴을 제공합니다.
