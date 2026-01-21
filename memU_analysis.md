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

## 5. 프롬프트 시스템

### 5.1 메모리 추출 프롬프트 (profile 예시)

```xml
<!-- 출력 형식 -->
<profile>
    <memory>
        <content>사용자가 인터넷 회사에서 제품 관리자로 일한다</content>
        <categories>
            <category>Basic Information</category>
        </categories>
    </memory>
    <memory>
        <content>사용자는 30살이다</content>
        <categories>
            <category>Basic Information</category>
        </categories>
    </memory>
</profile>
```

**핵심 규칙:**
- 사용자가 직접 언급/확인한 정보만 추출
- 임시/일회성 정보 제외 (날씨, 현재 기분 등)
- 30단어 이내의 간결한 표현
- 이벤트/일시적 상태는 profile에서 제외

### 5.2 검색 의도 판단 프롬프트

```xml
<decision>RETRIEVE 또는 NO_RETRIEVE</decision>
<rewritten_query>컨텍스트를 포함한 재작성된 쿼리</rewritten_query>
```

**NO_RETRIEVE 조건:**
- 인사, 일상 대화
- 현재 대화만 관련된 질문
- 일반 상식 질문
- 시스템 메타 질문

---

## 6. 통합 및 확장

### 6.1 LangGraph 통합

```python
from memu.integrations.langgraph import MemULangGraphTools

memu_tools = MemULangGraphTools(memory_service)
tools = memu_tools.tools()

# save_memory: 정보 저장
# search_memory: 메모리 검색
```

### 6.2 파이프라인 커스터마이징

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

### 6.3 지원 LLM 프로바이더

| 프로바이더 | 설명 |
|-----------|------|
| OpenAI | 기본 프로바이더 |
| Grok (X.AI) | 자동 설정 변경 |
| OpenRouter | 다중 프로바이더 라우팅 |
| Doubao | 바이트댄스 모델 |
| 커스텀 | base_url 지정으로 호환 API |

---

## 7. 성능 최적화 기법

### 7.1 지연 초기화
```python
# LLM 클라이언트 첫 사용 시에만 초기화
def _get_llm_base_client(profile):
    if name in self._llm_clients:
        return self._llm_clients[name]
    client = self._init_llm_client(cfg)
    self._llm_clients[name] = client
    return client
```

### 7.2 배치 임베딩
```python
# 여러 텍스트를 한 번에 임베딩
item_embeddings = await client.embed(summary_payloads)
```

### 7.3 병렬 LLM 호출
```python
# 메모리 타입별 병렬 추출
tasks = [client.summarize(prompt) for prompt in prompts]
responses = await asyncio.gather(*tasks)
```

### 7.4 효율적인 Top-K
```python
# O(n log n) 정렬 대신 O(n) argpartition
topk_indices = np.argpartition(scores, -k)[-k:]
```

---

## 8. 벤치마크 결과

MemU는 Locomo 벤치마크에서 **평균 92.09% 정확도**를 달성했습니다.

---

## 9. 에이전트 시스템 적용 시사점

### 9.1 컨텍스트 관리 전략

1. **계층적 구조**: Resource → Item → Category로 세분화하여 효율적 검색
2. **점진적 요약**: 각 레이어가 점차 추상화된 뷰 제공
3. **쿼리 재작성**: 대화 컨텍스트를 반영한 동적 쿼리 변환

### 9.2 메모리 추출 패턴

1. **선별적 추출**: 사용자가 직접 언급한 정보만 추출
2. **중복 제거**: 유사한 아이템 자동 병합
3. **카테고리 기반 분류**: 미리 정의된 카테고리로 자동 분류

### 9.3 검색 최적화

1. **의도 라우팅**: 검색이 필요한지 먼저 판단
2. **충분성 검사**: 각 계층에서 충분한 정보를 얻었는지 확인
3. **조기 종료**: 충분한 정보가 있으면 더 깊은 검색 생략

### 9.4 확장성 고려사항

1. **멀티 프로바이더**: 작업별 다른 LLM 사용 가능
2. **파이프라인 커스터마이징**: 도메인별 스텝 추가/교체
3. **인터셉터**: 모니터링, 로깅, 커스텀 처리 삽입

---

## 10. 주요 의존성

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

## 11. 결론

MemU는 에이전트 시스템을 위한 포괄적인 메모리 솔루션을 제공합니다:

- **3계층 아키텍처**로 원시 데이터부터 요약까지 완전한 추적성
- **RAG + LLM 이중 검색**으로 속도와 이해도 균형
- **멀티모달 지원**으로 다양한 입력 처리
- **파이프라인 기반 아키텍처**로 높은 확장성
- **인터셉터 시스템**으로 유연한 커스터마이징

에이전트 시스템 구현 시 장기 메모리, 컨텍스트 관리, 사용자 프로파일링 등의 요구사항에 참조할 수 있는 우수한 설계 패턴을 제공합니다.
