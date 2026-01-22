# Second-Me (Mindverse) 분석 보고서

> 저장소: https://github.com/mindverse/Second-Me
> 분석일: 2026-01-21

---

## 1. 개요

Second Me는 개인화된 "AI Self"(AI 자아)를 생성하기 위한 오픈소스 프레임워크입니다. 사용자의 정체성을 보존하고, 컨텍스트를 이해하며, 진정성 있게 사용자를 대변하는 디지털 트윈을 생성합니다. 계층적 메모리 모델링(HMM)과 Me-Alignment 알고리즘을 사용하여 사용자별 데이터로 로컬 LLM을 파인튜닝합니다.

### 핵심 차별점

- **로컬 파인튜닝**: 런타임에 메모리를 저장/검색하는 다른 메모리 시스템과 달리, Second Me는 실제로 사용자 데이터로 로컬 LLM을 파인튜닝
- **AI Self 개념**: 사용자를 대변할 수 있는 지속적인 디지털 아이덴티티 생성
- **탈중앙화 네트워크**: AI 자아들이 권한 기반으로 연결하고 협업 가능
- **완전한 프라이버시**: 100% 로컬 훈련 및 호스팅 - 데이터가 절대 기기를 떠나지 않음

### 연구 논문

- [AI-Native Memory v1](https://arxiv.org/abs/2406.18312)
- [AI-Native Memory v2](https://arxiv.org/abs/2503.08102)

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Second Me                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    사용자 데이터 입력                      │   │
│  │    (노트, 문서, 이미지, 오디오, 채팅, 할일)                 │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L0: 인사이트 생성                         │   │
│  │         (문서 파싱, 이미지 분석, 오디오 처리)              │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L1: 아이덴티티 모델링                      │   │
│  │    (바이오그래피, Shades, 토픽, 상태, 클러스터링)          │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L2: 모델 훈련                             │   │
│  │    (데이터 합성, 파인튜닝, DPO 정렬)                       │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    훈련된 AI Self                         │   │
│  │           (사용자에게 개인화된 로컬 LLM)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Second Me 네트워크                        │   │
│  │        (탈중앙화 AI 자아 협업)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 계층적 메모리 모델 (L0/L1/L2)

Second Me는 사용자 데이터를 개인화된 AI 모델로 처리하기 위한 3계층 구조를 구현합니다.

### 3.1 L0: 저수준 인사이트 생성

L0는 원시 데이터 파싱과 초기 인사이트 추출을 담당합니다.

```python
# /lpm_kernel/L0/l0_generator.py

class L0Generator:
    def insighter(self, inputs: InsighterInput) -> Dict[str, str]:
        """문서 입력에서 인사이트를 생성합니다."""
        datatype = DataType(inputs.file_info.data_type)

        if datatype == DataType.IMAGE:
            insight, title = self._insighter_image(bio, content, ...)
        elif datatype == DataType.AUDIO:
            insight, title = self._insighter_audio(bio, content, ...)
        else:  # DOCUMENT
            insight, title = self._insighter_doc(bio, content, ...)

        return {"title": title, "insight": insight}
```

**지원 데이터 타입:**
- TEXT: 일반 텍스트 문서
- MARKDOWN: 마크다운 파일
- PDF: PDF 문서
- LINK: 웹 링크
- IMAGE: 이미지 (비전 모델 사용)
- AUDIO: 오디오 파일 (Whisper 전사 사용)

### 3.2 L1: 아이덴티티 모델링 계층

L1은 바이오그래피(Biography)와 셰이드(Shades)를 통해 사용자 아이덴티티의 구조화된 표현을 생성합니다.

```python
# /lpm_kernel/L1/l1_generator.py

class L1Generator:
    def gen_global_biography(self, old_profile: Bio, cluster_list: List[Cluster]) -> Bio:
        """사용자 메모리에서 글로벌 바이오그래피를 생성합니다."""
        global_bio = self._global_bio_generate(global_bio)
        return global_bio

    def gen_shade_for_cluster(self, old_memory_list, new_memory_list, shade_info_list):
        """메모리 클러스터에 대한 Shade(성격 측면)를 생성합니다."""
        shade_generator = ShadeGenerator()
        return shade_generator.generate_shade(...)

    def gen_topics_for_shades(self, old_cluster_list, old_outlier_memory_list,
                               new_memory_list, cophenetic_distance=1.0):
        """계층적 클러스터링을 사용하여 메모리를 토픽으로 클러스터링합니다."""
        topics_generator = TopicsGenerator()
        return topics_generator.generate_topics_for_shades(...)
```

**핵심 L1 개념:**

| 개념 | 설명 |
|------|------|
| **Bio** | 글로벌 바이오그래피 - 종합적인 사용자 프로필 |
| **Shade** | 관심사/전문 분야를 나타내는 성격 측면 |
| **ShadeTimeline** | 참조된 메모리와 함께 Shade의 시간적 진화 |
| **Cluster** | 의미적으로 유사한 메모리 그룹 |
| **AttributeInfo** | 신뢰 수준이 있는 아이덴티티 속성 |

### 3.3 L2: 훈련 데이터 합성 및 모델 파인튜닝

L2는 훈련 데이터를 생성하고 기본 모델을 파인튜닝하여 개인화된 AI를 생성합니다.

```python
# /lpm_kernel/L2/l2_generator.py

class L2Generator:
    def gen_subjective_data(self, note_list, basic_info, ...):
        """개인화를 위한 주관적 훈련 데이터를 생성합니다."""

        # 1. 선호도 Q&A 데이터 생성
        self.gen_preference_data(...)

        # 2. 다양성 데이터 생성
        self.gen_diversity_data(...)

        # 3. 자기 Q&A 데이터 생성
        self.gen_selfqa_data(...)

        # 4. 모든 훈련 데이터 병합
        self.merge_json_files(data_output_base_dir)
```

**훈련 데이터 타입:**

1. **선호도 Q&A**: 사용자의 선호, 의견, 가치관에 대한 질문
2. **다양성 데이터**: 사용자의 의사소통 스타일을 보여주는 다양한 응답
3. **자기 Q&A**: 사용자의 아이덴티티와 경험에 대한 1인칭 응답

---

## 4. 데이터 모델

### 4.1 Note (메모리 단위)

```python
# /lpm_kernel/L1/bio.py

class Note:
    def __init__(
        self,
        noteId: int = None,
        content: str = "",
        createTime: str = "",
        memoryType: str = "",  # TEXT, MARKDOWN, PDF, LINK
        embedding: Optional[List[float]] = None,
        chunks: List[Chunk] = None,
        title: str = "",
        summary: str = "",
        insight: str = "",
        tags: List[str] = None,
        topic: str = None,
    ):
        ...
```

### 4.2 Memory & Cluster

```python
class Memory:
    def __init__(self, memoryId: int, embedding: List[float] = None):
        self.memory_id = memoryId
        self.embedding = np.array(embedding).squeeze() if embedding else None

class Cluster:
    def __init__(
        self,
        clusterId: int,
        memoryList: List[Memory] = [],
        centerEmbedding: List[float] = None,
        is_new=False,
    ):
        self.cluster_id = clusterId
        self.memory_list = memory_list
        self.cluster_center = np.array(centerEmbedding) if centerEmbedding else np.zeros(1536)

    def get_cluster_center(self):
        """메모리 임베딩의 중심점을 계산합니다."""
        self.cluster_center = np.mean(
            [memory.embedding for memory in self.memory_list], axis=0
        )

    def prune_outliers_from_cluster(self):
        """클러스터 중심에서 거리 기반으로 이상치를 제거합니다."""
        memory_list = sorted(
            self.memory_list,
            key=lambda x: np.linalg.norm(x.embedding - self.cluster_center),
        )
        # 가장 가까운 80% 메모리 유지
        self.memory_list = memory_list[: max(int(self.size * 0.8), 1)]
```

### 4.3 Shade (성격 측면)

```python
class ShadeInfo:
    def __init__(
        self,
        id: int = None,
        name: str = "",           # 예: "소프트웨어 엔지니어링 열정가"
        aspect: str = "",         # 예: "전문 기술"
        icon: str = "",           # 시각적 표현
        descThirdView: str = "",  # 3인칭 설명
        descSecondView: str = "", # 2인칭 설명
        contentThirdView: str = "",
        contentSecondView: str = "",
        timelines: List[ShadeTimeline] = [],
        confidenceLevel: str = None,  # VERY_LOW ~ VERY_HIGH
    ):
        ...
```

### 4.4 Bio (글로벌 아이덴티티)

```python
class Bio:
    def __init__(
        self,
        contentThirdView: str = "",
        content: str = "",
        summaryThirdView: str = "",
        summary: str = "",
        attributeList: List[AttributeInfo] = [],
        shadesList: List[ShadeInfo] = [],
    ):
        self.shades_list = sorted(
            [ShadeInfo(**shade) for shade in shadesList],
            key=lambda x: len(x.timelines),  # 증거 수로 정렬
            reverse=True,
        )

    def complete_content(self, second_view: bool = False) -> str:
        """완전한 바이오그래피 콘텐츠를 생성합니다."""
        interests = "\n### 사용자의 관심사와 선호도 ###\n"
        interests += "\n".join([shade._preview_(second_view) for shade in self.shades_list])
        conclusion = "\n### 결론 ###\n" + self.summary
        return f"## 종합 분석 보고서 ##\n{interests}\n{conclusion}"
```

---

## 5. 훈련 파이프라인

### 5.1 데이터 합성 흐름

```
사용자 데이터 → L0 인사이트 → L1 아이덴티티 → L2 훈련 데이터 → 파인튜닝된 모델

1. 선호도 Q&A 생성
   - L1의 토픽/클러스터 사용
   - 사용자 선호도에 대한 질문 생성
   - 일관된 선호도 응답 생성

2. 다양성 데이터 생성
   - 엔티티와 지식 그래프 사용
   - 다양한 Q&A 쌍 생성
   - 의사소통 스타일 다양성 보장

3. 자기 Q&A 생성
   - 1인칭 아이덴티티 질문
   - 글로벌 bio와 사용자 소개 사용
   - "나는 누구인가?" 유형 응답
```

### 5.2 DPO(Direct Preference Optimization) 훈련

```python
# /lpm_kernel/L2/dpo/dpo_train.py

def train_dpo_model(
    model,
    train_dataset,
    eval_dataset,
    beta=0.1,  # KL 페널티 계수
    learning_rate=5e-5,
    num_train_epochs=3,
):
    """DPO를 사용하여 선호도 정렬을 위한 모델 파인튜닝."""
    # DPO는 선택된 응답 vs 거부된 응답 쌍으로 훈련
    # 최적화: log P(선택) - log P(거부) 최대화
```

### 5.3 Me-Alignment 알고리즘

Me-Alignment 알고리즘은 AI 자아가 진정성 있게 사용자를 대변하도록 보장합니다:

1. **아이덴티티 기반**: 응답을 사용자의 바이오그래피와 셰이드에 고정
2. **선호도 일관성**: 상호작용 전반에 걸쳐 일관된 의견 유지
3. **스타일 보존**: 사용자의 의사소통 패턴 유지
4. **메모리 통합**: 관련 사용자 경험 참조

---

## 6. 메모리 관리

### 6.1 GPU 메모리 최적화

```python
# /lpm_kernel/L2/memory_manager.py

class MemoryManager:
    def get_optimal_training_config(self) -> Dict[str, Any]:
        """하드웨어 기반 권장 설정을 가져옵니다."""
        config = {
            "device_map": "auto",
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
        }

        if self.cuda_available:
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere+
                config["bf16"] = True
            elif capability[0] >= 7:  # Volta+
                config["fp16"] = True

            # VRAM에 따라 조정
            vram_gb = self.get_memory_info()["vram_total_gb"]
            if vram_gb < 8:
                config["gradient_accumulation_steps"] = 4
            elif vram_gb < 16:
                config["gradient_accumulation_steps"] = 2

        return config
```

### 6.2 모델 크기 권장사항

| 메모리 (GB) | Docker (Windows/Linux) | Docker (Mac) | 통합형 |
|-------------|------------------------|--------------|--------|
| 8           | ~0.8B                  | ~0.4B        | ~1.0B  |
| 16          | ~1.5B                  | ~0.5B        | ~2.0B  |
| 32          | ~2.8B                  | ~1.2B        | ~3.5B  |

---

## 7. 지식 그래프 통합 (GraphRAG)

Second Me는 지식 그래프 구축을 위해 Microsoft GraphRAG를 사용합니다:

```python
# /lpm_kernel/L2/data_pipeline/graphrag_indexing/

# GraphRAG가 추출하는 것:
# - 엔티티 (사람, 장소, 개념)
# - 엔티티 간 관계
# - 커뮤니티 요약 (계층적 클러스터링)
# - 주장과 사실
```

**통합 지점:**
- 다양성 데이터 생성을 위한 엔티티 추출
- 선호도 생성을 위한 토픽 모델링
- 추론 시 지식 검색

---

## 8. Second Me 네트워크

### 8.1 탈중앙화 AI 자아 공유

```
┌─────────────────────────────────────────────────────────────────┐
│                    Second Me 네트워크                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │ AI 자아 │    │ AI 자아 │    │ AI 자아 │    │ AI 자아 │    │
│   │  (나)   │◄──►│ (앨리스)│◄──►│  (밥)   │◄──►│ (찰리) │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│        │              │              │              │          │
│        │              │              │              │          │
│   ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    │
│   │  로컬   │    │  로컬   │    │  로컬   │    │  로컬   │    │
│   │  머신   │    │  머신   │    │  머신   │    │  머신   │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                 │
│   기능:                                                         │
│   • 권한 기반 공유                                               │
│   • AI 간 협업                                                   │
│   • 롤플레이 시나리오                                            │
│   • 집단 브레인스토밍                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 사용 사례

- **롤플레이**: AI 자아가 다양한 시나리오에서 사용자를 대변
- **AI 스페이스**: 여러 AI 자아가 협력하여 문제 해결
- **아이스브레이킹**: AI 자아가 소유자들을 서로 소개
- **브레인스토밍**: 여러 관점에서의 집단 아이디어 생성

---

## 9. 다른 메모리 시스템과의 비교

| 기능 | Second Me | Mem0 | MemU | OpenMemory | Cognee |
|------|-----------|------|------|------------|--------|
| **접근 방식** | 로컬 파인튜닝 | RAG 메모리 | RAG 메모리 | HSG 메모리 | 지식 그래프 |
| **메모리 타입** | 모델 가중치 | 벡터 저장소 | 파일 기반 | 멀티 섹터 | 그래프 + 벡터 |
| **개인화** | 모델 레벨 | 검색 | 검색 | 검색 | 그래프 구조 |
| **프라이버시** | 100% 로컬 | 클라우드/로컬 | 로컬 | 로컬 | 클라우드/로컬 |
| **훈련 필요** | 예 | 아니오 | 아니오 | 아니오 | 아니오 |
| **아이덴티티 모델링** | Shades/Bio | 사용자 메타데이터 | 카테고리 | 섹터 | 온톨로지 |
| **추론 속도** | 빠름 (RAG 없음) | 가변적 | 가변적 | 가변적 | 가변적 |

---

## 10. 강점과 약점

### 강점

1. **진정한 개인화**: 파인튜닝이 RAG보다 더 깊은 개인화 생성
2. **런타임 레이턴시 없음**: 추론 시 검색 단계 없음
3. **완전한 프라이버시**: 모든 훈련과 추론이 로컬
4. **아이덴티티 지속성**: 모델 가중치가 세션 간 아이덴티티 보존
5. **혁신적 "AI Self" 패러다임**: 최초의 디지털 트윈 접근법
6. **네트워크 효과**: AI 자아 간 탈중앙화 협업
7. **연구 기반**: 피어 리뷰된 AI-Native Memory 논문 기반

### 약점

1. **리소스 집약적**: 훈련에 GPU 필요
2. **훈련 시간**: 초기 AI 자아 생성에 수 시간 소요
3. **업데이트 비용**: 새 메모리는 재훈련 필요
4. **모델 크기 제한**: 소비자 하드웨어가 모델 능력 제한
5. **실시간 메모리 없음**: 재훈련 없이 메모리 추가 불가
6. **고정된 지식**: 모델이 훈련 후 외부 지식 접근 불가

---

## 11. 사용 예시

```bash
# 1. 클론 및 시작
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me
make docker-up

# 2. 웹 인터페이스 접속
# http://localhost:3000 열기

# 3. 메모리 업로드 (노트, 문서 등)
# 4. L0/L1/L2 처리 대기
# 5. AI 자아와 대화 시작
```

**훈련 파이프라인:**
```
사용자가 데이터 업로드
    ↓
L0: 문서 파싱, 인사이트 추출
    ↓
L1: 바이오그래피 생성, 셰이드 식별, 토픽 클러스터링
    ↓
L2: 훈련 데이터 합성 (선호도, 다양성, 자기 Q&A)
    ↓
LoRA로 기본 모델 (Qwen2.5) 파인튜닝
    ↓
효율적인 추론을 위해 GGUF로 변환
    ↓
AI Self 준비 완료!
```

---

## 12. 결론

Second Me는 검색 기반 메모리 시스템에서 모델 임베디드 메모리로의 패러다임 전환을 나타냅니다. 사용자 데이터로 로컬 LLM을 파인튜닝함으로써, 사용자를 진정으로 이해하고 대변하는 지속적인 "AI Self"를 생성합니다.

**적합한 사용 사례:**
- 깊은 개인화를 원하는 사용자
- 프라이버시를 중시하는 사용자
- 충분한 로컬 컴퓨팅 리소스가 있는 경우
- 일관된 AI 아이덴티티가 필요한 애플리케이션
- 멀티 에이전트 협업 시나리오

**부적합한 사용 사례:**
- 실시간 메모리 업데이트
- 저리소스 환경
- 훈련 데이터가 제한된 사용자
- 외부 지식 접근이 필요한 애플리케이션

---

*분석 완료일: 2026-01-21*
