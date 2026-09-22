# Jevlike vs mmBERT — 작은 선택기와 사전학습 인코더

분석일: 2026-09-18. 여기서 mmBERT는 JHU-CLSP의 multilingual ModernBERT를 뜻한다. 비교 대상인 Jevlike는 방금 로컬에서 실행한 Tiny 모델과 HF 인코더 경로를 구분한다.

- Jevlike: [`94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`](https://github.com/vinnylarouge/jevlike/commit/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452), `.repos/jevlike/`.
- mmBERT: [`108ffb376ad65c2512801bb88931933dcd286baf`](https://github.com/JHU-CLSP/mmBERT/commit/108ffb376ad65c2512801bb88931933dcd286baf), `.repos/mmbert/`.
- 두 클론 모두 기존 `/.repos/` ignore 규칙으로 Git 추적에서 제외된다. mmBERT 가중치는 이번 비교에서 다운로드하거나 실행하지 않았다. mmBERT와 Jevlike 조합은 코드 인터페이스에 근거한 호환성 판단이며 실측 결과가 아니다.

## 1. 비교의 핵심

**Jevlike는 후보별 점수를 학습하는 구조·코드이고, mmBERT는 다국어 텍스트의 표현을 만드는 사전학습 모델이다.** Jevlike Tiny와 mmBERT를 사용하는 분류기는 대안이 될 수 있고, Jevlike의 HF 인코더로 mmBERT를 사용하는 조합도 설계할 수 있다.

| 항목 | Jevlike Tiny — 현재 데모 | mmBERT 기반 분류·선택 모델 |
|---|---|---|
| 시작 상태 | 임의 초기화 후 배지 예제 2,000개로 학습 | 다국어 대규모 사전학습 가중치에서 시작 |
| 핵심 구조 | byte embedding, 위치 embedding, 후보 Attention head | 22층 양방향 Transformer 인코더와 업무용 head |
| 파라미터 | 현재 설정 41,280개 | Small 약 140M, Base 약 307M에 업무용 head 추가 |
| 텍스트 이해 기반 | 사용자가 제공한 학습 예제에서 습득 | 약 3T 규모의 사전학습 토큰에서 언어 표현 습득 |
| 입력 길이 | 현재 문맥 192 bytes, 후보 32 bytes | 공개 모델 최대 8,192 tokens; 애플리케이션 설정에 따라 더 짧게 사용 |
| 기본 출력 | 가변 후보별 점수 | 토큰별 벡터 또는 masked token 점수; 업무 head에 따라 분류·검색 점수 |
| 자연어 문장 생성 | 없음 | 기본 모델은 생성형 챗봇이 아님 |
| 업무 학습 | 문맥·후보·정답 인덱스로 학습 | 분류·검색·재순위화 목적에 맞게 추가 학습 |
| 예상 적합 분야 | 짧고 단순한 패턴 선택, 극소형 실험 | 한국어·다국어 문서 분류, 검색, 후보 재순위화 |

mmBERT의 공개 규모·학습 방식은 [모델 카드](https://huggingface.co/jhu-clsp/mmBERT-base)와 [논문](https://arxiv.org/abs/2509.06888)을 확인했다. 공식 언어 통계에는 한국어 `kor`도 포함된다. 언어 수는 저자의 학습 설명 기준 1,833개이며 언어별 품질이 같다는 뜻은 아니다. [언어 통계](https://github.com/JHU-CLSP/mmBERT/blob/108ffb376ad65c2512801bb88931933dcd286baf/statistics/language_counts.csv#L780).

## 2. 같은 상담 문의를 처리하는 예

입력이 “결제가 두 번 됐으니 하나를 취소해 주세요”이고, 분류가 “결제 / 배송 / 계정”이라고 가정한다.

현재 Tiny 데모는 영어 배지 선택만 학습했으므로 이 문의를 업무 의미에 따라 처리할 근거가 부족하다. 한글 데이터를 넣는다고 이미 학습된 상담 분류기가 되는 것은 아니다.

mmBERT는 한국어를 포함한 언어 표현을 사전학습했다. 여기에 상담 분류 head를 붙이고 정답 데이터로 추가 학습하면 사전학습된 표현을 해당 업무에 활용할 수 있다. 다만 기본 체크포인트를 내려받는 것만으로 회사의 분류 기준까지 학습되지는 않는다. 새로 초기화한 분류 head의 출력은 훈련 전 업무 예측으로 사용할 수 없다.

```mermaid
flowchart TB
    subgraph T["현재 Tiny 데모"]
        TI["문맥과 후보 문자열"] --> TE["작은 byte embedding"]
        TE --> TH["학습한 후보 Attention head"]
        TH --> TO["후보별 점수"]
    end
    subgraph M["mmBERT로 일반 분류기 구성"]
        MI["문의 문맥"] --> ME["사전학습된 mmBERT"]
        ME --> MH["업무 데이터로 학습한 분류 head"]
        MH --> MO["결제 · 배송 · 계정 점수"]
    end
    subgraph J["mmBERT와 Jevlike 조합 가능"]
        JI["문맥과 후보를 각각 입력"] --> JE["동결된 mmBERT 인코더"]
        JE --> JH["별도로 학습하는 Jevlike Attention head"]
        JH --> JO["가변 후보별 점수"]
    end
```

Jevlike의 장점인 **매 요청마다 후보 수와 텍스트가 달라지는 인터페이스**는 일반적인 고정 클래스 분류 head와 다르다. 하지만 mmBERT도 문맥·후보 쌍을 점수화하는 cross-encoder나 검색용 bi-encoder로 구성하면 가변 후보를 처리할 수 있다. mmBERT 자체가 고정 클래스만 지원하는 것은 아니다.

## 3. 학습 대상과 데이터의 차이

| 방식 | 학습 데이터 | 갱신 대상 |
|---|---|---|
| Jevlike Tiny | `context`, `options`, 정답 `label` | embedding과 후보 head 전체 |
| Jevlike HF | 같은 후보 선택 데이터 | 인코더는 동결, 후보 head만 갱신 |
| mmBERT 업무 파인튜닝 | 텍스트·클래스, 문장 쌍·관련성 등 | 기본적으로 인코더와 업무 head를 함께 갱신 가능; 인코더 동결도 선택 가능 |
| mmBERT 추가 사전학습 | 도메인 텍스트 | 일부 토큰을 가리고 맞히는 MLM 목적의 언어 모델 학습 |

mmBERT는 문서 자체에서 masked token 정답을 만들 수 있으므로 별도 사람 라벨 없이 도메인 추가 사전학습이 가능하다. 이것만으로 특정 업무의 “어떤 후보를 골라야 하는가”가 정의되는 것은 아니다. Jevlike의 기본 텍스트 학습은 정답 후보가 있는 지도학습이다.

mmBERT의 공식 모델 카드에는 Transformers 분류 파인튜닝과 Sentence Transformers를 이용한 검색·reranker 학습 예제가 있다. Jevlike의 기본 HF 경로는 `requires_grad_(False)`와 `torch.no_grad()`를 사용하므로 기반 모델을 자동으로 파인튜닝하지 않는다. [mmBERT 학습 예제](https://huggingface.co/jhu-clsp/mmBERT-base#fine-tuning-examples), [Jevlike 코드](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/model.py#L66-L94).

## 4. 소스와 기술 스택에서 확인한 차이

Jevlike의 `TinyScorer`는 문맥 byte에 위치 embedding을 더하고, 후보 byte embedding을 평균한 벡터로 Attention 점수를 구한다. 후보 내부에는 위치 표현이 없어 `ab`와 `ba`처럼 같은 byte 구성의 후보는 구분하지 못한다. mmBERT는 후보를 Transformer에 통과시키면 순서와 주변 문맥이 반영된 토큰 표현을 만든다. 이후 평균을 내더라도 Tiny의 원시 embedding 평균과는 다른 표현이다. 이것이 모든 후보를 반드시 정확히 구분한다는 보장은 아니다.

mmBERT 저장소는 주로 학습 설정, 데이터 접근 코드, 언어 통계와 모델 사용 예제를 제공한다. 독립적인 웹 서버나 후보 선택 CLI가 포함된 프로젝트는 아니다. 실제 모델 이용은 PyTorch·Hugging Face Transformers의 ModernBERT 구현을 통한다.

- `configs/mmBERT-small.yaml`: hidden size 384, 22층, Gemma 2 tokenizer, 256,000 vocabulary, MLM 설정.
- `configs/mmBERT-base-ext.yaml`: hidden size 768, 문맥 8,192, masking 15%, local attention과 주기적인 global attention 설정.
- `configs/mmBERT-base-decay.yaml`: masking 5%, 후반 학습률 감소 설정.
- `data/online_streaming.py`: Hugging Face에 저장된 데이터 shard를 `fsspec`과 streaming reader로 읽는 보조 코드. 현재 구현은 초기화 시 shard 파일을 내려받는 부분이 있어, 이름만 보고 순수한 지연 스트리밍으로 해석하면 안 된다.

mmBERT는 ModernBERT의 RoPE, local/global attention, 지원 환경에서의 Flash Attention과 unpadding을 활용한다. 실제 가속 방식은 장치·정밀도·Transformers 설정에 따라 달라진다. [공식 코드·설정](https://github.com/JHU-CLSP/mmBERT/tree/108ffb376ad65c2512801bb88931933dcd286baf), [ModernBERT API](https://huggingface.co/docs/transformers/model_doc/modernbert).

## 5. 장단점과 비용

| 대상 | 장점 | 단점·제약 |
|---|---|---|
| Jevlike Tiny | 극소형, CPU 학습·추론이 쉬움, 후보 선택 실험이 간단함 | 사전학습 언어 능력 없음, 표현력·길이 제약, 후보 내부 순서 소실 |
| mmBERT + 업무 head | 사전학습 표현 활용, 한국어·다국어, 다양한 분류·검색 용도로 확장, 기반 모델 파인튜닝 가능 | Tiny보다 훨씬 큰 모델, 목적별 데이터·학습 필요, 긴 문서와 후보 수에 따라 비용 증가 |
| Jevlike + mmBERT | 다국어 표현과 가변 후보 인터페이스 결합, head만 학습하면 학습 부담 감소 | 매 추론에서 mmBERT 계산이 필요, head도 새로 학습해야 함, end-to-end 파인튜닝은 기본 경로 밖 |

CPU 추론은 mmBERT에서도 가능하므로 GPU가 반드시 필요한 것은 아니다. 다만 처리량·긴 문서·전체 파인튜닝 요구에 따라 GPU를 사용하는 편이 유리할 수 있다. 이번에는 mmBERT를 실행하지 않았으므로 이 Mac에서 몇 ms인지, Jevlike보다 몇 배 느린지는 확인하지 않았다.

단순히 파라미터 수에 FP32의 4 bytes를 곱하면 Tiny의 가중치는 약 0.165 MB, mmBERT-small은 약 560 MB, Base는 약 1.23 GB다. 공개 모델의 반올림한 파라미터 수를 이용한 계산이며 head, activation, Python·PyTorch 및 학습 optimizer 메모리는 제외한다. Tiny 서버 역시 전체 프로세스가 0.165 MB만 쓰는 것은 아니다. 16-bit 저장은 이 가중치 계산을 대략 절반으로 줄이지만 장치에서의 속도 향상은 별도 문제다.

**두 방식 모두 긴 답변을 토큰 단위로 생성하지 않는다.** 따라서 Jevlike가 생성형 LLM 대비 얻는 “출력 생성을 생략하는 이점”을 mmBERT 대비 속도 배율로 적용할 수 없다. mmBERT 고정 클래스 분류는 보통 문맥을 한 번 인코딩하지만, Jevlike HF는 문맥 인코딩과 후보 배치 인코딩을 각각 실행한다. 같은 mmBERT를 쓴다고 Jevlike 구성이 일반 분류 head보다 반드시 빠른 것은 아니다.

mmBERT의 공개 “기존 다국어 모델 대비 2~4배” 역시 저자의 비교 모델·측정 조건에 해당하며 Tiny 대비 수치가 아니다. 기존 [Jevlike 속도 실측](performance-and-cost.md)과 작업·정확도·입력 길이를 맞춰야 직접 비교할 수 있다.

## 6. Jevlike에 mmBERT를 넣을 수 있는가

**코드 인터페이스상 가능성이 높다.** Jevlike는 HF 모델에서 `config.hidden_size`와 `last_hidden_state`를 읽고, mmBERT의 ModernBERT 구현은 이 인터페이스를 제공한다. 모델 이름을 Qwen에 고정한 코드가 아니다. [Jevlike 로더](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/model.py#L107-L126), [mmBERT-base 설정](https://huggingface.co/jhu-clsp/mmBERT-base/blob/main/config.json).

구성하려면 HF encoder 이름을 `jhu-clsp/mmBERT-small` 또는 `jhu-clsp/mmBERT-base`로 지정하고, 목적에 맞는 후보 선택 데이터로 Jevlike head를 새로 학습한다. ModernBERT를 지원하는 Transformers 버전과 해당 장치에서의 attention 구현 호환성을 확인해야 한다. 기존 Tiny 체크포인트는 인코더 폭·구조가 달라 그대로 재사용할 수 없다.

기본 HF 설정에서는 mmBERT가 동결되며, 학습 완료 후에도 추론 때 mmBERT 전체가 필요하다. 작은 head 파일만 저장된다고 실행 모델 전체가 Tiny 크기가 되는 것은 아니다. 이번 비교에서는 이 조합의 설치·학습·추론을 실제 실행하지 않았다.

## 7. 어떤 쪽으로 시작할 것인가

엔지니어링 관점에서 다음을 권한다. 아래는 구조와 사전학습 범위에 근거한 판단이며 특정 업무에서의 정확도 실측 결과는 아니다.

- **결제·배송·계정처럼 분류 목록이 고정된 한국어 업무**: mmBERT-small에 일반 분류 head를 붙여 파인튜닝하는 방식을 먼저 비교 기준으로 삼는다. 문맥만 인코딩하는 구성이 단순하다.
- **매 요청마다 선택할 도구·문서·행동 후보가 바뀌는 업무**: mmBERT cross-encoder와 Jevlike + mmBERT를 비교한다. 전자는 문맥과 후보를 함께 인코딩하고, 후자는 각각 인코딩한 뒤 작은 head로 결합한다.
- **영어 배지 맞추기처럼 단순하고 계산 예산이 매우 작은 실험**: Jevlike Tiny가 가볍다. 요구 정확도를 달성한다는 확인이 선행돼야 한다.
- **학습 없이 자연어 지시만으로 다양한 판단을 수행**: 기본 mmBERT나 미학습 Jevlike head보다 instruction-tuned LLM을 사용하는 OpenJev가 목적에 더 가깝다. [OpenJev 비교](jevlike-vs-openjev.md).
