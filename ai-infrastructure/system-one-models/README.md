# System One 모델 — Jev와 오픈 구현체 비교 분석

> 조사 기준일: **2026-09-30** · Jev 출시 2026-09-15

**System One 모델**은 텍스트를 생성하지 않고 "상태(state) + 타입이 정해진 질문 → 확률이 붙은 타입 안전한 답"만 돌려주는 의사결정 전용 모델이다. TypeSafe AI가 `Jev`로 이 범주를 열었고, 2주 만에 오픈 구현체 15종 이상과 커뮤니티 벤치마크가 생겼다. 이 문서는 **한 페이지에서 전체를 비교**하고, 세부는 하위 문서로 연결한다.

| 문서 | 내용 |
|---|---|
| [typesafe-jev.md](typesafe-jev.md) | Jev 본체 심층 분석 — RLCD, 프리미티브, API, 성능, 한계 |
| [open-implementations.md](open-implementations.md) | 역추론된 아키텍처 계보, 구현체별 심층 비교, 직접 학습 레시피 |
| [use-cases.md](use-cases.md) | 실제 사용 사례 카탈로그, 독립 검증 결과, 금융 도메인 적용 설계 |

---

## 1. 한눈에 보기

- **정체**: LLM과 같은 사전학습 Transformer를 쓰지만, post-training 목표가 "사람이 선호하는 텍스트"(RLHF)나 "검증 가능한 보상"(RLVR)이 아니라 **"캘리브레이션된 확률"(RLCD)** 이다.
- **인터페이스**: `POST /v1/systemone` — `state` + `questions{choice|score|noul}` → `answers{값 + probabilities + confidence}`. 타입 에러가 구조적으로 불가능.
- **효용**: 코드 안의 `if`문 자리에 "의미 판단"을 꽂는다. 답의 공간을 미리 열거할 수 있고, 호출량이 많고, 지연이 중요한 자리에서만 이긴다.
- **현재 상태**: Jev는 closed-weight 호스팅 API(early access). 반면 **인터페이스는 사실상 표준이 되어** 오픈 구현체·SDK 10개 언어·게이트웨이 3곳이 같은 계약을 따른다.

```mermaid
flowchart LR
  S["state — 문자열 · JSON 객체 · 배열"] --> M
  Q["questions — Choice · Score · Noul 여러 개"] --> M
  M["System One 모델 — 모든 질문·옵션을 병렬 평가"] --> A["answers — 타입값 + probabilities + confidence"]
  A --> C["코드가 결정 — 임계값 · 분기 · 에스컬레이션"]
  C -->|"confidence 낮음"| E["LLM 또는 사람에게 에스컬레이션"]
```

## 2. LLM과의 차이

| | 기존 LLM | System One 모델 |
|---|---|---|
| 학습 목표 | RLHF(사람 선호) · RLVR(검증 가능 보상) | **RLCD**(캘리브레이션된 결정) |
| 입력 | 순차 메시지 중심 | **구조화된 프로그램 상태**(JSON) 중심 |
| 출력 | 자유 문자열 → 파싱·검증 필요 | **사전 정의된 타입 값 + 확률분포 + confidence** |
| 샘플링 | 토큰 단위 autoregressive | **단일 forward pass**로 전부 병렬 (Jeeves는 예외) |
| 확률의 의미 | next-token 확률, RLHF 후 과신 경향 | 정답 여부(outcome)에 대해 캘리브레이션 |
| 못 하는 것 | — | 텍스트·코드 생성, 설명, 다단계 추론 |

핵심 트레이드오프: **"생성을 포기한 대가로 속도·비용·타입 안전성·확률의 신뢰성을 얻는다."** 단, 후술할 Jeeves는 이 전제를 의도적으로 깬다.

## 3. 아키텍처 계보

Jev의 내부는 비공개지만, API 행동 실험만으로 구조를 역추론한 글([Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked))이 사실상 오픈 구현체들의 설계도가 됐다.

```mermaid
flowchart TB
  JEV["Jev — TypeSafe, closed weights, 2026-09-15"]
  AH["역추론 분석 — Jev's Architecture Unmasked"]
  JEV -.->|"API 행동 실험"| AH

  AH --> PH["pointer head 계열"]
  AH --> LL["option-letter logit 계열"]
  AH --> ENC["encoder + scoring head 계열"]
  AH --> FZ["frozen logit readout 계열"]

  PH --> KEV["kev — Qwen3.5 0.8B·4B·9B, LoRA + pointer head"]
  KEV --> JEEVES["Jeeves — PostHog, 9B + thinking + CISPO + diffusion drafter"]
  LL --> AUTOJEV["AutoJev-27B — Qwen3.8-27B full-weight SFT"]
  AUTOJEV --> JEFF["Jeff — firelex, 0.8B·2B 축소판 + 로컬 합성데이터"]
  ENC --> LAYA["Laya — ModernBERT 421M, RLCD 구현"]
  ENC --> VERDICT["openJev-verdict-2.0 — ModernBERT-base 150M"]
  FZ --> SEMIF["SemIf — frozen Qwen3.5-4B, 학습 없음"]
  AH --> NANO["NanoJev — Qwen3-0.6B + decision heads"]
```

역추론에서 밝혀진 핵심 사실 4가지:

1. `output_tokens`는 생성량이 아니라 **사후 과금 계산값** — 255개 옵션 응답도 2개 옵션과 같은 속도
2. state 360→29,835 토큰은 지연 2배, 질문 1→1,500개는 sublinear(~600ms) → **state는 한 번 인코딩하고 질문을 배치**
3. "비밀 코드" 실험 — 다른 질문에 넣으면 확률 0.00, state에 넣으면 0.90 → **질문 간 attention 격리**
4. 무관한 옵션을 추가하면 기존 옵션의 log-odds가 변함 → **옵션끼리는 상호작용**(순서 효과 존재)

## 4. 전체 비교표

스타 수는 2026-09-30 기준이며 매우 빠르게 변동한다(참고용).

| 프로젝트 | 베이스 · 크기 | 학습 방식 | 확률 판독 | 지연 | 하드웨어 | 라이선스 |
|---|---|---|---|---|---|---|
| **Jev** (TypeSafe) | 비공개, "neither small nor an LLM" | RLCD (비공개) | 비공개 (prefill-only 추정) | 70~500ms | API 전용 | closed |
| **kev** (Jared Palmer) | Qwen3.5 0.8B·4B·9B | LoRA r16 + head, cross-entropy | **pointer head** (`</opt>` ↔ `<decide>`) | 수십 ms(H100)~2s(Mac) | CUDA · MPS | Apache-2.0 |
| **Jeff** (firelex) | Qwen3.5 0.8B·2B, Gemma4-E2B | **full-weight SFT** 1 epoch + temperature | **옵션 letter logit** (LM 헤드 유지) | **22ms**(RTX PRO 6000) · 28ms(M4 MLX) | CUDA · MLX · CPU | MIT · 가중치 Apache-2.0 |
| **Jeeves** (PostHog) | Qwen3.5-9B | **SFT + CISPO(RL)**, thinking | pointer head **+ reasoning chain** | 0.3s(no-think) ~ 3.3s(중앙값) | CUDA Hopper(FP8) | MIT |
| **Laya** | ModernBERT-large 421M · mmBERT 322M | **RLCD (GRPO + proper scoring rule)** | set attention head | **33ms**(T4) | CUDA · CPU · MLX 포트 | Apache-2.0 |
| **AutoJev-27B** | Qwen3.8-27B | full-weight SFT, 73k 예시 | 옵션 letter logit | 미공개 | GPU ~49GiB | MIT · Apache-2.0 |
| **SemIf** (구 OpenJev) | Qwen3.5-4B **frozen** | **학습 없음** | native option logit | ~50ms/건 | CUDA · MLX · MPS · llama.cpp · WebGPU · EXL3 | MIT |
| **NanoJev** | Qwen3-0.6B | decision heads, 18,760문항 | set attention + sigmoid | — | CUDA | — |
| **JevK5** | 4B · 9B | teacher 2종 증류 | 옵션 logit | **~13ms**(H100) | CUDA | Apache-2.0 |
| **openJev-verdict-2.0** | ModernBERT-base 150M | 8.8h on GTX 1660 Ti | dual calibration head | 20~25ms | CUDA · WebGPU | Apache-2.0 |

`/v1/systemone` 호환(TypeSafe SDK를 그대로 붙일 수 있음): kev, Jeff, Jeeves, AutoJev, JevK5, opendecision. Jeeves는 `jeeves_sdk`가 `typesafe-sdk`의 drop-in 대체다.

## 5. 성능 — 어디서 이기고 어디서 지는가

### 5-1. 태스크 성격별 (가장 중요한 표)

| 태스크 성격 | 승자 | 근거 |
|---|---|---|
| **분류 · 감성 · 라우팅** | 소형 오픈 모델 | Jeff-2B: Financial PhraseBank **96.3 vs Jev 77.0** |
| **근거 확인 (grounding)** | 소형 오픈 모델 | Jeff-2B: RAGTruth **88.9 vs Jev 77.3** |
| **추론이 섞인 판단** | Jev 또는 thinking 모델 | BBH: Jeff-2B 68.0 vs **Jev 94.3** / JevBench hard: Jeff 0.533, Jev 0.730, **Jeeves 0.865** |
| **지식 질문** | Jev | MMLU: Jeeves 0.793 vs **Jev 0.900**, MMLU-Pro 0.739 vs **0.840** |
| **고카디널리티 (수십~255 옵션)** | Jev | Banking77: Laya 0.425 vs **Jev 0.870** |
| **태스크 특화 (데이터 있음)** | 소형 특화 모델 | NanoJev ViZDoom Basic **128/128 vs Jev 56/128** |
| **제로샷 범용성** | Jev | 학습 없이 임의 도메인, 언어 커버리지 |

요약하면 **"분류·grounding은 2B로 충분히 이기고, 추론·지식·고카디널리티는 아직 Jev"**, 그리고 **"도메인 데이터가 있으면 소형 특화가 이긴다."**

### 5-2. Jeeves가 바꾼 지점

Jeeves는 "생성을 포기한다"는 전제를 깨고 **reasoning chain을 생성한 뒤 결정**한다. 소형 모델들이 전부 무너졌던 추론 계열을 이걸로 메꿨다.

| JevBench hard (111 public items) | 점수 |
|---|---|
| Jeff-2B (추론 없음) | 0.533 |
| Kev-8B (추론 없음) | 0.451 |
| Jev | 0.730 |
| **Jeeves (추론 있음)** | **0.865** |

그 대가는 지연이며, 같은 체크포인트에서 옵션으로 조절한다.

| 설정 | 정확도(dev 325문항) | 평균 reasoning 토큰 | median / p90 |
|---|---|---|---|
| full thinking | 0.825 | 1,138 | 3.3s / **17.1s** |
| `max_think` 768 + `nothink_threshold` 0.9 | 0.806 | 344 | 2.0s / 5.6s |
| no thinking | 0.775 | 0 | **~0.3s** |

`nothink_threshold`는 "no-think 확신도가 임계값 이상이면 생각을 건너뛴다"는 것으로, **우리가 코드에서 구현하던 confidence 게이트가 모델 안으로 들어온 형태**다.

### 5-3. JevBench — 커뮤니티 리더보드에서 Jev는 4위

[JevBench](https://github.com/fstandhartinger/jevbench) v1.4.2.2 기준 95개 시스템(91개 랭킹):

| 순위 | 시스템 | 점수 |
|---|---|---|
| 1 | Imajev-4B | 67.37 |
| 2 | Plumb-4B (JevK5 v0.2 + LoRA) | 65.84 |
| 3 | decider-4b v2 | 64.13 |
| **4** | **Jev 1.13.0 (TypeSafe)** | **63.29** |
| 5 | JevK5 v0.2.0 | 62.04 |
| 6 | Cygnet (frozen Gemma-4-12B-it) | 61.76 |

**해석 주의**: JevBench 점수는 Intelligence · Calibration · Speed · Cost 4축의 **등가중 조화평균**이다. 값싸고 빠른 로컬 4B가 구조적으로 유리하다. 실제로 Jev는 raw Intelligence에서 decider-4b v2를 앞선다(53.1 vs 49.4). 또 308개 fresh sealed 문항에서는 **JevK5 33.1%, Jev 36.7%** 로 모두 낮아 평가자 스스로 "비정상적으로 어려운 세트"라고 밝혔다. 즉 "오픈이 Jev를 추월했다"가 아니라 **"등가중 종합 지표에서는 로컬 4B가 경쟁력을 갖췄다"**가 정확한 독해다.

## 6. 캘리브레이션 — 이 범주의 진짜 쟁점

"확률이 정직하다"가 System One의 핵심 주장이므로, 독립 검증이 가장 중요하다.

| 출처 | 결과 |
|---|---|
| jev-ood-calibration | **in-domain**: OpenBookQA 94.2% ECE 0.024 · CommonsenseQA 88.1% ECE 0.032 · HellaSwag 86.1% ECE 0.029 (우수) |
| 같은 실험 (**OOD** 900건 합성 티켓) | 전체 75.1% **ECE 0.107** · Noul temperature 0.66(과소신) · **Choice 3.29(심한 과신)** · Score 3.40 |
| lindfors.no (노르웨이어 문서 24건) | 확률 0.7~0.9 구간 정답 일치 **97%**, 0.9~1.0 구간 **98%** |
| Archer Hume (MMLU 1,200문항) | ECE **0.031** |
| SemIf (오픈 재현) | authored raw ECE 0.068 → 보정 0.038 (T=1.23) · **WANLI 0.208 → 0.069 (T=2.50)** |
| Jeeves | JevBench public ECE **0.037** vs Jev 0.049 |

결론이 일관된다: **분포 안에서는 캘리브레이션이 훌륭하고, 분포 밖에서는 Jev든 오픈 재현이든 과신한다.** 따라서 자체 골든셋으로 confidence–정확도 곡선을 그리고 temperature를 다시 맞추는 작업은 선택이 아니라 필수다. 질문 타입별로 보정 방향이 다르다는 점도 중요하다(Choice는 과신, Noul은 과소신).

## 7. 무엇을 고를 것인가

```mermaid
flowchart TD
  Q0{"답의 공간을 미리 열거할 수 있나?"}
  Q0 -->|"아니오"| LLM["일반 LLM을 쓴다"]
  Q0 -->|"예"| Q1{"판단 기준이 룰·정규식으로 표현되나?"}
  Q1 -->|"예"| CODE["그냥 코드로 작성"]
  Q1 -->|"아니오"| Q2{"데이터 반출이 가능한가?"}
  Q2 -->|"아니오"| Q3{"추론이 필요한 판단인가?"}
  Q2 -->|"예"| Q4{"고카디널리티 또는 제로샷 범용성이 필요한가?"}
  Q4 -->|"예"| JEV["Jev"]
  Q4 -->|"아니오"| Q3
  Q3 -->|"예"| JEEVES["Jeeves — 9B CUDA, 0.3~3.3초"]
  Q3 -->|"아니오"| Q5{"한국어 또는 다국어인가?"}
  Q5 -->|"예"| LAYA["Laya-multilingual — mmBERT"]
  Q5 -->|"아니오"| Q6{"내 데이터로 학습할 것인가?"}
  Q6 -->|"예"| JEFFKEV["Jeff 또는 kev — 파인튜닝 경로 검증됨"]
  Q6 -->|"아니오"| SEMIF["SemIf — 학습 없이 즉시"]
```

| 목적 | 선택 | 이유 |
|---|---|---|
| 실시간 라우팅·게이트 (수십 ms) | **Jeff-0.8B** · Laya · Jev | 22~33ms |
| 어려운 판단, 지연 감수 가능 | **Jeeves** | JevBench hard 0.865 |
| 하나로 두 경우 모두 | **Jeeves + `nothink_threshold`** | 쉬운 건 0.3s, 어려운 건 생각 |
| 금융 텍스트 분류·감성 | **Jeff** | Financial PhraseBank 96.3 |
| 한국어·다국어 | **Laya-multilingual** | Jeff는 영어 전용 |
| 최고 정확도를 로컬에서 | **AutoJev-27B** | 84.6%, Jev(82.8%) 초과 |
| 학습 없이 오늘 확인 | **SemIf** | 브라우저 WebGPU 데모 |
| RLCD 알고리즘 연구 | **Laya** · [RLCR 논문](https://arxiv.org/abs/2507.16806) | 유일한 공개 RLCD 구현 |
| 데이터 출처 청결성이 중요 | **Jeff** | 클로즈드 모델 출력 미사용, 소스별 라이선스 명시 |

## 8. 종합 평가

**강점**
- `/v1/systemone`이라는 얇은 계약 하나가 사실상 표준이 되어, **모델을 갈아 끼울 수 있는 구조**가 확보됐다. 벤더 락인 리스크가 이례적으로 낮다.
- 타입 안전성은 수학적 보장이다. 파싱 실패·스키마 이탈이라는 LLM 통합의 만성 문제가 사라진다.
- 반복 평가(judge) 용도에서 LLM 대비 일관성이 압도적이다. LangSmith 실험에서 점수 분산이 LLM judge의 **1/92 ~ 1/913**.
- 비용 구조가 새 용도를 연다. 문서 1,000건 분류 $0.22, 777건 판정 0.7초 등, 기존에는 비용 때문에 시도할 수 없던 워크로드가 가능해진다.

**약점 · 리스크**
- **"0% hallucination"은 타입 안전성을 뜻할 뿐**이다. 유효하지만 틀린 값을 높은 confidence로 낼 수 있고, 그것을 알려주는 유일한 신호가 confidence다.
- **OOD 캘리브레이션이 무너진다.** Choice에서 temperature 3.29가 필요했다는 것은 "0.74 확률이 실제로는 훨씬 낮다"는 뜻이다.
- **추론이 필요한 판단에서 실패 양상이 위험하다.** 포커에서 넛츠를 들고 4배 팟 올인을 62% 선택(솔버는 100% 체크) — 틀리는 게 아니라 *자신 있게* 틀린다.
- Jev 자체는 closed weight · early access · 데이터 외부 전송 전제. 규제 도메인에서는 state 설계가 곧 컴플라이언스 문제가 된다.
- 생태계 지표에 거품이 있다. awesome 리스트조차 "같은 날 한 사람이 스캐폴드로 올린 벌크 제출을 주의하라"고 경고한다.

**엔지니어 관점 인사이트**
1. **이 범주의 해자는 아키텍처가 아니다.** 4B 프로즌 모델이 학습 없이 0.845(Jev 0.883)에 도달하고, 150M 인코더가 typed-decisions에서 앞선다. 남는 차별점은 **데이터 · 캘리브레이션 품질 · 단가 · 제로샷 범용성**이다.
2. **System One / System Two의 경계가 인터페이스와 내부 계산으로 분리되기 시작했다.** Jeeves가 증명했듯 계약(타입 출력 + 확률)은 유지하면서 내부에서 생성을 해도 된다. 클라이언트 코드를 바꾸지 않고 22ms ↔ 3.3s 모델을 교체할 수 있다.
3. **도입 시 첫 작업은 모델 선택이 아니라 골든셋 구축이다.** confidence–정확도 곡선 없이 임계값을 정하는 것은 이 범주에서 가장 흔하고 비싼 실수다.
4. **2단 구성이 현실적인 기본형이다.** 1차는 초저지연 소형 모델(Jeff·Laya), confidence 미달 시 2차로 thinking 모델(Jeeves) 또는 LLM·사람. 세 계층 모두 같은 API 계약을 쓴다.

---

## 참고 자료

**공식**
- [Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [TypeSafe 문서](https://docs.typesafe.ai/) · [Workflow evals](https://evals.typesafe.ai/) · [Manifesto](https://typesafe.ai/manifesto)
- [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) — LLM으로 같은 인터페이스를 흉내내는 공식 MIT 어댑터

**아키텍처 역추론**
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)

**오픈 구현체**
- [kev](https://github.com/jaredpalmer/kev) · [Jeff](https://github.com/firelex/jeff) · [Jeeves](https://github.com/PostHog/jeeves) · [Laya](https://github.com/NandhaKishorM/laya) · [SemIf](https://github.com/TheoLeeCJ/SemIf) · [NanoJev](https://github.com/TianyuCodings/NanoJev) · [AutoJev-27B](https://github.com/denis-pplx/autojev) · [JevK5](https://github.com/allebee/jevk5) · [openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) · [opendecision](https://github.com/konsumer/opendecision)

**벤치마크 · 검증**
- [JevBench v1](https://github.com/fstandhartinger/jevbench) · [jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration) · [LangChain — Jev as agent evaluator](https://www.langchain.com/blog/jev-agent-evals-langsmith) · [A first look at Jev](https://lindfors.no/blog/a-first-look-at-typesafes-jev/) · [Jev is the fish at the poker table](https://backnotprop.com/blog/jev-poker/)

**학술 배경**
- [Beyond Binary Rewards (RLCR)](https://arxiv.org/abs/2507.16806) · [GLiClass](https://arxiv.org/abs/2508.07662) · [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) · [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) · [Orthrus](https://arxiv.org/abs/2605.12825)
