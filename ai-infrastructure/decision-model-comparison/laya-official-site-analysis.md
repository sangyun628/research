# Laya 공식 사이트 분석 — 주장, 코드, 측정 조건

확인일: 2026-09-22. 대상은 [Laya 공식 사이트](https://laya.convaiinnovations.com/)다. 사이트가 연결한 논문·모델 카드와, 이미 복제한 Laya `573e5b62696ba441230cd6be71d593331b5d23af`의 소스·벤치마크를 대조했다. 전체 모델 구조와 Kev 비교는 [기존 분석](kev-vs-laya.md)에 있다.

이번에는 모델 가중치를 내려받거나 추론·학습 성능을 새로 측정하지 않았다. **가중치를 로드하지 않는 `Router.route()` 동작만 로컬 실행으로 확인**했다. 논문상의 과거 성능도 재현한 결과가 아니다.

## 1. 사이트에서 추가로 파악한 방향

사이트는 생성형 LLM이 맡던 분류·필터·업무 배정을 비생성형 모델로 처리하려는 개발 배경을 설명한다. 세 가지 출력 형식, 언어별 체크포인트 선택, 자체 호스팅, 분야별 추가 학습을 중심으로 소개한다. 저자의 2025년 판매 전환 예측·신뢰도 라우팅 연구를 현재 프로젝트의 배경으로 연결한다.

문서에 반영할 핵심은 **빠른 encoder 기반 판단 계층을 자체 학습해 배포할 수 있다는 방향**이다. 사이트의 속도·정확도·확신도 표현은 각각 측정 조건을 붙여 해석해야 한다. [사이트 원문](https://laya.convaiinnovations.com/)

## 2. 주요 표현과 엔지니어 관점의 해석

| 사이트가 강조하는 내용 | 대조 결과와 문서에 반영할 해석 |
|---|---|
| 약 33ms 추론 | 공개 원자료의 **T4 GPU 단일 질문 p50 32.8ms**다. CPU 보장치가 아니다. |
| 배치에서 질문당 7.2ms | 질문 열 개 배치의 전체 p50는 72.3ms다. 개별 요청의 지연시간과 배치 처리량을 구분한다. |
| Jev 대비 약 7.8배, 배치 20배 | Laya 로컬 실행과 외부 Jev 보고를 결합한 비교다. 공통 장비·프롬프트·배치 조건의 직접 실험으로 확인되지 않는다. |
| 76.6% 판단 정확도 | 영어 `laya-typed-decisions`를 해당 업무 데이터에 추가 학습한 결과다. 기본 다국어 모델의 일반 정확도가 아니다. |
| 보정된 확률 | 모델·업무에 따라 별도 보정이 필요하다. 다국어 공개 모델의 temperature는 모두 1이다. |
| 환각 없는 출력 | 후보 범위와 응답 형식을 코드가 구성하는 이점이다. 잘못된 후보를 높은 확률로 선택하는 오류는 남는다. |
| 100개 이상 언어 | 언어 지원 범위와 실용 정확도는 다르다. 공개된 51개 언어 실험의 조건을 함께 본다. |
| 자체 호스팅 비용 0 | API 사용료를 지불하지 않는다는 의미로 제한한다. CPU·GPU·메모리·전력·학습 비용은 남는다. |

첫 열은 [공식 사이트](https://laya.convaiinnovations.com/)의 주장 요약이다. 해석의 근거는 [T4 측정 원자료](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/results/t4_colab_benchmark.json), [비교 실험의 출처 설명](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/README.md), [다국어 모델 카드](https://huggingface.co/convaiinnovations/laya-multilingual), [특화 모델 카드](https://huggingface.co/convaiinnovations/laya-typed-decisions)다.

## 3. 선행 연구의 존재와 현재 구현의 관계

사이트가 인용한 두 논문은 실제로 공개되어 있다. 다만 내용과 범위를 구분해야 한다.

| 자료 | 공개 내용 | 현재 Laya와의 관계 |
|---|---|---|
| SalesRLAgent, 2025-03-30 | 판매 대화의 상태·embedding을 이용해 턴별 전환 확률을 예측하는 강화학습 접근 | 특정 업무의 수치 예측이라는 선행 연구. 현재 mmBERT 기반 가변 후보 모델과 같은 아키텍처라는 근거는 아님 |
| Confidence-Aware Routing, 2025-09-23 | 내부 표현의 의미 정렬·계층 수렴·학습한 신뢰도를 결합해 로컬 생성·RAG·큰 모델·사람 검토로 분기 | 신뢰도 기반 라우팅 연구. 현재 typed decision 모델의 RLCD 학습 알고리즘을 직접 기술한 논문으로 취급하면 안 됨 |

첫 논문은 Azure OpenAI embedding과 상태·정책·가치 표현을 설명한다. 연결된 [판매 예측 모델 카드](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning)는 Stable Baselines3 PPO 기반이라고 명시한다. 현재 Laya의 양방향 인코더와 후보 marker scorer와는 구분된다. [SalesRLAgent 논문](https://arxiv.org/html/2503.23303v1)

두 번째 논문의 학습 절차는 의미 정렬 손실·신뢰도 직접 지도·정규화를 설명하며, 신뢰도 임계값에 따른 결정적 라우팅을 사용한다. 따라서 사이트의 연구 계보 설명은 저자의 배경으로 기록하고, 이를 현재 Laya와 Jev가 동일한 기술이라는 증명이나 최초 발명에 대한 판단으로 확대하지 않는다. [Confidence-Aware Routing 논문](https://arxiv.org/html/2510.01237v1)

## 4. Router의 실제 역할과 기본값

`Router`는 질문의 답을 먼저 추론해서 최적 모델을 고르는 별도의 신경망이 아니다. `laya/lang.py`의 문자·언어 휴리스틱과 명시적 옵션으로 체크포인트를 선택한다. 업무 특화 자동 감지는 네 업무의 **질문 ID 집합**을 비교하며, `auto_task_detection=False`가 기본값이다.

분석한 코드에서 영어 고객지원 문장과 등록된 고객지원 질문 ID 집합으로 직접 확인한 결과는 다음과 같다. 실제 분류 예측이 아니라 `route()`만 호출했으며 모든 경우 `loaded=[]`였다.

| 호출 조건 | 선택된 체크포인트 |
|---|---|
| 기본 `Router().route(...)` | `english` |
| `route(..., model="typed-decisions")` | `typed-decisions` |
| `Router(auto_task_detection=True).route(...)` | `typed-decisions` |
| 기본 Router에 한국어 금융 문장 입력 | `multilingual` |

그러므로 “라우터를 쓰면 76.6% 모델을 자동으로 사용한다”는 해석은 현재 기본 동작과 맞지 않는다. **모델을 미리 올리는 `preload`도 선택 정책을 바꾸지 않는다.** 특화 모델을 쓰려면 명시적으로 선택하거나 해당 자동 감지를 켜야 한다. 업무 이름·언어보다 명시적 모델 지정이 우선한다는 점도 고려해야 한다. [Router 구현](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/router.py)

한국어 금융 서비스가 모델 하나만 쓸 계획이라면 언어별 자동 라우터보다 아래처럼 다국어 모델을 지정하는 구성이 명확하다. 이 예시는 이번 작업에서 실행하지 않았다.

```python
import laya

agent = laya.load(
    "convaiinnovations/laya",
    subfolder="multilingual",
    device="cpu",
)
# 서비스 시작 시 한 번 로드하고, 각 요청에서 agent.predict(state, questions)를 호출한다.
```

## 5. 선택 다운로드, 메모리, 오프라인 배포

사이트의 단일 Hub·subfolder 설명은 코드와 일치한다. `Agent`는 `allow_patterns`를 사용해 선택한 체크포인트의 설정·가중치·tokenizer·encoder 파일만 요청한다. 다국어 모델만 쓰기 위해 세 모델을 모두 내려받을 필요는 없다. 독립 저장소 ID `convaiinnovations/laya-multilingual`도 제공된다. [다운로드·로드 코드](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/agent.py)

사이트가 제시한 다국어 다운로드 약 647MB와 **CPU 실행 RAM은 다른 수치**다. 현재 SDK는 CPU에서 FP32 모델을 사용하므로 약 322M 파라미터의 가중치만 대략 1.29GB이며 activation·tokenizer·임시 로딩 tensor가 추가된다. 세 모델을 모두 상주시킬 경우 약 1.16B 파라미터, FP32 가중치만 약 4.66GB 규모다. 이는 크기로부터 계산한 값이지 실측 RSS가 아니다.

`Router(preload=True)`는 기본적으로 세 모델을 모두 올려 재로딩을 줄이는 대신 메모리를 사용한다. 반대로 기본 `max_loaded=1`은 언어가 바뀔 때 모델을 다시 로드할 수 있다. 한국어 전용 서버에 세 모델 preload를 적용할 이유는 약하다. [preload·LRU 구현](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/router.py)

오프라인 배포도 구성 가능하다. 필요한 패키지와 가중치·tokenizer·encoder 설정을 미리 확보하고 `laya.load()`에 로컬 경로를 넘겨야 한다. 처음부터 Hub ID를 넘겨 다운로드하는 quickstart가 인터넷 없는 서버에서도 그대로 작동한다는 뜻은 아니다.

## 6. 정확도, 보정, confidence의 차이

사이트의 ECE 개선을 모든 요청에 대한 정확도 보장으로 사용하면 안 된다. 공개 코드와 원자료에서 확인되는 조건은 다음과 같다.

- 영어 모델의 평균 ECE는 0.466에서 0.081, 다국어 모델은 0.314에서 0.106으로 보고되었다.
- 실험은 **각 데이터셋 안에서 절반을 보정용, 나머지를 평가용**으로 나누고, 질문 유형과 후보 개수 구간별로 temperature를 맞춘다.
- 모든 업무를 통틀어 단일 temperature 하나를 구해 새 업무에 그대로 적용한 결과가 아니다.
- 다국어 체크포인트는 기본 temperature가 모두 1이며, 영어 체크포인트는 기존 보정값이 있다. 모든 기본 모델이 동일한 보정 상태라고 묶을 수 없다.

또한 벤치마크의 ECE 계산은 **최대 후보 확률**을 사용하지만, SDK의 `choice.confidence`는 **정규화한 entropy**를 사용한다. ECE가 작아졌다는 이유만으로 `confidence >= 0.85`에서 오류율이 15% 이하라고 해석할 수 없다. [보정 실험 코드](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/scripts/build_benchmark_nb.py), [confidence 계산](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/common.py)

현재 SDK는 극단적인 temperature를 0.5~5 범위로 제한한다. 과거 측정 원자료에는 이보다 작은 영어 보정값도 기록되어 있어, 웹페이지의 모든 수치를 최신 SDK의 기본 동작으로 그대로 재현할 수 있다고 보장하지 않는다. [SDK 보정값 처리](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/agent.py)

공개 51개 언어 실험에서 “사용 가능”이라고 집계한 기준도 20개 후보의 무작위 정답률 5%보다 세 배 넘게 맞히는 것이다. 즉 15% 초과라는 실험상 기준이며 높은 제품 정확도를 뜻하지 않는다. 한국어의 45~49% 결과는 [기존 비교](kev-vs-laya.md#5-한국어와-공개-정확도-수치의-의미)에 상세히 기록했다.

## 7. 활용 사례는 분야별 근거로 나눠 본다

저장소의 업무별 표는 모든 활용 사례가 같은 수준으로 검증된 것이 아님을 보여준다. 아래는 제작자 보고이며 이번 조사에서 재측정하지 않았다.

| 업무 | 공개 결과와 조건 | 적용 판단 |
|---|---|---|
| 스팸 | 영어·다국어 99.3%, 학습에 포함된 데이터 출처 | 강점의 근거지만 새로운 업무·언어 전체의 정확도로 일반화하지 않음 |
| 피싱 | 영어 98.0%, 다국어 99.3%, 학습 출처 | 학습한 분야의 성능으로 해석 |
| 탈옥 탐지 | 다국어 75.5%, 특화 모델 76.2%, 미학습 출처 | 자동 보안 차단의 충분한 정확도라고 단정할 수 없음 |
| 유해성 분류 | 약 52.5~53.0%, 미학습 출처 | 탈옥 탐지와 다른 과제이며 더 약한 결과 |
| RAG 관련성 | 다국어 65.7% | 후보 기능은 있으나 운영 목표에 충분한지는 별도 판단 |
| 고객지원 10개 큐 | 다국어 52.2% | 금융 21개 클래스에 추가 학습 없이 충분하다는 근거가 아님 |

학습 출처라는 표기는 학습·평가에 정확히 같은 사례가 중복되었다는 확인이 아니다. 출처가 학습에 포함됐다는 구분이다. 사이트의 선택적 처리 성능도 **전체 요청 중 처리하는 비율과 그 부분집합의 정확도**를 함께 읽어야 한다. 절반만 처리한 결과를 모든 요청의 정확도로 쓰지 않는다. [업무별 벤치마크](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/BENCHMARKS.md), [업무 데이터 구성 코드](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/scripts/bench_apps.py)

분석 커밋에 포함된 원자료는 T4 비교와 CPU 언어 조사 두 파일이다. `BENCHMARKS.md`가 언급하는 `research/results/app_benchmark.json`은 이 체크아웃에 없어, 위 업무별 수치는 보고서·실험 스크립트 수준으로 확인했다.

## 8. 현재 금융 프로젝트에 반영할 내용

공식 사이트를 추가로 확인해도 **CPU·한국어·고정 21개 클래스에서는 mmBERT-small 분류기를 기준으로 두고 Laya-multilingual 추가 학습을 비교한다**는 판단은 유지한다.

1. 한국어에는 다국어 체크포인트 하나를 명시적으로 로드한다. 특화 영어 모델의 성능을 가져오지 않는다.
2. 최근 세 질문을 하나의 상태로 만들고, 분류 질문 하나에 21개 플레이북을 넣는다.
3. 20개를 넘는 후보에서 기본 길이 예산이 부족할 수 있으므로 설명 보존 여부를 확인한다. 21개가 하드 제한은 아니며, `head_max_len`과 전체 길이를 조정할 수 있다. 단지 20개로 줄이려고 `플레이북 없음`을 제거하거나 계층 분류를 필수로 도입하지 않는다.
4. mmBERT 기반의 최대 8k 길이 지원과 Laya 기본 1,024토큰을 구분한다. 길이를 늘려도 같은 속도·메모리·정확도가 유지된다고 볼 수 없다.
5. 정확도 향상을 위한 추가 학습과 확률 보정을 별도로 다룬다. temperature 조정만으로 분류 경계를 학습하는 것은 아니다.

공개 체크포인트를 그대로 실행해 보는 것은 가능하다. 그 초기 결과와 사용자 라벨 데이터를 이용해, 후보 설명을 읽는 구조가 일반 분류기보다 유리한지 판단하는 것이 현재 자료에서 지지되는 활용 방향이다.
