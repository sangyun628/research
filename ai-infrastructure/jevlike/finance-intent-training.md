# Jevlike Tiny로 금융 질문 의도 분류 학습하기

작성일: 2026-09-21. Jevlike `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`의 학습 코드를 기준으로 한다. 금융 거래를 수행하는 코드가 아니라 질문의 의도를 선택하는 모델을 학습하는 방법이다.

## 제공되는 코드

원본 프로젝트에 이미 다음 코드와 CLI가 있다.

| 기능 | 코드 | CLI |
|---|---|---|
| 데이터 로딩 | `jevlike/data.py` | JSONL의 문맥·후보·정답 읽기 |
| 학습 | `jevlike/train.py` | `jevlike-train` |
| 평가 | `jevlike/eval.py` | `jevlike-eval` |
| 추론 | `jevlike/predict.py` | `jevlike-predict` |

각 CLI는 `python -m jevlike.train`, `python -m jevlike.eval`, `python -m jevlike.predict`로도 실행할 수 있다. 원본 학습 코드는 PyTorch에서 후보 logits와 정답 인덱스의 cross entropy를 계산하고 AdamW로 갱신한다. [원본 학습 코드](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/train.py).

이 디렉터리의 [train_local.py](train_local.py)는 원본 학습 CLI를 호출하는 작은 실행 파일이다. 앞서 발견한 CPU의 최적 epoch 저장 문제를 피하도록 snapshot에 `.clone()`을 추가한다. 원본 소스는 수정하지 않고 실행 중인 프로세스에서 snapshot 함수만 교체한다. 데이터 형식, 모델, 손실 함수, CLI 인자는 원본과 같다.

## 1. 질문과 정답 데이터 준비

다음은 **형식 설명용 예시**이며 네 문장만으로 실무 분류기를 학습할 수 있다는 뜻은 아니다. 각 줄이 독립적인 JSON 객체인 `.jsonl` 파일로 저장한다.

```jsonl
{"context":"다른 계좌로 돈을 보내고 싶어요.","options":["계좌이체","잔액조회","이체취소","기타문의"],"label":0}
{"context":"통장에 얼마 남았는지 알려 주세요.","options":["계좌이체","잔액조회","이체취소","기타문의"],"label":1}
{"context":"방금 보낸 돈을 취소하고 싶어요.","options":["계좌이체","잔액조회","이체취소","기타문의"],"label":2}
{"context":"가까운 지점 영업시간을 알려 주세요.","options":["계좌이체","잔액조회","이체취소","기타문의"],"label":3}
```

- `context`: 사용자의 질문. 이전 대화가 판단에 필요하면 필요한 내용을 함께 넣는다.
- `options`: 선택 가능한 의도 이름. 고정 의도 분류라면 모든 행에서 같은 후보 목록을 사용하는 방식으로 시작할 수 있다.
- `label`: 정답 후보의 **0부터 시작하는 인덱스**. 후보 순서를 바꾸면 정답 인덱스도 맞춰야 한다.

“송금해 줘”, “다른 통장으로 옮기고 싶어”처럼 같은 의도의 다양한 표현과 “송금 말고 잔액만 확인할래”처럼 혼동하기 쉬운 질문을 포함한다. 실제 분류 기준은 프로젝트에 맞게 정의한다. 일반 금융 문서만 넣는 방식이 아니라, 각 질문에 맞는 의도 정답이 필요하다.

이 예시는 문장당 대표 의도 하나를 정한다. 복합 의도를 동시에 여러 개 출력하는 multi-label 분류는 기본 cross entropy 학습에 구현되어 있지 않다. `기타문의` 역시 실제 예제를 제공해 학습해야 하며, 후보에 이름만 추가한다고 미지의 질문을 자동으로 감지하지는 않는다.

데이터는 다음 세 파일로 준비한다. 같은 대화나 거의 같은 문장의 변형이 여러 파일에 섞이지 않게 나눈다.

| 파일 | 목적 |
|---|---|
| `.repos/jevlike/data/finance/train.jsonl` | 가중치를 갱신할 질문·정답 |
| `.repos/jevlike/data/finance/validation.jsonl` | 학습 중 모델을 선택할 질문·정답 |
| `.repos/jevlike/data/finance/test.jsonl` | 학습에 사용하지 않은 질문·정답 |

이 문서에는 실제 프로젝트 데이터를 만들거나 수집하지 않았다. 필요한 데이터 양은 의도 수와 표현 다양성에 따라 달라지며 보편적인 최소 건수는 제시하지 않는다.

## 2. Tiny 학습

이 컴퓨터에는 `.repos/jevlike/.venv`와 Jevlike가 설치되어 있다. 아래는 **research 루트에서 실행하는 명령**이다. 먼저 위 데이터 파일들을 준비해야 한다.

```bash
cd /Users/cleave-sangyun/opensource/research

OMP_NUM_THREADS=2 .repos/jevlike/.venv/bin/python \
  ai-infrastructure/jevlike/train_local.py \
  .repos/jevlike/data/finance/train.jsonl \
  --validation .repos/jevlike/data/finance/validation.jsonl \
  --output .repos/jevlike/runs/finance-tiny.pt \
  --encoder tiny \
  --context-tokens 768 \
  --option-tokens 192 \
  --epochs 8 \
  --batch-size 32 \
  --device cpu
```

GPU나 Qwen 다운로드는 필요하지 않다. `768/192`, 8 epochs와 batch size 32는 시작 설정 예시이며 금융 의도 분류에 최적화한 값이 아니다.

Tiny의 길이 단위는 UTF-8 **bytes**다. 768 bytes는 한글 음절만 있을 경우 대략 256음절에 해당하며, 공백·숫자·기호에 따라 달라진다. 긴 질문은 잘릴 수 있으므로 실제 입력 길이에 맞춰 설정한다.

이 명령은 새 Tiny 모델을 초기화해 embedding과 후보 선택 head를 학습한다. 배지 데모 체크포인트를 금융용으로 자동 전환하거나, Qwen을 파인튜닝하는 명령이 아니다. 완료 후 `finance-tiny.pt`에 모델 설정과 가중치가 저장된다. 같은 `--output` 파일이 이미 있으면 덮어쓰며, 이어서 학습하지 않는다.

## 3. 학습한 모델 평가와 추론

```bash
.repos/jevlike/.venv/bin/python -m jevlike.eval \
  .repos/jevlike/runs/finance-tiny.pt \
  .repos/jevlike/data/finance/test.jsonl \
  --device cpu

.repos/jevlike/.venv/bin/python -m jevlike.predict \
  .repos/jevlike/runs/finance-tiny.pt \
  --context "다른 은행 계좌로 돈을 보내고 싶어요." \
  --option "계좌이체" \
  --option "잔액조회" \
  --option "이체취소" \
  --option "기타문의" \
  --device cpu
```

`predict`는 각 후보의 점수를 JSON으로 출력한다. 가장 높은 점수의 후보를 선택하는 부분은 애플리케이션에서 처리하거나 기존 로컬 데모 서버를 사용할 수 있다. 점수는 후보 사이의 softmax 값이며 실제 정답 확률을 보장하지 않는다.

학습된 모델을 기존 화면에 연결하려면 다음 명령을 직접 실행한다.

```bash
sh ai-infrastructure/jevlike/start-demo.sh \
  --checkpoint .repos/jevlike/runs/finance-tiny.pt
```

화면의 기본 배지 예제는 금융 질문과 의도 후보로 바꿔 입력한다. 이 문서를 작성하면서 서버를 다시 실행하지 않았다.

## 추가 튜닝

원본 텍스트 학습 CLI에는 `--resume`이 없다. 기존 금융 체크포인트를 불러온 후 새 데이터로 이어 학습하려면 [자체 학습 가이드의 warm start 코드](training-and-finetuning.md#41-텍스트-모델의-warm-start)를 사용한다. 이는 가중치를 시작점으로 삼는 방식이며 optimizer 상태까지 복원하는 완전한 resume는 아니다.

금융 의미를 이미 학습한 모델이나 충분한 정확도를 이 실행 절차가 제공하는 것은 아니다. Tiny 구조의 제약과 사전학습 모델 대안은 [mmBERT 비교](../decision-model-comparison/jevlike-vs-mmbert.md)를 참고한다.
