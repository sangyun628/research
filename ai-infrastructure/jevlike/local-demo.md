# Jevlike 로컬 데모

구현·실행일: 2026-09-18. 원본 Jevlike 커밋 `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`의 추론 코드를 호출하는 로컬 웹 화면이다. 서버와 화면만 이 research 저장소에 추가했으며 원본 모델 코드는 수정하지 않았다.

## 실행

이 컴퓨터에는 Jevlike 클론, Python 가상환경, 학습된 기본 체크포인트가 준비되어 있다.

```bash
cd /Users/cleave-sangyun/opensource/research
sh ai-infrastructure/jevlike/start-demo.sh
```

브라우저에서 [http://127.0.0.1:8765](http://127.0.0.1:8765)를 연다. 이미 같은 포트에 데모가 떠 있으면 바로 접속하면 된다. 직접 터미널에서 실행한 서버는 `Ctrl+C`로 종료한다.

다른 포트로 실행하려면 다음과 같이 지정한다.

```bash
sh ai-infrastructure/jevlike/start-demo.sh --port 8766
```

## 화면 사용법

1. 접속하면 기본 예제를 자동으로 평가한다.
2. **다른 예제**는 기존 합성 평가 데이터에서 예제를 골라 다시 평가한다.
3. 문맥과 선택지를 직접 수정한 뒤 **선택지 평가하기**를 누른다. 단축키는 `⌘+Enter` 또는 `Ctrl+Enter`다.
4. 오른쪽에서 가장 높은 점수의 선택지, 전체 후보의 점수, 처리시간을 확인한다.
5. 하단 **요청·응답 JSON 보기**에서 API 입력과 출력을 확인한다.

선택지는 한 줄에 하나씩 2~16개 입력한다. 16개는 데모 서버의 제한이며 원본 모델의 고정 클래스 수가 아니다. 기본 모델의 학습 데이터에는 선택지 2~8개가 사용됐다.

예제 문맥에서 `azure falcon`을 두 곳 모두 `green falcon`으로 바꾸면 같은 후보 목록을 두고 다른 배지를 선택하는지 볼 수 있다. 사용자가 입력을 바꾸면 예제 정답 표시는 해제된다.

## 현재 모델의 범위

| 항목 | 기본값 |
|---|---|
| 모델 | Jevlike `TinyScorer`, 41,280 파라미터 |
| 실행 장치 | CPU, PyTorch 연산 스레드 2개 |
| 체크포인트 | `.repos/jevlike/runs/research/synthetic.pt` |
| 학습 데이터 | 영어 색상·동물 배지 매칭, 2,000개, 8 epochs |
| 문맥 길이 | 앞 192 UTF-8 bytes |
| 선택지 길이 | 각 선택지의 앞 32 UTF-8 bytes |
| 출력 | 후보별 softmax 점수와 최고 점수 후보 |
| 추가 모델 다운로드·API 키 | 기본 Tiny 데모에는 불필요 |

**현재 데모는 Qwen을 사용하지 않는다.** Tiny는 사전학습된 범용 언어 모델이 아니다. 한글 상담 분류, 일반 상식, 복잡한 추론 등을 하려면 목적에 맞는 데이터로 따로 학습해야 한다. 입력은 가능하지만 그 결과를 업무 정확도로 해석할 수 없다.

기본 체크포인트 설정에는 `hf_model`이라는 기본 문자열이 남아 있지만, `encoder=tiny` 분기에서는 Hugging Face 모델을 로드하지 않는다.

길이는 글자 수가 아니라 UTF-8 byte 수다. 한글 음절은 대체로 3 bytes이므로 기본 문맥에는 약 64음절만 들어간다. 초과 입력은 원본 collator가 앞부분만 사용하며, 데모는 이 사실을 경고한다. 후보 내부의 byte 순서가 평균 풀링으로 사라지는 Tiny의 제약도 그대로 유지된다.

100%에 가까운 점수도 **제시된 후보 사이의 상대적인 값**이다. 실제 정답 확률로 보정된 값이 아니며, 학습 범위 밖의 입력에도 높게 나올 수 있다. 배경과 자세한 한계는 [분석 보고서](jevlike-analysis.md)를 참고한다.

## 처리시간과 서버 동작

`demo_server.py`는 Python 표준 라이브러리 `ThreadingHTTPServer`를 사용하고, 원본 `load_checkpoint`, collator, 모델 `forward`를 호출한다. 모델을 서버 시작 때 한 번 로드하고 계속 재사용한다. 추론은 lock으로 직렬화하고 `torch.inference_mode()`에서 실행한다.

- **모델 처리**: 입력 tensor 준비, 모델 추론, softmax, 결과를 CPU list로 변환하는 시간. GPU 사용 시 동기화가 포함된다.
- **로컬 요청 왕복**: 브라우저 요청부터 응답 JSON 읽기까지. 서버 대기·HTTP·직렬화 비용이 더해진다.

모델 다운로드·최초 로딩은 표시 시간에 포함되지 않는다. 작은 모델에서는 브라우저·서버 비용의 비중이 크므로 두 시간을 구분한다. 엄밀한 반복 측정 결과는 [성능 문서](../decision-model-comparison/performance-and-cost.md)를 참고한다.

서버는 `127.0.0.1`에만 바인딩한다. 입력 본문은 파일에 저장하지 않으며, 콘솔에는 HTTP 경로·상태 코드 등 접속 로그가 남는다. 학습·외부 API 호출은 요청 처리에 포함되지 않는다. 일반 서비스 배포용 서버가 아닌 로컬 실험용이다.

## HTTP API

| 경로 | 기능 |
|---|---|
| `GET /` | 데모 화면 |
| `GET /api/status` | 모델·장치·길이 한도·학습 안내 |
| `GET /api/example` | 합성 배지 예제 한 개와 정답 인덱스 |
| `POST /api/predict` | 문맥과 선택지 평가 |

```bash
curl http://127.0.0.1:8765/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"context":"Choose the exact badge amber badger. Notes: north east west south. Badge: amber badger.","options":["amber badger","azure falcon","green gecko"]}'
```

응답의 `prediction.index`와 `scores[].index`는 0부터 시작한다. 화면에서 보여주는 후보 번호는 1부터 시작한다. `inference_ms`는 처리시간이고, `warnings`는 입력 잘림 안내다. 정답 `label`은 추론 API에 필요하지 않다.

## 직접 학습한 모델 연결

```bash
sh ai-infrastructure/jevlike/start-demo.sh \
  --checkpoint /absolute/path/to/my-model.pt \
  --device cpu \
  --port 8766
```

원본 텍스트 학습 CLI가 저장한 체크포인트 형식을 사용한다. 모델 선택은 서버 시작 옵션으로 지정하며 브라우저에서 파일을 업로드하지 않는다. 체크포인트는 신뢰할 수 있는 본인 생성 파일을 사용한다. 원본 로더가 `torch.load(..., weights_only=False)`를 사용하기 때문이다.

HF 인코더로 학습한 체크포인트도 원본 로더를 통해 연결할 수 있다. 이 경우 `.repos/jevlike/.venv`에 `transformers`가 추가로 필요하고, 기반 모델이 캐시에 없으면 로딩 시 다운로드된다. CPU·MPS·CUDA 장치 옵션을 전달할 수 있지만 **이번 데모의 실행 확인 범위는 Tiny + CPU**다. 게임용 Doom·체스 체크포인트는 텍스트 모델과 형식이 다르므로 이 화면에서 사용하지 않는다.

데모에는 학습 버튼이 없다. 데이터 준비와 학습 절차는 [자체 학습 가이드](training-and-finetuning.md)를 따르고 학습 결과를 `--checkpoint`로 연결한다.

## 새 환경에서 기본 데모 준비

아래는 research 루트에서 **클론·가상환경·데이터가 없는 새 환경**에 적용하는 명령이다. `uv`와 Git이 설치되어 있어야 한다.

```bash
git clone https://github.com/vinnylarouge/jevlike.git .repos/jevlike
git -C .repos/jevlike checkout 94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452
uv venv .repos/jevlike/.venv
uv pip install --python .repos/jevlike/.venv/bin/python -e .repos/jevlike

PYTHONHASHSEED=0 .repos/jevlike/.venv/bin/python -m jevlike.data synthetic \
  --output .repos/jevlike/runs/research/synthetic \
  --train 2000 --validation 400 --test 400 --seed 17

OMP_NUM_THREADS=2 .repos/jevlike/.venv/bin/python -m jevlike.train \
  .repos/jevlike/runs/research/synthetic/train.jsonl \
  --validation .repos/jevlike/runs/research/synthetic/validation.jsonl \
  --output .repos/jevlike/runs/research/synthetic.pt \
  --encoder tiny --epochs 8 --device cpu

sh ai-infrastructure/jevlike/start-demo.sh
```

원본 저장소의 CPU 체크포인트 snapshot 문제는 이 데모에서 변경하지 않았다. 학습을 재현·확장할 때는 [관련 분석](jevlike-analysis.md#63-cpu의-최적-체크포인트-보존-버그--직접-재현)을 함께 참고한다. `.repos/`는 research의 `.gitignore`에 포함되어 소스 클론·가중치·가상환경은 커밋되지 않는다.
