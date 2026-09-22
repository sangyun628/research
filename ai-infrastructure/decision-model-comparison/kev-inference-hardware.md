# Kev 추론 하드웨어 — CPU와 GPU

확인일: 2026-09-22. 소스는 `.repos/kev`의 `90990a5fac2995b9faa3190f7d437e84f2067768`, 공개 모델 카드는 당일 확인했다. 아래는 **추론용** 사양이며 학습용 사양과 다르다. 직접 실행해 측정한 하드웨어 최소 요구사항은 아니다.

## 1. 판단

Kev는 CPU 추론을 지원한다. `kev/device.py`는 CUDA, MPS, CPU 순으로 장치를 선택하며, `Checkpoint.load("cpu", ...)`로도 로드할 수 있다. 최적화된 커널을 사용하는 NVIDIA GPU에서는 CPU보다 낮은 지연시간과 높은 처리량을 기대할 수 있다. 다만 GPU 모델·입력 길이·배치·커널에 따라 차이가 달라지므로 배율은 실측 없이 정할 수 없다.

한국어 금융 플레이북 21개를 고르는 현재 요구에서 CPU로 시작한다면 **Kev-0.8B, 8 vCPU, RAM 16GB, 모델 프로세스 하나**를 첫 실험 구성으로 제안한다. 이는 해당 업무의 정확도나 지연시간을 보장하는 운영 규격이 아니다.

## 2. 모델별 시작 사양

아래 표는 짧거나 중간 길이 입력, 배치 1, 모델 인스턴스 하나를 전제로 한 **설계 추정치**다. 긴 입력, 큰 배치, 여러 worker, 다른 서비스와의 자원 공유에는 추가 메모리가 필요하다. CPU는 FP32, GPU는 BF16을 기준으로 한다.

| 모델 | CPU만 사용할 때 시작 구성 | GPU VRAM 시작 구성 | 적합성 판단 |
|---|---|---|---|
| Kev-0.8B | 8 vCPU, RAM 16GB | 8GB | CPU 첫 비교 후보 |
| Kev-4B | 16 vCPU, RAM 32~64GB | 12GB에서 검토, 16GB 이상으로 여유 확보 | CPU도 가능하지만 저지연 서비스는 GPU 우선 검토 |
| Kev-9B | 16~32 vCPU, RAM 64~96GB | 24GB에서 검토, 32GB 이상으로 여유 확보 | 현재 CPU 우선 의도 분류에는 우선순위 낮음 |

GPU 구성의 host는 0.8B·4B에 4~8 vCPU와 RAM 32GB, 9B에 8 vCPU와 RAM 64GB를 준비하는 출발안을 생각할 수 있다. host 메모리와 VRAM은 별개다. 임시 로딩 메모리와 CPU tokenization도 필요하다. SSD에는 기반 모델·adapter·Python 환경·다운로드 캐시 공간을 확보하며, 모델·revision을 여러 개 보관하면 그만큼 늘어난다.

GPU 표는 **아래 4절의 FP32 로딩 문제를 피하는 구성**을 전제로 한다. VRAM 숫자만 맞는 임의의 구형 GPU가 동일 성능을 내는 것은 아니다. BF16 연산 및 현재 Qwen3.5 커널이 지원되는 GPU·소프트웨어 조합이 필요하다.

vCPU 수 역시 공식 요구사항이 아니다. 물리 코어와 vCPU의 차이, CPU 세대, 메모리 대역폭, 공유 인스턴스 여부에 따라 같은 숫자의 서버 성능이 달라진다.

## 3. 메모리 산정 근거

| 모델 | 이름에 표시된 파라미터 수로 계산한 FP32 가중치 규모 | BF16 가중치 규모 | 제작자의 BF16 serving 메모리 보고 |
|---|---:|---:|---|
| 0.8B | 약 3.2GB | 약 1.6GB | 이번에 확인한 카드에는 별도 수치 없음 |
| 4B | 약 16GB | 약 8GB | 약 9GB |
| 9B | 약 36GB | 약 18GB | 약 19GB |

위 계산은 파라미터당 4바이트 또는 2바이트를 곱한 대략적인 규모다. Kev가 사용하는 backbone, 제외하는 생성 head, adapter, pointer head 때문에 정확한 실사용 가중치 바이트와 일치하지 않는다. 전체 메모리에는 activation, 상태 cache, allocator, 런타임과 로딩 중 복사본도 포함된다.

공식 9GB·19GB는 제작자 조건의 보고이며 모든 입력 길이와 시작 과정까지 포함한 최대치가 아니다. 특히 LoRA만 학습했다고 추론에서 기반 Qwen을 생략할 수는 없다.

근거: [Kev-4B 모델 카드](https://huggingface.co/jaredpalmer/kev-4b#known-limits), [Kev-9B 모델 카드](https://huggingface.co/jaredpalmer/kev-9b#known-limits).

## 4. 시작할 때 FP32가 먼저 올라가는 문제

`Checkpoint.load()`의 기본값은 `merge=True`다. 다음 순서로 로드한다.

1. merge를 할 경우 기반 모델을 FP32로 생성한다.
2. `DecisionModel.__init__()`에서 지정 장치로 옮긴다.
3. LoRA를 로드하고 FP32에서 합친다.
4. 마지막에 요청한 BF16 등으로 변환한다.

따라서 `KEV_DTYPE=bf16`만 지정해도 GPU에는 먼저 FP32 기반 모델이 올라갈 수 있다. 예를 들어 BF16 상태에서 약 19GB라는 9B 모델이 24GB GPU에서 기본 로딩 경로로 반드시 시작된다는 뜻은 아니다.

현재 코드의 `KEV_MERGE=0`을 함께 지정하면 이 FP32 merge 경로를 피하고 기반 모델을 요청 dtype으로 바로 로드한다. 다음은 **실행하지 않은 GPU 서버 시작 예시**다. 의존성과 Qwen3.5용 GPU 커널은 먼저 설치되어 있어야 한다.

```bash
KEV_DTYPE=bf16 KEV_MERGE=0 \
  uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-4b --port 8009
```

LoRA를 합치지 않는 경로에는 adapter 추가 연산이 남으며, 합친 경로와 저정밀 수치 차이도 생길 수 있다. 이것이 모든 메모리 부족을 해결하는 설정이라는 의미는 아니다. 장기적으로 CPU에서 merge한 후 저장·변환하는 별도 로딩 경로도 가능하지만, 이번 조사에서 구현하거나 검증한 것은 아니다.

CPU 전용 Linux 서버에서는 기본 장치 선택이 CPU가 된다. 기존 CLI에는 `--device cpu`가 없으므로 그런 인자를 추가한 명령을 그대로 사용해서는 안 된다. GPU가 있는 환경에서 CPU를 명시하려면 Python의 `Checkpoint.load("cpu", ...)` 경로 등을 사용한다.

근거: [Checkpoint.load](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/checkpoint.py#L116-L140), [모델 생성·장치 배치](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/model.py#L148-L177), [장치 선택](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/device.py).

## 5. GPU 속도 근거와 한계

공식 README는 `flash-linear-attention`을 설치한 H100·MI300X 환경에서 질문 다섯 개의 요청을 수십 ms에 처리한다고 설명한다. 이는 제작자의 고성능 가속기 보고이며 일반 GPU와 현재 금융 입력의 보장치는 아니다.

동일 README의 Apple M5 GPU 결과는 약 230토큰 상태, 질문 다섯 개, 질문당 후보 세 개, BF16의 median 모델 시간이다.

| 모델 | Apple M5 GPU의 제작자 보고 |
|---|---:|
| Kev-0.8B | 329ms |
| Kev-4B | 779ms |
| Kev-9B | 약 2초 |

이는 CPU 결과가 아니다. Qwen3.5의 DeltaNet 빠른 커널이 MPS에 없어서 GPU 종류에 따라 성능 차이가 크다는 사례다. 모델이 VRAM에 들어가는지와 빠르게 실행되는지는 별개다. [공식 Serving Performance](https://github.com/jaredpalmer/kev#serving-performance).

현재 업무는 **최근 세 질문을 상태로 받고, 21개 후보 중 하나를 고르는 분류 질문 하나**다. 질문 다섯 개·각 후보 세 개의 공개 측정과 길이·구성이 다르다. 특정 CPU에서 100ms, 특정 GPU에서 10ms라고 환산할 근거는 없다.

기본 HTTP 서버는 lock으로 모델 실행을 한 번에 하나씩 처리하며 사용자 요청 간 자동 배치를 하지 않는다. 동시에 요청이 들어오면 대기시간이 생기므로 GPU를 올리는 것만으로 동시 처리 능력이 GPU의 이론 처리량까지 늘어나지는 않는다. 다중 worker는 보통 모델을 각각 로드하므로 메모리도 증가한다. [서버 구현](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/serve.py).

## 6. 현재 프로젝트의 출발안

- CPU 우선: Kev-0.8B를 8 vCPU·RAM 16GB에서 검토한다. 실제로는 라벨 정확도와 요청 지연시간을 함께 봐야 한다.
- GPU를 쓸 경우: 0.8B는 VRAM 8GB, 4B는 VRAM 16GB를 시작 후보로 둔다. BF16과 커널 지원·로딩 경로를 함께 맞춘다.
- 9B는 작은 모델의 정확도가 부족하다는 근거가 생겼을 때 비교한다. 현재 고정 21개 라벨 라우터의 첫 CPU 배포 대상으로 삼을 이유는 약하다.
- 학습은 별도 GPU에서 수행하고 CPU 추론용으로 배포할 수 있다. 파라미터 규모를 늘리지 않는 일반 LoRA 추가 학습은 추론 하드웨어를 학습 GPU 수준으로 요구하지 않는다.

모델 선택과 학습 구조는 [Kev와 Laya 비교](kev-vs-laya.md)를 참고한다.
