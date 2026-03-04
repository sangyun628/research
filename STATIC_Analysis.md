# STATIC: Vectorizing the Trie — 논문 및 코드 분석 보고서

> **논문**: "Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators"
> **저자**: Zhengyang Su, Isay Katsman, Yueqi Wang 외 (YouTube / Google DeepMind)
> **arXiv**: 2602.22647v1 (2026년 2월)
> **코드**: https://github.com/youtube/static-constraint-decoding

---

## 1. 개요 (Executive Summary)

**STATIC** (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding)은 LLM 기반 생성형 검색(Generative Retrieval)에서 **제약 조건 디코딩(Constrained Decoding)을 하드웨어 가속기(TPU/GPU)에서 초고속으로 수행**하기 위한 기술이다.

### 핵심 아이디어
기존의 포인터 기반 Trie(접두사 트리) 순회를 **CSR(Compressed Sparse Row) 희소 행렬 연산**으로 변환하여, 불규칙한 트리 탐색을 **완전히 벡터화된 행렬 연산**으로 대체한다.

### 핵심 성과
| 지표 | 수치 |
|------|------|
| 디코딩 스텝당 지연 시간 | **0.033ms** (추론 시간의 0.25%) |
| CPU Trie 대비 속도 향상 | **948배** |
| PPV Exact 대비 속도 향상 | **1,033배** |
| PPV Approximate 대비 속도 향상 | **47배** |
| YouTube 프로덕션 배포 규모 | 수십억 사용자, 2,000만 제약 항목 |

---

## 2. 문제 정의: 왜 STATIC이 필요한가?

### 2.1 Generative Retrieval (생성형 검색)이란?

전통적 추천 시스템은 임베딩 + ANN(Approximate Nearest Neighbor) 검색 방식을 사용한다. 생성형 검색은 이를 **LLM의 자기회귀적 토큰 생성**으로 대체한다:

1. 각 아이템(영상, 상품 등)을 **Semantic ID (SID)** — RQ-VAE로 생성된 이산 토큰 시퀀스 `(y₁, y₂, ..., yL)` — 로 표현
2. Transformer가 사용자 컨텍스트를 기반으로 SID를 **토큰 단위로 자기회귀적 생성**
3. 생성된 SID를 실제 아이템에 매핑

### 2.2 문제: "Validity Gap" (유효성 격차)

제약 없는 LLM은 **존재하지 않는 아이템의 SID를 생성**("환각")할 수 있다. 이는 특히 다음과 같은 비즈니스 로직이 필요할 때 문제가 된다:

- **신선도 제약**: "7일 이내 업로드된 영상만"
- **지역 제약**: "한국 지역 추천만"
- **카테고리 제약**: "여름 의류만"
- **재고 제약**: "재고 있는 상품만"

### 2.3 기존 해결책의 한계

| 방법 | 문제점 |
|------|--------|
| **Post-filtering** | 생성 예산 전부를 무효 아이템에 낭비할 수 있음 |
| **CPU Trie** | 포인터 추적 → TPU↔CPU 라운드트립으로 31.3ms 지연 |
| **PPV (Binary Search)** | O(log\|C\|) I/O 복잡도 → 대규모에서 병목 |
| **FST (Finite State Transducer)** | 상태 폭발 문제, TPU 미지원 |

**하드웨어 가속기에 적대적인 이유 2가지:**
1. **메모리 지연**: 포인터 기반 구조 → 비연속 랜덤 메모리 접근 → HBM 버스트 불가
2. **컴파일 비호환**: XLA는 정적 계산 그래프 필요 → 데이터 의존적 제어 흐름 불가

---

## 3. STATIC 알고리즘 심층 분석

### 3.1 핵심 변환: Trie → CSR 희소 전이 행렬

Trie의 모든 고유 접두사 노드를 정수 상태 `s ∈ [S]`로 매핑하고, 희소 전이 행렬 **T ∈ Z^{S×|V|}** 를 정의:

```
T[s, v] = s_next   (상태 s에서 토큰 v를 선택하면 s_next로 전이)
         = 0        (해당 전이가 없음 = 터미널)
```

이 행렬을 **CSR 형식**으로 표현:
- **Row Pointers (P)**: 각 상태의 전이 시작 인덱스
- **Column Indices (C)**: 유효한 토큰 ID
- **Values (V)**: 다음 상태 노드 ID

#### 예시 (논문 Figure 1)
```
제약 어휘: C = {(A1,B2,C1), (A3,B1,C2), (A3,B1,C3)}

CSR 표현:
  Node IDs:     0  1  2  3  4  5  6  7
  Row Pointers: 0  2  3  4  5  7  7  7
  Columns:      1  3  2  1  1  2  3  Terminal
  Values:       1  2  3  4  5  6  7  Terminal
```

### 3.2 하이브리드 Dense + Sparse 전략

STATIC은 순수 CSR만 사용하지 않는다. **하이브리드 접근법**을 사용:

| 레이어 깊이 | 전략 | 이유 |
|-----------|------|------|
| 0 ~ d-1 (초기 d개 레이어) | **Dense 텐서 마스크** `D ∈ R^{|V|^d}` | 초기 레이어는 분기 팩터가 높아 Dense 조회가 O(1)로 더 빠름 |
| d ~ L-1 (나머지 레이어) | **CSR 희소 행렬** | 깊은 레이어는 분기 팩터가 낮아 희소 표현이 메모리 효율적 |

YouTube에서는 `d=2`, `|V|=2048` → Dense 마스크: 2048×2048 = ~17.3MB

### 3.3 Algorithm 1: Hardware-Accelerated Constrained Decoding Step

```
입력: 로짓 L_t, 빔 상태 S_{t-1}, 스텝 t
입력: Dense 텐서 D, 전이 행렬 T, 현재 노드 n_{t-1}
출력: 업데이트된 상태 S_t, 새 노드 n_t

Phase 1: Log-Space Projection
  P_t ← LogSoftmax(L_t)

Phase 2: Constraint Masking
  if t-1 < d:
    (n_next, m) ← DenseLookup(n_{t-1}, D, t-1)    // O(1) Dense 조회
  else:
    (n_next, m) ← VNTK(n_{t-1}, T, t-1)            // 벡터화된 CSR 조회
  P'_t ← Where(m, P_t, -∞)                         // 무효 토큰 마스킹

Phase 3: Beam Search
  (S_best, I_best) ← BeamSearch(P'_t, S.scores, M)

Phase 4: State Update
  S_t.tokens ← Gather(S_{t-1}.tokens, I_best)
  n_t ← Gather(n_next, I_best)
```

### 3.4 Algorithm 2: Vectorized Node Transition Kernel (VNTK)

**VNTK는 STATIC의 핵심 혁신**이다. 분기 프리(branch-free) 설계로 동적 제어 흐름을 완전히 제거:

```
입력: 현재 노드 n_curr, CSR 행렬 (P, C, V), 스텝 t
입력: 토큰 카디널리티 |V|, 최대 분기 팩터 벡터 B

Phase 1: Boundary Lookup
  idx_start ← P[n_curr]
  N_child ← P[n_curr + 1] - idx_start

Phase 2: Speculative Slicing (투기적 슬라이싱)
  // 실제 자식 수와 무관하게 항상 B_t개를 고정 크기로 슬라이싱
  d_col ← DynamicSlice(C, start=idx_start, len=B_t)
  d_val ← DynamicSlice(V, start=idx_start, len=B_t)

Phase 3: Sanitization (Branch-Free)
  J ← Range(B_t)              // [0, 1, ..., B_t-1]
  m_valid ← (J < N_child)     // 유효 슬롯 식별
  t_valid ← Where(m_valid, d_col, {})
  n_next ← Where(m_valid, d_val, 0)

Phase 4: Projection
  m ← Scatter(indices=t_valid, values=m_valid)  // |V| 크기 Dense 마스크 생성
```

**핵심 설계 원리:**
- **투기적 슬라이싱**: 노드마다 자식 수가 다르지만, 항상 해당 레벨의 최대 분기 팩터 `B_t`만큼 고정 크기로 읽음
- **Range + Where로 무효 엔트리 제거**: 분기(branch) 없이 마스크 산술만으로 처리
- **결과**: 전체 디코딩 스텝이 **단일 정적 XLA 계산 그래프**로 유지됨

### 3.5 I/O 복잡도 비교

| 방법 | I/O 복잡도 (제약 집합 크기 \|C\| 기준) |
|------|--------------------------------------|
| CPU Trie | 통신 오버헤드 지배적 |
| PPV (Binary Search) | **O(log\|C\|)** |
| **STATIC** | **O(1)** — 단일 coalesced 읽기 |

---

## 4. 코드 분석: 구현 아키텍처

### 4.1 프로젝트 구조

```
static-constraint-decoding/
├── static_decoding/           # 핵심 라이브러리
│   ├── csr_utils.py           # 오프라인 인덱스 구축 (NumPy)
│   ├── decoding_jax.py        # JAX/TPU 온라인 디코딩
│   └── decoding_pt.py         # PyTorch/GPU 온라인 디코딩
├── benchmarks/                # 벤치마크 스크립트
│   ├── baselines_jax.py       # Trie, Hash Bitmap, PPV 베이스라인
│   ├── run_comparative_benchmark_jax.py
│   ├── run_branch_benchmark_jax.py
│   └── run_branch_benchmark_pt.py
├── tests/                     # 단위/통합 테스트
│   ├── test_csr_builder.py
│   ├── test_jax_decoding.py
│   ├── test_pt_decoding.py
│   └── test_baselines_jax.py
└── example.ipynb              # 사용 예제 노트북
```

### 4.2 핵심 모듈 1: `csr_utils.py` — 오프라인 인덱스 구축

`build_static_index()` 함수가 전체 오프라인 파이프라인을 수행:

```python
def build_static_index(
    fresh_sids: np.ndarray,      # (N, L) — 정렬된 Semantic ID 배열
    vocab_size: int = 2048,       # 토큰 어휘 크기
    dense_lookup_layers: int = 2, # Dense 레이어 수 (d)
) -> tuple[packed_csr, indptr, layer_max_branches, start_mask, dense_mask, dense_states]:
```

**구축 과정 (8단계):**

1. **Level-0 마스크 생성**: 유효한 첫 번째 토큰을 boolean 마스크로 생성
2. **벡터화된 Trie 노드 식별**: 정렬된 SID 배열에서 인접 행 비교로 고유 접두사 노드 탐지 (`diff → first_diff → is_new`)
3. **상태 ID 할당**: Level-0 토큰은 `token + 1`로 매핑, 이후 레벨은 순차 할당 + `maximum.accumulate`로 중복 접두사 채움
4. **에지 수집**: 모든 부모→자식 전이를 `(parent_id, token, child_id)` 형태로 수집
5. **Dense 특수화**: 처음 d개 레이어를 `|V|^d` Dense 텐서로 구성
6. **CSR 구축**: `bincount → cumsum`으로 `indptr` 생성
7. **레이어별 최대 분기 팩터 계산**: 각 레벨의 최대 자식 수 벡터 `B` 계산
8. **최종 패킹**: `[token_id, next_state_id]` 쌍을 인터리빙하여 단일 `(N_edges, 2)` 텐서로 패킹 — **Stacked CSR Layout**

**Stacked CSR Layout의 의미:**
논문 Appendix A.1.1에서 설명하듯, 일반 CSR은 column indices와 values를 별도 배열에 저장하여 2번의 메모리 접근이 필요하다. STATIC은 이를 `(N_edges, 2)` 텐서로 인터리빙하여 **단일 메모리 트랜잭션으로 토큰 ID와 다음 상태를 동시 로드** → 랜덤 메모리 접근 횟수 50% 감소.

### 4.3 핵심 모듈 2: `decoding_jax.py` — JAX 온라인 디코딩

**`generate_and_apply_logprobs_mask()`** — VNTK의 JAX 구현:

```python
def generate_and_apply_logprobs_mask(flat_logprobs, flat_states, packed_csr, csr_indptr, limit, vocab_size):
    # 1. CSR 행 경계 조회
    starts = csr_indptr[flat_states]
    actual_lens = csr_indptr[flat_states + 1] - starts

    # 2. 고정 크기 Burst Read (투기적 슬라이싱)
    offsets = jnp.arange(limit)
    gather_indices = starts[:, None] + offsets[None, :]  # Broadcasting: (B*M, K)
    gathered_vals = jnp.take(packed_csr, gather_indices, axis=0, mode="fill", fill_value=0)

    # 3. 토큰 ID와 다음 상태 분리
    candidate_token_ids = gathered_vals[:, :, 0]
    candidate_next_states = gathered_vals[:, :, 1]

    # 4. 유효성 마스킹 & 로그 확률 수집
    valid_mask = offsets[None, :] < actual_lens[:, None]
    candidate_logprobs = jnp.take_along_axis(flat_logprobs, safe_token_ids, axis=1)
    safe_logprobs = jnp.where(valid_mask, candidate_logprobs, -jnp.inf)

    return safe_logprobs, candidate_token_ids, candidate_next_states
```

**`sparse_transition_jax()`** — 전체 빔 서치 루프:

1. **초기 스텝**: `start_mask`로 유효한 첫 토큰 선택
2. **자기회귀 루프 (스텝 1 ~ L-1)**:
   - `step < d_dense - 1`: **Dense 경로** — `dense_mask[parent_tokens]`로 마스킹 + `dense_states`로 상태 전이
   - `step >= d_dense - 1`: **Sparse 경로** — `generate_and_apply_logprobs_mask()` 호출
3. **빔 업데이트**: `lax.top_k` → 원-핫 수축(`einsum`)으로 TPU 최적화된 Gather

**JAX 특수 최적화:**
- `_gather_beams`: `torch.gather` 대신 **원-핫 행렬 + einsum** 사용 — TPU의 행렬 곱셈 유닛에 최적화
- `@jax.jit` + `static_argnames`: 정적 인자로 XLA가 최적화된 단일 계산 그래프 생성
- `jnp.take(..., mode="fill", fill_value=0)`: OOB 접근 시 안전한 기본값 반환

### 4.4 핵심 모듈 3: `decoding_pt.py` — PyTorch 온라인 디코딩

JAX 버전과 동일한 알고리즘을 PyTorch로 구현. 주요 차이점:

| JAX | PyTorch |
|-----|---------|
| `jnp.take(mode="fill")` | `packed_csr[safe_gather_indices]` + `clamp(max=max_idx)` |
| `jax.nn.one_hot` + `einsum` | `torch.gather` |
| `lax.top_k` | `torch.topk` |
| `@jax.jit` | `@torch.inference_mode()` |

### 4.5 베이스라인 구현: `baselines_jax.py`

4개의 비교 알고리즘이 동일한 빔 서치 하니스(`generic_beam_search_jax`)로 구현:

1. **CPU Trie**: Python 딕셔너리 기반 Trie + `jax.pure_callback`으로 TPU↔CPU 콜백
2. **Hash Bitmap**: Bloom 필터 스타일 — 30비트 해시로 비트맵 조회 (false positive 있음)
3. **PPV Exact**: 정렬된 SID 배열에서 `lax.while_loop` 기반 이진 탐색 (모든 2048 로짓)
4. **PPV Approximate**: 상위 50개 로짓만 이진 탐색 (원래 논문 방식)

---

## 5. 실험 결과 분석

### 5.1 YouTube 대규모 배포

**실험 환경:**
- 모델: Gemini 기반 3B 파라미터 생성형 검색 모델 (PLUM 유사)
- SID: L=8, |V|=2048
- 제약 집합: ~2,000만 개 (7일 이내 업로드 신선 영상)
- 하드웨어: Google TPU v6e
- 빔 크기: M=70, 배치 크기: 2

**지연 시간 비교:**

| 방법 | 지연 시간 (ms) | 추론 시간 대비 |
|------|:---:|:---:|
| Unconstrained | +0.0 | — |
| **STATIC (Ours)** | **+0.033** | **0.25%** |
| PPV Approximate | +1.56 | 11.9% |
| Hash Bitmap | +12.3 | 94.0% |
| CPU Trie | +31.3 | 239% |
| PPV Exact | +34.1 | 260% |

### 5.2 확장성 (Scalability)

**제약 집합 크기 |C|에 따른 확장:**
- STATIC: |C| = 10⁵ → 10⁸ 범위에서 **0.023ms → 0.039ms** (거의 상수)
- PPV Exact: 6.4ms → 38.7ms (로그 스케일링)
- CPU Trie: |C| = 10⁸에서 OOM 발생

**어휘 크기 |V|에 따른 확장:**
- STATIC: |V| = 256 → 32K 범위에서 **0.034ms → 0.041ms** (거의 상수)
- Hash Bitmap: 1.55ms → 196ms (선형)
- PPV Exact: 6.0ms → 578ms (선형 이상)

### 5.3 메모리 사용량

상한: `U_max = (1/8 + K₂)|V|^d + K₁ · Σ min(|V|^ℓ, |C|)`

YouTube 설정 (|V|=2048, L=8, d=2, |C|=20M):
- Dense 마스크: ~17.3 MB
- CSR (레벨 3~8): 6 × 20M × 12B = 1.44 GB
- **총합: ~1.5 GB** (실제 사용량 ≤ 75%)
- **경험 법칙: 100만 제약당 ~90 MB**

### 5.4 온라인 A/B 테스트 (YouTube)

"7일 신선도 제약"으로 숏폼 비디오 홈 피드에 배포:

| 지표 | 개선 | 95% 신뢰 구간 |
|------|:---:|:---:|
| 7일 신선 영상 조회수 | **+5.1%** | [5.0%, 5.2%] |
| 3일 신선 영상 조회수 | **+2.9%** | [2.8%, 3.0%] |
| 클릭률 (CTR) | **+0.15%** | [0.01%, 0.29%] |
| 전략적 사용자 세그먼트 만족도 | **+0.15%** | [0.03%, 0.27%] |

### 5.5 Cold-Start 성능 (Amazon Reviews)

설정: L=4, |V|=256, Gemma 1B 모델

| 데이터셋 | Unconstrained | Random | **STATIC** |
|---------|:---:|:---:|:---:|
| Beauty (2%) | 0.00% | 0.42% | **4.29%** |
| Sports (2%) | 0.00% | 0.27% | **1.24%** |
| Toys (2%) | 0.00% | 0.42% | **4.39%** |

→ Constrained decoding만으로도 cold-start 문제를 크게 개선할 수 있음을 증명

---

## 6. 기술적 혁신 요약

### 6.1 왜 O(1)인가?

기존 PPV는 각 후보 토큰마다 정렬된 SID 배열에서 이진 탐색 → O(log|C|) 블록 전송.
STATIC은 현재 상태의 CSR 행 포인터를 직접 조회 → **단일 coalesced burst read**로 모든 유효 자식을 한 번에 가져옴.

### 6.2 XLA 호환성의 핵심

XLA 컴파일러는 모든 텐서의 정확한 차원을 컴파일 타임에 알아야 한다. Trie 노드마다 자식 수가 다르면 `ConcretizationTypeError` 발생. STATIC은:

1. **레벨별 최대 분기 팩터 `B_ℓ`을 사전 계산** (오프라인)
2. 항상 **고정 크기 `B_ℓ`만큼 슬라이싱** (투기적)
3. `Range + Where`로 **무효 엔트리를 산술적으로 제거** (분기 없음)

→ 전체 디코딩 스텝이 **단일 정적 XLA 그래프**로 컴파일 가능

### 6.3 크로스 플랫폼 이식성

- **JAX/XLA**: `jnp.take(mode='fill')` → scatter-gather 하드웨어 활용
- **PyTorch/Inductor**: `torch.gather` → GPU warp 32개 스레드가 coalesced 접근
- 코드 구현에서 확인: `decoding_jax.py`와 `decoding_pt.py`가 동일 알고리즘을 각 프레임워크 관용구로 구현

---

## 7. 한계 및 향후 연구

1. **정적 구축**: CSR 전이 행렬 구축이 오프라인 프로세스 → 실시간 재고 변동에 대응하려면 **동적 희소 업데이트** 필요
2. **Dense 레이어 제한**: d ≥ 3이면 |V|^d가 기하급수적 증가 → 실질적으로 d ≤ 2로 제한
3. **단일 모델 제약**: 현재는 하나의 전이 행렬을 모든 디바이스에 복제 → 수십억 규모 코퍼스에서는 **계층적 샤딩 전략** 필요

---

## 8. 코드 사용법 (Quick Start)

```bash
# 클론 및 설치
git clone https://github.com/youtube/static-constraint-decoding.git
cd static-constraint-decoding
pip install -e .

# 테스트 실행
python tests/test_csr_builder.py
python tests/test_jax_decoding.py
python tests/test_pt_decoding.py

# 벤치마크 실행
python -m benchmarks.run_comparative_benchmark_jax
```

### 기본 사용 예시 (Python):

```python
import numpy as np
from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_jax import sparse_transition_jax, RandomModel
import jax

# 1. 제약 어휘 정의 (정렬된 SID 배열)
vocab_size, sid_length = 256, 4
num_constraints = 10000
fresh_sids = np.sort(
    np.random.randint(0, vocab_size, (num_constraints, sid_length)),
    axis=0
)

# 2. STATIC 인덱스 구축 (오프라인)
packed_csr, indptr, max_branches, start_mask, dense_mask, dense_states = \
    build_static_index(fresh_sids, vocab_size=vocab_size, dense_lookup_layers=2)

# 3. 제약 빔 서치 실행 (온라인)
import jax.numpy as jnp
model = RandomModel(vocab_size)
results = sparse_transition_jax(
    model=model,
    key=jax.random.PRNGKey(0),
    batch_size=4, beam_size=10, tokens_per_beam=20,
    start_token=0, max_sample_len=sid_length, vocab_size=vocab_size,
    max_branch_factors=max_branches,
    packed_csr=jnp.array(packed_csr),
    csr_indptr=jnp.array(indptr),
    start_mask=jnp.array(start_mask),
    dense_mask=jnp.array(dense_mask),
    dense_states=jnp.array(dense_states),
    d_dense=2,
)
# results.shape: (4, 10, 4) — 모든 생성된 SID가 fresh_sids에 속함을 보장
```

---

## 9. 결론

STATIC은 **"효율적인 GPU Trie 알고리즘이 존재하는가?"** 라는 개방된 질문에 긍정적으로 답한 최초의 프로덕션급 시스템이다. 포인터 추적 기반 Trie를 벡터화된 희소 행렬 연산으로 변환함으로써:

- **47~1,033배** 속도 향상 달성
- **O(1)** I/O 복잡도로 확장성 확보
- **수십억 사용자** 규모의 YouTube 프로덕션 배포 성공
- Cold-start 추천 품질 유의미한 개선 증명

이 기술은 LLM 기반 생성형 검색이 산업 현장에서 **제어 가능한 출력 공간**을 갖추는 데 필수적인 빌딩 블록이 될 것이다.
