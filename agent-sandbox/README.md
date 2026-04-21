# Agent Sandbox 기술 조사

> LLM 기반 에이전트가 임의의 코드/명령을 실행할 때 쓸 수 있는 **격리 런타임** 후보들의 지도(landscape)와 심층 분석.
>
> 이 디렉토리는 "에이전트가 안전하게 코드를 실행할 수 있는 환경" 이라는 목표 아래 오픈소스 기술들을 카테고리별로 정리하고, 개별 기술에 대한 심층 분석 문서를 함께 둡니다.

---

## 이 디렉토리의 구성

| 파일 | 내용 |
|---|---|
| [README.md](./README.md) (본 문서) | 에이전트 샌드박스 기술 landscape, 선정 기준, 추천 순서 |
| [smolvm_analysis.md](./smolvm_analysis.md) | **SmolVM** 심층 분석 — libkrun 기반 OCI-native 마이크로VM |

> 새로운 기술 분석을 추가할 때마다 위 표에 항목을 추가해 주세요.

---

## 왜 "에이전트 샌드박스" 인가

LLM 기반 에이전트는 본질적으로 **임의의 자연어 지시를 임의의 코드로 변환**해서 실행한다. 이 특성은 전통적 "사용자가 작성한 코드를 사용자 권한으로 실행"과 질적으로 다르다:

1. **예측 불가능성** — tool call이 시스템에 어떤 영향을 미칠지 사전에 모두 열거 불가
2. **프롬프트 인젝션** — 입력 데이터(웹 스크랩 결과, 문서 등)가 에이전트의 실행 의도를 조작 가능
3. **공급망 취약점 증폭** — `pip install foo` 한 줄이 무엇을 당겨오는지 에이전트가 검증하지 않음
4. **자격증명 경로 의존** — 에이전트가 `~/.aws/`, `~/.ssh/`, 브라우저 쿠키 등에 접근하면 곧 횡적 이동 경로

따라서 에이전트 플랫폼은 **"코드가 실행되는 경계"** 를 명시적 인프라 계층으로 가져야 한다. 이 디렉토리는 그 경계를 만드는 오픈소스 후보들을 다룬다.

### 좋은 에이전트 샌드박스의 공통 요건
- [ ] 커널/하이퍼바이저 경계의 **강한 격리** (컨테이너 이스케이프 표면 최소화)
- [ ] **네트워크 기본 차단** + 도메인/CIDR 단위 egress allowlist
- [ ] **자격증명 격리** (SSH 키, 클라우드 credential 미노출)
- [ ] **저지연 콜드 스타트** (tool call 레이턴시 요구를 만족)
- [ ] **재현 가능한 환경** (이미지/스냅샷/매니페스트로 고정)
- [ ] **프로그래밍 인터페이스** (REST/SDK) — CLI-only 시스템은 에이전트 하네스와 결합이 어렵다
- [ ] **관측성** (감사 로그, 메트릭, trace) — 사후 분석을 위한 최소 기반
- [ ] **멀티 세션** — 세션당/에이전트당 격리 인스턴스 스케일

---

## Landscape: 카테고리별 오픈소스

### 1. 마이크로VM / 하드웨어 격리 (★ 가장 강한 격리, smolvm 계열)

진짜 하이퍼바이저 + 자체 커널. 컨테이너 이스케이프 전 계열이 의미 없어진다.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **SmolVM** | [smol-machines/smolvm](https://github.com/smol-machines/smolvm) | libkrun 래퍼. OCI-native, macOS 1급, `.smolmachine` 이식 포맷 | ★★★★★ (macOS 로컬) |
| **libkrun** | [containers/libkrun](https://github.com/containers/libkrun) | Red Hat 주도 VMM 라이브러리. 직접 임베드 가능 | ★★★★ (저수준 빌딩블록) |
| **krunkit** | [containers/krunkit](https://github.com/containers/krunkit) | macOS용 libkrun 런처. Podman Machine에 사용 | ★★★ |
| **Firecracker** | [firecracker-microvm/firecracker](https://github.com/firecracker-microvm/firecracker) | AWS Lambda 기반, <125ms 콜드 스타트, Linux/KVM 전용 | ★★★★★ (Linux 서버) |
| **Cloud Hypervisor** | [cloud-hypervisor/cloud-hypervisor](https://github.com/cloud-hypervisor/cloud-hypervisor) | Intel 주도. Firecracker 대비 기능 풍부 | ★★★★ |
| **Kata Containers** | [kata-containers/kata-containers](https://github.com/kata-containers/kata-containers) | OCI 호환 VM 컨테이너. K8s 통합 성숙 | ★★★★ (멀티테넌트) |
| **Ignite** | [weaveworks/ignite](https://github.com/weaveworks/ignite) | Firecracker + OCI 래퍼. 현재 아카이브 | ★★ (참고용) |
| **QEMU microvm** | [qemu/qemu](https://github.com/qemu/qemu) | 가장 보편적인 VMM의 "microvm" 머신 타입 | ★★★ (저수준) |

### 2. LLM 에이전트 전용 샌드박스 (★ 직접 경쟁/대체)

"AI agent" 를 1급 사용 사례로 설계한 것들. smolvm과 가장 직접적으로 비교된다.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **microsandbox** | [microsandbox/microsandbox](https://github.com/microsandbox/microsandbox) | libkrun 기반 LLM 샌드박스. smolvm과 설계 겹침/차이 흥미 | ★★★★★ |
| **E2B Infra** | [e2b-dev/infra](https://github.com/e2b-dev/infra) | Firecracker 기반. SaaS + 셀프호스팅. LLM 샌드박스 de facto | ★★★★★ |
| **Daytona** | [daytonaio/daytona](https://github.com/daytonaio/daytona) | "AI agent sandbox infrastructure" 표방 | ★★★★ |
| **OpenHands Runtime** | [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) | 에이전트 전용 Docker 기반 실행 환경 | ★★★ |
| **Agent Infra Sandbox** | [agent-infra/sandbox](https://github.com/agent-infra/sandbox) | 에이전트 실행용 통합 샌드박스 | ★★★ |
| **CodeInterpreter API** | [shroominic/codeinterpreter-api](https://github.com/shroominic/codeinterpreter-api) | OpenAI code interpreter 오픈 대안 | ★★★ |
| **Riza** | [rizaio/code-interpreter](https://github.com/rizaio/code-interpreter) | WASM 기반 상용/오픈 혼재 | ★★ |

### 3. 사용자공간 커널 / 시스템콜 필터 (중간 격리)

진짜 VM은 아니지만 syscall을 가로채서 게스트 커널을 흉내낸다. 성능과 격리의 절충점.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **gVisor** | [google/gvisor](https://github.com/google/gvisor) | Go로 쓴 사용자공간 Linux 커널. GCP Cloud Run 기반 | ★★★★ |
| **Nabla Containers** | [nabla-containers/runnc](https://github.com/nabla-containers/runnc) | Solo5 유니커널 기반 OCI 런타임 | ★★ |

### 4. 컨테이너 기반 (가장 널리 쓰이지만 격리는 약함)

커널을 공유하기 때문에 신뢰할 수 없는 코드에는 약하지만, 생태계가 풍부하고 성능이 좋다. 에이전트 샌드박스 PoC/내부용에 자주 쓰인다.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **Docker / Moby** | [moby/moby](https://github.com/moby/moby) | 사실상 표준 컨테이너 런타임 | ★★ |
| **Podman** | [containers/podman](https://github.com/containers/podman) | rootless 컨테이너 + Podman Machine | ★★★ |
| **nerdctl** | [containerd/nerdctl](https://github.com/containerd/nerdctl) | containerd CLI. rootless 강점 | ★★★ |
| **Sysbox** | [nestybox/sysbox](https://github.com/nestybox/sysbox) | 시스템 컨테이너. 중첩 컨테이너/systemd 강점 | ★★★ |
| **Bubblewrap** | [containers/bubblewrap](https://github.com/containers/bubblewrap) | Flatpak 기반 네임스페이스 샌드박스 | ★★ |
| **Firejail** | [netblue30/firejail](https://github.com/netblue30/firejail) | SUID 기반 데스크톱 샌드박스 | ★★ |
| **youki** | [youki-dev/youki](https://github.com/youki-dev/youki) | Rust로 쓴 OCI 런타임. runc 대체 | ★★ |
| **crun** | [containers/crun](https://github.com/containers/crun) | C로 쓴 OCI 런타임 (smolvm 게스트 내부에서도 사용) | ★★ |

### 5. WebAssembly 기반 (초경량 + 언어 차원 안전성)

"프로세스 없이 샌드박스된 함수 실행" 모델. 임의 바이너리 실행엔 한계가 있으나 도구 함수 실행엔 유리.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **Wasmtime** | [bytecodealliance/wasmtime](https://github.com/bytecodealliance/wasmtime) | Bytecode Alliance 레퍼런스 WASM 런타임 | ★★★ (tool 실행) |
| **WasmEdge** | [WasmEdge/WasmEdge](https://github.com/WasmEdge/WasmEdge) | LLM 추론/에이전트 확장 풍부 | ★★★ |
| **Wasmer** | [wasmerio/wasmer](https://github.com/wasmerio/wasmer) | 멀티 런타임, 언어 바인딩 다수 | ★★★ |
| **Extism** | [extism/extism](https://github.com/extism/extism) | 플러그인용 WASM 프레임워크 | ★★★ (플러그인) |
| **Spin** | [fermyon/spin](https://github.com/fermyon/spin) | WASM 기반 서버리스 런타임 | ★★ |
| **WasmCloud** | [wasmCloud/wasmCloud](https://github.com/wasmCloud/wasmCloud) | 분산 WASM 액터 런타임 | ★★ |

> WASM은 파일시스템/네트워크/프로세스 실행 능력이 WASI 스펙 진화 중. 현재로선 "코드 실행" 에이전트용으론 부분 적용이 현실적.

### 6. 개발 환경 / 머신 관리 (SmolVM의 create/start/stop 유사 모델)

에이전트에게 "전용 개발 머신"을 통째로 제공하는 모델. 세션 지속성이 특징.

| 프로젝트 | 리포지토리 | 한 줄 요약 | 에이전트 적합도 |
|---|---|---|---|
| **Lima** | [lima-vm/lima](https://github.com/lima-vm/lima) | macOS에서 Linux VM 간편 관리 | ★★★ |
| **Colima** | [abiosoft/colima](https://github.com/abiosoft/colima) | Lima + 컨테이너 런타임. Docker Desktop 대체 | ★★★ |
| **DevPod** | [loft-sh/devpod](https://github.com/loft-sh/devpod) | Dev Container 기반 워크스페이스 매니저 | ★★★★ |
| **Dev Containers CLI** | [devcontainers/cli](https://github.com/devcontainers/cli) | VS Code dev container 표준 구현 | ★★★ |
| **Coder** | [coder/coder](https://github.com/coder/coder) | 셀프호스팅 원격 개발 환경 | ★★★ |
| **Gitpod** | [gitpod-io/gitpod](https://github.com/gitpod-io/gitpod) | 클라우드 IDE + 워크스페이스 | ★★★ |
| **OrbStack** | (orbstack.dev) | macOS 상용. 참고용 (비오픈소스) | N/A |

### 7. 오케스트레이션 / 멀티 세션 (상위 스케줄러)

개별 샌드박스 런타임 위에 멀티 테넌트/스케줄링을 얹는 층.

| 프로젝트 | 리포지토리 | 한 줄 요약 |
|---|---|---|
| **Kubernetes (+Kata/gVisor)** | [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) | 샌드박스 RuntimeClass로 멀티 테넌트 |
| **Nomad + firecracker-task-driver** | [cneira/firecracker-task-driver](https://github.com/cneira/firecracker-task-driver) | 경량 멀티 VM 스케줄러 |
| **Unikraft / kraftkit** | [unikraft/kraftkit](https://github.com/unikraft/kraftkit) | 유니커널 빌드/배포 툴체인 |
| **Flatcar / Bottlerocket** | [flatcar/flatcar](https://github.com/flatcar/flatcar), [bottlerocket-os/bottlerocket](https://github.com/bottlerocket-os/bottlerocket) | 컨테이너/VM 전용 호스트 OS |

---

## 비교 요약: 에이전트 샌드박스로서의 속성

| 기술 | 격리 강도 | 콜드 스타트 | OCI 호환 | 자격증명 격리 | macOS 네이티브 | 이식 단위 | 대표 사용처 |
|---|---|---|---|---|---|---|---|
| **SmolVM** | VM | <200ms | ✅ | SSH agent fwd | ✅ | .smolmachine | 로컬 에이전트, 엣지 |
| **microsandbox** | VM | <수백ms | ✅ | 유사 | ✅ (예상) | 이미지 | LLM 에이전트 |
| **E2B** | VM (FC) | ~수백ms | 제한적 | 클라우드 API | ❌ | 템플릿 | LLM SaaS |
| **Firecracker** | VM | <125ms | 수동 | 수동 | ❌ | rootfs | Lambda, 서버 |
| **Kata Containers** | VM | ~500ms | ✅ | 컨테이너 동일 | ❌ | 이미지 | K8s 멀티테넌트 |
| **gVisor** | 사용자공간 커널 | ~수 ms | ✅ | 컨테이너 동일 | ❌ | 이미지 | GCP Cloud Run |
| **Docker** | 공유 커널 | ~100ms | ✅ | bind mount | Docker Desktop | 이미지 | 내부 도구, CI |
| **Sysbox** | 공유 커널+ | ~100ms | ✅ | 강화됨 | ❌ | 이미지 | 중첩 컨테이너 |
| **Bubblewrap** | 네임스페이스 | 즉시 | 부분 | bind mount | ❌ | 없음 | Flatpak |
| **Wasmtime** | WASM | 즉시 | ❌ | N/A | ✅ | .wasm | tool 실행 |
| **DevPod** | 컨테이너/VM 위임 | 위임 | ✅ | dev container | ✅ | devcontainer.json | 원격 개발 |

---

## 도입 의사결정: 우리 팀은 어디서 시작해야 하나

### 시나리오별 추천 조합

| 시나리오 | 1순위 | 대안 |
|---|---|---|
| macOS 로컬 개발자 PC에서 에이전트 tool 실행 | **SmolVM**, microsandbox | Colima + Docker, Lima |
| Linux 서버(자체 인프라)에서 에이전트 서비스 | **Firecracker + E2B** 또는 **Kata on K8s** | Cloud Hypervisor |
| 클라우드 네이티브 멀티 테넌트 SaaS | **Kata Containers on K8s** | gVisor RuntimeClass |
| 에이전트에 도구 함수만 제공 (네트워크 I/O 없음) | **Wasmtime / Extism** | WasmEdge |
| 에이전트에게 지속 가능한 개발 환경 전체 제공 | **DevPod**, **Coder** | SmolVM persistent machine |
| 최소 의존성, 빠른 PoC | **Docker + bubblewrap + seccomp** | Podman rootless |

### 향후 심층 분석 추천 순서
smolvm과 **직접 비교/상호 보완** 관점에서 우선순위:

1. **microsandbox** — 같은 libkrun 기반 LLM 샌드박스. 설계 차이 분석이 가장 유익
2. **E2B Infra** — LLM 샌드박스의 de facto. 셀프호스팅 옵션 확인 필요
3. **Firecracker** — VM 진영의 표준. smolvm이 참조한 설계 철학 원류
4. **Kata Containers** — 프로덕션 검증된 VM-per-workload 모델
5. **gVisor** — 사용자공간 커널이라는 다른 접근법의 trade-off
6. **Daytona / OpenHands Runtime** — "AI 에이전트 전용" 패키징 관점 비교
7. **DevPod** — 에이전트에게 "전용 개발 환경"을 주는 모델의 참고

---

## 참고 자료

- **OCI Runtime Spec**: https://github.com/opencontainers/runtime-spec
- **OCI Image Spec**: https://github.com/opencontainers/image-spec
- **Kernel virtio 문서**: https://docs.kernel.org/driver-api/virtio/index.html
- **KVM API**: https://docs.kernel.org/virt/kvm/api.html
- **Apple Hypervisor.framework**: https://developer.apple.com/documentation/hypervisor
- **WASI 스펙**: https://github.com/WebAssembly/WASI

---

## 기여 가이드 (이 디렉토리에)

새로운 기술 분석을 추가할 때:

1. `{technology}_analysis.md` 파일명 사용 (소문자 + 언더스코어)
2. 이 README의 **"이 디렉토리의 구성"** 표에 한 줄 추가
3. Landscape 테이블에도 항목을 추가 또는 갱신
4. 분석 문서 상단에 메타 정보(분석일, 대상 버전, 관점) 명시
5. smolvm 분석 문서 구조를 참고 — 개요, 아키텍처, 격리 메커니즘, 보안 체크리스트, 대안 비교, 결론 순서
