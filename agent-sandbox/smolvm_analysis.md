# SmolVM 심층 분석: AI 에이전트 샌드박스 기술로서의 평가

> **분석 대상**: [smol-machines/smolvm](https://github.com/smol-machines/smolvm) (v0.5.19, Apache 2.0, Rust 82.9%)
> **조사일**: 2026-04-20
> **관점**: LLM 기반 에이전트가 임의의 코드/명령을 실행할 때 사용할 수 있는 샌드박스 런타임 후보
> **저장소 메타**: Stars 2,098 / Forks 80 / 2025-12-18 생성, 활발히 개발 중
> **홈페이지**: https://smolmachines.com

---

## 0. 한 눈에 보는 결론

| 질문 | 답 |
| --- | --- |
| 이게 뭔가? | libkrun 기반 하드웨어 가상화 마이크로VM CLI/라이브러리/HTTP API |
| 격리 강도 | **VM-per-workload** (컨테이너보다 강하고, 전통적 QEMU보다 훨씬 가벼움) |
| 콜드 스타트 | **< 200ms** (에이전트 한 턴에 새 VM을 매번 띄워도 부담 없음) |
| 호스트 플랫폼 | macOS (Apple Silicon/Intel, Hypervisor.framework) + Linux (KVM) — Windows 미지원 |
| 게스트 | Linux (arm64/x86_64), OCI 이미지 그대로 부팅 (Docker 데몬 불필요) |
| 에이전트 대상 인터페이스 | CLI / Rust 라이브러리 / **REST + SSE HTTP API (OpenAPI)** / Node.js NAPI |
| 네트워크 기본값 | **차단** (명시적 opt-in, CIDR/호스트 allowlist, DNS 필터) |
| 자격증명 격리 | **SSH agent forward via vsock** (개인키 자체는 게스트에 노출되지 않음) |
| 이식 단위 | `.smolmachine` — 커널/루트FS/OCI 레이어/매니페스트 포함 단일 바이너리 |
| 적합도 (에이전트 샌드박스) | **매우 높음** — 보안/성능/포터빌리티/API 모두 갖춤. 단 Windows 호스트 미지원, 아직 초기 프로젝트 |

---

## 1. 프로젝트 개요와 포지셔닝

### 1.1 슬로건과 문제 정의
> "Ship and run software with isolation by default."

smolvm은 **"격리가 기본값인 소프트웨어 실행 환경"** 을 지향한다. 컨테이너는 가볍지만 커널을 공유해서 신뢰할 수 없는 코드에 약하고, 기존 VM(QEMU, VirtualBox 등)은 격리는 강하지만 무겁고 부팅이 느리다. smolvm은 libkrun을 백엔드로 써서 두 세계의 장점을 결합한다:

- **VM per workload**: 각 워크로드가 자체 Linux 커널을 가진다
- **OCI 네이티브**: Docker Hub 이미지를 데몬 없이 바로 부팅
- **Sub-second 콜드 스타트**: 200ms 미만
- **탄성 메모리**: virtio-balloon으로 실제 쓰는 만큼만 호스트에서 점유

### 1.2 비교표 (README 기반)

| 특성 | smolvm | 컨테이너 | Colima | QEMU | Firecracker | Kata |
|---|---|---|---|---|---|---|
| 격리 단위 | VM/워크로드 | 공유 커널 | 단일 VM | 개별 VM | 개별 VM | VM/컨테이너 |
| 부팅 시간 | <200ms | ~100ms | ~초 | 15–30s | <125ms | ~500ms |
| macOS 네이티브 | ✅ | Docker VM 경유 | ✅ | ✅ | ❌ | ❌ |
| 이식 아티팩트 | `.smolmachine` | 이미지만 | ❌ | ❌ | ❌ | ❌ |
| 워크로드별 VM | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |

핵심 차별점은 **`.smolmachine` 단일 파일 포터빌리티**와 **macOS 1급 지원**이다. AI 에이전트 플랫폼 입장에서 macOS 개발자 데스크톱에서 그대로 쓸 수 있다는 건 드문 장점.

---

## 2. 아키텍처 전체 조감

```
┌──────────────────────────────────────────────────────────────────────┐
│ 호스트 프로세스 (smolvm CLI / serve / 라이브러리 / NAPI)                 │
│                                                                      │
│ ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│ │ src/cli/*    │  │ src/api/*    │  │ src/agent/ │  │ src/vm/*    │  │
│ │ (Clap 명령)   │  │ (Axum HTTP)  │  │  (매니저)    │  │ (libkrun    │  │
│ │              │  │  + OpenAPI   │  │             │  │  backend)  │  │
│ └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  │
│        └─────────────────┴─────────────────┴────────────────┘         │
│                                    │                                  │
│                  posix_spawn / fork + dlopen libkrun                  │
│                                    │                                  │
│ ┌──────────────────────────────────┴───────────────────────────────┐  │
│ │   libkrun (VMM, C 라이브러리)                                      │  │
│ │   ├─ Hypervisor.framework (macOS) │ KVM (Linux)                   │  │
│ │   ├─ libkrunfw: 임베디드 Linux 커널                                │  │
│ │   ├─ TSI (transparent socket impersonation) 네트워킹               │  │
│ │   └─ virtiofs, virtio-block, virtio-vsock                         │  │
│ └─────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                    │ vsock (CID 2 ↔ CID 3, port 6000)
┌──────────────────────────────────────────────────────────────────────┐
│ 게스트 VM (Linux, Alpine + init.krun)                                 │
│                                                                      │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ smolvm-agent (Rust 바이너리, 크기 최적화 빌드 release-small)      │  │
│ │  ├─ vsock::VsockListener (port 6000, AgentRequest 수신)         │  │
│ │  ├─ storage: OCI 레이어 풀링(crane) + overlayfs 준비             │  │
│ │  ├─ crun: OCI 런타임 (컨테이너 create/exec/kill)                  │  │
│ │  ├─ oci: config.json 생성 (capabilities, masked paths, rlimits) │  │
│ │  ├─ ssh_agent: vsock ↔ /tmp/ssh-agent.sock 브릿지               │  │
│ │  ├─ dns_proxy: UDP-over-vsock 브릿지 (resolv.conf 덮어쓰기)      │  │
│ │  └─ pty: 인터랙티브 세션용 pseudo-terminal                         │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│ ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│ │ OCI 컨테이너│  │ virtiofs     │  │ overlay disk │  │ storage disk  │  │
│ │ (crun)    │  │ 바인드 마운트  │  │ (persistent) │  │ (OCI 레이어 캐시)│  │
│ └──────────┘  └──────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 레이어 책임
1. **사용자 인터페이스**: CLI 4대 명령(`machine`, `serve`, `pack`, `config`) + HTTP API + NAPI + 임베디드 Rust 라이브러리
2. **VM 추상화**: `VmBackend`/`VmHandle` 트레이트로 향후 다른 백엔드 이식 여지 확보 — 현재 백엔드는 libkrun 단일
3. **libkrun FFI**: C API를 Rust에서 직접 호출 (`krun_set_vm_config`, `krun_add_disk2`, `krun_add_vsock_port2`, `krun_start_enter` 등)
4. **게스트 에이전트**: vsock으로 명령을 받아 컨테이너를 띄우는 사용자공간 데몬. 매 요청마다 crun으로 독립 OCI 런타임 실행
5. **호스트 보조 스택**: `smolvm-network` (smoltcp 기반 TCP 릴레이), `dns_filter` (호스트 측 allowlist 판정)

---

## 3. Rust 워크스페이스 구조

```
smolvm/
├─ Cargo.toml                (workspace + binary + lib)
├─ Makefile.toml             (cargo-make: dev/build/build-agent/dist/agent-rootfs)
├─ docs/DEVELOPMENT.md
├─ src/                       ← 호스트 측 CLI + 라이브러리
│  ├─ main.rs                 (진입점, packed 바이너리 자동 감지)
│  ├─ lib.rs                  (re-exports)
│  ├─ config.rs               (SmolvmConfig, VmRecord, RestartConfig)
│  ├─ db.rs                   (redb 기반 상태 DB)
│  ├─ storage.rs              (storage/overlay 디스크 레이아웃, 경로 규칙)
│  ├─ registry.rs             (OCI 레지스트리 클라이언트 래퍼)
│  ├─ smolfile.rs
│  ├─ process.rs              (호스트 측 프로세스 관리)
│  ├─ dns_filter.rs           (호스트 측 DNS allowlist 판정 + NXDOMAIN 반환)
│  ├─ dns_filter_listener.rs
│  ├─ disk_utils.rs
│  ├─ log_rotation.rs
│  ├─ util.rs
│  ├─ agent/                  ← "게스트 VM의 에이전트를 호스트에서 제어"
│  │  ├─ mod.rs
│  │  ├─ boot_config.rs       (서브프로세스로 전달할 VM 부팅 설정)
│  │  ├─ client.rs            (AgentClient: vsock 클라이언트)
│  │  ├─ launcher.rs          (fork 후 libkrun 호출)
│  │  ├─ launcher_dynamic.rs  (dlopen 기반 동적 로딩)
│  │  ├─ manager.rs           (AgentManager: 생명주기 오케스트레이션)
│  │  ├─ state_probe.rs       (DB / PID / vsock 3-way 상태 판정)
│  │  └─ terminal.rs
│  ├─ api/                    ← Axum HTTP 서버
│  │  ├─ mod.rs               (라우트, CORS, 트레이싱, Prometheus)
│  │  ├─ errors.rs / state.rs / supervisor.rs / types.rs
│  │  └─ handlers/{exec,files,health,images,machines}.rs
│  ├─ cli/
│  │  ├─ machine.rs           (run/exec/create/start/stop/ls/rm)
│  │  ├─ serve.rs             (HTTP 서버 런처)
│  │  ├─ pack.rs              (.smolmachine 패킹/배포)
│  │  ├─ pack_run.rs / internal_boot.rs
│  │  ├─ config.rs            (smolvm 자체 설정)
│  │  └─ parsers.rs
│  ├─ network/                ← 네트워크 정책/백엔드 선택
│  │  ├─ backend.rs           (NetworkBackend: Tsi | VirtioNet)
│  │  ├─ launch.rs            (plan_launch_network)
│  │  └─ policy.rs            (NetworkPolicy, CIDR allowlist)
│  ├─ platform/               (macOS/Linux, arm64/x86_64 추상화)
│  ├─ vm/                     (VmBackend 트레이트 + libkrun 구현체)
│  │  └─ backend/libkrun.rs   (FFI, fork 실행)
│  ├─ data/                   (데이터 타입)
│  └─ embedded/               ("language-neutral embedded runtime")
│
├─ crates/
│  ├─ smolvm-agent/           ← 게스트 VM 안에서 도는 바이너리
│  │  └─ src/{main,vsock,crun,oci,pty,storage,network,process,ssh_agent,dns_proxy,retry,paths}.rs
│  ├─ smolvm-protocol/        ← 공유 와이어 프로토콜 (AgentRequest/Response, Host/Guest Message)
│  ├─ smolvm-pack/            ← .smolmachine 포맷 직렬화 + Mach-O 섹션 임베드
│  ├─ smolvm-registry/        ← OCI 레지스트리 클라이언트 (push/pull/cache)
│  ├─ smolvm-network/         ← 호스트 측 smoltcp 게이트웨이 (프레임 ↔ 소켓)
│  ├─ smolvm-smolfile/        ← Smolfile TOML 파서/검증
│  └─ smolvm-napi/            ← Node.js 네이티브 바인딩
│
├─ examples/                  (python-app, node-app, openclaw-app, doom-web)
├─ libkrun/                   (서브모듈)
├─ libkrunfw/                 (서브모듈)
└─ tests/
```

---

## 4. 격리 메커니즘: 왜 "진짜 하드웨어 격리"인가

### 4.1 libkrun + libkrunfw
- **libkrun**은 Red Hat이 주도하는 오픈소스 VMM 라이브러리. Podman Machine(krunkit)의 기반 기술이다
- **libkrunfw**는 초소형 Linux 커널 + 부팅에 필요한 최소 구성이 임베디드된 라이브러리. initramfs를 쓰지 않고 호스트가 주입한 `init.krun` 바이너리를 PID 1로 실행
- macOS에서는 Apple의 **Hypervisor.framework**, Linux에서는 **KVM**을 이용해 게스트 코드를 CPU의 VT-x/ARMv8-A 가상화 모드에서 직접 실행 — 에뮬레이션 아님
- 즉 **게스트 커널 = 호스트 커널과 독립**. 컨테이너의 `runc`/`crun`이 공유 커널 위에 네임스페이스/cgroup만 두는 것과 근본적으로 다르다

### 4.2 VM 부팅 시퀀스 (`LibkrunVm::create()` → `exec_vm()`)
`src/vm/backend/libkrun.rs`에서 다음 순서로 VM을 구성한다:

1. rootfs 경로 결정 → `init.krun` 바이너리 주입 (libkrunfw 커널이 initramfs를 지원하지 않으므로)
2. `krun_set_vm_config(ctx, vcpus, mem_mib)` — CPU/메모리
3. `krun_set_root(ctx, rootfs_path)`
4. `krun_set_exec(ctx, cmd, args, envp)`
5. `krun_add_disk2()` — storage/overlay 디스크 (raw, qcow2 지원)
6. `krun_add_vsock_port2()` — 제어 채널 (port 6000 = 에이전트), SSH agent, DNS
7. virtiofs 태그 설정 (호스트 디렉토리 공유)
8. Rosetta (macOS x86_64 변환) — 선택
9. `setrlimit()`로 파일 디스크립터 한계 상향
10. `fork()` → 자식 프로세스에서 `krun_start_enter()` 호출 (성공 시 반환하지 않음) → 부모는 exit code 수집

### 4.3 macOS에서의 특수 처리 (중요)
- macOS에서 `fork()` 후 Hypervisor.framework를 쓰면 Apple 프레임워크가 멀티스레드 fork 상태를 감지하고 abort시킨다
- 해결책: **`posix_spawn`으로 `smolvm _boot-vm` 서브프로세스를 새로 띄운다**. 이 서브프로세스는 싱글스레드 상태에서 안전하게 VM을 초기화
- `BootConfig` 구조체(JSON 직렬화)로 부팅에 필요한 모든 정보를 새 프로세스에 전달
- 두 모드 (`start_with_full_config` / `start_via_subprocess`) 모두 동일한 `prepare_for_launch` / `finalize_launch`를 거쳐 일관성 확보

### 4.4 "에이전트 샌드박스" 관점의 의미
- **커널 공유가 없다** → 게스트 커널 익스플로잇이 호스트 커널로 직접 이어지지 않음 (Hypervisor.framework/KVM 탈출이 필요)
- **컨테이너 이스케이프 전체 계열 무력화**: runc CVE, cgroup 관련 취약점, `/proc/*/status` 누출 등이 호스트에 영향을 주지 못함
- 즉 **"LLM이 임의의 bash/Python/빌드 툴체인을 실행하는 상황"** 에서 컨테이너보다 훨씬 보수적인 격리 경계를 제공

---

## 5. 호스트 ↔ 게스트 통신: vsock + 길이 프리픽스 JSON

### 5.1 프로토콜 (`crates/smolvm-protocol`)
- **와이어 포맷**: `[4바이트 빅엔디안 길이][JSON 페이로드]`
- **최대 프레임**: 32 MB, 바이너리는 base64 인코딩
- **Envelope**: `Envelope<T>` 래퍼로 trace_id 부착 가능 → 분산 추적/로그 상관관계

### 5.2 AgentRequest (호스트 → 에이전트 VM)
```
// 이미지 관련
Pull, Query, ListImages, GarbageCollect

// 오버레이
PrepareOverlay, CleanupOverlay

// 파일 I/O
FileWrite, FileWriteBegin, FileWriteChunk, FileRead

// 명령 실행
Run, VmExec, Stdin, Resize

// 스토리지/운영
FormatStorage, StorageStatus, ExportLayer, NetworkTest, Ping, Shutdown
```

### 5.3 AgentResponse (에이전트 VM → 호스트)
```
Ok(JSON), Pong, Progress, Error(code, message)
Started, Stdout, Stderr, Exited           // 인터랙티브 모드
Completed                                 // 배치 모드
DataChunk                                 // 스트리밍 다운로드
```

### 5.4 Workload VM 프로토콜 (별도)
에이전트 VM이 아닌, 실제 워크로드 VM에는 더 단순한 메시지 세트가 쓰인다:
- Host→Guest: `Auth, Run, Exec, Signal, Stop`
- Guest→Host: `AuthOk/AuthFailed, Ready, Started, Stdout, Stderr, Exit, Error`

**에이전트 샌드박스 관점**: 이 프로토콜은 LLM 에이전트의 tool-use 패턴과 아주 매끄럽게 매핑된다. `run_code(image, cmd)` 툴 하나가 `Pull → PrepareOverlay → Run` 시퀀스로 치환되고, SSE 스트림(아래 §8)으로 실시간 stdout/stderr를 받을 수 있다.

---

## 6. 게스트 에이전트 (`crates/smolvm-agent`) 해부

`smolvm-agent`는 크기 최적화로 빌드되는 전용 Rust 바이너리다 (`profile.release-small`: opt-level `"z"`, panic=abort, LTO, strip).

### 6.1 초기화 시퀀스 (`main.rs`)
1. 필수 파일시스템 마운트 (로깅보다 먼저)
2. vsock 리스너를 **즉시** 생성 → 호스트가 이때부터 연결 시도 가능
3. `ready marker`(`.smolvm-ready`) 작성 → 호스트가 이 파일로 readiness 판정
4. 네트워크/스토리지 구성 (아래 5초 동안 호스트는 접속 유예)
5. accept loop 진입

### 6.2 핵심 모듈
| 모듈 | 역할 |
|---|---|
| `vsock` | Linux vsock 전용 리스너/스트림. 비 Linux 스텁은 에러 반환 |
| `storage` | `crane` CLI로 OCI 이미지 풀링 → tar 스트리밍 추출. lowerdir 다중 마운트 또는 병합 디렉토리 fallback. `libc::sync()`로 내구성 확보 |
| `oci` | 아래 §7 참조 — OCI config.json 생성 |
| `crun` | crun runtime 호출 빌더. cgroup-manager/root 경로 일관성, PATH env 자동 주입 |
| `pty` | `openpty()` + `setsid`/`TIOCSCTTY`로 인터랙티브 세션 지원 |
| `ssh_agent` | `/tmp/ssh-agent.sock` 리스너 → vsock 브릿지 (poll 멀티플렉싱) |
| `dns_proxy` | `/etc/resolv.conf`를 127.0.0.1:53으로 덮어쓰고 UDP-over-vsock으로 호스트에 전달. 연결 실패 시 SERVFAIL 생성 |
| `process` | EINTR 재시도, 10ms 폴링 기반 timeout, 클라이언트 연결 끊김 시 자식 kill (고아 프로세스 방지) |
| `network` | 게스트 네트워크 구성 (인터페이스/라우트) |
| `retry` | 공유 재시도 로직 |

### 6.3 명령 실행 경로
`Run` 요청 하나의 처리 흐름:
1. `storage.prepare_overlay()` — OCI 레이어가 캐시에 없으면 풀링 → overlayfs 구성
2. `oci.generate_config()` — capabilities, namespaces, rlimit, masked/read-only paths, env/cmd 채움
3. `crun create` → `crun start`
4. stdout/stderr를 `AgentResponse::Stdout/Stderr`로 스트리밍 (base64 바이너리 세이프)
5. `crun` 종료 → `Exited{code}` / `Completed` 전송
6. `crun delete`로 컨테이너 정리

---

## 7. OCI 런타임 사양 및 보안 한계 (`oci.rs`)

게스트 에이전트가 crun에 넘기는 `config.json`에 하드코딩된 보안 제약:

### 7.1 Capabilities (root 컨테이너)
14개 기본 Linux capability만 허용:
`CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FSETID, CAP_FOWNER, CAP_MKNOD, CAP_NET_RAW, CAP_SETGID, CAP_SETUID, CAP_SETFCAP, CAP_SETPCAP, CAP_NET_BIND_SERVICE, CAP_SYS_CHROOT, CAP_KILL, CAP_AUDIT_WRITE`
- **Inheritable/ambient은 비어 있음** → setuid 바이너리로의 권한 상승 차단
- Docker 기본(14종)과 동등 수준 — 컨테이너 관행을 따라감

### 7.2 Masked paths (읽어도 빈 값)
`/proc/asound, /proc/acpi, /proc/kcore, /proc/keys, /sys/firmware`

### 7.3 Read-only paths
`/proc/bus, /proc/fs, /proc/irq, /proc/sys, /proc/sysrq-trigger`

### 7.4 Mount 플래그
- `/proc`, `/dev`, `/dev/pts`, `/dev/shm`, `/sys`, `cgroup` 모두 `nosuid,noexec,nodev` 기본 적용
- `/sys`는 read-only

### 7.5 Rlimits
- `RLIMIT_NOFILE` soft/hard = 1024

### 7.6 입력 검증
- 이미지 레퍼런스 최대 512자, 셸 메타문자 (`$`, `` ` ``, `|`, `;`, `&`, `<`, `>`) 차단
- env 키 256자 / 값 32KB 제한
- 볼륨 마운트 경로: 심링크 탈출 / 디렉토리 트래버설 방지 (`storage.rs`에서 검증)

### 7.7 에이전트 샌드박스 관점 평가
- VM 경계 + 컨테이너 기본 권한 + 경로 마스킹의 **3중 방어**
- 단, seccomp 프로필은 코드에서 별도로 본 게 없음 — crun 기본 seccomp를 쓸 가능성이 높으나, LLM 에이전트에 쓰려면 **튜닝할 여지** 있음 (예: `ptrace`, `clone3` 제한)

---

## 8. 호스트 HTTP API (Axum)

에이전트 하네스에서 가장 중요한 인터페이스다. OpenAPI 스펙과 Swagger UI까지 제공.

### 8.1 주요 라우트
```
GET  /health                                    헬스체크
GET  /metrics                                   Prometheus

POST /api/v1/machines                           VM 생성
GET  /api/v1/machines                           VM 목록
GET  /api/v1/machines/{id}                      VM 조회
POST /api/v1/machines/{id}/start                시작
POST /api/v1/machines/{id}/stop                 정지
DELETE /api/v1/machines/{id}                    삭제

POST /api/v1/machines/{id}/exec                 명령 실행 (버퍼링)
POST /api/v1/machines/{id}/exec/stream          명령 실행 (SSE 스트림)
POST /api/v1/machines/{id}/run                  컨테이너 실행
GET  /api/v1/machines/{id}/logs                 로그 스트림 (SSE, 무기한)

PUT  /api/v1/machines/{id}/files/{path}         파일 업로드
GET  /api/v1/machines/{id}/files/{path}         파일 다운로드

GET  /api/v1/machines/{id}/images               이미지 목록
POST /api/v1/machines/{id}/images/pull          OCI 풀
```

### 8.2 특징
- 기본 **5분 타임아웃** (logs는 무기한)
- CORS 기본값은 localhost
- 요청마다 **trace_id** 자동 생성 → 로그/응답 상관관계
- **메트릭 경로 정규화**로 machine ID에 의한 카디널리티 폭증 방지
- exec/stream의 SSE 포맷: `stdout`, `stderr`, `exit`, `error` 이벤트 타입

### 8.3 에이전트 하네스 연동 시 이점
- **SSE 스트리밍**이 처음부터 설계에 포함 → LLM에 부분 출력을 바로 태울 수 있음
- REST + JSON은 파이썬/TS 클라이언트 자동 생성이 쉬움 (OpenAPI 제공)
- `/run` (컨테이너) vs `/exec` (VM 직접)의 구분 — tool을 **"컨테이너 run"**과 **"호스트 네임스페이스의 Alpine에서 exec"** 두 층으로 설계 가능

---

## 9. 스토리지 모델

### 9.1 디스크 2종
- **storage.raw (sparse ext4)**: OCI 레이어 캐시
  - `layers/`: 콘텐츠 주소 기반 추출 (SHA256)
  - `configs/`: 이미지 config
  - `overlays/{wl}/`: upper/work/merged
  - `manifests/`: 매니페스트 캐시
- **overlay.raw (sparse ext4)**: 루트FS 변경을 지속화하는 overlayfs의 upper layer
  - `apk add`, `pip install` 같은 것이 재부팅 이후에도 살아 있다

### 9.2 경로 규칙 (macOS 기준)
- 기본 VM: `~/Library/Application Support/smolvm/storage.raw`
- 명명된 VM: `~/Library/Caches/smolvm/vms/{name}/storage.raw`

### 9.3 Virtiofs + 바인드 마운트
- `-v HOST:GUEST[:ro]` 로 호스트 디렉토리를 읽기/읽기전용으로 공유
- 태그별로 고유 virtiofs 디바이스 → 게스트 측에서 staging 위치로 마운트한 뒤 컨테이너 rootfs에 bind-mount
- **심링크 탈출 방어**가 코드에 명시됨

### 9.4 에이전트 관점
- **"ephemeral run"**: `machine run`은 자동 삭제되어 한 번의 tool call에 격리된 세계를 보장
- **"persistent dev env"**: `machine create` + overlay disk로 state를 유지 — 장기 세션에서 에이전트가 설치한 라이브러리가 보존

---

## 10. 네트워크 모델 (중요)

### 10.1 기본 정책: 차단
- Smolfile에 `net = true`를 명시하지 않으면 네트워크 자체가 없다
- 즉 **기본값이 오프라인 샌드박스** — 외부 통신 없는 코드 분석에 즉시 쓸 수 있음

### 10.2 네트워크 백엔드
- **TSI (Transparent Socket Impersonation)**: libkrun 내장. 게스트 소켓 호출이 호스트 소켓으로 "의인화"되어 통과 (빠르지만 기능 제한)
- **virtio-net**: virtio NIC + 호스트측 `smolvm-network` crate의 smoltcp 사용자공간 TCP 스택 + UDP 게이트웨이 + DNS 포워딩
  - `crates/smolvm-network/`: `FrameStreamBridge`(frame ↔ queue), `NetworkFrameQueues`, smoltcp poll thread
  - Guest ↔ libkrun (Unix socket) ↔ frame bridge ↔ smoltcp ↔ 호스트 소켓

### 10.3 Egress 제한 3단 콤보
- `--allow-cidr 10.0.0.0/8` — CIDR allowlist
- `--allow-host api.stripe.com` — 호스트명을 DNS로 풀어 CIDR로 확장 + DNS 필터에도 등록. **실패 시 hard error** (코멘트 "should not silently weaken the security policy")
- `--outbound-localhost-only` — 127.0.0.0/8 + ::1/128만

### 10.4 DNS 필터 (양방향)
- 게스트: `dns_proxy.rs` — resolv.conf를 127.0.0.1:53로 지정하고 실제 UDP 패킷을 vsock 프레이밍(2바이트 빅엔디안 길이)으로 호스트에 넘김
- 호스트: `dns_filter.rs` — vsock로 받은 raw DNS 패킷을 **외부 라이브러리 없이** 파싱, allowlist에 대해:
  - 정확 매칭 (`api.stripe.com` ⇔ `api.stripe.com`)
  - 와일드카드 (`stripe.com` → `*.stripe.com`)
  통과하면 업스트림(1.1.1.1 등)으로 재시도, 차단하면 **NXDOMAIN** 응답
- 즉 **앱 바이너리를 수정 없이 두고도 네트워크 allowlist 강제**

### 10.5 에이전트 샌드박스 관점 평가
- 공급망 공격 대응으로 훌륭: `pip install`이 npm/pypi 외의 곳에 콜백하는 걸 DNS 레벨에서 차단 가능
- 프록시/VPN 없이 **도메인 단위의 egress 정책**을 하드웨어 격리 경계 안에서 운영 가능 — 클라우드 에이전트 플랫폼에서 가치가 큼

---

## 11. 자격 증명과 SSH Agent 포워딩

- 에이전트가 `git clone git@github.com:...`을 해야 하지만 **개인키를 VM 안에 복사하고 싶지 않은** 고전적 문제
- smolvm의 접근:
  1. 호스트의 `SSH_AUTH_SOCK`을 호스트 측 "브릿지"에 노출
  2. 게스트 에이전트가 `/tmp/ssh-agent.sock`을 만들고, 여기 오는 SSH 에이전트 프로토콜 요청을 **vsock으로 호스트 브릿지에 전달**
  3. 컨테이너 실행 시 이 소켓을 bind-mount + `SSH_AUTH_SOCK` 환경변수 설정
  4. `poll()` 멀티플렉싱으로 요청/응답을 바이트 단위로 중계
- **개인키는 절대 게스트에 들어오지 않는다** — sign 요청만 호스트로 올라와서 수행된 서명 결과만 돌아온다
- Docker 자격증명도 동일 관점: `--docker-config`로 `~/.docker/` 마운트하되 일반적으로 읽기전용 바인드

**에이전트 관점 평가**: LLM이 git push 같은 권한 있는 동작을 수행해도, 키 유출 위험은 호스트 SSH agent 단에서 관리 가능 (예: `ssh-add -c` confirm, 세션 타임아웃)

---

## 12. Smolfile 설정 스키마 (`crates/smolvm-smolfile`)

```toml
# 최상위
image = "python:3.12-alpine"       # OCI ref (옵션 → bare Alpine)
entrypoint = ["python3"]
cmd = []
env = ["PYTHONDONTWRITEBYTECODE=1"]
workdir = "/workspace"
cpus = 2
memory = 512                       # MiB (기본 256)
storage = 10                       # GiB
overlay = 2                        # GiB
net = true
ports = ["8080:8080"]
volumes = ["./src:/workspace/src:ro"]
init = ["pip install -r requirements.txt"]

[dev]                              # dev override (pack 시 제외)
volumes = ["./local-secrets:/secrets:ro"]
init    = ["pip install ipython"]

[artifact]                         # .smolmachine용 설정
cpus = 1
memory = 256
oci_platform = "linux/arm64"

[network]
allow_hosts = ["api.openai.com", "pypi.org"]
allow_cidrs = ["10.0.0.0/8"]

[health]
exec = ["curl", "-f", "http://localhost:8080/health"]
interval = "30s"
timeout = "5s"
retries = 3
startup_grace = "10s"

[restart]
policy = "on-failure"   # never | always | on-failure | unless-stopped
max_retries = 3
max_backoff = "30s"

[auth]
ssh_agent = true

[service]               # 배포 메타
port = 8080
protocol = "http"
```

- `#[serde(deny_unknown_fields)]` — 오타는 에러
- **[dev] vs [artifact]** 분리: 개발 편의(디버그 마운트)가 배포물(.smolmachine)에 안 섞임
- **Configuration-as-code**: 에이전트 도구 설명에 "이 Smolfile을 줄 테니 run 해라" 수준으로 추상화 가능

---

## 13. `.smolmachine` 포터블 아티팩트

### 13.1 내용물 (`crates/smolvm-pack`)
- stub 실행파일 = smolvm 바이너리 자체
- libkrun / libkrunfw
- 에이전트 rootfs
- OCI 레이어 또는 VM 오버레이 디스크 (tar)
- 매니페스트 (CPU/mem 기본, entrypoint/env/workdir 등)
- **Mach-O 섹션 임베드** (macOS) — 단일 파일 서명 가능
- 서명 지원 (`signing.rs`), 풋터 magic bytes + manifest + 체크섬

### 13.2 생성 방식
- 소스: OCI 이미지, 정지된 VM 스냅샷, Smolfile
- 이미지 기반 팩: 임시 에이전트 VM을 띄워 이미지 풀 → 레이어 병합(다중 lowerdir 성능 이슈 회피) → tar export
- VM 기반 팩: overlay 디스크 export

### 13.3 배포
- `pack push/pull/inspect`: OCI Distribution 스펙 호환 레지스트리로 임의의 `.smolmachine` 전송 — 기존 컨테이너 레지스트리 인프라를 그대로 사용
- **bin + sidecar** 또는 **단일 파일** 모두 지원

### 13.4 에이전트 샌드박스 관점
- **재현성**: "개발자 PC에서 잘 되는데 CI에서 실패" 문제의 상당 부분이 사라짐. 커널까지 핀 고정.
- **플랫폼 포팅**: 에이전트 실행 환경을 `.smolmachine` 하나로 배포 가능 → 신규 팀원/CI 노드에 설치 붐 없이 격리 환경 보급

---

## 14. 언어별 바인딩과 임베디드 런타임

- **`crates/smolvm-napi`**: Node.js 네이티브 바인딩 (NAPI-RS). TypeScript/Node 에이전트 하네스에서 직접 호출 가능
- **`src/embedded/`**: "language-neutral embedded runtime support for SDK bindings"
  - `MachineSpec`, `EmbeddedRuntime`, `runtime()` 추상화
  - 향후 Python/Go/Ruby 바인딩의 공유 진입점으로 보임
- **HTTP API는 언제나 작동**: 언어 무관. OpenAPI → openapi-generator로 어떤 스택에서도 클라이언트 자동 생성

---

## 15. 에이전트 샌드박스로서의 종합 평가

### 15.1 강점 (★ = 에이전트 샌드박스 핵심 요구사항과의 적합도)
1. ★★★ **진짜 격리**: VM-per-workload + 자체 커널. LLM이 예상 못 한 익스플로잇을 시도해도 호스트로 뚫리려면 하이퍼바이저 레벨 버그가 필요
2. ★★★ **오프라인 기본값 + 도메인 allowlist**: 데이터 유출/공급망 공격 대응. 코드 수정 없이 정책 enforcement
3. ★★★ **저지연 콜드 스타트 < 200ms**: 에이전트 tool call마다 새 VM을 만들어도 된다는 의미. 상태 비 의존 샌드박싱이 실용적
4. ★★★ **OCI 호환**: `python:3.12`, `node:20-alpine` 등 기성 이미지 그대로 사용. 에이전트에게 "python 실행기"를 주는 것이 1줄
5. ★★ **개인키 분리**: SSH agent forwarding — git 작업 자동화를 안전하게
6. ★★ **HTTP API + SSE**: 원격 제어 및 스트리밍. LLM orchestrator와의 궁합이 좋음
7. ★★ **macOS 1급 지원**: 개발자 로컬에서 Docker Desktop 없이 바로 작동
8. ★ **`.smolmachine` 이식 단위**: 환경 drift 제거. "에이전트 런타임 이미지"를 파일로 서명/배포
9. ★ **Smolfile 선언형 구성**: 에이전트 도구 정의와 잘 매핑 (image, env, allow_hosts가 tool metadata)

### 15.2 약점 / 주의점
1. **Windows 호스트 미지원** — 팀 내 Windows 개발자가 있다면 WSL2에서 KVM을 쓰거나 대안 필요
2. **libkrun 의존** — 신뢰해야 하는 C 코드 표면이 있음 (Red Hat/Podman 상용/오픈 에코 활발하지만 여전히 단일 의존)
3. **seccomp 프로필 미확인** — crun 기본값에 의존. LLM을 위해 커스텀 seccomp를 얹으려면 소스 수정 필요 가능성
4. **게스트 agent는 root 수준 권한**을 vsock 통신에서 행사 — 프로토콜 메시지의 **인증**은 Workload VM 프로토콜엔 `Auth`가 있지만 Agent VM엔 vsock 자체 신뢰에 의존
5. **성숙도**: 2025-12 생성, 스타 2k로 관심 높지만 아직 프로덕션 전쟁터 검증이 길지 않음. 버전 0.5.x
6. **GPU 미지원** (README가 "in progress"로 기재) — LLM 서빙/트레이닝 용도엔 아직 한계
7. **네트워크 성능**: smoltcp 사용자공간 TCP는 정책성이 좋지만 throughput은 TSI 대비 낮다. 대용량 파일 전송 워크로드에선 튜닝 필요
8. **다중 VM 리소스 관리**: 글로벌 CPU/메모리 풀, 스케줄링은 OS에 맡기는 구조. 수십~수백 에이전트를 한 박스에서 돌릴 때는 외부 오케스트레이터(쿠버네티스 등) 역할이 필요
9. **관측성**: Prometheus metrics와 tracing은 있으나, 감사 로그(어느 명령이 어떤 VM에서 언제 실행되었는지의 중앙집중 로그)는 기본 제공 범위 밖으로 보임

### 15.3 보안 표면 체크리스트

| 항목 | 상태 |
|---|---|
| 게스트 커널 ↔ 호스트 커널 격리 | ✅ Hypervisor.framework / KVM |
| 컨테이너 이스케이프 계열 차단 | ✅ (VM 경계 때문에 의미 없음) |
| 네트워크 egress allowlist | ✅ CIDR + 호스트명 + DNS 필터 |
| 파일시스템 공유 심링크 탈출 방어 | ✅ 코드에 명시 |
| 이미지 입력 샤니타이즈 | ✅ 길이/메타문자 검증 |
| Capabilities 최소화 | ✅ (Docker 기본 14종, ambient 비어 있음) |
| masked/ro paths | ✅ |
| rlimits | ✅ (NOFILE=1024) |
| seccomp 프로필 커스터마이즈 | ⚠️ 기본 crun 의존. 확인/강화 필요 |
| SSH 키 분리 | ✅ vsock 에이전트 포워딩 |
| 게스트→호스트 프로토콜 인증 | ⚠️ Agent VM 프로토콜은 신뢰된 vsock 가정 |
| 감사 로그 | ⚠️ 기본 트레이싱만 제공, 집계는 사용자 몫 |
| 서명된 이식 단위 | ✅ `.smolmachine` signing 지원 |

---

## 16. 에이전트 하네스 통합 설계 제안

### 16.1 레퍼런스 배포 토폴로지
```
┌──────────────────┐      HTTPS(TLS)       ┌────────────────────────────┐
│ LLM Orchestrator │─────────────────────▶│ smolvm serve (HTTP API)    │
│ (Python/TS)      │◀─── SSE stdout ──────│ • REST + OpenAPI           │
└──────────────────┘                       │ • Prometheus + tracing     │
                                           │ • 호스트별 1개 프로세스       │
                                           └──────────┬─────────────────┘
                                                      │ posix_spawn
                                  ┌───────────────────┴────────────────────┐
                                  │                                          │
                         ┌────────▼────────┐                       ┌────────▼────────┐
                         │ Agent VM        │  …  workload VMs      │ Workload VM N   │
                         │ (smolvm-agent)  │◀─ vsock 6000 ────────▶│ image X, net:off│
                         └─────────────────┘                       └─────────────────┘
```
- 한 호스트에 `smolvm serve` 하나를 띄우고, LLM orchestrator는 HTTP로만 말한다 → 하네스와 VMM 간 분리
- tool call = `POST /machines/{sessionId}/run` (ephemeral) 또는 `/exec/stream` (persistent 세션)

### 16.2 LLM tool 정의 예 (의사코드)
```python
@tool
def run_in_sandbox(image: str, cmd: list[str], timeout_s: int = 60,
                   allow_hosts: list[str] | None = None) -> RunResult:
    """격리된 마이크로VM에서 명령을 실행하고 stdout/stderr/exit를 반환."""
    return smolvm.post("/api/v1/machines", {
        "image": image, "cmd": cmd,
        "network": {"allow_hosts": allow_hosts or []},
        "ephemeral": True, "timeout": timeout_s
    }).stream_sse()
```

### 16.3 운영 팁
- **세션별 VM**: 한 agent 세션 = 한 persistent VM. overlay disk가 파이썬 venv 등을 보존
- **단발 코드 실행**: tool마다 새 ephemeral VM. 200ms면 LLM 턴당 오버헤드로 허용 가능
- **작업별 이미지**: 브라우저 자동화용, 빌드용, 분석용 — `.smolmachine` 미리 빌드해 레지스트리에 비축
- **CIDR 대신 호스트명**: `allow_hosts = ["pypi.org", "files.pythonhosted.org"]` 같은 선언이 DNS 필터에서 동작
- **`[auth].ssh_agent = true`**로만 키 접근 허용, 절대 `-v ~/.ssh:/root/.ssh`를 쓰지 말 것
- **Conductor 같은 멀티워크스페이스 환경**: 각 워크스페이스가 자기 `smolvm serve` 포트 하나씩 점유하도록 하면 교차 오염 제거

---

## 17. 대안 기술과의 비교 (에이전트 샌드박싱 관점)

| 기술 | 격리 강도 | 콜드 스타트 | OCI 호환 | 자격증명 격리 | macOS 네이티브 | 이식 단위 | 메모 |
|---|---|---|---|---|---|---|---|
| **smolvm** | VM | <200ms | ✅ | SSH agent fwd | ✅ | .smolmachine | 본 문서 대상 |
| Docker 컨테이너 | 공유 커널 | ~100ms | ✅ | bind mount | Docker Desktop 필요 | 이미지 | 컨테이너 이스케이프 위험 |
| Firecracker | VM | <125ms | 제한적 | 수동 | ❌ Linux 전용 | rootfs | AWS Lambda 기반, 프로덕션 검증, macOS X |
| Kata Containers | VM | ~500ms | ✅ | 컨테이너와 동일 | ❌ | 이미지 | Kubernetes 통합 성숙 |
| gVisor | 사용자공간 커널 | ~수 ms | ✅ | bind mount | ❌ | 이미지 | 진짜 VM 아님, 일부 syscall 호환 이슈 |
| E2B Sandbox | VM(Firecracker) | ~수백 ms | 제한적 | 클라우드 API | 해당 없음 (SaaS) | 템플릿 | LLM 전용 SaaS, 로컬 셀프호스팅 제약 |
| Cloudflare Sandbox | Workers isolate | <수 ms | ❌ (JS/WASM) | 제한적 | N/A | JS 번들 | 코드 실행 언어 제한 |
| QEMU microvm | VM | ~수 초 | 수동 | 수동 | ✅ | 디스크 | 너무 저수준 |
| Bubblewrap/bwrap | 네임스페이스 | 즉시 | 부분 | bind mount | ❌ | 없음 | Flatpak 기반, 강한 샌드박스 아님 |

**요약**: 로컬/데스크톱에서 macOS를 1급으로 지원하면서 VM 격리 + OCI + 이식 바이너리 + HTTP API + egress allowlist를 전부 갖춘 조합은 smolvm이 드물다. Firecracker는 성능은 좋으나 macOS 미지원 & Lambda 류 클라우드 전제. E2B는 SaaS 의존이 강함.

---

## 18. 도입 시 고려할 질문들 (의사결정 체크리스트)

1. **대상 호스트가 macOS/Linux만으로 충분한가?** (Yes → smolvm 강력 후보, No/Windows 필요 → WSL2 기반 Linux 호스트 경유 설계 필요)
2. **에이전트가 인터넷 접근이 필요한가?** (도메인 단위 필터 요구면 적합)
3. **세션 상태를 VM에 얼마나 보존할 것인가?** (ephemeral = run, persistent = create + overlay disk)
4. **LLM이 호스트 권한 파일을 건드리면 안 되는가?** (ro virtiofs + `--allow-host`만 쓰고 `-v $HOME` 같은 건 금지)
5. **오디트/컴플라이언스 요구?** (외부 로그 수집기에 trace_id와 Prometheus 메트릭을 연동해야 함 — 기본 제공 아님)
6. **패키징/배포 주기?** (자주 이미지를 바꾼다면 `.smolmachine` 빌드를 CI에 추가)
7. **성능 목표?** (네트워크 대역폭이 중요하면 TSI, 정책이 중요하면 virtio-net + smoltcp)
8. **다중 테넌트 클러스터?** (smolvm 자체는 호스트당 서비스. 쿠버네티스 수준 스케줄러는 상위에서 조립 필요)

---

## 19. 후속 조사/검증 아이템

- [ ] 게스트 `crun` 호출 시 적용되는 **seccomp 프로필 상세** 확인 (기본인지, 커스텀인지)
- [ ] Hypervisor.framework 관련 CVE 대응 주기 (smolvm 릴리즈 노트와 Apple 업데이트 간격)
- [ ] vsock 프로토콜에 대한 **대역폭/지연 벤치마크** (수천 tool call 시나리오)
- [ ] `.smolmachine` 서명 검증 플로우 및 공격 시나리오 (매니페스트 변조, 풋터 재작성)
- [ ] 다중 VM 동시 기동 시 **메모리 풋프린트** 실측 (virtio-balloon의 실제 효과)
- [ ] NAPI 바인딩 성숙도 (TS 에이전트 하네스 채택 전 타입/에러 경계 확인)
- [ ] `/api/v1/machines/{id}/files/{path}` 대용량 전송 시 성능/타임아웃
- [ ] **관측성 확장** 방법 (OpenTelemetry exporter, 구조화 audit log 파이프라인)
- [ ] libkrun 업스트림과 smolvm 포크/패치 관리 방식 (하위 호환성 리스크)

---

## 20. 결론

smolvm은 "**컨테이너의 개발자 경험 + VM의 격리 강도 + Firecracker의 속도 + macOS 1급 지원**" 을 한 덩어리로 가진 드문 조합이다. 특히 **네트워크 기본 차단 + 도메인 allowlist + SSH agent 포워딩** 같은 세 가지는 LLM 에이전트가 임의 코드를 실행할 때 필요한 **"데이터가 안 새고, 키는 누수되지 않고, 외부로 달라붙지도 못한다"** 를 하나의 런타임에서 제공한다.

프로젝트 자체는 아직 초기(v0.5.x, 2026-04 기준 4개월 된 리포지토리)이지만, 구성요소(libkrun, crun, OCI, smoltcp)는 모두 운용 실적이 있는 것들을 조합한 것이라 코어 위험은 제한적이다. 에이전트 샌드박스 기술 후보로서 **PoC에 우선순위 높게 올릴 가치**가 있다 — 특히 macOS 개발자 PC에서 도는 로컬 에이전트나, Mac Mini/Mac Studio 기반 엣지 에이전트 인프라에 적합하다.

Windows 데스크톱 지원이 필요하거나 수십 테넌트를 공유 호스트에 겹치는 대규모 클라우드 서비스 용도라면, smolvm을 **컴포넌트 중 하나**로 쓰고 위에 스케줄러/오케스트레이터를 얹는 설계가 현실적이다.

---

### 부록 A. 주요 파일 빠른 참조

| 주제 | 파일 |
|---|---|
| libkrun FFI/부팅 | `src/vm/backend/libkrun.rs` |
| 게스트 에이전트 진입 | `crates/smolvm-agent/src/main.rs` |
| 프로토콜 | `crates/smolvm-protocol/src/lib.rs` |
| CLI 머신 명령 | `src/cli/machine.rs` |
| HTTP API 진입 | `src/api/mod.rs` |
| SSE 실행 | `src/api/handlers/exec.rs` |
| DNS 필터 (호스트) | `src/dns_filter.rs` |
| DNS 프록시 (게스트) | `crates/smolvm-agent/src/dns_proxy.rs` |
| OCI 스펙 생성 | `crates/smolvm-agent/src/oci.rs` |
| SSH 에이전트 브릿지 | `crates/smolvm-agent/src/ssh_agent.rs` |
| 스토리지/오버레이 | `src/storage.rs`, `crates/smolvm-agent/src/storage.rs` |
| 네트워크 스택 (호스트) | `crates/smolvm-network/src/lib.rs` |
| 패킹 포맷 | `crates/smolvm-pack/src/lib.rs` |
| Smolfile 스키마 | `crates/smolvm-smolfile/src/lib.rs` |
| 설정 DB | `src/config.rs`, `src/db.rs` |

### 부록 B. CLI 치트시트

```bash
# 일회성 실행 (네트워크 꺼짐, 자동 정리)
smolvm machine run --image alpine -- sh -c "echo hello"

# 도메인 allowlist만 허용
smolvm machine run --image python:3.12-alpine \
  --allow-host pypi.org --allow-host files.pythonhosted.org \
  -- python -c "import requests; print(requests.get('https://pypi.org').status_code)"

# 지속형 개발 환경
smolvm machine create dev -I python:3.12-alpine --cpus 2 --mem 1024
smolvm machine start dev
smolvm machine exec dev -i -t -- bash

# Smolfile 기반 선언형 구성
smolvm machine create myvm -s ./Smolfile

# 포터블 아티팩트
smolvm pack create --image python:3.12-alpine -o ./python.smolmachine
./python.smolmachine  # 어디서든 실행

# HTTP API 서버
smolvm serve --port 7777
# Swagger UI: http://localhost:7777/swagger-ui/
```
