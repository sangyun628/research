# Prometheus vs VictoriaMetrics: 운영자 관점 심층 비교

> **TL;DR** — Prometheus는 단일 클러스터·중간 규모 워크로드에서는 여전히 최선의 선택이며 LGTM/Mimir와 자연스럽게 통합되는 사실상 표준이다. 그러나 멀티 클러스터 EKS, 장기 보관, 카디널리티 폭발 또는 비용 절감이 핵심 KPI라면 VictoriaMetrics(VM)는 단일 노드만으로도 Mimir 클러스터를 대체할 수 있을 만큼 자원 효율이 뛰어나다. 단, VM의 핵심 운영 기능(downsampling, retention filter, multiple retentions)은 Enterprise이며 WAL이 없는 설계로 인한 트레이드오프가 있다는 점을 분명히 인지하고 도입해야 한다.

---

## 1. 역사적 맥락 — 왜 두 프로젝트가 따로 존재하는가

### 1.1 Prometheus (2012, SoundCloud)

Prometheus는 2012년 2월 Matt T. Proud의 사이드 프로젝트로 시작되어, 같은 해 9월 Matt가 SoundCloud에 합류하고 10월 Julius Volz가 합류하면서 본격적으로 개발되었다(공식 commit 기록은 2012년 11월). 두 사람 모두 이전 직장이 Google이었기 때문에 Google 내부 모니터링 시스템 **Borgmon**의 데이터 모델·alerting 패턴을 강하게 의식했다. 당시 SoundCloud는 Docker 이전부터 자체 컨테이너 오케스트레이터 위에서 수백 개 마이크로서비스를 돌리고 있었는데, StatsD/Graphite/RRDtool 기반 도구들은 이런 *동적이고 high-cardinality한 환경*을 다루지 못했다는 것이 출발점이다(Volz, "Prometheus Turns 10").

핵심 설계 원칙은 다음과 같다:

- **다차원 데이터 모델**(label 기반) — Borgmon의 핵심 사상
- **Operationally simple** — 외부 의존성 없음, 단일 바이너리, 로컬 디스크
- **Pull-based scraping** — Brian Brazil은 "engineering 관점에서 push/pull은 거의 무관하지만, pull이 *아주 약간* 더 낫다"고 정리한다(SE-Radio 270, robustperception.io). pull의 실질적 이점:
  1. 타깃의 up/down 자체가 자연스러운 헬스 체크
  2. service discovery가 모니터링과 동일 경로로 통합
  3. 부하 분산 게이트웨이 없이도 sharding이 자연 분리됨(각 Prometheus가 자기 영역만 scrape)
  4. push→pull은 어렵지만 pull→push는 쉬움(`/metrics` 엔드포인트를 캐시한 뒤 forward)

타임라인:
- 2015년 1월 — 공식 발표, 같은 날 Hacker News 1위
- 2016년 5월 — CNCF의 두 번째 incubating 프로젝트(Kubernetes 다음)
- 2016년 7월 — v1.0 (API stability)
- 2017년 11월 — **v2.0**: TSDB v3 도입(Fabian Reinartz 주도), churn에 강한 새 storage 엔진
- 2018년 8월 — **CNCF Graduated** (두 번째 졸업 프로젝트)

### 1.2 VictoriaMetrics (2018, Kyiv)

VictoriaMetrics는 Aliaksandr Valialkin이 시작했다. 그는 Go 생태계에서 `fasthttp`, `quicktemplate`, `fastcache` 등 high-performance 라이브러리 저자로 이미 유명한 엔지니어였고, 이전 직장(adtech)에서 Prometheus의 scaling 한계, 특히 **수직 확장만 되는 점, high-cardinality에서 OOM 발생, 장기 보관 부재**를 직접 겪었다. 그는 2018년 회사를 그만두고 ~2년치 생활비를 모은 뒤 풀타임 개발에 들어갔다(Underdog Founders interview).

VM의 가장 결정적인 영감은 **ClickHouse**였다(Yandex의 OLAP 엔진). Valialkin은 "ClickHouse의 MergeTree 아키텍처를 Prometheus처럼 시계열에 맞게 재해석한다면 어떻게 될까?"를 출발점으로 삼았다(공식 FAQ: "VictoriaMetrics core is written in Go from scratch... storage uses certain ideas from ClickHouse. Special thanks to Alexey Milovidov"). 따라서 VM은 Thanos/Cortex처럼 Prometheus 코드를 재사용하지 않고 처음부터 다시 짰다.

타임라인:
- 2018년 9월 — 첫 비공개 릴리스
- 2019년 — single-node 및 cluster 버전 모두 Apache 2.0으로 오픈소스화
- 2019~2021 — 공동창업자(Roman Khavronenko/前 Cloudflare, Dzmitry Lazerka/前 Lyft, Artem Navoiev/前 Google) 합류
- 2022~2024 — KubeCon/PromCon 정규 발표, **VictoriaLogs**·**vmanomaly**·**VictoriaTraces**까지 확장
- 2024~2026 — 다운로드 1B+, 사용자 사례(Roblox, Wix, Grammarly, Adidas, CERN, Spotify, Criteo, **Naver**, NetEase 등)

VM이 해결하려고 한 Prometheus의 구체적 한계:

| Prometheus 한계 | VM의 답 |
|---|---|
| 단일 노드 수직 확장만 | vminsert/vmselect/vmstorage 분리, shared-nothing 수평 확장 |
| 장기 보관 시 disk·RAM 비용 | Gorilla + ZSTD 추가 압축으로 2~10x 작은 디스크 |
| High-cardinality에서 OOM | inverted index 별도 분리, in-memory 데이터 최소화 |
| 멀티 테넌시 부재 | URL-based `accountID/projectID` native 지원 |
| HA·downsampling·long-term은 외부 도구 필요 | 단일 바이너리에 내장(downsampling은 Enterprise) |

---

## 2. 아키텍처 — 두 엔진은 디스크에 어떻게 다르게 쓰는가

### 2.1 Prometheus TSDB

Prometheus 데이터 디렉터리 레이아웃(공식 문서):

```
data/
├── wal/                              # 128MB segment 파일들 + checkpoint
│   ├── 00000003
│   └── checkpoint.00000002/00000000
├── chunks_head/                      # mmap된 head 청크 (v2.19+)
│   └── 000001
├── 01BKGV7JBM69T2G1BGBGM6KB12/       # ULID 이름의 persistent block (보통 2h)
│   ├── meta.json
│   ├── index                         # postings + label index
│   ├── chunks/000001                 # ≤512 MiB segment
│   └── tombstones
└── lock
```

**Write path**:
1. scrape된 sample은 메모리의 **head block**에 쓰여지고 동시에 **WAL**(append-only, sequential)에 기록됨 — 크래시 복구용
2. head 안의 chunk는 120 sample 또는 chunkRange(기본 2h)가 차면 닫혀 `chunks_head/`로 mmap flush(v2.19+ 도입으로 RAM 사용량 대폭 감소)
3. 2h마다 head의 닫힌 chunk들이 **persistent block**으로 컴팩션, WAL은 truncation 후 checkpoint
4. 백그라운드 컴팩터가 인접 block을 더 큰 block(최대 10% retention)으로 다단계 머지

**압축 — Gorilla 알고리즘** (Pelkonen et al., Facebook 2015):
- Timestamp: **delta-of-delta** 인코딩(고정 scrape interval에서는 대부분 0~몇 비트)
- Value: 직전 값과의 **XOR** 후 leading/trailing zero를 가변 길이로 인코딩
- 결과: production node_exporter 데이터에서 평균 **~1.37 bytes/sample**(공식 PromCon 2017 자료, "Storing 16 bytes at scale")

**Index**: block마다 **inverted index**를 가진다. label `pair (k=v) → postings list (series ID 정렬 리스트)`. PromQL의 selector는 여러 postings list의 intersection으로 시리즈를 찾고, 그 다음 chunk를 mmap으로 읽어 디코딩. 이 inverted index는 head에서는 in-memory, persistent block에서는 디스크.

**WAL 비용**(robustperception 분석): 100k samples/s 환경에서 약 **6시간치, 26GB**, 13 bytes/sample(WAL 자체는 비압축). v2.11부터 `--storage.tsdb.wal-compression`로 압축 가능. WAL replay는 OOM 후 재시작 시 잘 알려진 큰 운영 통증 — 실제 EKS 환경에서 Prometheus가 OOMKilled → WAL replay하다가 다시 OOM되는 crash loop가 자주 보고된다(이 글의 사용자 후기 섹션 참고).

**PromQL 엔진**(`prometheus/promql/engine.go`): 쿼리는 AST로 파싱 → AST를 walk하면서 selector마다 storage의 `Select()` 호출 → series 반환 → step 단위로 sample 읽고 함수/operator 적용. binary operator의 vector matching은 매 step마다 hash table을 다시 빌드해 매칭하므로 비싸다(Grafana 블로그, "Inside PromQL"). 최근에는 thanos-io/promql-engine이 Volcano 모델 기반의 streaming/multi-threaded 엔진을 별도로 구현 중이며, Prometheus 본체에도 점진 도입 논의가 진행되고 있다.

### 2.2 VictoriaMetrics 엔진

VM 클러스터는 **shared-nothing** 3-tier 아키텍처다:

```
       remote_write / scrape
              │
        ┌─────▼─────┐    consistent hashing
        │ vminsert  │──────── shard ─────────┐
        └─────┬─────┘                        │
              │                              ▼
        ┌─────▼─────┐                  ┌──────────┐
        │ vminsert  │  ◄── replicate ──│ vmstorage│ ── on-disk parts
        └─────┬─────┘   factor=N       └──────────┘
              │                              ▲
        ┌─────▼─────┐                        │
        │ vmselect  │ ─────────── fan-out ───┘
        └───────────┘   PromQL/MetricsQL     parallel scatter-gather
```

- **vminsert** (stateless): remote_write/InfluxDB/Graphite/OTLP 등 다양한 프로토콜 수신, label hash로 vmstorage 노드에 분배. `-replicationFactor=N`이면 N개 노드에 동일 sample 기록.
- **vmstorage** (stateful): 실제 시계열 저장. shared-nothing이라 노드끼리 서로의 존재를 모른다. 디스크는 자기 디스크만 본다. 따라서 vmstorage 1대 죽어도 vminsert는 다른 노드로 reroute, vmselect는 부분 응답을 마킹.
- **vmselect** (stateless): 쿼리 받으면 모든 vmstorage에 fan-out, 결과를 머지 후 PromQL/MetricsQL 실행.

**Storage(vmstorage 내부) — ClickHouse MergeTree 변형**(dbdb.io DB of DBs 분류 + VM 공식 internals):

```
storage/data/
├── small/<partition>/<part>/        # column-oriented, ≤ 100MB
│   ├── values.bin                   # 압축된 값
│   ├── timestamps.bin               # 압축된 타임스탬프
│   ├── index.bin
│   └── metaindex.bin
├── big/<partition>/<part>/          # 머지된 큰 part
└── indexdb/<gen>/                   # mergeset(inverted index)
```

핵심 차이:
1. **WAL이 없다**. 대신 vminsert에서 받은 데이터를 in-memory `raw-row shards`(120MB 버퍼) → 1~5초마다 in-memory part로 flush → 디스크의 **inmemoryPart**로 atomic flush + fsync. fsync 대상이 이미 압축된 ~50KB짜리 작은 payload라 SSD에서 부담이 작다(VM CTO 글, "WAL Usage Looks Broken"). **트레이드오프**: 비정상 종료 시 **최대 ~1~5초의 in-RAM 데이터 손실 가능** — 대신 replication과 vmagent의 디스크 백업 큐로 보완.
2. **MergeTree 스타일 part 머지**: 각 part는 immutable. 백그라운드 머지(merge multiplier ≈ 7.5, 최대 15 part 동시)로 작은 part들을 큰 part로 합치고 동시에 dedup 수행. ClickHouse처럼 strict level은 없음. snapshot은 hardlink 한 번으로 즉시 생성 가능(매우 큰 디스크에서도 O(1)).
3. **Inverted index는 별도 mergeset**: `lib/mergeset`이 자체 LSM 트리. label→TSID lookup이 시계열 데이터와 분리돼 있어 high-cardinality에서 메인 데이터를 건드리지 않는다.

**압축 — Gorilla 위에 두 단계 더 쌓음**(Valialkin, "Achieving better compression than Gorilla"):
1. **float→int 정규화**: 시계열 값에 10^X를 곱해 정수화(소수점 12자리 한도)
2. **Counter→Gauge**: counter는 delta encoding으로 변환
3. **Gorilla 인코딩**(timestamp delta-of-delta + value XOR)
4. **ZSTD 일반 압축**으로 한 번 더(`lib/encoding/compress.go`의 `CompressZSTDLevel`). 데이터 종류에 따라 압축 레벨을 동적으로 선택.

결과: production node_exporter 트래픽에서 **~0.4 bytes/sample**, Prometheus 대비 **~2.5~7x** 작은 디스크(공식 자체 벤치 24h, 80K samples/s, 1000 targets).

**MetricsQL**: PromQL 슈퍼셋. 의도적 차이가 있다(공식 docs/metricsql + Khavronenko의 PromQL Compliance 분석).
- `rate()/increase()`가 lookbehind window의 직전 sample을 포함 → `increase(metric[$__interval])`이 끊기지 않음(Prometheus는 extrapolation으로 분수값 반환, VM은 정수 반환)
- 함수 적용 후 `__name__` 라벨 보존(Prometheus는 drop)
- step < scrape interval인 경우에도 빈 응답 대신 합리적 값 반환
- `WITH templates`(SQL의 CTE처럼 재사용 가능 표현식), Graphite filter `{__graphite__="dc*.appA.*"}`, `keep_last_value`, `running_sum`, `histogram_quantiles` 등 다수 추가 함수
- Prometheus Conformance Program 테스트 약 17%가 fail하는데, 대부분 **위 의도된 동작 차이**(라벨 보존, rate 의미 차이) 때문이며 실제 사용자 영향은 미미

**왜 디스크·RAM 차이가 발생하는가** — 한 줄 요약:

| 요인 | Prometheus | VictoriaMetrics |
|---|---|---|
| Active series in RAM | 모든 active series가 head + chunks_head + index | inverted index만 메모리에 캐시, 데이터 자체는 inmemoryPart 작은 버퍼만 |
| WAL | 비압축 6시간치 | 없음 |
| 압축 | Gorilla(1.37 B/sample) | Gorilla + 정수화 + ZSTD(0.4 B/sample) |
| 인덱스 | block마다 인덱스 + head index | mergeset 단일 LSM, 데이터/인덱스 분리 |
| 컴팩션 시 메모리 | head→2h block 컴팩션이 RAM-heavy | 작은 part들을 stream merge, RAM peak 작음 |

VM 자체 벤치(Aliaksandr Valialkin, Medium)에서 동일 1000 node_exporter 타깃 24h scrape 시 Prometheus가 최대 **~23GB RAM**, VM은 **~4.3GB RAM**으로 약 5.3x 차이를 보고했다. 이는 VM이 발표한 수치임을 명시.

---

## 3. 운영 / 스케일 / HA / 멀티테넌시

### 3.1 Prometheus의 HA 전략

Prometheus 본체는 **HA 메커니즘이 없다**. 일반적인 패턴:

1. **Pair of identical Prometheus** — 같은 scrape config 두 인스턴스 → Alertmanager가 dedupe(가장 단순, Brazil 권장 방식). 그러나 historical 데이터는 서로 약간 어긋남(scrape 시점 차이).
2. **Hierarchical federation**(`/federate`) — leaf Prometheus가 cluster별 메트릭을, 상위 Prometheus가 aggregated만 pull. Flipkart는 이걸로 80M metrics를 운영(InfoQ 2025). 한계: federate 자체가 무거운 scrape, alerting은 leaf에 두는 것이 권장, raw 시리즈를 위로 끌어올리지 말 것.
3. **Remote write → 외부 storage** — 가장 일반적. 옵션:
   - **Thanos**: Prometheus 옆 sidecar가 2h block을 S3/GCS에 업로드, Querier가 sidecar+Store Gateway를 통합 쿼리. Thanos sidecar는 Prometheus의 compaction을 끄게 만든다(RAM 증가 부작용).
   - **Cortex/Mimir**: distributor → ingester → 객체 저장소. Mimir는 Cortex fork(2022), 더 정리된 도큐먼트와 운영 모델, 자체 발표로 1B active series, 50M samples/s까지 스케일(7000 vCPU, 30TiB RAM 클러스터). Mimir는 멀티테넌시 first-class — `X-Scope-OrgID` 헤더로 분리, per-tenant rate limit, shuffle sharding.
   - **VictoriaMetrics** (long-term storage 또는 완전 대체)

### 3.2 Prometheus의 멀티테넌시 한계

Prometheus 본체는 **single-tenant 설계**다. 한 인스턴스가 받는 모든 메트릭은 동일한 namespace에 들어가며 per-tenant rate limit·query quota가 없다. 멀티테넌시가 필요하면 사실상 Mimir/Cortex 또는 VM-cluster로 가야 한다. Chronosphere의 분석은 "Prometheus has no concept of multi-tenancy or per-user overload controls. A single source or heavy query can take out the entire server"라고 명확히 한다.

### 3.3 VictoriaMetrics 클러스터 — 의미 있는 분리

3개 컴포넌트가 *왜* 분리되어 있는지가 중요하다:

- **vminsert와 vmstorage 분리**: ingest 트래픽이 폭증해도 vminsert만 HPA로 늘리면 됨. vmstorage는 stateful이라 신중히 늘려야 하지만, vminsert는 stateless라 무한 가까이 확장 가능.
- **vmselect와 vmstorage 분리**: query workload(특히 dashboard refresh)와 ingest workload의 자원 경합을 막음. 무거운 쿼리가 ingest를 죽이지 않는다 — 이는 EKS dashboard burst 시점에 매우 중요.
- **vmstorage 수평 확장**: shared-nothing이라 노드 추가만으로 active series 용량 증가. 단점: 새 노드의 부하 균형이 자연 도달까지 시간 필요(역사적 데이터는 기존 노드에).

**Native multi-tenancy**: URL path 기반.
```
http://vminsert:8480/insert/<accountID>/prometheus/api/v1/write
http://vminsert:8480/insert/<accountID>:<projectID>/...
http://vmselect:8481/select/<accountID>/prometheus/api/v1/query
```
모든 테넌트 데이터는 같은 vmstorage 노드들에 spread되며, 성능은 active series 총량에만 의존(테넌트 수가 아님). Mimir와 비교하면 **OSS edition에 multi-tenancy가 포함**된다는 것이 큰 차이(Mimir의 multi-tenant feature도 OSS이지만 운영 복잡도가 훨씬 높다).

**Replication**: `-replicationFactor=N`을 vminsert/vmselect에 넘기면 vminsert가 N개 노드에 sample을 동시에 쓰고, vmselect는 dedupe. 단, **VM 공식 권장은 RF=1 + 클라우드 디스크 replication에 의존하기**다. 이유: RF=N은 자원 사용량을 N배로 늘리는 반면, GCE Persistent Disk·EBS·Azure Premium SSD는 이미 5-nine durability를 보장한다. Multi-AZ가 정말 필요하면 AZ별로 독립 클러스터를 운영하고 vmagent로 fan-out 복제하는 것이 권장 토폴로지(`victoria-metrics-distributed` Helm chart 참고).

### 3.4 카디널리티 폭발 처리

이 부분이 EKS 운영자에게 가장 실용적 차이가 난다.

- **Prometheus**: 모든 active series가 head index + WAL에 들어가 있어 RAM이 series 수에 거의 선형 비례. cardinality 폭발 시 OOM → 재시작 시 WAL replay하다 다시 OOM되는 crash loop가 잘 알려진 실패 모드. 한 사용자 보고(seifrajhi 블로그): "high load 시 Prometheus가 200GB RAM까지 쓰다 OOMKilled, 그동안 metric도 alert도 다 잃음".
- **VictoriaMetrics**: VM 자체 보고로 1M active series에 850MB RAM, 10M에 4GB 수준. inverted index가 디스크 mergeset이고 in-memory에는 캐시만 두기 때문. 또한 vmagent에 `cardinality_limit`, `series_limit_per_target`, **stream aggregation**(scrape time에 사전 집계)으로 *문제가 storage에 도달하기 전에* 차단할 수 있다. 운영 도구로 **vmui Cardinality Explorer**가 내장 — 메트릭 이름·라벨·라벨 값별 series 수 상위 N을 즉시 보여줌(Prometheus는 같은 작업이 `count by (__name__)({__name__=~".+"})` 같은 자체 무거운 쿼리). 단, HN에서 일부 코멘트는 "VM도 진짜 cardinality 문제가 무한정 풀리는 건 아니다, circuit breaker가 있다"고 지적 — 무제한이 아니라 *훨씬 더 멀리 간다*는 표현이 정확하다.

### 3.5 다운샘플링·장기 보관

- **Prometheus 자체**: 다운샘플링 **없음**. retention만 있음. 길게 보관하려면 외부로 remote_write(Thanos compactor, Mimir compactor가 5m·1h downsampling 지원).
- **VictoriaMetrics**: OSS는 **retention period 한 개**만. **Downsampling, retention per label/filter, multiple retentions는 모두 Enterprise 기능**(`-downsampling.period=30d:5m,180d:1h,1y:6h,2y:1d` 같은 형태). 일부 OSS 사용자는 vmagent의 stream aggregation으로 다운샘플 비슷한 효과를 얻을 수 있지만 1:1 대체는 아님. 이는 도입 전 **반드시 검토할** 라이센스 이슈다(GitHub issue #36에서 처음 요청된 이래 OSS화 거부 입장 유지). Percona PMM은 Enterprise 라이센스 없이 다운샘플링 통합한 사례.

---

## 4. 비용·성능 벤치마크 — 수치를 보되 출처를 명시한다

### 4.1 VictoriaMetrics 자체 발표(편향 가능성 있음)

**Prometheus vs VM, node_exporter 24h, 1000 targets**(Valialkin, 자체 벤치):
- Prometheus: 최대 ~23 GB RAM
- VictoriaMetrics: ~4.3 GB RAM (5.3x 차이)
- 디스크: VM이 ~2.5x 작음
- CPU: 비슷한 수준

**VM vs Grafana Mimir, 동일 하드웨어 24h**(VM 블로그 "Grafana Mimir and VictoriaMetrics: performance tests"; Mimir 팀의 사전 검토 거쳤다고 함):
- 워크로드: 360K samples/s, 5.5M initial series + 6K series/min churn, 24h
- Mimir(RF=3): avg 20/43 cores CPU, 120GiB/283GiB max RAM, p50 read 95ms, p99 35s
- VM(RF=2): avg 12/44 cores CPU, 19GiB/208GiB max RAM, p50 read 165ms, p99 17s
- 5x 부하 증가 시(1.8M samples/s): Mimir는 자원 한계로 alert firing, VM은 26 cores·69 GiB로 처리(p99 120s)

**100M samples/s 스케일 벤치**(VM 자체, OSMC 2022): 1B active series에서 100M samples/s 달성 — 이 수치는 자체 발표이며 reproduce 비용(7000 vCPU 수준)이 매우 높다.

### 4.2 제3자·사용자 보고

- **Roblox**(VM 공식 case study, Datanami 2023 인터뷰): 기존 Prometheus + InfluxDB siloed 환경에서 VM cluster + Grafana 단일화. "VM이 우리 규모에서 잘 동작했다", 정확한 수치는 비공개.
- **DFKI(독일 인공지능 연구센터)**: Prometheus → VM 전환 후 storage ~1/3, CPU/RAM 감소, "WAL replay 크래시 루프가 사라진 게 day-to-day의 가장 큰 변화".
- **DreamHost**(VM case study): RAM 80% 절감, 76M time series로 확장.
- **Grammarly**: Graphite 대비 10x 비용 절감(VM CTO 인용 사례).
- **OpenAI engineer LinkedIn 글에서 인용된 이전 직장 사례**: 76 vCPU/368 GB Prometheus → 16 vCPU/50 GB VM(약 5~7배 절감, 단일 사례).
- **Last9, Chronosphere, Onidel, Apprecode** 등 third-party 비교 글은 대체로 "VM이 자원 효율 우위, Mimir가 multi-tenancy 성숙도 우위"로 수렴.

### 4.3 주의해서 읽어야 할 점

자체 벤치는 거의 모두 VM·Mimir·Grafana **자기 회사가 발표**한 것이며, prometheus-benchmark 도구도 VM이 만든 것이다. **편향 가능성을 인정하고**, 실제 도입 전에는 본인 워크로드(특정 라벨 분포, 쿼리 패턴, scrape 주기)를 그대로 replay해 active series·p95 query latency·CPU·RAM·storage를 직접 재라는 권고가 모든 third-party 분석의 공통 결론이다(Apprecode가 정확히 이 점을 지적).

### 4.4 인프라 비용 사례(공개된 dev0ps.tech의 추정 — 출처 신뢰도는 중간)

| 규모 | Prometheus 3y TCO | VictoriaMetrics 3y TCO | 절감 |
|---|---|---|---|
| Small(1M series) | $27,900 | $6,876 | ~75% |
| Large(10M) | $208,800 | $54,720 | ~74% |
| Enterprise(100M+) | $720,000 | $180,000 | ~75% |

이 수치는 단일 블로그(dev0ps.tech)의 모델링이며 VM Enterprise·운영 인건비·실제 디스크 종류에 따라 크게 흔들릴 수 있다. **방향성**(VM가 매우 비용 효율적)으로만 받아들이고, 자기 환경 PoC가 필수.

---

## 5. 실제 사용자 리뷰 — 균형 있게

### 5.1 긍정적 후기

- **HN top comment**(item 39940057): "We migrated and replaced 6 prometheus servers with only 1 VictoriaMetrics server, it was crazy efficient."
- **HN(item 32779662)**: 단일 VM 인스턴스로 230B metrics, 4GB RAM, 200m CPU 운영. "Cortex/Thanos에 비해 운영이 정말 단순하다, 디스크 모니터링만 하면 됨".
- **Grafana community forums, Onidel 분석**: VM이 single binary deployment로 운영 복잡도 가장 낮다는 데 광범위한 합의.
- **Naver, NetEase, xiaohongshu** 등이 VM 공식 case study에 등재 — 한국에서는 **Naver**가 공개된 대표 케이스다(VM `casestudies` 페이지에 회사명 등재; 공개된 자세한 한국어 포스트는 발견되지 않음). 한국어 포스트 중에서는 개인 블로그·기술 미디엄(LinuxEA 등)의 도입기가 다수 있으며, "kube-prometheus-stack의 long-term remote storage를 VM single로 두는 패턴"이 가장 흔하게 보인다.

### 5.2 비판적·부정적 의견

이 부분이 보통 VM 마케팅에서는 잘 다뤄지지 않는다. **실제로 알려진 이슈**:

1. **WAL이 없다 = 크래시 시 1~5초 데이터 손실 가능**. damnever 블로그(중국 엔지니어, 매우 분석적)는 "WAL은 적어도 *옵션*이어야 했다, 신뢰성에서 9 하나가 빠지는 셈"이라고 비판. 공식 GitHub issue(#6606)에서도 VM 메인테이너가 직접 "vmagent 비정상 종료 시 데이터 손실 가능, WAL 대신 replication을 추천"이라고 답변.
2. **다운샘플링/multiple retentions/retention filter가 Enterprise**. Mimir, Thanos는 OSS에 포함이라 비용 비교 시 라이센스를 빼먹으면 잘못된 결론으로 갈 수 있다.
3. **rate/increase 의미가 Prometheus와 미묘하게 다르다**. dashboard·alert 마이그레이션 시 일부 그래프가 *조금* 다르게 보일 수 있다(공식적으로 의도된 차이지만 첫 도입 시 혼란).
4. **HN(item 37603421)**: "VictoriaMetrics가 cardinality 문제를 진짜 푼 게 아니다, circuit breaker로 막는 것"이라는 지적. 메인테이너도 동의하며 한계 존재 인정.
5. **Hacker News(item 45935666)**: "object storage 기반(Mimir/Thanos)은 일정 규모에서 'ingestion vs query' 둘 중 하나를 선택해야 하는 지점이 오는데, VM은 block storage 의존이라 다른 트레이드오프가 있다"는 댓글.
6. **VictoriaMetrics 마이그레이션 후 후회한 사례**: CECG(컨설팅) 블로그의 평가는 multi-tenant K8s 플랫폼에서 VM 대신 **Mimir를 선택**한 사례를 보여준다. 이유: "per-tenant alerting과 downsampling이 VM Enterprise에서만 제공되어 비용이 부담", "Mimir는 OSS에서 native multi-tenancy 풀세트 제공". 즉 *멀티테넌시·다운샘플링이 진짜 OSS로 필요한* 환경에서는 VM이 이상적인 선택이 아닐 수 있다.
7. **GitHub Issues에 자주 보이는 패턴**: vmctl Prometheus snapshot import에서 데이터가 안 들어가는 케이스(#7530), vmstorage LSM part 수가 매월 1일에 점프하는 안정성 이슈(#3069), assisted merge로 인한 ingestion slowdown 등 — 운영 가능한 수준이지만 "0-day 무사고"는 아니다.

### 5.3 양쪽 다 쓰는 패턴

EKS 멀티 클러스터 환경에서 가장 흔한 hybrid 패턴:

```
[EKS cluster A] kube-prometheus-stack
                  └─ remote_write ──┐
[EKS cluster B] kube-prometheus-stack
                  └─ remote_write ──┼─► VictoriaMetrics cluster (long-term, global view)
[EKS cluster C] kube-prometheus-stack                                   │
                  └─ remote_write ──┘                                   ▼
                                                                    Grafana
                                                                  (dual datasource:
                                                                   - cluster-local Prom
                                                                   - global VM)
```

이는 Prometheus의 alerting 책임은 leaf에 남기고(latency·신뢰성), VM은 dashboarding·long-term·global query view를 맡기는 구성이다. Apprecode와 다수 third-party 글이 추천하는 안전한 점진 전환 경로이기도 하다.

---

## 6. 어떤 경우에 무엇을 써야 하는가 — 운영자용 결정 가이드

### 6.1 결정 매트릭스

| 시나리오 | 추천 | 이유 |
|---|---|---|
| 단일 EKS 클러스터, retention < 30d, < 1M active series | **kube-prometheus-stack** | 도구·문서·생태계가 가장 두텁고, VM의 이점이 비용 차이로 의미 있게 드러나지 않는 규모 |
| 멀티 EKS 클러스터(3~10+), 통합 dashboard 필요 | **Prometheus(scrape) + VM single/cluster(remote_write)** | leaf alerting은 Prom, global query는 VM. 가장 운영 부담 낮은 hybrid |
| > 6개월 장기 보관, 1년+ 데이터 비교가 핵심 | **VM Enterprise**(downsampling) **또는 Thanos/Mimir** | OSS만으로는 VM이 raw retention만 가능. Mimir는 객체 스토리지로 무제한이지만 운영 복잡 |
| High-cardinality(K8s 라벨 폭발, queryid·trace_id 등) | **VM** | inverted index 분리 + cardinality explorer + stream aggregation |
| Native multi-tenancy(SaaS, internal platform) | **Mimir** if OSS-only로 per-tenant downsample/alert 필요, **VM** if URL-path 분리로 충분 | Mimir는 OSS에 limit/quota·shuffle sharding 다 포함 |
| 비용 최적화가 1순위 KPI | **VM** | 디스크 5~7x, RAM 5x 절감 사례 다수, 단일 바이너리 운영비 절감 |
| LGTM 스택 풀 도입(Mimir 이미 운영 중) | **Mimir 유지** 또는 점진 VM 전환 | Mimir도 PromQL 호환, 이미 운영 노하우가 있다면 굳이 옮길 필요 없음 |
| Edge/IoT, 저사양 머신, HDD 스토리지 | **VM** | "optimized for high-latency IO, low IOPS"가 명시 설계 목표 |
| 가장 단순한 long-term backend 한 가지만 | **VM single-node** | 단일 바이너리, 의존성 0, 1대 인스턴스가 medium-size Mimir/Thanos 클러스터를 대체 가능 |

### 6.2 LGTM 스택과의 호환성 (사용자 컨텍스트 직접 답변)

이미 Loki/Grafana/Tempo/Mimir 스택을 운영 중인 7년차 한국 플랫폼 엔지니어 관점에서:

- **VM은 Grafana Prometheus datasource로 그대로 붙는다**. URL만 `http://vmselect:8481/select/0/prometheus`로 바꾸면 끝. dashboard 99% 그대로 동작.
- **Mimir와 VM은 직접 경쟁하는 레이어**다(모두 long-term, multi-tenant, PromQL). Mimir를 이미 운영 중이라면 VM으로의 *전체 교체*는 운영 노하우 sunk cost를 버리는 의미가 있으니, 다음 두 가지만 정량적으로 비교하라:
  1. 같은 active series·query qps에 대한 **인프라 비용** (object storage I/O 비용 포함)
  2. **per-tenant alerting/downsampling 의존도** — 이게 강하면 Mimir 우위
- **Loki/Tempo는 VM 도입과 무관하다**. metrics 백엔드 교체이지 logs/traces 교체가 아니다. VictoriaLogs/VictoriaTraces가 별도로 있긴 하나 성숙도는 Loki/Tempo보다 낮다(2023~2025 단계).
- **vmagent는 Grafana Agent/Alloy의 metrics 부분을 부분적으로 대체** 가능. VM 자체 벤치(공식)에서 vmagent가 OTel Collector·Prometheus Agent 대비 CPU 1.6~3.2x, RAM 2.7~3x 절감 보고. 그러나 logs/traces가 필요하면 Alloy/OTel Collector를 빼기 어렵다.

### 6.3 권장 PoC 계획(EKS, 한 달)

```yaml
# 1주차: 기준선 측정
- 현재 Prometheus의 active series, ingestion rate, p95 query latency, 디스크/RAM 7일 베이스라인
- 가장 무거운 PromQL alert/recording rule, 가장 자주 보는 dashboard top 10 식별

# 2주차: VM single-node 사이드바이사이드
- VMSingle 또는 Helm chart `victoria-metrics-single` 1 replica, 동일 EKS에 배포
- Prometheus의 remote_write 추가:
  remote_write:
    - url: http://vmsingle.monitoring:8428/api/v1/write
      queue_config:
        max_samples_per_send: 10000
        capacity: 20000
- Grafana에 VM datasource 추가, 같은 dashboard를 두 datasource에 띄워 그래프 시각 비교

# 3주차: 부하·장애 시나리오
- vmstorage Pod kill 시나리오 (RF=2 권장)
- cardinality 폭발 시뮬레이션 (라벨에 timestamp 주입)
- vmagent로 scrape 일부 옮기고 stream_aggr로 사전 집계 적용

# 4주차: 의사결정
- 디스크/RAM/CPU 절감 실측치
- rate/increase 그래프 차이가 운영에 영향 있는지
- downsampling이 정말 필요한지(필요 시 Enterprise 또는 stream aggregation으로 해결 가능한지)
```

### 6.4 운영 시 알아둘 다이어그램·설정 스니펫

**EKS 멀티 클러스터에서 VM cluster 도입 토폴로지** (다이어그램 그려두면 좋음):

```
AZ-a / EKS cluster A          AZ-b / EKS cluster B
┌──────────────────┐          ┌──────────────────┐
│ Prom (alerts)    │          │ Prom (alerts)    │
│ vmagent          │          │ vmagent          │
└────────┬─────────┘          └────────┬─────────┘
         │ remote_write(zstd)          │
         ▼                             ▼
   ┌────────────────────────────────────────┐
   │  vmauth (LB / authN / multi-tenant 라우팅) │
   └─────────┬────────────────┬───────────────┘
        /insert/X            /select/X
             ▼                    ▼
        vminsert × N         vmselect × M
             │                    │
             └────► vmstorage StatefulSet × K (PVC, RF=2) ◄──┘
                          ▲
                          │
                    vmbackup → S3 (snapshot)
```

**Helm 핵심 값 예시(victoria-metrics-k8s-stack)**:
```yaml
vmcluster:
  enabled: true
  spec:
    replicationFactor: 2
    retentionPeriod: "12"  # 12 months (Enterprise면 다단 가능)
    vmstorage:
      replicaCount: 3
      storageDataPath: /vm-data
      storage:
        volumeClaimTemplate:
          spec:
            storageClassName: gp3
            resources: { requests: { storage: 500Gi } }
      resources:
        requests: { cpu: "2", memory: "8Gi" }
    vminsert:
      replicaCount: 3
      extraArgs:
        replicationFactor: "2"
    vmselect:
      replicaCount: 2
      cacheMountPath: /select-cache
      extraArgs:
        dedup.minScrapeInterval: 30s   # HA pair 중복 제거
```

**Prometheus의 remote_write 튜닝(VM 권장값)**:
```yaml
remote_write:
  - url: http://vminsert.monitoring:8480/insert/0/prometheus/api/v1/write
    queue_config:
      max_samples_per_send: 10000
      capacity: 20000
      max_shards: 30
    metadata_config:
      send: false   # VM은 메타데이터 활용 안함, 트래픽 절감
```

---

## 마무리 — 7년차 운영자에게 드리는 한 줄 결론

**Prometheus는 폐기하지 말고, 그 옆 long-term 백엔드를 VM으로 두는 것**이 멀티 클러스터 EKS·LGTM 스택을 이미 운영하는 한국 엔지니어 입장에서 가장 위험이 낮고 ROI가 높은 경로다. Mimir 운영 노하우가 이미 깊다면 Mimir를 유지하되 VM과 정량 비교 PoC를 4주만 돌려보면 의사결정에 충분한 데이터가 모인다. 만약 카디널리티 폭발이 *이미* 통증 포인트라면 VM 도입의 정당성은 바로 그 자리에서 입증된다. 단, **Enterprise 라이센스 라인**(downsampling, retention filter, multiple retentions)은 도입 전 반드시 확인하고, **WAL 부재로 인한 1~5초 손실 가능성**은 vmagent persistent queue + replication으로 보완하는 것이 표준 권장 구성임을 잊지 말 것.

---

### 출처 신뢰도 메모
- **공식 1차 자료**: prometheus.io, victoriametrics.com/docs, github.com/prometheus/prometheus, github.com/VictoriaMetrics/VictoriaMetrics, CNCF 공식 발표
- **창립자/메인테이너 직접 글**: Julius Volz(PromLabs), Brian Brazil(Robust Perception), Aliaksandr Valialkin(Medium @valyala), Roman Khavronenko(Medium), Ganesh Vernekar(Prometheus TSDB 시리즈)
- **컨퍼런스 발표**: PromCon 2016/2017/2019/2023, KubeCon, OSMC 2022
- **주요 엔지니어링 블로그**: Grafana Labs(Inside PromQL), Chronosphere, Last9, InfoQ(Flipkart), Datanami(Roblox)
- **벤치마크 출처는 모두 명시했고**, VM 자체 벤치는 "self-published, 편향 가능성 있음"으로 표기. Mimir vs VM 벤치는 Mimir 팀 사전 검토를 거쳤다는 점에서 third-party보다는 양쪽 합의 자료에 가까움.
- **사용자 리뷰**: HN, reddit(직접 검색 결과 부족), GitHub issues, dbdb.io(Carnegie Mellon DB 분류 — 학술 사이트), damnever 블로그(비판적 분석), CECG(Mimir 선택 사례)
- **한국어 자료**: VictoriaMetrics 공식 case studies에 Naver 등재 외, 깊이 있는 한국어 도입 후기 블로그는 검색 시점에 발견하지 못했음(개인 블로그·LinuxEA 중국어 번역본 정도). 사내 자료가 더 있을 가능성은 있으나 공개 검색에서는 부족함을 명시.