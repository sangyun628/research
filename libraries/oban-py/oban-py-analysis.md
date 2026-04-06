# Oban-py 심층 분석 리포트

> **분석 대상**: [oban-bg/oban-py](https://github.com/oban-bg/oban-py)
> **분석 날짜**: 2026-01-30
> **버전**: 최신 main 브랜치

## 1. 개요

### 1.1 Oban이란?

Oban-py는 **PostgreSQL 기반의 견고한 백그라운드 작업 처리 프레임워크**입니다. 원래 Elixir 생태계에서 시작된 Oban을 Python으로 포팅한 프로젝트로, 두 언어 버전이 **동일한 데이터베이스 스키마를 공유**하여 같은 시스템에서 함께 실행할 수 있습니다.

### 1.2 핵심 철학

```
신뢰성(Reliability) + 일관성(Consistency) + 관측성(Observability)
```

- **작업 데이터 보존**: 다른 백그라운드 작업 도구와 달리, 작업 완료 후에도 데이터를 유지하여 히스토리 메트릭과 검사가 가능
- **트랜잭션 제어**: 데이터베이스 변경과 함께 작업을 원자적으로 커밋/롤백
- **적은 의존성**: Redis 등 추가 인프라 없이 PostgreSQL만으로 운영

### 1.3 다른 도구와의 차별점

| 특징 | Oban-py | Celery | RQ | Dramatiq |
|------|---------|--------|----|---------|
| 브로커 | PostgreSQL | Redis/RabbitMQ | Redis | Redis/RabbitMQ |
| 트랜잭션 지원 | O | X | X | X |
| 작업 히스토리 | O (자동 보존) | X | X | X |
| Async Native | O | X | X | O |
| 데이터베이스 백업과 통합 | O | X | X | X |

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Application                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   @worker   │    │    @job     │    │  직접 호출   │             │
│  │  decorator  │    │  decorator  │    │  enqueue()  │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         Oban Core                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │  Query   │  │ Notifier │  │  Leader  │  │ Producer │    │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │   │
│  │       │             │             │             │           │   │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐    │   │
│  │  │  Stager  │  │Scheduler │  │  Pruner  │  │Refresher │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  │       │             │             │             │           │   │
│  │  ┌────┴─────┐  ┌────┴─────────────┴─────────────┴─────┐    │   │
│  │  │ Executor │  │               Lifeline               │    │   │
│  │  └──────────┘  └──────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PostgreSQL Database                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   oban_jobs     │  │  oban_leaders   │  │ oban_producers  │     │
│  │  (작업 테이블)   │  │  (리더 선출)     │  │  (큐 프로듀서)   │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              LISTEN/NOTIFY (실시간 알림)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 실행 모드

#### 서버 모드 (Server Mode)
- 큐가 설정된 경우 활성화
- 작업 처리 + 작업 추가 모두 가능
- 리더십 선거 자동 활성화

#### 클라이언트 모드 (Client Mode)
- 큐 설정 없이 실행
- 작업 추가만 가능 (enqueue only)
- 리더십 비활성화

```python
# 서버 모드
oban = Oban(pool=pool, queues={"default": 10, "mailers": 5})

# 클라이언트 모드
oban = Oban(pool=pool)  # 큐 없이
```

---

## 3. 핵심 컴포넌트 상세 분석

### 3.1 Job (작업)

**위치**: `oban/job.py`

Job은 실행될 작업의 모든 정보를 담는 데이터 클래스입니다.

```python
class Job:
    __slots__ = (
        "worker",        # 실행할 워커 클래스 경로
        "id",            # 데이터베이스 ID
        "state",         # 상태 (JobState enum)
        "queue",         # 큐 이름
        "attempt",       # 현재 시도 횟수
        "max_attempts",  # 최대 재시도 횟수
        "priority",      # 우선순위 (0-9, 낮을수록 높음)
        "args",          # 작업 인자
        "meta",          # 메타데이터
        "errors",        # 에러 히스토리
        "tags",          # 태그 리스트
        "scheduled_at",  # 예약 실행 시간
        # ... 타임스탬프들
    )
```

#### Job 상태 (JobState)

```
┌──────────┐     enqueue      ┌───────────┐
│ (생성)   │ ─────────────▶  │ available │
└──────────┘                  └─────┬─────┘
                                    │ fetch
                                    ▼
                              ┌───────────┐
                              │ executing │
                              └─────┬─────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
     ┌───────────┐           ┌───────────┐           ┌───────────┐
     │ completed │           │ retryable │           │ discarded │
     └───────────┘           └─────┬─────┘           └───────────┘
                                   │ 재시도
                                   ▼
                             ┌───────────┐
                             │ scheduled │
                             └───────────┘
```

- **available**: 즉시 실행 가능
- **scheduled**: 예약된 시간에 실행 대기
- **executing**: 현재 실행 중
- **completed**: 성공적으로 완료
- **retryable**: 실패 후 재시도 대기
- **discarded**: 최대 시도 횟수 초과로 폐기
- **cancelled**: 명시적으로 취소됨
- **suspended**: 일시 중지됨

#### 작업 결과 타입

```python
@dataclass(frozen=True, slots=True)
class Snooze:
    """작업을 지정 시간 후 다시 실행"""
    seconds: int

@dataclass(frozen=True, slots=True)
class Cancel:
    """작업을 취소하고 이유 기록"""
    reason: str

@dataclass(slots=True)
class Record:
    """결과값을 메타에 저장"""
    value: Any
    limit: int = 64_000_000  # 64MB
```

### 3.2 Worker & Decorators (워커 및 데코레이터)

**위치**: `oban/decorators.py`, `oban/_worker.py`

두 가지 방식으로 워커를 정의할 수 있습니다:

#### @worker 데코레이터 (클래스 기반)

```python
@worker(queue="exports", max_attempts=5, priority=1)
class ExportWorker:
    async def process(self, job: Job) -> Result[Any]:
        if job.cancelled():
            return Cancel("Export cancelled by user")

        report = await generate_report(job.args["report_id"])

        if report.status == "pending":
            return Snooze(seconds=30)  # 30초 후 재시도

        await send_to_user(job.args["email"], report)
        return Record({"file_size": report.size})

    def backoff(self, job: Job) -> int:
        """커스텀 재시도 백오프 (선택사항)"""
        return 2 * job.attempt
```

#### @job 데코레이터 (함수 기반)

```python
@job(queue="mailers", priority=1)
def send_email(to: str, subject: str, body: str):
    """함수 시그니처가 그대로 보존됨"""
    print(f"Sending to {to}: {subject}")
    return {"sent": True}

# 호출 시 함수 시그니처 그대로 사용
await send_email.enqueue("user@example.com", "Hello", "World")
```

#### 워커 등록 메커니즘

```python
# _worker.py
_registry: dict[str, type] = {}

def register_worker(cls) -> None:
    """워커를 전역 레지스트리에 등록"""
    key = f"{cls.__module__}.{cls.__qualname__}"
    _registry[key] = cls

def resolve_worker(path: str) -> type:
    """경로로 워커 클래스 해결"""
    if path in _registry:
        return _registry[path]
    # 동적 임포트 폴백
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    register_worker(cls)
    return cls
```

### 3.3 Producer (프로듀서)

**위치**: `oban/_producer.py`

각 큐마다 하나의 Producer가 생성되어 작업을 가져오고 실행합니다.

```python
class Producer:
    def __init__(self, *, queue: str, limit: int = 10, ...):
        self._queue = queue
        self._limit = limit          # 동시 실행 제한
        self._paused = False
        self._running_jobs = {}      # {job_id: (job, task)}
        self._pending_acks = []      # 완료 대기 작업들
        self._debounce_interval = 0.005  # 디바운스 간격
```

#### 작업 처리 루프

```python
async def _loop(self) -> None:
    while True:
        try:
            await asyncio.wait_for(self._notified.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        self._notified.clear()
        await self._debounce()
        await self._produce()

async def _produce(self) -> None:
    # 1. 완료된 작업 ACK
    await self._ack_jobs()

    # 2. 일시 정지 또는 용량 초과 시 스킵
    if self._paused or (self._limit - len(self._running_jobs)) <= 0:
        return

    # 3. 새 작업 가져오기
    jobs = await self._get_jobs()

    # 4. 비동기 실행
    for job in jobs:
        task = self._dispatcher.dispatch(self, job)
        task.add_done_callback(lambda _, job_id=job.id: self._on_job_complete(job_id))
        self._running_jobs[job.id] = (job, task)
```

### 3.4 Notifier (알리미)

**위치**: `oban/_notifier.py`

PostgreSQL의 LISTEN/NOTIFY를 활용한 실시간 Pub/Sub 시스템입니다.

```python
class PostgresNotifier:
    """PostgreSQL LISTEN/NOTIFY 기반 알리미"""

    def __init__(self, *, prefix: str = "public", beat_interval: float = 30.0):
        self._subscriptions = defaultdict(dict)  # {channel: {token: callback}}
        self._conn = None  # 전용 알림 연결
```

#### 채널 종류

- **insert**: 새 작업 삽입 시 알림
- **signal**: 큐 제어 신호 (pause, resume, scale, pkill)
- **leader**: 리더십 이벤트

#### 페이로드 압축

```python
def encode_payload(payload: dict) -> str:
    """딕트를 gzip + base64로 압축"""
    dumped = orjson.dumps(payload)
    zipped = gzip.compress(dumped)
    return base64.b64encode(zipped).decode("ascii")

def decode_payload(payload: str) -> dict:
    """압축 해제"""
    if payload.startswith("{"):  # SQL에서 직접 생성된 경우
        return orjson.loads(payload)
    decoded = base64.b64decode(payload)
    unzipped = gzip.decompress(decoded)
    return orjson.loads(unzipped.decode("utf-8"))
```

### 3.5 Leader (리더)

**위치**: `oban/_leader.py`

분산 환경에서 단일 노드만 특정 작업을 수행하도록 리더 선출을 관리합니다.

```python
class Leader:
    async def _election(self) -> None:
        """리더십 선거 시도"""
        self._is_leader = await self._query.attempt_leadership(
            self._name,          # 인스턴스 이름
            self._node,          # 노드 호스트명
            int(self._interval), # TTL (초)
            self._is_leader      # 현재 리더 여부
        )
```

#### 리더 선출 SQL

```sql
-- elect_leader.sql (새 리더 선출)
INSERT INTO oban_leaders (name, node, elected_at, expires_at)
VALUES (%(name)s, %(node)s, timezone('UTC', now()),
        timezone('UTC', now()) + interval '1 second' * %(ttl)s)
ON CONFLICT (name) DO NOTHING
RETURNING node;

-- reelect_leader.sql (기존 리더 갱신)
UPDATE oban_leaders
SET expires_at = timezone('UTC', now()) + interval '1 second' * %(ttl)s
WHERE name = %(name)s AND node = %(node)s
RETURNING node;
```

#### 리더 전용 작업

- **Pruner**: 오래된 완료 작업 정리
- **Lifeline**: 고아 작업 구출
- **Refresher Cleanup**: 만료된 프로듀서 정리
- **Scheduler**: CRON 작업 실행

### 3.6 Stager (스테이저)

**위치**: `oban/_stager.py`

예약된 작업(`scheduled`, `retryable`)을 실행 가능 상태(`available`)로 전환합니다.

```python
class Stager:
    async def _stage(self) -> None:
        queues = list(self._producers.keys())
        (staged, active) = await self._query.stage_jobs(self._limit, queues)

        # 활성화된 큐의 프로듀서에게 알림
        for queue in active:
            self._producers[queue].notify()
```

#### 스테이징 SQL

```sql
-- stage_jobs.sql
WITH locked_jobs AS (
  SELECT id
  FROM oban_jobs
  WHERE state = ANY('{scheduled,retryable}')
    AND scheduled_at <= coalesce(%(before)s, timezone('UTC', now()))
  ORDER BY scheduled_at ASC, id ASC
  LIMIT %(limit)s
  FOR UPDATE SKIP LOCKED  -- 동시성 안전
),
updated_jobs AS (
  UPDATE oban_jobs
  SET state = 'available'::oban_job_state
  FROM locked_jobs
  WHERE oban_jobs.id = locked_jobs.id
)
SELECT DISTINCT q.queue
FROM unnest(%(queues)s::text[]) AS q(queue)
WHERE EXISTS (
  SELECT 1 FROM oban_jobs
  WHERE state = 'available' AND queue = q.queue
)
```

### 3.7 Executor (실행기)

**위치**: `oban/_executor.py`

작업을 실제로 실행하고 결과를 처리합니다.

```python
class Executor:
    async def execute(self) -> Executor:
        self._report_started()      # 텔레메트리: 시작
        await self._process()       # 실제 실행
        self._record_stopped()      # 결과 기록
        self._report_stopped()      # 텔레메트리: 종료
        self._reraise_unsafe()      # 안전 모드가 아니면 예외 재발생
        return self

    async def _process(self) -> None:
        token = _current_job.set(self.job)  # 컨텍스트 변수 설정
        try:
            self.worker = resolve_worker(self.job.worker)()
            self.result = await self.worker.process(self.job)
        except Exception as error:
            self.result = error
            self._traceback = traceback.format_exc()
        finally:
            _current_job.reset(token)
```

#### 결과 처리

```python
def _record_stopped(self) -> None:
    match self.result:
        case Exception() as error:
            if self.job.attempt >= self.job.max_attempts:
                self.action = AckAction(job=self.job, state="discarded", error=...)
            else:
                self.action = AckAction(
                    job=self.job,
                    state="retryable",
                    schedule_in=self._retry_backoff()
                )

        case Cancel(reason=reason):
            self.action = AckAction(job=self.job, state="cancelled", error=...)

        case Snooze(seconds=seconds):
            self.action = AckAction(
                job=self.job,
                attempt_change=-1,  # 시도 횟수 유지
                state="scheduled",
                schedule_in=seconds
            )

        case Record(encoded=encoded):
            self.action = AckAction(
                job=self.job,
                state="completed",
                meta={"recorded": True, "return": encoded}
            )

        case _:  # 정상 완료
            self.action = AckAction(job=self.job, state="completed")
```

### 3.8 Scheduler (스케줄러)

**위치**: `oban/_scheduler.py`

CRON 표현식을 파싱하고 주기적 작업을 관리합니다.

```python
@dataclass(slots=True, frozen=True)
class Expression:
    input: str
    minutes: set
    hours: set
    days: set
    months: set
    weekdays: set

    @classmethod
    def parse(cls, expression: str) -> Expression:
        """CRON 표현식 파싱"""
        # 닉네임 지원: @daily, @hourly, @weekly 등
        normalized = NICKNAMES.get(expression, expression)
        # 5개 필드 파싱: minute hour day month weekday
        ...
```

#### 지원 CRON 닉네임

```python
NICKNAMES = {
    "@annually": "0 0 1 1 *",
    "@yearly": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@midnight": "0 0 * * *",
    "@daily": "0 0 * * *",
    "@hourly": "0 * * * *",
}
```

### 3.9 Pruner (정리기)

**위치**: `oban/_pruner.py`

완료/취소/폐기된 오래된 작업을 주기적으로 삭제합니다.

```python
class Pruner:
    def __init__(self, *, max_age: int = 86_400, interval: float = 60.0, limit: int = 20_000):
        self._max_age = max_age     # 보존 기간 (초, 기본 1일)
        self._interval = interval   # 실행 주기
        self._limit = limit         # 배치 삭제 제한
```

### 3.10 Lifeline (라이프라인)

**위치**: `oban/_lifeline.py`

비정상 종료로 고아가 된 `executing` 상태의 작업을 구출합니다.

```python
class Lifeline:
    def __init__(self, *, rescue_after: float = 300.0):
        self._rescue_after = rescue_after  # 구출 대기 시간 (초)

    async def _rescue(self) -> None:
        if not self._leader.is_leader:
            return
        await self._query.rescue_jobs(self._rescue_after)
```

### 3.11 Refresher (리프레셔)

**위치**: `oban/_refresher.py`

프로듀서 상태를 주기적으로 갱신하고 만료된 프로듀서를 정리합니다.

```python
class Refresher:
    async def _refresh(self) -> None:
        """프로듀서 하트비트 갱신"""
        uuids = [p._uuid for p in self._producers.values()]
        await self._query.refresh_producers(uuids)

    async def _cleanup(self) -> None:
        """리더만: 만료된 프로듀서 정리"""
        if self._leader.is_leader:
            await self._query.cleanup_expired_producers(self._max_age)
```

---

## 4. 데이터베이스 스키마

### 4.1 oban_jobs 테이블

```sql
CREATE TABLE IF NOT EXISTS oban_jobs (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    state oban_job_state NOT NULL DEFAULT 'available',
    queue text NOT NULL DEFAULT 'default',
    worker text NOT NULL,
    attempt smallint NOT NULL DEFAULT 0,
    max_attempts smallint NOT NULL DEFAULT 20,
    priority smallint NOT NULL DEFAULT 0,
    args jsonb NOT NULL DEFAULT '{}',
    meta jsonb NOT NULL DEFAULT '{}',
    tags jsonb NOT NULL DEFAULT '[]',
    errors jsonb NOT NULL DEFAULT '[]',
    attempted_by text[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    inserted_at timestamp WITHOUT TIME ZONE NOT NULL,
    scheduled_at timestamp WITHOUT TIME ZONE NOT NULL,
    attempted_at timestamp WITHOUT TIME ZONE,
    cancelled_at timestamp WITHOUT TIME ZONE,
    completed_at timestamp WITHOUT TIME ZONE,
    discarded_at timestamp WITHOUT TIME ZONE,

    CONSTRAINT attempt_range CHECK (attempt >= 0 AND attempt <= max_attempts),
    CONSTRAINT positive_max_attempts CHECK (max_attempts > 0),
    CONSTRAINT non_negative_priority CHECK (priority >= 0)
);
```

### 4.2 인덱스 전략

```sql
-- 작업 페칭용 복합 인덱스 (핵심)
CREATE INDEX oban_jobs_state_queue_priority_scheduled_at_id_index
ON oban_jobs (state, queue, priority, scheduled_at, id)
WITH (fillfactor = 90);

-- 스테이징용 부분 인덱스
CREATE INDEX oban_jobs_staging_index
ON oban_jobs (scheduled_at, id)
WHERE state IN ('scheduled', 'retryable');

-- 정리용 부분 인덱스
CREATE INDEX oban_jobs_completed_at_index ON oban_jobs (completed_at)
WHERE state = 'completed';
```

### 4.3 Autovacuum 최적화

```sql
ALTER TABLE oban_jobs SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 50,
  autovacuum_analyze_scale_factor = 0.02,
  autovacuum_analyze_threshold = 100,
  autovacuum_vacuum_cost_limit = 2000,
  autovacuum_vacuum_cost_delay = 1,
  autovacuum_vacuum_insert_scale_factor = 0.02,
  autovacuum_vacuum_insert_threshold = 1000,
  fillfactor = 85
);
```

### 4.4 oban_leaders 테이블 (UNLOGGED)

```sql
CREATE UNLOGGED TABLE IF NOT EXISTS oban_leaders (
    name text PRIMARY KEY DEFAULT 'oban',
    node text NOT NULL,
    elected_at timestamp WITHOUT TIME ZONE NOT NULL,
    expires_at timestamp WITHOUT TIME ZONE NOT NULL
);
```

`UNLOGGED` 테이블은 WAL 로깅을 하지 않아 성능이 좋지만 크래시 시 데이터 손실 가능. 리더십은 재선거로 복구되므로 적합.

### 4.5 oban_producers 테이블 (UNLOGGED)

```sql
CREATE UNLOGGED TABLE IF NOT EXISTS oban_producers (
    uuid uuid PRIMARY KEY,
    name text NOT NULL DEFAULT 'oban',
    node text NOT NULL,
    queue text NOT NULL,
    meta jsonb NOT NULL DEFAULT '{}',
    started_at timestamp WITHOUT TIME ZONE NOT NULL,
    updated_at timestamp WITHOUT TIME ZONE NOT NULL
);
```

---

## 5. 디자인 패턴 분석

### 5.1 데코레이터 패턴 (Decorator Pattern)

워커 클래스에 메타프로그래밍으로 기능을 추가합니다.

```python
def worker(*, oban: str = "oban", cron: str | None = None, **overrides):
    def decorate(cls: type) -> type:
        # 1. new() 클래스 메서드 추가
        @classmethod
        def new(cls, args=None, /, **params):
            return Job(worker_name(cls), args=args or {}, **{**cls._opts, **params})

        # 2. enqueue() 클래스 메서드 추가
        @classmethod
        async def enqueue(cls, args=None, /, **overrides):
            job = cls.new(args, **overrides)
            return await Oban.get_instance(cls._oban_name).enqueue(job)

        setattr(cls, "new", new)
        setattr(cls, "enqueue", enqueue)
        register_worker(cls)

        return cls
    return decorate
```

### 5.2 싱글톤/레지스트리 패턴 (Registry Pattern)

전역 인스턴스 관리로 어디서나 접근 가능합니다.

```python
_instances: dict[str, Oban] = {}

class Oban:
    def __init__(self, *, name: str = "oban", ...):
        self._name = name
        _instances[self._name] = self  # 자동 등록

    @classmethod
    def get_instance(cls, name: str = "oban") -> Oban:
        if name not in _instances:
            raise RuntimeError(f"Oban instance '{name}' not found")
        return _instances[name]
```

### 5.3 옵저버/Pub-Sub 패턴 (Observer Pattern)

PostgreSQL NOTIFY를 통한 이벤트 기반 아키텍처입니다.

```python
# 발행
await self._notifier.notify("insert", {"queue": "default"})

# 구독
token = await self._notifier.listen("insert", self._on_notification)

# 콜백
async def _on_notification(self, channel: str, payload: dict) -> None:
    queue = payload["queue"]
    if queue in self._producers:
        self._producers[queue].notify()  # 프로듀서에게 알림
```

### 5.4 전략 패턴 (Strategy Pattern)

백오프 전략을 교체 가능하게 설계했습니다.

```python
# 기본 백오프
def jittery_clamped(attempt: int, max_attempts: int) -> int:
    clamped = round(attempt / max_attempts * 20)
    time = 15 + pow(2, min(clamped, 100))
    return jitter(time, mode="inc")

# 커스텀 백오프 (워커에서 오버라이드)
@worker()
class MyWorker:
    def backoff(self, job: Job) -> int:
        return 2 * job.attempt  # 선형 백오프
```

### 5.5 확장 포인트 패턴 (Extension Points)

Pro 버전 확장을 위한 훅 시스템입니다.

```python
# _extensions.py
_extensions: dict[str, Callable] = {}

def use_ext(name: str, default: Callable, *args, **kwargs) -> Any:
    func = _extensions.get(name, default)
    return func(*args, **kwargs)

# 사용 예시
async def _get_jobs(producer):
    demand = producer._limit - len(producer._running_jobs)
    return await producer._query.fetch_jobs(demand=demand, ...)

# 호출 시
jobs = await use_ext("producer.get_jobs", _get_jobs, self)
```

### 5.6 컨텍스트 매니저 패턴

자원 관리를 위한 비동기 컨텍스트 매니저 활용입니다.

```python
async with Oban(pool=pool, queues={"default": 10}) as oban:
    await oban.enqueue(MyWorker.new({"id": 1}))
    # ... 애플리케이션 실행 ...
# 자동으로 stop() 호출

# 내부 구현
async def __aenter__(self) -> Oban:
    return await self.start()

async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
    await self.stop()
```

### 5.7 명령 패턴 (Command Pattern)

작업을 명령 객체로 캡슐화합니다.

```python
@dataclass(frozen=True, slots=True)
class AckAction:
    """실행 결과를 나타내는 명령 객체"""
    job: Job
    state: str
    attempt_change: int | None = None
    error: dict | None = None
    meta: dict | None = None
    schedule_in: int | None = None
```

---

## 6. 동시성 및 안전성

### 6.1 작업 페칭의 동시성 안전

```sql
-- fetch_jobs.sql: FOR UPDATE SKIP LOCKED 사용
WITH locked_jobs AS (
  SELECT priority, scheduled_at, id
  FROM oban_jobs
  WHERE state = 'available' AND queue = %(queue)s
  ORDER BY priority ASC, scheduled_at ASC, id ASC
  LIMIT %(demand)s
  FOR UPDATE SKIP LOCKED  -- 잠긴 행은 건너뜀
)
UPDATE oban_jobs oj
SET attempt = oj.attempt + 1,
    attempted_at = timezone('UTC', now()),
    state = 'executing'
FROM locked_jobs
WHERE oj.id = locked_jobs.id
RETURNING ...
```

`FOR UPDATE SKIP LOCKED`는:
- 이미 잠긴 행은 건너뛰어 다른 노드와 충돌 방지
- 대기 없이 즉시 사용 가능한 작업만 가져옴
- 분산 환경에서 안전한 작업 분배

### 6.2 리더십 선거의 원자성

```sql
-- 새 리더 선출 시도 (원자적)
INSERT INTO oban_leaders (name, node, elected_at, expires_at)
VALUES (%(name)s, %(node)s, ...)
ON CONFLICT (name) DO NOTHING  -- 이미 리더가 있으면 무시
RETURNING node;
```

### 6.3 작업 취소의 안전성

```python
async def cancel_job(self, job: Job | int) -> None:
    # 1. 데이터베이스에서 취소 상태로 업데이트
    count, executing_ids = await self._query.cancel_many_jobs([job_id])

    # 2. 실행 중인 작업에게 취소 신호 전송
    if executing_ids:
        payloads = [{"action": "pkill", "job_id": id} for id in executing_ids]
        await self._notifier.notify("signal", payloads)

# 워커에서 취소 확인
async def process(self, job):
    for item in large_dataset:
        if job.cancelled():  # 안전 지점에서 확인
            return Cancel("Job was cancelled")
        await process_item(item)
```

---

## 7. 텔레메트리 시스템

### 7.1 이벤트 기반 계측

```python
# telemetry/core.py
def attach(id: str, events: List[str], handler: Handler) -> None:
    """핸들러 등록"""
    for name in events:
        _handlers[name].append((id, handler))

def execute(name: str, metadata: Metadata) -> None:
    """이벤트 발행"""
    for handler in _handlers.get(name, []):
        handler(name, metadata.copy())

@contextmanager
def span(prefix: str, start_metadata: Metadata):
    """시작/종료/예외 이벤트 자동 발행"""
    start_time = time.monotonic_ns()
    execute(f"{prefix}.start", {"system_time": start_time, **start_metadata})

    try:
        yield collector
        execute(f"{prefix}.stop", {"duration": end - start, ...})
    except Exception:
        execute(f"{prefix}.exception", {"error_message": ..., ...})
        raise
```

### 7.2 주요 이벤트

| 이벤트 | 설명 |
|--------|------|
| `oban.job.start` | 작업 실행 시작 |
| `oban.job.stop` | 작업 정상 완료 |
| `oban.job.exception` | 작업 예외 발생 |
| `oban.producer.get` | 작업 페칭 |
| `oban.producer.ack` | 작업 ACK |
| `oban.stager.stage` | 작업 스테이징 |
| `oban.leader.election` | 리더 선거 |
| `oban.pruner.prune` | 작업 정리 |
| `oban.lifeline.rescue` | 고아 작업 구출 |

### 7.3 로깅 통합

```python
# telemetry/logger.py
def attach() -> None:
    """기본 로거 핸들러 등록"""
    telemetry.attach(
        "oban-logger",
        ["oban.job.start", "oban.job.stop", "oban.job.exception"],
        _log_job_event
    )
```

---

## 8. CLI 도구

### 8.1 명령어 구조

```bash
# 스키마 설치
oban install --dsn postgresql://localhost/mydb

# 스키마 제거
oban uninstall --dsn postgresql://localhost/mydb

# 워커 프로세스 시작
oban start --dsn postgresql://localhost/mydb --queues default:10,mailers:5

# 버전 확인
oban version
```

### 8.2 설정 우선순위

1. CLI 인자 (최우선)
2. 환경 변수 (`OBAN_DSN`, `OBAN_QUEUES` 등)
3. `oban.toml` 설정 파일 (최하위)

```python
@classmethod
def load(cls, path: str | None = None, **overrides: Any) -> Config:
    tml_conf = cls.from_toml(path)
    env_conf = cls.from_env()
    cli_conf = cls(**overrides)
    return tml_conf.merge(env_conf).merge(cli_conf)
```

### 8.3 CRON 워커 자동 발견

```bash
# 자동 발견 (현재 디렉토리)
oban start --dsn ...

# 특정 모듈 지정
oban start --cron-modules myapp.workers,myapp.jobs

# 특정 경로 검색
oban start --cron-paths myapp/workers,myapp/jobs
```

---

## 9. 테스팅 지원

### 9.1 테스트 모드

```python
import oban.testing

# 인라인 모드: 작업 즉시 실행
with oban.testing.mode("inline"):
    await EmailWorker.enqueue({"to": "user@example.com"})
    # 바로 실행됨
```

### 9.2 테스트 헬퍼

```python
from oban.testing import assert_enqueued, refute_enqueued, drain_queue, process_job

# 작업 추가 확인
await assert_enqueued(worker=EmailWorker, args={"to": "user@example.com"})

# 작업 없음 확인
await refute_enqueued(worker=EmailWorker)

# 큐 드레인 (동기 실행)
result = await drain_queue(queue="default")
# {'completed': 5, 'discarded': 1, 'cancelled': 0, ...}

# 단위 테스트용 직접 실행
job = EmailWorker.new({"to": "test@example.com"})
result = await process_job(job)
```

### 9.3 데이터 초기화

```python
from oban.testing import reset_oban

@pytest.fixture(autouse=True)
async def clean_oban():
    yield
    await reset_oban()  # 테스트 후 테이블 초기화
```

---

## 10. 다른 프로젝트에 적용할 수 있는 패턴

### 10.1 PostgreSQL을 메시지 브로커로 활용

**장점**:
- 추가 인프라 불필요
- 트랜잭션과 함께 작업 추가 가능
- 단일 백업 지점

**적용 시 고려사항**:
```sql
-- 효율적인 인덱스 설계
CREATE INDEX idx_queue_priority ON jobs (state, queue, priority, scheduled_at, id)
WITH (fillfactor = 90);

-- FOR UPDATE SKIP LOCKED로 경합 방지
SELECT * FROM jobs WHERE state = 'available'
FOR UPDATE SKIP LOCKED LIMIT 10;

-- LISTEN/NOTIFY로 폴링 최소화
LISTEN job_inserted;
```

### 10.2 리더 선출 패턴

**구현 핵심**:
```python
class Leader:
    async def attempt_leadership(self):
        # INSERT ... ON CONFLICT DO NOTHING + TTL
        # 만료된 리더는 자동 정리
        # 절반 간격으로 갱신하여 유지
```

**사용 사례**:
- 분산 CRON 작업
- 싱글톤 백그라운드 태스크
- 데이터 정리/집계 작업

### 10.3 확장 포인트 시스템

**패턴**:
```python
_extensions = {}

def use_ext(name: str, default: Callable, *args, **kwargs):
    func = _extensions.get(name, default)
    return func(*args, **kwargs)

# Pro/Enterprise 버전에서 오버라이드
_extensions["producer.get_jobs"] = pro_get_jobs
```

### 10.4 작업 결과 타입 시스템

**Algebraic Data Types 활용**:
```python
@dataclass
class Snooze:
    seconds: int

@dataclass
class Cancel:
    reason: str

@dataclass
class Record:
    value: Any

# 패턴 매칭으로 처리
match result:
    case Snooze(seconds=s): reschedule(s)
    case Cancel(reason=r): mark_cancelled(r)
    case Record(value=v): store_result(v)
    case _: mark_completed()
```

### 10.5 텔레메트리 설계

**이벤트 기반 계측**:
```python
with telemetry.span("operation.name", {"context": "data"}) as ctx:
    result = await do_work()
    ctx.add({"result_size": len(result)})
# 자동으로 start/stop/exception 이벤트 발생
```

---

## 11. 결론

### 11.1 강점

1. **PostgreSQL 단일 의존성**: Redis 없이 운영 가능, 인프라 단순화
2. **완전한 Async Native**: asyncio 기반으로 현대적인 Python 생태계와 호환
3. **견고한 동시성 처리**: `FOR UPDATE SKIP LOCKED`로 경합 조건 방지
4. **훌륭한 관측성**: 텔레메트리, 작업 히스토리, 메트릭 내장
5. **Elixir 버전과 호환**: 폴리글랏 환경 지원
6. **확장성**: Pro 버전을 위한 깔끔한 확장 포인트

### 11.2 학습 포인트

- PostgreSQL의 고급 기능 활용 (LISTEN/NOTIFY, SKIP LOCKED, UNLOGGED)
- asyncio 기반 복잡한 시스템 설계
- 분산 시스템의 리더 선출 구현
- 데코레이터를 활용한 메타프로그래밍
- 테스트 용이성을 고려한 설계

### 11.3 적용 가능한 프로젝트

- 이메일/알림 발송 시스템
- 보고서 생성 파이프라인
- 데이터 처리/ETL 작업
- 주기적 정리/집계 작업
- 분산 태스크 스케줄링

---

## 참고 자료

- **GitHub**: [oban-bg/oban-py](https://github.com/oban-bg/oban-py)
- **공식 문서**: [oban.pro/docs/py](https://oban.pro/docs/py)
- **Elixir 버전**: [oban-bg/oban](https://github.com/oban-bg/oban)
- **Hacker News 토론**: [Oban for Python](https://news.ycombinator.com/item?id=46797594)
