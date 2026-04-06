# Claude Code 소스코드 분석 — 외부 리포트에 없는 핵심 발견사항

> 공식 문서 기반 외부 분석에서는 다루지 않았지만, 실제 소스코드에서 확인된 중요 설계 요소 10가지

---

## 1. 시스템 리마인더 주입 패턴

외부 리포트는 시스템 프롬프트를 "대화 시작 시 한 번 조립"한다고 설명한다. 실제로는 **대화 중간에 반복적으로 동적 정보를 주입**한다.

```
// print.ts에서 확인된 실제 패턴
도구 결과나 사용자 메시지에 <system-reminder> 태그를 삽입

<system-reminder>
사용 가능한 도구: Read, Edit, Bash...
현재 권한 모드: auto-accept
</system-reminder>
```

**왜 중요한가:** 긴 대화에서 LLM은 초반 시스템 프롬프트를 점점 "잊어버린다". 시스템 리마인더는 이걸 방지하는 핵심 기법으로, 자체 에이전트를 만들 때 가장 먼저 차용해야 할 패턴이다.

---

## 2. 도구 지연 로드 (Deferred Tools)

외부 리포트는 "사용 가능한 도구 목록"을 시스템 프롬프트에 넣는다고만 설명한다. 실제로는 **2단계 로딩**이다.

```
즉시 로드: Read, Edit, Bash, Grep, Glob, Write (스키마 전체 노출)
지연 로드: 나머지 도구들 (이름만 노출, ToolSearch로 스키마 온디맨드 로드)
```

시스템 프롬프트에 도구 스키마를 전부 넣으면 토큰을 수천 개 소모한다. 지연 로드로 **~50% 토큰 절약** + 불필요한 도구 호출 방지 효과를 얻는다.

---

## 3. 텔레메트리/분석 스택

외부 리포트는 "파일·셸·인증 정보가 머신 밖으로 나가지 않는다"고 말한다. 실제로는 **3개 분석 시스템이 외부로 데이터를 전송**한다.

| 시스템 | 용도 | 증거 |
|--------|------|------|
| **Datadog** | 운영 모니터링 | `shutdownDatadog()`, 23개 파일 137건 호출 |
| **1st Party Logger** | 사용자 행동 분석 | `shutdown1PEventLogging()` |
| **GrowthBook** | 피처 플래그/A·B 테스트 | 폴링 간격·브릿지 설정 원격 제어 |

파일 내용이나 코드를 전송하는 건 아니지만, 세션 이벤트·도구 사용 패턴·에러율·인증 흐름 등은 외부로 나간다.

전송되는 이벤트 예시:

```
tengu_update_check           — 업데이트 체크 시
tengu_oauth_flow_start       — OAuth 로그인 시작
tengu_oauth_success          — 로그인 성공
tengu_bridge_session_done    — 세션 종료 (상태, 소요시간)
tengu_bridge_reconnected     — 재연결 (끊김 시간)
tengu_bridge_token_refreshed — 토큰 갱신
tengu_bridge_message_received — 메시지 수신
```

`tengu_` 접두사로 미루어 내부 프로젝트 코드네임은 **"Tengu(천구)"**로 추정된다.

---

## 4. 멀티 트랜스포트 아키텍처

외부 리포트는 "로컬 터미널 프로세스에서만 돌아간다"고 한다. 실제로는 **3가지 트랜스포트를 전략 패턴으로 선택**하는 분산 통신 구조다.

```
우선순위: SSE+POST (CCR v2) > Hybrid (WS읽기+POST쓰기) > WebSocket (양방향)

각 트랜스포트 공통 기능:
├─ 지수 백오프 재연결 (최대 10분)
├─ 수면 감지 (60초 갭 → 예산 초기화)
├─ 메시지 버퍼링 + 재전송
└─ Keep-alive (프록시 타임아웃 방지)
```

로컬 CLI뿐 아니라 IDE 확장, 웹앱, 데스크톱 앱에서도 동일 엔진이 동작하기 위한 설계다.

---

## 5. 브릿지 시스템 (원격 실행)

유출본에서 **가장 큰 모듈** (33개 파일)인데 외부 리포트에는 아예 없다.

```
3가지 브릿지 구현:
├─ Environment Bridge: 멀티세션 (최대 32+), 폴링 기반 작업 수신
├─ REPL Bridge: 직접 WebSocket 세션
└─ Envless Bridge: 경량 OAuth→JWT

핵심 기능:
├─ QR 코드로 원격 세션 연결
├─ Git worktree로 세션 간 파일 격리
├─ JWT 만료 5분 전 사전 갱신 (세대 카운터로 오래된 비동기 작업 무효화)
├─ 세션별 타임아웃 + 30초 유예 → SIGKILL
└─ 용량 도달 시 heartbeat 루프로 전환
```

---

## 6. 메시지 중복 제거 메커니즘

안정적 에이전트 운영에 필수적인 패턴이지만 외부 리포트에 언급이 없다.

```typescript
// BoundedUUIDSet: 고정 메모리로 무한 중복 제거
// WebSocket 재연결, SSE 히스토리 재전송 시 같은 메시지가 반복 수신됨
// 이걸 막지 않으면 도구가 중복 실행되거나 응답이 이중 표시됨

두 가지 중복 제거 레이어:
1. recentPostedUUIDs → 자신이 보낸 메시지의 에코 차단
2. recentInboundUUIDs → 서버 히스토리 재전송 중복 차단
```

구현 패턴:

```typescript
class BoundedUUIDSet {
  private _set = new Set<string>()
  private _queue: string[] = []

  add(uuid: string): void {
    if (this._set.has(uuid)) return
    this._set.add(uuid)
    this._queue.push(uuid)
    if (this._queue.length > MAX_SIZE) {
      const old = this._queue.shift()!
      this._set.delete(old)
    }
  }

  has(uuid: string): boolean {
    return this._set.has(uuid)
  }
}
```

---

## 7. 피처 플래그 시스템

Claude Code의 기능 진화 방식을 이해하는 데 핵심이지만 외부 리포트에 없다.

```
25+ 컴파일 타임 플래그:
KAIROS           — 어시스턴트/데일리 로그 모드
PROACTIVE        — 자율 선행 실행
VOICE_MODE       — 음성 입력
COORDINATOR_MODE — 멀티에이전트 스웜
BUDDY            — 컴패니언 스프라이트
WEB_BROWSER_TOOL — 인프로세스 브라우저
CHICAGO_MCP      — 컴퓨터 사용 (화면 제어)
ULTRAPLAN        — 울트라 계획 모드
EXTRACT_MEMORIES — 백그라운드 메모리 자동 추출
TEAMMEM          — 팀 공유 메모리
...
```

비활성 플래그 이름만으로도 Anthropic이 준비 중인 기능을 알 수 있다. `COORDINATOR_MODE`(멀티에이전트 스웜)이나 `CHICAGO_MCP`(컴퓨터 사용)은 아직 공개되지 않은 기능들이다.

---

## 8. 권한 결정의 레이스 조건 패턴

외부 리포트는 "allow/ask/deny 중 하나 결정"이라고 단순화한다. 실제 구현은 **두 소스를 병렬 실행해서 먼저 결정된 쪽이 승리**하는 레이스 패턴이다.

```
도구 호출 → hasPermissionsToUseTool() → 'requires_action'
    ├─ 훅 실행 (settings.json에 설정된 셸 스크립트)  ─┐
    └─ SDK 권한 프롬프트 (사용자에게 질문)           ─┤
                                                     ├─ 먼저 결정된 쪽이 승리
                                                     └─ 나머지는 AbortController로 취소
```

**왜 중요한가:** 팀/조직에서 에이전트를 쓸 때, **사전 정의된 정책(훅)**과 **사용자 판단**을 동시에 실행하여 지연을 최소화하면서도 안전성을 확보하는 기법이다.

---

## 9. SerialBatchEventUploader와 백프레셔

프로덕션 안정성의 핵심이지만 외부 리포트에 없다.

```
문제: 도구 실행이 빠르면 이벤트가 폭주 → 서버 과부하/메모리 폭발
해결: 직렬 배칭 + 백프레셔

├─ 동시 POST 최대 1개 (직렬)
├─ 100ms 동안 text_delta 축적 후 배치 전송
├─ 큐 크기 100,000 초과 시 enqueue() 블로킹
├─ Retry-After 헤더 존중
└─ 연속 실패 시 배치 드롭 후 복구
```

배치 제한:

```
maxBatchSize:  100 아이템
maxBatchBytes: 10MB
maxQueueSize:  100,000
```

---

## 10. 메모리 자동 추출 (Extract Memories)

외부 리포트는 CLAUDE.md를 "메모리 파일"이라고만 언급한다. 실제로는 **대화 종료 시 백그라운드로 메모리를 자동 추출하는 별도 서비스**가 있다.

```
services/extractMemories/:
├─ 대화 전체를 분석
├─ 4가지 유형으로 분류 (user, feedback, project, reference)
├─ 기존 메모리와 중복 체크
├─ 마크다운 파일로 저장 + MEMORY.md 인덱스 업데이트
└─ EXTRACT_MEMORIES 피처 플래그로 게이트
```

CLAUDE.md와 메모리 시스템은 별개다:

```
CLAUDE.md (프로젝트 지침)        ≠  메모리 시스템
├─ git 추적 가능                    ├─ ~/.claude/projects/<hash>/memory/
├─ 수동 작성                        ├─ 대화에서 자동 추출
├─ 프로젝트 규칙/지침               ├─ 사용자 프로필, 피드백, 프로젝트 맥락
└─ 팀원과 공유 가능                 └─ 개인별 로컬 저장
```

---

## 중요도 순 정리

| 순위 | 요소 | 자체 에이전트 구축 시 영향 |
|------|------|--------------------------|
| 1 | **시스템 리마인더 주입** | 긴 대화 품질 유지의 핵심, 즉시 적용 가능 |
| 2 | **도구 지연 로드** | 시스템 프롬프트 토큰 50% 절약 |
| 3 | **메모리 자동 추출** | 대화 간 맥락 유지의 차별점 |
| 4 | **권한 레이스 패턴** | 팀 환경 에이전트에 필수 |
| 5 | **메시지 중복 제거** | 네트워크 불안정 환경에서 안정성 확보 |
| 6 | **백프레셔/배칭** | 프로덕션 배포 시 필수 |
| 7 | **피처 플래그** | 점진적 기능 배포 전략 |
| 8 | **멀티 트랜스포트** | 다양한 클라이언트 지원 시 필요 |
| 9 | **브릿지 시스템** | 원격 실행 필요 시 참고 |
| 10 | **텔레메트리** | 운영 모니터링, 보안 관점에서 인지 필요 |
