# K8sGPT 프로젝트 심층 분석

> K8sGPT는 Kubernetes 클러스터를 AI로 분석하고 문제를 진단하는 도구입니다.
> 이 문서는 K8sGPT의 아키텍처, K8s 사용 방식, LLM 통합 방식, 그리고 코딩 에이전트와의 차이점을 분석합니다.

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Kubernetes 사용 방식](#2-kubernetes-사용-방식)
3. [LLM 통합 및 제어 방식](#3-llm-통합-및-제어-방식)
4. [코딩 에이전트와의 핵심 차이점](#4-코딩-에이전트와의-핵심-차이점)
5. [구현 상세](#5-구현-상세)
6. [결론](#6-결론)

---

## 1. 프로젝트 개요

### 1.1 K8sGPT란?

K8sGPT는 **Kubernetes 클러스터 진단 도구**입니다. SRE(Site Reliability Engineering) 경험을 기반으로 한 분석 규칙과 AI를 결합하여 클러스터 문제를 자동으로 감지하고 설명합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      K8sGPT 아키텍처                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────┐     ┌──────────────┐     ┌──────────────┐  │
│    │  사용자   │────▶│   K8sGPT    │────▶│  Kubernetes  │  │
│    │  (CLI)   │     │   분석기     │     │   API 서버   │  │
│    └──────────┘     └──────────────┘     └──────────────┘  │
│                            │                                │
│                            ▼                                │
│                     ┌──────────────┐                       │
│                     │   LLM API    │                       │
│                     │ (설명 생성)   │                       │
│                     └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 특징

| 특징 | 설명 |
|------|------|
| **30+ 분석기** | Pod, Deployment, Service 등 K8s 리소스별 전문 분석 |
| **15+ AI 백엔드** | OpenAI, Azure, Google, AWS, Ollama 등 다양한 LLM 지원 |
| **MCP 프로토콜** | Claude Desktop 등 AI 도구와의 표준화된 통합 |
| **데이터 익명화** | 민감 정보 마스킹 후 LLM 전송 |

---

## 2. Kubernetes 사용 방식

### 2.1 K8s 클라이언트 구조

K8sGPT는 **읽기 전용(Read-Only)** 방식으로 Kubernetes와 상호작용합니다.

```go
// pkg/kubernetes/kubernetes.go
type Client struct {
    Client        kubernetes.Interface      // 표준 K8s API 클라이언트
    CtrlClient    ctrl.Client              // Controller-runtime 클라이언트
    Config        *rest.Config             // REST 설정
    ServerVersion *version.Info            // 클러스터 버전
    DynamicClient dynamic.Interface        // 동적 리소스 접근
}
```

**핵심 포인트**: K8sGPT는 **K8s 리소스를 수정하지 않습니다**. 오직 조회(List/Get)만 수행합니다.

### 2.2 클라이언트 초기화 프로세스

```
1. InCluster 설정 시도 (Pod 내부 실행 시)
   └─▶ ServiceAccount 토큰 사용

2. kubeconfig 기반 설정 (외부 실행 시)
   └─▶ ~/.kube/config 또는 지정 경로

3. 네 가지 클라이언트 생성
   ├─ kubernetes.Clientset (표준 API)
   ├─ controller-runtime Client (고수준 조작)
   ├─ Dynamic Client (CRD 포함 모든 리소스)
   └─ Discovery Client (API 메타데이터)
```

### 2.3 K8s API 호출 패턴

**Pod 목록 조회 예제:**
```go
// pkg/analyzer/pod.go
func (PodAnalyzer) Analyze(a common.Analyzer) ([]common.Result, error) {
    // K8s API 호출 - List 연산만 수행
    list, err := a.Client.GetClient().CoreV1().Pods(a.Namespace).List(
        a.Context,
        metav1.ListOptions{LabelSelector: a.LabelSelector},
    )

    // 조회된 데이터를 분석기 규칙으로 검사
    for _, pod := range list.Items {
        if pod.Status.Phase == "Pending" {
            // 문제 감지 로직
        }
    }
}
```

### 2.4 지원 리소스 범위

| 카테고리 | 리소스 |
|----------|--------|
| **워크로드** | Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob |
| **네트워킹** | Service, Ingress, NetworkPolicy, Gateway API |
| **설정** | ConfigMap, Secret |
| **스토리지** | PersistentVolumeClaim, PersistentVolume |
| **노드** | Node |
| **확장** | HPA, PDB, ValidatingWebhook, MutatingWebhook |

---

## 3. LLM 통합 및 제어 방식

### 3.1 핵심 개념: LLM은 "설명자"이지 "제어자"가 아님

```
┌────────────────────────────────────────────────────────────────┐
│                    K8sGPT의 LLM 사용 방식                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   [K8s 클러스터]                                               │
│        │                                                       │
│        ▼                                                       │
│   ┌─────────────────┐                                         │
│   │  사전정의 분석기  │  ◀── SRE 경험 기반 규칙               │
│   │  (30+ 규칙)      │                                         │
│   └─────────────────┘                                         │
│        │                                                       │
│        ▼ (감지된 문제)                                         │
│   ┌─────────────────┐                                         │
│   │  익명화 모듈     │  ◀── 민감 정보 마스킹                   │
│   └─────────────────┘                                         │
│        │                                                       │
│        ▼ (마스킹된 문제)                                       │
│   ┌─────────────────┐                                         │
│   │     LLM API     │  ◀── "설명"만 생성                      │
│   └─────────────────┘                                         │
│        │                                                       │
│        ▼ (설명 텍스트)                                         │
│   ┌─────────────────┐                                         │
│   │  역익명화 모듈   │  ◀── 원래 이름 복원                     │
│   └─────────────────┘                                         │
│        │                                                       │
│        ▼                                                       │
│   [사용자에게 출력]                                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**중요**: LLM은 K8s를 직접 제어하지 않습니다. LLM의 역할은 오직 **문제에 대한 설명 생성**입니다.

### 3.2 지원 AI 백엔드

```go
// pkg/ai/iai.go
var clients = []IAI{
    &OpenAIClient{},         // OpenAI GPT 시리즈
    &AzureAIClient{},        // Azure OpenAI
    &LocalAIClient{},        // 자체 호스팅 LocalAI
    &OllamaClient{},         // Ollama (로컬 LLM)
    &CohereClient{},         // Cohere
    &AmazonBedRockClient{},  // AWS Bedrock
    &SageMakerAIClient{},    // AWS SageMaker
    &GoogleGenAIClient{},    // Google AI (Gemini)
    &GoogleVertexAIClient{}, // Google Vertex AI
    &HuggingfaceClient{},    // Hugging Face
    &OCIGenAIClient{},       // Oracle Cloud
    &CustomRestClient{},     // 커스텀 REST API
    &IBMWatsonxAIClient{},   // IBM Watsonx
    &GroqClient{},           // Groq
}
```

### 3.3 AI 인터페이스 설계

```go
// pkg/ai/iai.go
type IAI interface {
    Configure(config IAIConfig) error
    GetCompletion(ctx context.Context, prompt string) (string, error)
    GetName() string
    Close()
}
```

**단순함이 핵심**: 인터페이스는 `GetCompletion()` 하나로 통일됩니다. 모든 백엔드가 동일한 방식으로 작동합니다.

### 3.4 프롬프트 구성

LLM에 전달되는 프롬프트는 다음 형식입니다:

```
다음 Kubernetes 문제를 분석하고 해결 방안을 제시해주세요:

- 리소스 타입: Pod
- 이름: tGLcCRcHa1Ce5Rs (익명화됨)
- 문제: CrashLoopBackOff - Container exited with code 137
```

LLM은 이 정보를 바탕으로 **설명과 권장사항**을 생성합니다:
- "이 오류는 메모리 부족(OOMKilled)으로 인해 발생합니다..."
- "권장사항: 리소스 제한을 늘리거나 메모리 누수를 점검하세요"

---

## 4. 코딩 에이전트와의 핵심 차이점

### 4.1 아키텍처 비교

| 항목 | K8sGPT | Claude Code / Gemini CLI |
|------|--------|--------------------------|
| **LLM 역할** | 설명 생성 (보조) | 의사결정 및 실행 (주체) |
| **도메인** | K8s 전용 | 범용 코딩/시스템 |
| **실행 제어** | 사전정의 규칙 | LLM이 동적 결정 |
| **수정 권한** | 읽기 전용 | 파일/코드 수정 가능 |
| **자율성** | 낮음 (규칙 기반) | 높음 (LLM 기반) |

### 4.2 상세 비교

#### K8sGPT 방식 (규칙 기반 + LLM 설명)

```
┌──────────────────────────────────────────────────────────┐
│                    K8sGPT 흐름                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   사용자: "k8sgpt analyze"                               │
│        │                                                 │
│        ▼                                                 │
│   [하드코딩된 분석기 실행]                               │
│   if pod.Status == "Pending" → 문제 감지                │
│   if deployment.Replicas != Ready → 문제 감지           │
│        │                                                 │
│        ▼                                                 │
│   [LLM: 감지된 문제 설명]                               │
│   "이 Pod는 스케줄링되지 않았습니다. 노드 리소스를      │
│    확인하세요..."                                        │
│        │                                                 │
│        ▼                                                 │
│   [사용자가 직접 수정]  ◀── LLM은 수정하지 않음         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### 코딩 에이전트 방식 (LLM 주도)

```
┌──────────────────────────────────────────────────────────┐
│               Claude Code / Gemini CLI 흐름              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   사용자: "이 버그를 고쳐줘"                             │
│        │                                                 │
│        ▼                                                 │
│   [LLM이 상황 분석]                                     │
│   "파일을 읽어서 문제를 파악해야겠다"                   │
│        │                                                 │
│        ▼                                                 │
│   [LLM이 도구 호출 결정]                                │
│   Read("src/main.py") → Grep("error") → ...            │
│        │                                                 │
│        ▼                                                 │
│   [LLM이 직접 코드 수정]                                │
│   Edit("src/main.py", old_code, new_code)              │
│        │                                                 │
│        ▼                                                 │
│   [LLM이 검증]                                          │
│   Bash("pytest") → 결과 확인 → 추가 수정               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.3 핵심 차이점 분석

#### 1) 문제 감지 방식

**K8sGPT:**
```go
// 하드코딩된 규칙
if pod.Status.Phase == "Pending" {
    for _, condition := range pod.Status.Conditions {
        if condition.Reason == "Unschedulable" {
            failures = append(failures, Failure{
                Text: "Pod is unschedulable",
            })
        }
    }
}
```

**코딩 에이전트:**
```
LLM 판단: "코드를 읽어보니 null 체크가 없어서
          NullPointerException이 발생할 수 있겠군"
```

#### 2) 실행 권한

**K8sGPT:**
- K8s API: `List`, `Get` 만 사용
- 파일 시스템: 캐시 저장 외 없음
- **K8s 리소스 수정 불가**

**코딩 에이전트:**
- 파일 읽기/쓰기/삭제
- 셸 명령어 실행
- Git 커밋/푸시
- **시스템 전반적 수정 가능**

#### 3) LLM 의존도

**K8sGPT:**
```
분석 정확도 = 분석기 규칙 품질 (90%) + LLM 설명 품질 (10%)
```
- LLM 없이도 문제 감지 가능 (`--explain` 옵션 비활성화)
- LLM은 선택적 "설명 레이어"

**코딩 에이전트:**
```
작업 성공률 = LLM 판단 품질 (100%)
```
- LLM이 모든 결정을 주도
- LLM 없이는 작동 불가

### 4.4 MCP를 통한 브릿지

K8sGPT는 **MCP(Model Context Protocol)**를 통해 코딩 에이전트와 연동됩니다:

```json
// Claude Desktop 설정 (claude_desktop_config.json)
{
  "mcpServers": {
    "k8sgpt": {
      "command": "k8sgpt",
      "args": ["serve", "--mcp"]
    }
  }
}
```

이렇게 하면:
- Claude가 K8sGPT의 분석 기능을 **도구로 호출** 가능
- `analyze`, `list-resources`, `get-logs` 등 12개 도구 제공
- Claude의 범용 능력 + K8sGPT의 K8s 전문성 결합

```
┌─────────────────────────────────────────────────────────┐
│              MCP를 통한 통합 아키텍처                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────┐     MCP 프로토콜     ┌──────────┐       │
│   │  Claude  │◀───────────────────▶│  K8sGPT  │       │
│   │  Desktop │   tools/call        │   MCP    │       │
│   └──────────┘   "analyze"         │  Server  │       │
│        │                           └──────────┘       │
│        │                                │              │
│        ▼                                ▼              │
│   [LLM 판단 능력]              [K8s 분석 전문성]       │
│   - 대화 이해                   - 30+ 분석기           │
│   - 복합 추론                   - K8s API 접근         │
│   - 행동 계획                   - 규칙 기반 진단       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 구현 상세

### 5.1 분석 흐름

```
k8sgpt analyze --explain --namespace default
        │
        ▼
┌───────────────────────────────────────┐
│ 1. 설정 로드                          │
│    - kubeconfig                       │
│    - AI 백엔드 설정                   │
│    - 활성 필터 확인                   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 2. 분석기 선택                        │
│    - 코어 분석기 (13개)               │
│    - 추가 분석기 (17개)               │
│    - 통합 분석기 (Prometheus 등)      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 3. 병렬 분석 실행                     │
│    - MaxConcurrency (기본값: 10)      │
│    - 각 분석기가 K8s API 호출         │
│    - 결과 수집                        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4. AI 설명 생성 (--explain 시)        │
│    - 문제 익명화                      │
│    - LLM API 호출                     │
│    - 설명 역익명화                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 5. 결과 출력                          │
│    - JSON / Text 형식                 │
│    - 캐시 저장                        │
└───────────────────────────────────────┘
```

### 5.2 분석기 구현 예시 (Pod Analyzer)

```go
// pkg/analyzer/pod.go

func (PodAnalyzer) Analyze(a common.Analyzer) ([]common.Result, error) {
    // 1. K8s API로 Pod 목록 조회
    list, err := a.Client.GetClient().CoreV1().Pods(a.Namespace).List(
        a.Context,
        metav1.ListOptions{LabelSelector: a.LabelSelector},
    )

    for _, pod := range list.Items {
        var failures []common.Failure

        // 2. Pending 상태 검사
        if pod.Status.Phase == "Pending" {
            for _, condition := range pod.Status.Conditions {
                if condition.Type == v1.PodScheduled &&
                   condition.Reason == "Unschedulable" {
                    failures = append(failures, common.Failure{
                        Text: condition.Message,
                        KubernetesDoc: getDocLink(pod),
                    })
                }
            }
        }

        // 3. 컨테이너 상태 검사
        for _, status := range pod.Status.ContainerStatuses {
            if status.State.Waiting != nil {
                reason := status.State.Waiting.Reason
                // CrashLoopBackOff, ImagePullBackOff 등 검사
                if reason == "CrashLoopBackOff" {
                    failures = append(failures, common.Failure{
                        Text: fmt.Sprintf("Container %s: %s",
                            status.Name, reason),
                    })
                }
            }
        }

        // 4. 결과 저장
        if len(failures) > 0 {
            a.Results = append(a.Results, common.Result{
                Kind:   "Pod",
                Name:   fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
                Error:  failures,
            })
        }
    }

    return a.Results, nil
}
```

### 5.3 AI 설명 생성 프로세스

```go
// pkg/analysis/analysis.go

func (a *Analysis) GetAIResults(output string, anonymize bool) error {
    // 1. 프롬프트 구성
    prompt := fmt.Sprintf(
        "다음 Kubernetes 문제를 분석하고 해결 방안을 제시해주세요:\n%s",
        output,
    )

    // 2. 익명화 (선택적)
    if anonymize {
        prompt = a.anonymizer.Anonymize(prompt)
    }

    // 3. LLM 호출
    response, err := a.AIClient.GetCompletion(a.Context, prompt)

    // 4. 역익명화 및 결과 저장
    if anonymize {
        response = a.anonymizer.Deanonymize(response)
    }

    // 결과를 각 Result에 할당
}
```

### 5.4 MCP 서버 구현

```go
// pkg/server/mcp.go

func NewMCPServer(...) (*K8sGptMCPServer, error) {
    // MCP 서버 생성
    mcpServer := server.NewMCPServer("k8sgpt", "1.0.0",
        server.WithToolCapabilities(true),
        server.WithResourceCapabilities(true, false),
        server.WithPromptCapabilities(false),
    )

    // 도구 등록
    analyzeTool := mcp.NewTool("analyze",
        mcp.WithDescription("Analyze Kubernetes resources for issues"),
        mcp.WithString("namespace", ...),
        mcp.WithBoolean("explain", ...),
    )
    mcpServer.AddTool(analyzeTool, handleAnalyze)

    // HTTP 또는 Stdio 모드로 시작
    if useHTTP {
        httpServer := server.NewStreamableHTTPServer(mcpServer)
        httpServer.Start(":" + port)
    } else {
        server.ServeStdio(mcpServer)
    }
}
```

### 5.5 프로젝트 구조

```
k8sgpt/
├── cmd/
│   ├── analyze/          # 분석 CLI 명령어
│   ├── auth/             # AI 백엔드 인증 관리
│   ├── cache/            # 캐시 관리
│   ├── filters/          # 분석기 필터 관리
│   ├── integration/      # 외부 시스템 통합
│   └── serve/            # 서버 모드 (gRPC/MCP)
│
├── pkg/
│   ├── ai/               # LLM 백엔드 (15개)
│   │   ├── iai.go        # 공통 인터페이스
│   │   ├── openai.go
│   │   ├── azure.go
│   │   └── ...
│   │
│   ├── analyzer/         # 분석기 (30+)
│   │   ├── pod.go
│   │   ├── deployment.go
│   │   ├── service.go
│   │   └── ...
│   │
│   ├── kubernetes/       # K8s 클라이언트
│   │   ├── kubernetes.go # 클라이언트 초기화
│   │   └── types.go
│   │
│   ├── server/           # 서버 구현
│   │   ├── server.go     # gRPC 서버
│   │   ├── mcp.go        # MCP 서버
│   │   └── mcp_handlers.go
│   │
│   └── integration/      # 외부 시스템 통합
│       ├── prometheus/
│       ├── kyverno/
│       └── keda/
│
└── main.go
```

---

## 6. 결론

### 6.1 K8sGPT의 위치

K8sGPT는 **코딩 에이전트와 다른 카테고리**의 도구입니다:

| 카테고리 | 특성 | 예시 |
|----------|------|------|
| **도메인 특화 진단 도구** | 규칙 기반, 읽기 전용, LLM 보조 | K8sGPT |
| **범용 AI 에이전트** | LLM 주도, 수정 권한, 자율적 | Claude Code, Gemini CLI |

### 6.2 K8sGPT의 강점

1. **안전성**: K8s 클러스터를 수정하지 않음
2. **예측 가능성**: 하드코딩된 규칙으로 일관된 결과
3. **전문성**: SRE 경험이 녹아든 30+ 분석기
4. **유연성**: 15+ AI 백엔드, MCP 통합 지원

### 6.3 한계점

1. **새로운 문제 감지 어려움**: 사전정의 규칙에 없는 문제는 감지 불가
2. **자동 수정 불가**: 문제 감지만 하고 수정은 사용자 몫
3. **LLM 의존성**: 설명 품질은 LLM에 의존

### 6.4 활용 시나리오

**K8sGPT가 적합한 경우:**
- 클러스터 헬스 체크 자동화
- CI/CD 파이프라인에서 K8s 배포 검증
- 운영 환경 모니터링

**코딩 에이전트가 적합한 경우:**
- 코드 작성/수정
- 복잡한 문제 해결
- 시스템 구성 변경

**통합 사용 (MCP):**
- Claude Desktop + K8sGPT MCP 서버
- 대화형 K8s 트러블슈팅
- AI 기반 운영 자동화

---

## 참고 자료

- [K8sGPT GitHub](https://github.com/k8sgpt-ai/k8sgpt)
- [K8sGPT 공식 문서](https://docs.k8sgpt.ai/)
- [MCP 프로토콜 명세](https://modelcontextprotocol.io/)
- [K8sGPT 지원 모델](https://github.com/k8sgpt-ai/k8sgpt/blob/main/SUPPORTED_MODELS.md)
