# Kagent 프로젝트 심층 분석

> Kagent는 Kubernetes 네이티브 AI 에이전트 프레임워크입니다.
> 이 문서는 Kagent의 아키텍처, CRD 설계, 에이전트 실행 방식, MCP 통합 메커니즘을 분석합니다.

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처](#2-아키텍처)
3. [Kubernetes CRD 분석](#3-kubernetes-crd-분석)
4. [Go 컨트롤러](#4-go-컨트롤러)
5. [Python 에이전트 런타임](#5-python-에이전트-런타임)
6. [MCP 도구 통합](#6-mcp-도구-통합)
7. [에이전트 정의 및 실행 흐름](#7-에이전트-정의-및-실행-흐름)
8. [K8sGPT와의 비교](#8-k8sgpt와의-비교)
9. [결론](#9-결론)

---

## 1. 프로젝트 개요

### 1.1 Kagent란?

Kagent는 **Kubernetes 네이티브 AI 에이전트 개발 및 배포 프레임워크**입니다. Solo.io가 개발하여 CNCF(Cloud Native Computing Foundation)에 기증했습니다.

**핵심 가치:**
- **K8s 네이티브**: 에이전트를 CRD(Custom Resource Definition)로 정의
- **선언적 설정**: YAML 기반 에이전트 정의 및 관리
- **MCP 표준**: Model Context Protocol을 통한 도구 통합
- **다중 LLM**: OpenAI, Anthropic, Google, AWS Bedrock, Ollama 등 지원
- **A2A 프로토콜**: 에이전트 간 통신 표준화

### 1.2 프로젝트 구조

```
kagent/
├── go/                  # K8s 컨트롤러 (관리 평면)
│   ├── api/            # CRD 정의 (Agent, ModelConfig, RemoteMCPServer)
│   ├── internal/       # 컨트롤러 구현
│   └── pkg/            # 공유 패키지
│
├── python/              # 에이전트 런타임 (실행 평면)
│   ├── packages/
│   │   ├── kagent-adk/     # ADK 기반 에이전트 엔진
│   │   ├── kagent-core/    # 핵심 라이브러리
│   │   └── kagent-openai/  # OpenAI 특화 기능
│   └── samples/        # 샘플 코드
│
├── helm/                # K8s 배포 차트
│   ├── kagent/         # 메인 차트
│   ├── agents/         # 사전 정의 에이전트 (k8s, istio, helm 등)
│   └── tools/          # MCP 도구 서버
│
├── design/              # 설계 문서 (EP-685 등)
└── examples/            # 예제 파일
```

---

## 2. 아키텍처

### 2.1 4계층 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        1. UI Layer                               │
│                   Web Dashboard / CLI                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. API Layer (Go)                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │   Agent     │  │   ModelConfig   │  │  RemoteMCPServer  │   │
│  │ Controller  │  │   Controller    │  │    Controller     │   │
│  └─────────────┘  └─────────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. Execution Layer (Python)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ADK Agent Executor                       │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐    │   │
│  │  │   LLM    │  │  MCP Client  │  │   A2A Server    │    │   │
│  │  │  Client  │  │  (도구 호출)  │  │  (에이전트 통신) │    │   │
│  │  └──────────┘  └──────────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  4. Infrastructure Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ K8s API  │  │   MCP    │  │   LLM    │  │    Other     │   │
│  │  Server  │  │  Servers │  │   APIs   │  │   Agents     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 컴포넌트 역할

| 컴포넌트 | 언어 | 역할 |
|----------|------|------|
| **Agent Controller** | Go | Agent CRD 감시, Deployment 생성 |
| **ModelConfig Controller** | Go | LLM 설정 관리 |
| **ADK Translator** | Go | Agent CRD → Python 설정 변환 |
| **Agent Executor** | Python | LLM 호출, 도구 실행, 응답 생성 |
| **MCP Client** | Python | MCP 서버 연결 및 도구 호출 |
| **A2A Server** | Python | 에이전트 간 통신 처리 |

---

## 3. Kubernetes CRD 분석

### 3.1 Agent CRD (핵심)

Agent CRD는 AI 에이전트를 정의하는 핵심 리소스입니다.

**두 가지 타입:**
1. **Declarative**: YAML로 완전히 정의
2. **BYO (Bring Your Own)**: 커스텀 이미지 사용

```yaml
# 선언형 에이전트 예시
apiVersion: kagent.dev/v1alpha2
kind: Agent
metadata:
  name: k8s-troubleshooter
  namespace: kagent
spec:
  type: Declarative
  description: "K8s 클러스터 분석 및 트러블슈팅 에이전트"

  declarative:
    # 시스템 프롬프트
    systemMessage: |
      당신은 Kubernetes 전문가입니다.
      클러스터 상태를 분석하고 문제를 해결합니다.

    # LLM 설정 참조
    modelConfig: gpt4-config

    # 응답 스트리밍
    stream: true

    # 사용할 도구들
    tools:
      - type: McpServer
        mcpServer:
          kind: RemoteMCPServer
          apiGroup: kagent.dev
          name: k8s-tools
          toolNames:
            - list_pods
            - get_logs
            - describe_resource

    # A2A 설정 (다른 에이전트에서 호출 가능)
    a2aConfig:
      skills:
        - name: analyze_cluster
          description: "클러스터 상태 분석"
```

### 3.2 ModelConfig CRD

LLM 프로바이더 설정을 중앙화합니다.

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: gpt4-config
  namespace: kagent
spec:
  provider: OpenAI
  model: gpt-4

  # API 키 (Secret에서 참조)
  apiKeySecret: openai-secret
  apiKeySecretKey: api-key

  # OpenAI 특화 설정
  openAI:
    baseUrl: https://api.openai.com/v1  # 또는 LiteLLM 게이트웨이
    temperature: 0.7
    maxTokens: 4096

  # TLS 설정 (내부 서버용)
  tls:
    caCertSecretRef: custom-ca
    caCertSecretKey: ca.crt
```

**지원 프로바이더:**
- OpenAI
- AzureOpenAI
- Anthropic
- Google Gemini / Vertex AI
- AWS Bedrock
- Ollama (로컬)

### 3.3 RemoteMCPServer CRD

외부 MCP 서버를 K8s 리소스로 정의합니다.

```yaml
apiVersion: kagent.dev/v1alpha2
kind: RemoteMCPServer
metadata:
  name: k8s-tools
  namespace: kagent
spec:
  description: "Kubernetes 도구 서버"
  protocol: STREAMABLE_HTTP  # 또는 SSE
  url: "http://mcp-server.kagent.svc:8080/mcp"
  timeout: 30s

  # 인증 헤더
  headersFrom:
    - secretRef:
        name: mcp-auth
        key: token
```

### 3.4 Tool 정의 구조

```go
// Tool 정의
type Tool struct {
    Type      ToolProviderType    // McpServer | Agent
    McpServer *McpServerTool      // MCP 서버 도구
    Agent     *TypedLocalReference // 다른 에이전트를 도구로 사용
}

// MCP 서버 참조
type McpServerTool struct {
    Kind      string    // MCPServer | RemoteMCPServer | Service
    ApiGroup  string    // kmcp.dev | kagent.dev
    Name      string    // 리소스 이름
    ToolNames []string  // 선택할 도구 목록
}
```

---

## 4. Go 컨트롤러

### 4.1 Agent Controller 동작 원리

```
Agent CRD 생성/변경
        │
        ▼
┌───────────────────────────────────────────┐
│           Agent Controller                 │
│                                            │
│  1. Agent 검증                            │
│     - 참조 리소스 확인                    │
│     - 순환 참조 검사                      │
│                                            │
│  2. AdkApiTranslator 호출                 │
│     - Agent → AgentConfig JSON 변환       │
│     - K8s 리소스 생성 (Deployment 등)     │
│                                            │
│  3. 리소스 배포                           │
│     - ConfigMap (에이전트 설정)           │
│     - Secret (API 키, 인증서)             │
│     - Deployment (에이전트 Pod)           │
│     - Service (A2A 엔드포인트)            │
│                                            │
│  4. 상태 업데이트                         │
│     - Ready 조건 설정                     │
└───────────────────────────────────────────┘
```

### 4.2 ADK API Translator

Agent CRD를 Python 런타임이 이해할 수 있는 JSON으로 변환합니다.

```go
// go/internal/controller/translator/agent/adk_api_translator.go

type AgentOutputs struct {
    Manifest    []client.Object     // K8s 리소스
    Config      *adk.AgentConfig    // Python ADK 설정
    AgentCard   server.AgentCard    // A2A 에이전트 카드
}

func TranslateAgent(agent *v1alpha2.Agent) (*AgentOutputs, error) {
    // 1. ModelConfig 조회
    modelConfig := getModelConfig(agent.Spec.Declarative.ModelConfig)

    // 2. 도구 설정 생성
    httpTools := []HttpMcpServerConfig{}
    for _, tool := range agent.Spec.Declarative.Tools {
        if tool.Type == McpServer {
            httpTools = append(httpTools, translateMcpTool(tool))
        }
    }

    // 3. AgentConfig 생성
    config := &AgentConfig{
        Model:       translateModel(modelConfig),
        Instruction: agent.Spec.Declarative.SystemMessage,
        HttpTools:   httpTools,
        Stream:      agent.Spec.Declarative.Stream,
    }

    // 4. Deployment 생성
    deployment := createDeployment(agent, config)

    return &AgentOutputs{
        Manifest: []client.Object{deployment, configMap, secret},
        Config:   config,
    }, nil
}
```

### 4.3 도구 라우팅

```
Tool 정의
    │
    ▼
┌──────────────────────────────────────────────────┐
│              도구 타입별 처리                     │
├──────────────────────────────────────────────────┤
│                                                   │
│  MCPServer (kmcp.dev)                            │
│  └─▶ HTTP MCP URL 생성                           │
│      http://server.namespace.svc:8080/mcp        │
│                                                   │
│  RemoteMCPServer (kagent.dev)                    │
│  └─▶ 설정된 URL 사용                             │
│      https://external-mcp.example.com/mcp        │
│                                                   │
│  Service (core/v1)                               │
│  └─▶ K8s Service DNS 기반 URL                   │
│      http://service.namespace.svc:port/mcp       │
│                                                   │
│  Agent (kagent.dev)                              │
│  └─▶ A2A 엔드포인트                             │
│      http://agent.namespace.svc:8083/a2a         │
│                                                   │
└──────────────────────────────────────────────────┘
```

---

## 5. Python 에이전트 런타임

### 5.1 패키지 구조

```
python/packages/
├── kagent-adk/              # 핵심 에이전트 엔진
│   └── src/kagent/adk/
│       ├── types.py         # AgentConfig, ModelConfig 타입
│       ├── _agent_executor.py # A2A 실행기
│       ├── tools/           # MCP 도구 클라이언트
│       └── models/          # LLM 래퍼
│
├── kagent-core/             # 공유 유틸리티
└── kagent-openai/           # OpenAI 특화 기능
```

### 5.2 AgentConfig 타입

```python
# python/packages/kagent-adk/src/kagent/adk/types.py

class AgentConfig(BaseModel):
    """에이전트 설정 모델"""

    # LLM 설정 (discriminated union)
    model: Union[
        OpenAI, Anthropic, Gemini,
        AzureOpenAI, Ollama, Bedrock
    ] = Field(discriminator="type")

    # 에이전트 정보
    description: str
    instruction: str          # 시스템 프롬프트

    # 도구 설정
    http_tools: list[HttpMcpServerConfig] | None
    sse_tools: list[SseMcpServerConfig] | None
    remote_agents: list[RemoteAgentConfig] | None

    # 실행 옵션
    stream: bool | None
    execute_code: bool | None

    def to_agent(self, name: str) -> Agent:
        """AgentConfig → Google ADK Agent 변환"""
        tools = []

        # HTTP MCP 도구
        for http_tool in self.http_tools or []:
            tools.append(McpToolset(
                connection_params=http_tool.params,
                tool_filter=http_tool.tools
            ))

        # 원격 에이전트 도구
        for remote in self.remote_agents or []:
            tools.append(AgentTool(
                agent=RemoteA2aAgent(
                    name=remote.name,
                    agent_card=remote.url + "/.well-known/agent"
                )
            ))

        # LLM 생성
        model = self._create_model()

        return Agent(
            name=name,
            model=model,
            instruction=self.instruction,
            tools=tools
        )
```

### 5.3 에이전트 실행기

```python
# python/packages/kagent-adk/src/kagent/adk/_agent_executor.py

class A2aAgentExecutor(AgentExecutor):
    """A2A 요청을 ADK Agent로 실행"""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # 1. A2A 요청 → ADK 실행 인자 변환
        run_args = convert_a2a_request(context.request)

        # 2. ADK Agent 실행
        async for event in self.agent.run(**run_args):

            # 3. ADK Event → A2A Event 변환
            a2a_events = convert_event(event)

            # 4. 이벤트 발행
            for a2a_event in a2a_events:
                await event_queue.put(a2a_event)
```

### 5.4 LLM 프로바이더별 설정

```python
class OpenAI(BaseLLM):
    type: Literal["openai"] = "openai"
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None  # o1 시리즈

class Anthropic(BaseLLM):
    type: Literal["anthropic"] = "anthropic"
    max_tokens: int | None = None

class Bedrock(BaseLLM):
    type: Literal["bedrock"] = "bedrock"
    region: str | None = None  # AWS 리전

# TLS 설정 (모든 모델 공통)
class BaseLLM(BaseModel):
    model: str
    headers: dict[str, str] | None = None
    tls_disable_verify: bool | None = None
    tls_ca_cert_path: str | None = None
```

---

## 6. MCP 도구 통합

### 6.1 MCP 프로토콜 지원

Kagent는 두 가지 MCP 프로토콜을 지원합니다:

```
┌────────────────────────────────────────────────────┐
│              MCP 프로토콜 지원                      │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. Streamable HTTP                                │
│     - POST /mcp                                    │
│     - JSON-RPC over HTTP                           │
│     - 단일 요청-응답                               │
│                                                     │
│  2. SSE (Server-Sent Events)                       │
│     - 양방향 스트리밍                              │
│     - 실시간 이벤트                                │
│                                                     │
└────────────────────────────────────────────────────┘
```

### 6.2 도구 검색 및 호출

```python
# MCP 도구 검색
mcp_client = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://mcp-server:8080/mcp"
    ),
    tool_filter=["list_pods", "get_logs"]  # 특정 도구만 선택
)

# 사용 가능한 도구 확인
tools = await mcp_client.get_tools()
# [Tool(name="list_pods", description="..."), ...]

# 도구 호출 (LLM이 결정)
result = await mcp_client.call_tool(
    name="list_pods",
    arguments={"namespace": "default"}
)
```

### 6.3 도구 헤더 커스터마이제이션

```yaml
tools:
  - type: McpServer
    mcpServer:
      name: secure-api
      toolNames: [fetch_data]
    # 도구별 인증 헤더
    headersFrom:
      - secretRef:
          name: api-token
          key: token
```

```python
# 자동으로 헤더 추가
headers = {
    "Authorization": "Bearer secret_token"
}
```

---

## 7. 에이전트 정의 및 실행 흐름

### 7.1 전체 라이프사이클

```
┌──────────────────────────────────────────────────────────────┐
│                    에이전트 배포 흐름                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 사용자가 Agent YAML 작성                                 │
│        │                                                      │
│        ▼                                                      │
│  2. kubectl apply -f agent.yaml                              │
│        │                                                      │
│        ▼                                                      │
│  3. Agent Controller가 CRD 감지                              │
│        │                                                      │
│        ▼                                                      │
│  4. AdkApiTranslator 변환                                    │
│     Agent CRD → AgentConfig JSON                             │
│        │                                                      │
│        ▼                                                      │
│  5. K8s 리소스 생성                                          │
│     - ConfigMap (에이전트 설정)                              │
│     - Secret (API 키, 인증서)                                │
│     - Deployment (에이전트 Pod)                              │
│     - Service (A2A 엔드포인트)                               │
│        │                                                      │
│        ▼                                                      │
│  6. Pod 시작 → Python 런타임 초기화                          │
│     - AgentConfig 로드                                        │
│     - LLM 클라이언트 생성                                    │
│     - MCP 연결 설정                                          │
│        │                                                      │
│        ▼                                                      │
│  7. A2A 서버 시작                                            │
│     http://agent.namespace.svc:8083                          │
│        │                                                      │
│        ▼                                                      │
│  8. Ready 상태 (사용 가능)                                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 요청 처리 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                    요청 처리 흐름                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  사용자 요청: "nginx Pod가 왜 안 되는지 확인해줘"            │
│        │                                                      │
│        ▼                                                      │
│  ┌────────────────────────────────────────────────────┐      │
│  │              A2A 프로토콜 처리                      │      │
│  │  POST /api/a2a/kagent/k8s-agent                   │      │
│  │  Body: {"task": "nginx Pod가 왜 안 되는지..."}    │      │
│  └────────────────────────────────────────────────────┘      │
│        │                                                      │
│        ▼                                                      │
│  ┌────────────────────────────────────────────────────┐      │
│  │              LLM 호출 (1차)                         │      │
│  │  "사용자가 Pod 문제를 확인하고 싶어합니다.         │      │
│  │   list_pods 도구를 사용하겠습니다"                 │      │
│  └────────────────────────────────────────────────────┘      │
│        │                                                      │
│        ▼ tool_call: list_pods                                │
│  ┌────────────────────────────────────────────────────┐      │
│  │              MCP 도구 호출                          │      │
│  │  POST http://k8s-mcp:8080/mcp                     │      │
│  │  {"method": "tools/call", "params": {...}}        │      │
│  └────────────────────────────────────────────────────┘      │
│        │                                                      │
│        ▼ 도구 결과                                           │
│  ┌────────────────────────────────────────────────────┐      │
│  │              LLM 호출 (2차)                         │      │
│  │  "nginx-abc123 Pod가 ImagePullBackOff 상태입니다.  │      │
│  │   이미지를 찾을 수 없습니다..."                    │      │
│  └────────────────────────────────────────────────────┘      │
│        │                                                      │
│        ▼                                                      │
│  ┌────────────────────────────────────────────────────┐      │
│  │              응답 스트리밍                          │      │
│  │  Event: TaskStatusUpdateEvent (진행 중)           │      │
│  │  Event: TaskArtifactUpdateEvent (분석 결과)       │      │
│  │  Event: TaskStatusUpdateEvent (완료)              │      │
│  └────────────────────────────────────────────────────┘      │
│        │                                                      │
│        ▼                                                      │
│  사용자에게 응답 반환                                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 에이전트 간 통신 (A2A)

```yaml
# 에이전트 A: 다른 에이전트를 도구로 사용
apiVersion: kagent.dev/v1alpha2
kind: Agent
metadata:
  name: coordinator-agent
spec:
  declarative:
    tools:
      # 다른 에이전트를 도구로 참조
      - type: Agent
        agent:
          kind: Agent
          apiGroup: kagent.dev
          name: k8s-agent
```

```
┌────────────────┐         A2A 호출         ┌────────────────┐
│  Coordinator   │ ───────────────────────▶ │   K8s Agent    │
│     Agent      │                          │                │
│                │ ◀─────────────────────── │                │
└────────────────┘        결과 반환          └────────────────┘
```

---

## 8. K8sGPT와의 비교

### 8.1 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                         K8sGPT                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [사용자] → [Go 코드 (하드코딩 규칙)] → [K8s API 조회]        │
│                      ↓                                           │
│              [문제 감지 (규칙 기반)]                             │
│                      ↓                                           │
│              [LLM: 설명만 생성]                                  │
│                                                                  │
│   특징: 읽기 전용, 규칙 기반, LLM은 보조                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Kagent                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [사용자] → [A2A 프로토콜] → [Python 런타임]                   │
│                      ↓                                           │
│              [LLM: 도구 선택 결정]                               │
│                      ↓                                           │
│              [MCP 도구 호출]                                     │
│                      ↓                                           │
│              [K8s API 조회/수정]                                 │
│                      ↓                                           │
│              [LLM: 결과 분석 및 응답]                            │
│                                                                  │
│   특징: 읽기/쓰기, LLM 주도, 에이전틱                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 상세 비교표

| 항목 | K8sGPT | Kagent |
|------|--------|--------|
| **분류** | 진단 도구 | 에이전트 프레임워크 |
| **LLM 역할** | 설명 생성 (보조) | 도구 선택 + 실행 (주도) |
| **K8s 접근** | 읽기 전용 (List/Get) | 읽기/쓰기 (CRUD) |
| **문제 감지** | 하드코딩 규칙 | LLM이 동적 판단 |
| **확장성** | 분석기 추가 (Go 코드) | CRD로 에이전트 정의 |
| **K8s 네이티브** | 아니오 (독립 실행) | 예 (CRD, Controller) |
| **에이전트 통신** | 없음 | A2A 프로토콜 |
| **도구 표준** | 자체 구현 | MCP 표준 |
| **커스터마이징** | 제한적 | YAML로 자유롭게 |

### 8.3 사용 시나리오

**K8sGPT가 적합한 경우:**
- 빠른 클러스터 헬스 체크
- CI/CD 파이프라인 검증
- 수정 권한 없이 진단만 필요

**Kagent가 적합한 경우:**
- 자동화된 문제 해결
- 복잡한 멀티스텝 작업
- 여러 에이전트 협업
- K8s 리소스 수정 필요

---

## 9. 결론

### 9.1 Kagent의 핵심 가치

1. **Kubernetes 네이티브**
   - CRD로 에이전트 정의
   - kubectl로 완전한 관리
   - GitOps 친화적

2. **진정한 에이전틱 AI**
   - LLM이 도구 선택 및 실행 결정
   - 멀티스텝 작업 자동화
   - 에이전트 간 협업

3. **표준 기반**
   - MCP (Model Context Protocol)
   - A2A (Agent-to-Agent) 프로토콜
   - 확장성과 상호운용성

4. **프로덕션 레디**
   - CNCF 프로젝트
   - TLS/인증 지원
   - 다중 LLM 백엔드

### 9.2 K8sGPT vs Kagent 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   K8sGPT = "K8s 진단 결과를 AI가 설명해주는 도구"               │
│                                                                  │
│   Kagent = "AI가 K8s를 직접 분석하고 조작하는 에이전트"         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 아키텍처 다이어그램 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kagent 전체 아키텍처                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Kubernetes Cluster                      │   │
│  │                                                            │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │   │
│  │  │    Agent    │   │ ModelConfig │   │  MCP Server │     │   │
│  │  │     CRD     │   │     CRD     │   │     CRD     │     │   │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘     │   │
│  │         │                 │                 │             │   │
│  │         ▼                 ▼                 ▼             │   │
│  │  ┌────────────────────────────────────────────────┐      │   │
│  │  │              Kagent Controller (Go)             │      │   │
│  │  └────────────────────────────────────────────────┘      │   │
│  │                        │                                  │   │
│  │                        ▼                                  │   │
│  │  ┌────────────────────────────────────────────────┐      │   │
│  │  │              Agent Deployment                   │      │   │
│  │  │  ┌──────────────────────────────────────────┐  │      │   │
│  │  │  │         Python Runtime (ADK)              │  │      │   │
│  │  │  │  ┌────────┐  ┌──────────┐  ┌──────────┐ │  │      │   │
│  │  │  │  │  LLM   │  │   MCP    │  │   A2A    │ │  │      │   │
│  │  │  │  │ Client │  │  Client  │  │  Server  │ │  │      │   │
│  │  │  │  └────────┘  └──────────┘  └──────────┘ │  │      │   │
│  │  │  └──────────────────────────────────────────┘  │      │   │
│  │  └────────────────────────────────────────────────┘      │   │
│  │                        │                                  │   │
│  │          ┌─────────────┼─────────────┐                   │   │
│  │          ▼             ▼             ▼                   │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐             │   │
│  │  │  K8s API │   │  MCP     │   │  Other   │             │   │
│  │  │  Server  │   │  Servers │   │  Agents  │             │   │
│  │  └──────────┘   └──────────┘   └──────────┘             │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 참고 자료

- [Kagent GitHub](https://github.com/kagent-dev/kagent)
- [Kagent 공식 문서](https://kagent.dev/docs/kagent)
- [Solo.io Kagent 발표](https://www.solo.io/blog/bringing-agentic-ai-to-kubernetes-contributing-kagent-to-cncf)
- [MCP 프로토콜](https://modelcontextprotocol.io/)
- [A2A 프로토콜](https://google.github.io/A2A/)
- [CNCF Kagent 페이지](https://landscape.cncf.io/)
