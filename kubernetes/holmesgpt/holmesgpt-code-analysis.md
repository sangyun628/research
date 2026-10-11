# HolmesGPT 코드 분석 보고서

## 1. 프로젝트 개요

**HolmesGPT**는 Robusta.Dev에서 개발한 CNCF Sandbox 프로젝트로, 클라우드 환경의 문제를 조사하고 근본 원인을 찾아 해결책을 제안하는 AI 에이전트입니다.

### 핵심 기능
- 클라우드/Kubernetes 환경의 알림 자동 조사
- 다양한 데이터 소스(Prometheus, Loki, Datadog 등)와 통합
- LLM 기반 에이전트 루프를 통한 문제 분석
- PagerDuty, OpsGenie, Jira 등 온콜 시스템 연동

---

## 2. 프로젝트 구조

```
holmesgpt/
├── holmes/                    # 핵심 코드
│   ├── core/                  # 핵심 비즈니스 로직
│   │   ├── tool_calling_llm.py   # LLM과 도구 호출 통합
│   │   ├── llm.py                # LLM 추상화 및 구현
│   │   ├── tools.py              # 도구 정의 및 실행
│   │   ├── toolset_manager.py    # 도구셋 관리
│   │   ├── config.py             # 설정 경로
│   │   ├── supabase_dal.py       # 데이터 액세스 레이어
│   │   └── transformers/         # 출력 변환기
│   ├── plugins/               # 플러그인 시스템
│   │   ├── toolsets/            # 통합 도구셋들
│   │   ├── sources/             # 데이터 소스 (PagerDuty, OpsGenie 등)
│   │   ├── destinations/        # 결과 출력 대상 (Slack 등)
│   │   ├── runbooks/            # 런북 정의
│   │   └── prompts/             # 프롬프트 템플릿
│   ├── config.py              # 전역 설정
│   ├── main.py                # CLI 진입점
│   └── interactive.py         # 대화형 모드
├── server.py                  # FastAPI 서버
├── pyproject.toml             # Python 의존성
└── helm/                      # Kubernetes 배포
```

---

## 3. 핵심 아키텍처

### 3.1 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 입력                               │
│    (CLI ask / investigate 명령 또는 API 요청)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Config 로드                               │
│  - YAML 설정 파일 파싱                                            │
│  - LLM 모델 레지스트리 초기화                                       │
│  - Toolset Manager 초기화                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ToolCallingLLM / IssueInvestigator            │
│  - 시스템/사용자 프롬프트 구성                                      │
│  - 에이전트 루프 실행 (max_steps 까지)                              │
│  - LLM 응답에서 도구 호출 추출 및 실행                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐      ┌─────────────────────────────────┐
│      LLM (litellm)     │      │        Tool Executor            │
│  - OpenAI/Azure/       │      │  - 도구 실행                      │
│    Anthropic/Bedrock   │      │  - 결과 변환 (Transformers)       │
│  - 토큰 카운팅          │      │  - 승인 확인                      │
└───────────────────────┘      └─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         결과 출력                                │
│  (CLI / Slack / JSON / 티켓 시스템 업데이트)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 에이전트 루프 (Agentic Loop)

`holmes/core/tool_calling_llm.py:417-609`에 구현된 핵심 루프:

```python
def call(self, messages, ...):
    while i < max_steps:
        # 1. 컨텍스트 윈도우 제한 적용
        limit_result = limit_input_context_window(llm, messages, tools)

        # 2. LLM 호출
        full_response = self.llm.completion(
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # 3. 도구 호출 추출 및 병렬 실행
        tools_to_call = response_message.tool_calls
        if not tools_to_call:
            return LLMResult(result=text_response, ...)

        # 4. 도구 실행 (ThreadPoolExecutor로 병렬화)
        for tool in tools_to_call:
            tool_result = self._invoke_llm_tool_call(tool, ...)
            messages.append(tool_result.as_tool_call_message())
```

---

## 4. 핵심 모듈 분석

### 4.1 LLM 통합 (`holmes/core/llm.py`)

#### DefaultLLM 클래스
```python
class DefaultLLM(LLM):
    def __init__(self, model, api_key, api_base, api_version, ...):
        self.model = model
        self.api_key = api_key
        # litellm을 통한 다양한 프로바이더 지원

    def completion(self, messages, tools, ...):
        return litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            ...
        )

    def count_tokens(self, messages, tools):
        # litellm.token_counter를 통한 토큰 카운팅
        return TokenCountMetadata(...)
```

#### 지원 LLM 프로바이더
- **OpenAI**: GPT-4, GPT-4o
- **Azure OpenAI Service**
- **Anthropic Claude**
- **AWS Bedrock**
- **IBM WatsonX**
- **Robusta AI** (프록시 서비스)

#### LLMModelRegistry
모델 설정을 관리하고 동적으로 로드:
```python
class LLMModelRegistry:
    def _init_models(self):
        # 1. YAML 파일에서 모델 로드
        self._llms = self._parse_models_file(MODEL_LIST_FILE_LOCATION)

        # 2. Robusta AI 모델 설정 (선택적)
        if self._should_load_robusta_ai():
            self.configure_robusta_ai_model()

        # 3. 환경변수/설정에서 모델 로드
        if self._should_load_config_model():
            self._llms[self.config.model] = self._create_model_entry(...)
```

### 4.2 도구 시스템 (`holmes/core/tools.py`)

#### Tool 추상 클래스
```python
class Tool(ABC, BaseModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter] = {}
    transformers: Optional[List[Transformer]] = None
    restricted: bool = False  # 런북 승인 필요 여부

    def invoke(self, params, context: ToolInvokeContext):
        # 1. 승인 확인
        if not context.user_approved:
            approval_check = self._get_approval_requirement(params, context)
            if approval_check.needs_approval:
                return StructuredToolResult(
                    status=StructuredToolResultStatus.APPROVAL_REQUIRED,
                    ...
                )

        # 2. 도구 실행
        result = self._invoke(params=params, context=context)

        # 3. 변환기 적용
        return self._apply_transformers(result)
```

#### StructuredToolResult
도구 실행 결과의 표준화된 구조:
```python
class StructuredToolResult(BaseModel):
    schema_version: str = "robusta:v1.0.0"
    status: StructuredToolResultStatus  # SUCCESS, ERROR, NO_DATA, APPROVAL_REQUIRED
    error: Optional[str] = None
    return_code: Optional[int] = None
    data: Optional[Any] = None
    url: Optional[str] = None
    invocation: Optional[str] = None
```

#### YAMLTool
YAML 파일에서 정의된 도구를 동적으로 로드:
```python
class YAMLTool(Tool):
    command: Optional[str] = None  # 쉘 명령어
    script: Optional[str] = None   # 멀티라인 스크립트

    def _invoke(self, params, context):
        # Jinja2 템플릿 렌더링 후 명령 실행
        rendered_command = Template(self.command).render(params)
        output, return_code = self.__execute_subprocess(rendered_command)
        return StructuredToolResult(...)
```

### 4.3 Toolset 관리 (`holmes/core/toolset_manager.py`)

#### ToolsetManager
```python
class ToolsetManager:
    def _list_all_toolsets(self, dal, ...):
        # 1. 빌트인 도구셋 로드
        builtin_toolsets = load_builtin_toolsets(dal)

        # 2. 설정에서 도구셋 로드 (오버라이드 가능)
        toolsets_from_config = self._load_toolsets_from_config(...)

        # 3. 커스텀 도구셋 로드
        custom_toolsets = self.load_custom_toolsets(...)

        # 4. fast_model 주입 (LLM 요약용)
        self._inject_fast_model_into_transformers(final_toolsets)

        # 5. 사전 조건 확인 (병렬)
        self.check_toolset_prerequisites(enabled_toolsets)
```

### 4.4 빌트인 Toolsets (`holmes/plugins/toolsets/`)

#### Python 기반 도구셋
| 도구셋 | 파일 | 설명 |
|--------|------|------|
| CoreInvestigationToolset | `investigator/core_investigation.py` | 핵심 조사 도구 |
| PrometheusToolset | `prometheus/prometheus.py` | Prometheus 메트릭 쿼리 |
| GrafanaLokiToolset | `grafana/loki/toolset_grafana_loki.py` | Loki 로그 쿼리 |
| GrafanaTempoToolset | `grafana/toolset_grafana_tempo.py` | Tempo 트레이스 조회 |
| DatadogLogsToolset | `datadog/toolset_datadog_logs.py` | Datadog 로그 |
| KubernetesLogsToolset | `kubernetes_logs.py` | K8s 로그 조회 |
| RobustaToolset | `robusta/robusta.py` | Robusta 플랫폼 연동 |
| BashExecutorToolset | `bash/bash_toolset.py` | 쉘 명령 실행 |
| RunbookToolset | `runbook/runbook_fetcher.py` | 런북 조회 |

#### YAML 기반 도구셋
```yaml
# kubernetes.yaml 예시
toolsets:
  kubernetes:
    enabled: true
    description: "Kubernetes 리소스 조회"
    tools:
      - name: "kubectl_describe"
        description: "Get details of a Kubernetes resource"
        command: "kubectl describe {{ resource_type }} {{ resource_name }} -n {{ namespace }}"
        parameters:
          resource_type:
            description: "리소스 유형 (pod, deployment, service 등)"
          resource_name:
            description: "리소스 이름"
          namespace:
            description: "네임스페이스"
```

---

## 5. 설정 시스템 (`holmes/config.py`)

### Config 클래스
```python
class Config(RobustaBaseConfig):
    # LLM 설정
    model: Optional[str] = None
    api_key: Optional[SecretStr] = None
    fast_model: Optional[str] = None
    max_steps: int = 40

    # 알림 소스 설정
    alertmanager_url: Optional[str] = None
    pagerduty_api_key: Optional[SecretStr] = None
    opsgenie_api_key: Optional[SecretStr] = None
    jira_url: Optional[str] = None

    # 도구 설정
    toolsets: Optional[dict[str, dict[str, Any]]] = None
    custom_toolsets: Optional[List[FilePath]] = None
    mcp_servers: Optional[dict[str, dict[str, Any]]] = None

    @classmethod
    def load_from_file(cls, config_file, **kwargs):
        # 파일 설정 + CLI 옵션 병합
        config_from_file = load_model_from_file(cls, config_file)
        merged_config = config_from_file.dict()
        merged_config.update(cli_options)
        return cls(**merged_config)

    def create_toolcalling_llm(self, dal, model, tracer):
        tool_executor = self.create_tool_executor(dal)
        return ToolCallingLLM(tool_executor, self.max_steps, self._get_llm(model))
```

### 설정 파일 예시 (`~/.holmes/config.yaml`)
```yaml
model: gpt-4o
api_key: ${OPENAI_API_KEY}
max_steps: 50

toolsets:
  prometheus:
    enabled: true
    config:
      url: http://prometheus:9090

  grafana/loki:
    enabled: true
    config:
      url: http://loki:3100

custom_toolsets:
  - /path/to/custom_toolset.yaml
```

---

## 6. 런북 시스템 (`holmes/plugins/runbooks/`)

### Runbook 모델
```python
class Runbook(RobustaBaseConfig):
    match: IssueMatcher      # 이슈 매칭 조건 (정규식)
    instructions: str        # AI에게 전달할 지시사항

class IssueMatcher(RobustaBaseConfig):
    issue_id: Optional[Pattern] = None
    issue_name: Optional[Pattern] = None
    source: Optional[Pattern] = None
```

### RunbookCatalog
```python
class RunbookCatalog(BaseModel):
    catalog: List[Union[RunbookCatalogEntry, RobustaRunbookInstruction]]

    def to_prompt_string(self):
        # 프롬프트에 포함할 문자열 생성
```

### RunbookManager (`holmes/core/runbooks.py`)
```python
class RunbookManager:
    def get_instructions_for_issue(self, issue: Issue) -> List[str]:
        # 이슈에 매칭되는 런북 지시사항 반환
        matching_instructions = []
        for runbook in self.runbooks:
            if runbook.match.issue_name and runbook.match.issue_name.match(issue.name):
                matching_instructions.append(runbook.instructions)
        return matching_instructions
```

---

## 7. CLI 인터페이스 (`holmes/main.py`)

### 주요 명령어

#### `holmes ask`
```bash
holmes ask "what pods are unhealthy and why?" --interactive
```
- 자유 형식 질문
- 파이프 입력 지원
- 대화형 모드 지원

#### `holmes investigate alertmanager`
```bash
holmes investigate alertmanager --alertmanager-url http://localhost:9093
```
- AlertManager에서 알림 가져와 조사

#### `holmes investigate pagerduty`
```bash
holmes investigate pagerduty --pagerduty-api-key $PAGERDUTY_API_KEY --update
```
- PagerDuty 인시던트 조사 및 결과 업데이트

#### `holmes toolset list/refresh`
```bash
holmes toolset list      # 도구셋 상태 확인
holmes toolset refresh   # 도구셋 상태 새로고침
```

---

## 8. 서버 모드 (`server.py`)

### FastAPI 엔드포인트
```python
app = FastAPI()

@app.post("/api/investigate")
async def investigate(request: InvestigateRequest):
    # 이슈 조사 API

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # 채팅 API (스트리밍 지원)

@app.get("/healthz")
async def health():
    # 헬스 체크
```

### 주기적 작업
```python
def _toolset_status_refresh_loop():
    # 도구셋 상태 주기적 갱신
    while True:
        time.sleep(TOOLSET_STATUS_REFRESH_INTERVAL_SECONDS)
        changes = config.refresh_server_tool_executor(dal)
```

---

## 9. 주요 설계 패턴

### 9.1 플러그인 아키텍처
- **Sources**: 알림/이슈 소스 (PagerDuty, OpsGenie, Jira, GitHub)
- **Destinations**: 결과 출력 대상 (Slack, CLI)
- **Toolsets**: 데이터 수집 도구 (Prometheus, Loki, kubectl 등)

### 9.2 팩토리 패턴
```python
class SourceFactory:
    @staticmethod
    def create_source(source: SupportedTicketSources, ...):
        if source == SupportedTicketSources.JIRA_SERVICE_MANAGEMENT:
            return config.create_jira_service_management_source()
        elif source == SupportedTicketSources.PAGERDUTY:
            return config.create_pagerduty_source()
```

### 9.3 변환기 패턴 (Transformers)
도구 출력을 후처리하는 파이프라인:
```python
class Transformer(BaseModel):
    name: str
    config: dict = {}

# llm_summarize: 대용량 출력을 LLM으로 요약
# 다른 변환기도 확장 가능
```

### 9.4 승인 시스템
보안을 위한 도구 실행 승인 메커니즘:
```python
class ApprovalRequirement(BaseModel):
    needs_approval: bool
    reason: str
    prefixes_to_save: Optional[List[str]] = None  # Bash용
```

---

## 10. 보안 고려사항

### 10.1 읽기 전용 접근
- HolmesGPT는 설계상 **읽기 전용** 접근만 수행
- RBAC 권한 존중

### 10.2 민감 정보 보호
- API 키는 `SecretStr` 타입으로 관리
- `ToolInvokeContext`에서 민감 정보 마스킹

### 10.3 Bash 명령 실행 제어
```python
class BashExecutorToolset:
    # 허용 목록/차단 목록 기반 명령 필터링
    # 사용자 승인 필요 옵션
```

---

## 11. 확장 가이드

### 11.1 커스텀 도구셋 추가
```yaml
# custom_toolset.yaml
toolsets:
  my-custom-tool:
    enabled: true
    description: "My custom integration"
    prerequisites:
      - env: [MY_API_KEY]
    tools:
      - name: "fetch_data"
        description: "Fetch data from my service"
        command: "curl -H 'Authorization: Bearer ${MY_API_KEY}' {{ url }}"
```

### 11.2 Python 도구셋 추가
```python
class MyCustomToolset(Toolset):
    def __init__(self):
        tools = [MyCustomTool()]
        super().__init__(
            name="my-custom",
            description="My custom toolset",
            tools=tools,
            prerequisites=[...]
        )
```

### 11.3 MCP 서버 연동
```yaml
mcp_servers:
  my-mcp-server:
    url: "http://localhost:8080/mcp"
    enabled: true
```

---

## 12. 의존성

### 핵심 라이브러리
| 라이브러리 | 용도 |
|-----------|------|
| `litellm` | 다양한 LLM 프로바이더 통합 |
| `pydantic` | 데이터 검증 및 설정 |
| `typer` | CLI 인터페이스 |
| `fastapi` | REST API 서버 |
| `rich` | 터미널 출력 포맷팅 |
| `jinja2` | 템플릿 렌더링 |
| `benedict` | YAML 설정 로딩 |

---

## 13. 결론

HolmesGPT는 잘 설계된 에이전틱 AI 시스템으로:

1. **모듈화된 아키텍처**: 플러그인 시스템을 통해 쉽게 확장 가능
2. **다양한 LLM 지원**: litellm을 통해 여러 프로바이더 통합
3. **풍부한 데이터 소스**: Kubernetes, Prometheus, Grafana 등 클라우드 네이티브 스택 지원
4. **보안 중심 설계**: 읽기 전용 접근, 승인 시스템, 민감 정보 마스킹
5. **유연한 설정**: YAML 기반 설정과 CLI 옵션 결합

클라우드 운영 환경에서 AI 기반 문제 해결 자동화를 구현하는 좋은 참고 사례입니다.
