# HolmesGPT Runbook & Toolset 시스템 심층 분석

## 목차
1. [개요](#1-개요)
2. [Toolset 시스템](#2-toolset-시스템)
3. [Tool 시스템](#3-tool-시스템)
4. [Runbook 시스템](#4-runbook-시스템)
5. [Transformers (변환기)](#5-transformers-변환기)
6. [실제 사용 예시](#6-실제-사용-예시)
7. [확장 가이드](#7-확장-가이드)

---

## 1. 개요

HolmesGPT는 **Toolset**과 **Runbook** 두 가지 핵심 시스템을 통해 AI 에이전트가 클라우드 환경을 효과적으로 조사할 수 있도록 합니다.

| 시스템 | 역할 | 비유 |
|--------|------|------|
| **Toolset** | 데이터 수집을 위한 도구 모음 | 엔지니어의 "도구 상자" |
| **Runbook** | 문제 해결을 위한 지침서 | 엔지니어의 "체크리스트" |

### 시스템 간 관계

```
┌─────────────────────────────────────────────────────────────┐
│                      알림/이슈 발생                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────────┐
│   Runbook 매칭       │         │      Toolset 로딩            │
│                     │         │                             │
│ • 이슈 이름 매칭     │         │ • 사전 조건 확인              │
│ • 지침 추출         │         │ • 활성화된 도구 수집           │
└─────────┬───────────┘         └──────────────┬──────────────┘
          │                                    │
          └────────────────┬───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM 에이전트                             │
│                                                             │
│  시스템 프롬프트:                                             │
│  - 사용 가능한 도구 목록 (Toolset에서)                         │
│  - 조사 지침 (Runbook에서)                                    │
│                                                             │
│  에이전트 루프:                                               │
│  1. Runbook 지침 참조                                        │
│  2. 적절한 Tool 선택 및 호출                                   │
│  3. 결과 분석                                                │
│  4. 다음 단계 결정 또는 완료                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Toolset 시스템

### 2.1 Toolset 클래스 구조

**파일 위치**: `holmes/core/tools.py:500-700`

```python
class Toolset(ABC, BaseModel):
    """도구셋 기본 클래스"""

    # 기본 정보
    name: str                              # 도구셋 이름 (예: "kubernetes/core")
    description: str                       # 설명
    docs_url: Optional[str] = None         # 문서 URL
    icon_url: Optional[str] = None         # 아이콘 URL

    # 상태 관리
    enabled: bool = False                  # 활성화 여부
    status: ToolsetStatusEnum = UNKNOWN    # 상태 (ENABLED, DISABLED, FAILED)
    error: Optional[str] = None            # 에러 메시지

    # 도구 및 설정
    tools: List[Tool] = []                 # 포함된 도구들
    prerequisites: List[ToolsetPrerequisite] = []  # 사전 조건
    tags: List[ToolsetTag] = []            # 태그 (CORE, CLI, CLUSTER)

    # 고급 설정
    transformers: Optional[List[Transformer]] = None  # 출력 변환기
    llm_instructions: Optional[str] = None            # LLM 추가 지침
    restricted_tools: List[str] = []                  # 승인 필요 도구 패턴

    # 타입 정보
    type: ToolsetType = BUILTIN            # BUILTIN, CUSTOMIZED, MCP
    path: Optional[str] = None             # 커스텀 도구셋 파일 경로
```

### 2.2 ToolsetType (도구셋 유형)

```python
class ToolsetType(str, Enum):
    BUILTIN = "built_in"      # 내장 도구셋 (Python/YAML)
    CUSTOMIZED = "customized"  # 사용자 정의 도구셋
    MCP = "mcp"               # MCP (Model Context Protocol) 서버
```

### 2.3 ToolsetTag (도구셋 태그)

```python
class ToolsetTag(str, Enum):
    CORE = "core"      # 핵심 도구셋 (CLI + 서버 모두 사용)
    CLI = "cli"        # CLI 전용 도구셋
    CLUSTER = "cluster" # 클러스터/서버 전용 도구셋
```

### 2.4 Prerequisites (사전 조건)

도구셋 활성화 전 확인하는 조건들:

```python
# 1. 환경변수 확인
class ToolsetEnvironmentPrerequisite(ToolsetPrerequisite):
    env: List[str]  # 필요한 환경변수 목록

    def is_satisfied(self):
        return all(os.environ.get(var) for var in self.env)

# 2. 명령어 실행 확인
class ToolsetCommandPrerequisite(ToolsetPrerequisite):
    command: str  # 실행할 명령어

    def is_satisfied(self):
        return_code, _ = execute_command(self.command)
        return return_code == 0

# 3. 정적 조건
class StaticPrerequisite(ToolsetPrerequisite):
    enabled: bool
    reason: str
```

**YAML 정의 예시:**
```yaml
prerequisites:
  # 환경변수 확인
  - env:
      - PROMETHEUS_URL
      - PROMETHEUS_TOKEN

  # 명령어 실행 확인
  - command: "kubectl version --client"

  # 복합 조건 (명령어에서 환경변수 사용)
  - command: "curl -s ${PROMETHEUS_URL}/api/v1/status/config"
```

### 2.5 빌트인 Toolsets 목록

**Python 기반 (동적 로직 필요):**

| Toolset | 파일 위치 | 설명 |
|---------|----------|------|
| `prometheus` | `plugins/toolsets/prometheus/prometheus.py` | PromQL 쿼리, 알림 조회 |
| `grafana/loki` | `plugins/toolsets/grafana/loki/toolset_grafana_loki.py` | 로그 쿼리 |
| `grafana/tempo` | `plugins/toolsets/grafana/toolset_grafana_tempo.py` | 트레이스 조회 |
| `datadog/*` | `plugins/toolsets/datadog/` | 로그, 메트릭, 트레이스 |
| `robusta` | `plugins/toolsets/robusta/robusta.py` | Robusta 플랫폼 연동 |
| `kafka` | `plugins/toolsets/kafka.py` | Kafka 메타데이터 |
| `rabbitmq` | `plugins/toolsets/rabbitmq/toolset_rabbitmq.py` | RabbitMQ 상태 |
| `kubernetes/logs` | `plugins/toolsets/kubernetes_logs.py` | Pod 로그 조회 |
| `bash` | `plugins/toolsets/bash/bash_toolset.py` | 쉘 명령 실행 |
| `runbook` | `plugins/toolsets/runbook/runbook_fetcher.py` | 런북 조회 |

**YAML 기반 (선언적 명령어):**

| Toolset | 파일 위치 | 설명 |
|---------|----------|------|
| `kubernetes/core` | `plugins/toolsets/kubernetes.yaml` | kubectl 명령어 |
| `kubernetes/live-metrics` | `plugins/toolsets/kubernetes.yaml` | kubectl top |
| `helm` | `plugins/toolsets/helm.yaml` | Helm 릴리스 관리 |
| `argocd` | `plugins/toolsets/argocd.yaml` | ArgoCD 앱 상태 |
| `docker` | `plugins/toolsets/docker.yaml` | Docker 컨테이너 |
| `confluence` | `plugins/toolsets/confluence.yaml` | Confluence 문서 |
| `slab` | `plugins/toolsets/slab.yaml` | Slab 지식베이스 |

### 2.6 ToolsetManager

**파일 위치**: `holmes/core/toolset_manager.py`

도구셋의 로딩, 상태 관리, 캐싱을 담당:

```python
class ToolsetManager:
    def __init__(
        self,
        toolsets: Optional[dict[str, dict[str, Any]]] = None,      # 설정에서 로드
        mcp_servers: Optional[dict[str, dict[str, Any]]] = None,   # MCP 서버
        custom_toolsets: Optional[List[FilePath]] = None,          # 커스텀 파일
        custom_toolsets_from_cli: Optional[List[FilePath]] = None, # CLI 옵션
        toolset_status_location: Optional[FilePath] = None,        # 상태 캐시
        global_fast_model: Optional[str] = None,                   # 요약용 모델
    ):
        ...

    def _list_all_toolsets(self, dal, check_prerequisites, enable_all_toolsets, toolset_tags):
        """모든 도구셋 로드 (우선순위 순서)"""

        # 1. 빌트인 도구셋 로드
        builtin_toolsets = load_builtin_toolsets(dal)

        # 2. 설정 파일에서 도구셋 로드 (오버라이드 가능)
        toolsets_from_config = self._load_toolsets_from_config(self.toolsets, ...)
        self.add_or_merge_onto_toolsets(toolsets_from_config, toolsets_by_name)

        # 3. 커스텀 도구셋 로드
        custom_toolsets = self.load_custom_toolsets(builtin_toolsets_names)
        self.add_or_merge_onto_toolsets(custom_toolsets, toolsets_by_name)

        # 4. Transformer에 fast_model 주입
        self._inject_fast_model_into_transformers(final_toolsets)

        # 5. 사전 조건 확인 (병렬 실행)
        self.check_toolset_prerequisites(enabled_toolsets)

        return final_toolsets

    def refresh_toolset_status(self, dal, enable_all_toolsets, toolset_tags):
        """도구셋 상태 새로고침 및 캐시 저장"""
        all_toolsets = self._list_all_toolsets(...)

        # 상태를 JSON 파일로 캐시
        with open(self.toolset_status_location, "w") as f:
            toolset_status = [
                {"name": t.name, "status": t.status, "enabled": t.enabled, ...}
                for t in all_toolsets
            ]
            json.dump(toolset_status, f)
```

### 2.7 Toolset 오버라이드 메커니즘

설정 파일에서 빌트인 도구셋의 속성을 오버라이드할 수 있습니다:

```yaml
# ~/.holmes/config.yaml
toolsets:
  # 빌트인 도구셋 오버라이드
  kubernetes/core:
    enabled: true
    llm_instructions: "Always check events before logs"

  # 빌트인 도구셋 비활성화
  prometheus:
    enabled: false

  # 새 커스텀 도구셋 추가
  my-custom-tool:
    enabled: true
    description: "My custom monitoring"
    tools:
      - name: "check_status"
        command: "curl http://my-service/status"
```

**오버라이드 로직** (`tools.py:override_with`):

```python
def override_with(self, other: "Toolset") -> None:
    """다른 도구셋의 설정으로 현재 도구셋을 오버라이드"""

    # 기본 필드 오버라이드
    if other.enabled is not None:
        self.enabled = other.enabled
    if other.description:
        self.description = other.description
    if other.llm_instructions:
        self.llm_instructions = other.llm_instructions

    # 도구 병합 (이름 기준)
    if other.tools:
        for new_tool in other.tools:
            existing = next((t for t in self.tools if t.name == new_tool.name), None)
            if existing:
                existing.override_with(new_tool)
            else:
                self.tools.append(new_tool)
```

---

## 3. Tool 시스템

### 3.1 Tool 클래스 구조

**파일 위치**: `holmes/core/tools.py:191-443`

```python
class Tool(ABC, BaseModel):
    """도구 기본 클래스"""

    # 기본 정보
    name: str                              # 도구 이름
    description: str                       # LLM에게 보여줄 설명
    user_description: Optional[str] = None # 사용자용 설명 (UI 표시)

    # 파라미터 정의
    parameters: Dict[str, ToolParameter] = {}

    # 보안 및 승인
    restricted: bool = False               # 승인 필요 여부

    # 출력 처리
    transformers: Optional[List[Transformer]] = None
    _transformer_instances: List[Any] = []

    @abstractmethod
    def _invoke(self, params: Dict, context: ToolInvokeContext) -> StructuredToolResult:
        """실제 도구 실행 (하위 클래스에서 구현)"""
        pass

    def invoke(self, params: Dict, context: ToolInvokeContext) -> StructuredToolResult:
        """도구 호출 (승인 확인 + 실행 + 변환)"""

        # 1. 승인 확인
        if not context.user_approved:
            approval_check = self._get_approval_requirement(params, context)
            if approval_check.needs_approval:
                return StructuredToolResult(
                    status=StructuredToolResultStatus.APPROVAL_REQUIRED,
                    error=f"Approval required: {approval_check.reason}",
                    invocation=self._format_invocation(params),
                )

        # 2. 실제 도구 실행
        result = self._invoke(params=params, context=context)

        # 3. Transformer 적용
        return self._apply_transformers(result)
```

### 3.2 ToolParameter (파라미터 정의)

```python
class ToolParameter(BaseModel):
    description: str           # 파라미터 설명 (LLM이 읽음)
    type: str = "string"       # 데이터 타입
    required: bool = True      # 필수 여부
    default: Optional[str] = None  # 기본값
    enum: Optional[List[str]] = None  # 허용 값 목록
```

### 3.3 StructuredToolResult (도구 실행 결과)

```python
class StructuredToolResultStatus(str, Enum):
    SUCCESS = "success"                    # 성공
    ERROR = "error"                        # 에러 발생
    NO_DATA = "no_data"                    # 데이터 없음
    TIMEOUT = "timeout"                    # 타임아웃
    APPROVAL_REQUIRED = "approval_required" # 승인 필요

class StructuredToolResult(BaseModel):
    schema_version: str = "robusta:v1.0.0"
    status: StructuredToolResultStatus
    error: Optional[str] = None            # 에러 메시지
    return_code: Optional[int] = None      # 명령어 반환 코드
    data: Optional[Any] = None             # 결과 데이터
    url: Optional[str] = None              # 관련 URL
    invocation: Optional[str] = None       # 실행된 명령어
```

### 3.4 YAMLTool (YAML 기반 도구)

**파일 위치**: `holmes/core/tools.py:680-850`

YAML 파일에서 정의된 도구를 동적으로 실행:

```python
class YAMLTool(Tool):
    command: Optional[str] = None   # 단일 명령어 (Jinja2 템플릿)
    script: Optional[str] = None    # 멀티라인 스크립트

    def _invoke(self, params: Dict, context: ToolInvokeContext) -> StructuredToolResult:
        # 1. Jinja2 템플릿 렌더링
        if self.command:
            rendered_command = self._render_template(self.command, params)
            output, return_code = self.__execute_subprocess(rendered_command)
        elif self.script:
            rendered_script = self._render_template(self.script, params)
            output, return_code = self.__execute_script(rendered_script)

        # 2. 결과 반환
        if return_code != 0:
            return StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=output,
                return_code=return_code,
            )

        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=output,
            invocation=rendered_command,
        )

    def _render_template(self, template: str, params: Dict) -> str:
        """Jinja2 템플릿 렌더링"""
        # 환경변수 주입
        env_vars = {k: v for k, v in os.environ.items()}
        context = {**env_vars, **params}

        return Template(template).render(context)
```

### 3.5 YAML Tool 정의 예시

```yaml
tools:
  # 1. 단순 명령어
  - name: "kubectl_get_pods"
    description: "List all pods in a namespace"
    command: "kubectl get pods -n {{ namespace }}"
    parameters:
      namespace:
        description: "Kubernetes namespace"
        type: "string"
        required: true

  # 2. 조건부 파라미터
  - name: "kubectl_describe"
    description: "Describe a Kubernetes resource"
    command: "kubectl describe {{ kind }} {{ name }}{% if namespace %} -n {{ namespace }}{% endif %}"
    parameters:
      kind:
        description: "Resource type (pod, deployment, service, etc.)"
      name:
        description: "Resource name"
      namespace:
        description: "Namespace (optional for cluster-scoped resources)"
        required: false

  # 3. 복잡한 스크립트
  - name: "kubernetes_jq_query"
    description: "Query Kubernetes resources with jq"
    script: |
      #!/bin/bash

      LIMIT=500
      CONTINUE=""

      while true; do
        if [ -z "$CONTINUE" ]; then
          QUERY="${API_PATH}?limit=${LIMIT}"
        else
          QUERY="${API_PATH}?limit=${LIMIT}&continue=${CONTINUE}"
        fi

        OUTPUT=$(kubectl get --raw "$QUERY")
        MATCHES=$(echo "$OUTPUT" | jq -r {{ jq_expr }})

        echo "$MATCHES"

        CONTINUE=$(echo "$OUTPUT" | jq -r '.metadata.continue // empty')
        if [ -z "$CONTINUE" ]; then
          break
        fi
      done
    parameters:
      kind:
        description: "Resource kind (plural form: pods, services, etc.)"
      jq_expr:
        description: "jq expression to filter results"

  # 4. Transformer 적용
  - name: "kubectl_get_all"
    description: "Get all resources in cluster"
    command: "kubectl get {{ kind }} -A"
    transformers:
      - name: llm_summarize
        config:
          input_threshold: 1000
          prompt: |
            Summarize this output:
            - Group similar items
            - Highlight errors and warnings
            - Be concise
```

### 3.6 Python Tool 구현 예시

```python
# holmes/plugins/toolsets/prometheus/prometheus.py

class PromqlQueryTool(Tool):
    def __init__(self, toolset: "PrometheusToolset"):
        super().__init__(
            name="promql_query",
            description="Execute a PromQL query against Prometheus",
            parameters={
                "query": ToolParameter(
                    description="PromQL query to execute",
                    type="string",
                    required=True,
                ),
                "time": ToolParameter(
                    description="Evaluation timestamp (optional)",
                    type="string",
                    required=False,
                ),
            },
        )
        self.toolset = toolset

    def _invoke(self, params: Dict, context: ToolInvokeContext) -> StructuredToolResult:
        try:
            query = params["query"]
            time = params.get("time")

            # Prometheus API 호출
            result = self.toolset.prometheus_client.query(
                query=query,
                time=time,
            )

            if not result:
                return StructuredToolResult(
                    status=StructuredToolResultStatus.NO_DATA,
                    data="No data returned for query",
                )

            return StructuredToolResult(
                status=StructuredToolResultStatus.SUCCESS,
                data=json.dumps(result, indent=2),
                url=f"{self.toolset.prometheus_url}/graph?g0.expr={query}",
            )

        except Exception as e:
            return StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=str(e),
            )
```

---

## 4. Runbook 시스템

### 4.1 Runbook 클래스 구조

**파일 위치**: `holmes/plugins/runbooks/__init__.py`

```python
class IssueMatcher(RobustaBaseConfig):
    """이슈 매칭 조건 (정규식 패턴)"""
    issue_id: Optional[Pattern] = None     # 이슈 ID 패턴
    issue_name: Optional[Pattern] = None   # 이슈/알림 이름 패턴
    source: Optional[Pattern] = None       # 소스 시스템 패턴 (prometheus, pagerduty 등)

class Runbook(RobustaBaseConfig):
    """런북 정의"""
    match: IssueMatcher      # 매칭 조건
    instructions: str        # AI에게 전달할 조사 지침
```

### 4.2 Runbook Catalog (런북 카탈로그)

다양한 형태의 런북을 통합 관리:

```python
class RunbookCatalogEntry(BaseModel):
    """마크다운 파일 기반 런북"""
    id: str                    # 런북 ID
    update_date: date          # 업데이트 날짜
    description: str           # 런북 설명
    link: str                  # 파일 경로 (.md)

class RobustaRunbookInstruction(BaseModel):
    """Robusta 플랫폼 연동 런북"""
    id: str                    # UUID
    symptom: str               # 증상 설명
    title: str                 # 런북 제목
    instruction: Optional[str] = None  # 지침 내용
    alerts: List[str] = []     # 관련 알림 목록

class RunbookCatalog(BaseModel):
    catalog: List[Union[RunbookCatalogEntry, RobustaRunbookInstruction]]

    def to_prompt_string(self) -> str:
        """프롬프트에 포함할 문자열 생성"""
        lines = []
        for entry in self.catalog:
            if isinstance(entry, RunbookCatalogEntry):
                lines.append(f"- {entry.id}: {entry.description} (link: {entry.link})")
            else:
                lines.append(f"- {entry.id}: {entry.symptom} - {entry.title}")
        return "\n".join(lines)
```

### 4.3 RunbookManager

**파일 위치**: `holmes/core/runbooks.py`

```python
class RunbookManager:
    def __init__(self, runbooks: List[Runbook]):
        self.runbooks = runbooks

    def get_instructions_for_issue(self, issue: Issue) -> List[str]:
        """이슈에 매칭되는 런북 지침 반환"""
        matching_instructions = []

        for runbook in self.runbooks:
            if self._matches(runbook.match, issue):
                matching_instructions.append(runbook.instructions)

        return matching_instructions

    def _matches(self, matcher: IssueMatcher, issue: Issue) -> bool:
        """이슈가 매처 조건에 맞는지 확인"""
        if matcher.issue_name and not matcher.issue_name.match(issue.name):
            return False
        if matcher.issue_id and not matcher.issue_id.match(issue.id):
            return False
        if matcher.source and not matcher.source.match(issue.source_type):
            return False
        return True
```

### 4.4 RunbookToolset (런북 조회 도구셋)

**파일 위치**: `holmes/plugins/toolsets/runbook/runbook_fetcher.py`

```python
class RunbookToolset(Toolset):
    """런북 조회를 위한 도구셋"""

    def __init__(self, dal: Optional[SupabaseDal] = None, additional_search_paths: Optional[List[str]] = None):
        self.dal = dal
        self.additional_search_paths = additional_search_paths

        # 런북 카탈로그 로드
        self.catalog = self._load_catalog()

        tools = []
        if self.catalog and self.catalog.catalog:
            tools.append(RunbookFetcher(self.catalog))

        super().__init__(
            name="runbook",
            description="Fetch runbook content for troubleshooting",
            tools=tools,
            tags=[ToolsetTag.CORE],
        )

    def _load_catalog(self) -> Optional[RunbookCatalog]:
        """런북 카탈로그 로드"""
        entries = []

        # 1. 로컬 마크다운 파일에서 로드
        for search_path in self._get_search_paths():
            for md_file in glob.glob(f"{search_path}/**/*.md", recursive=True):
                entry = self._parse_md_runbook(md_file)
                if entry:
                    entries.append(entry)

        # 2. Robusta 플랫폼에서 로드 (DAL 사용)
        if self.dal and self.dal.enabled:
            robusta_runbooks = self.dal.get_runbook_instructions()
            entries.extend(robusta_runbooks)

        return RunbookCatalog(catalog=entries) if entries else None

class RunbookFetcher(Tool):
    """런북 내용 조회 도구"""

    def __init__(self, catalog: RunbookCatalog):
        self.catalog = catalog
        runbook_list = ", ".join([e.id for e in catalog.catalog])

        super().__init__(
            name="fetch_runbook",
            description=f"Fetch runbook content by ID. Available runbooks: {runbook_list}",
            parameters={
                "runbook_id": ToolParameter(
                    description=f"Runbook ID. Must be one of: {runbook_list}",
                    type="string",
                    required=True,
                ),
            },
        )

    def _invoke(self, params: Dict, context: ToolInvokeContext) -> StructuredToolResult:
        runbook_id = params.get("runbook_id")

        # 카탈로그에서 런북 찾기
        entry = next(
            (e for e in self.catalog.catalog if e.id == runbook_id),
            None
        )

        if not entry:
            return StructuredToolResult(
                status=StructuredToolResultStatus.NO_DATA,
                error=f"Runbook '{runbook_id}' not found",
            )

        # 런북 내용 로드
        if isinstance(entry, RunbookCatalogEntry):
            content = self._load_md_content(entry.link)
        else:
            content = entry.instruction or entry.symptom

        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=content,
        )
```

### 4.5 Runbook 정의 예시 (YAML)

```yaml
# custom_runbooks.yaml

runbooks:
  # 1. 특정 알림에 대한 런북
  - match:
      issue_name: "KubePodCrashLooping"
    instructions: |
      ## CrashLoopBackOff 조사 지침

      1. **Pod 상태 확인**
         - kubectl describe pod로 현재 상태 확인
         - 재시작 횟수와 마지막 종료 이유 확인

      2. **로그 분석**
         - kubectl logs로 현재 로그 확인
         - kubectl logs --previous로 이전 컨테이너 로그 확인

      3. **리소스 제한 확인**
         - memory/CPU limits가 충분한지 확인
         - OOMKilled 여부 확인

      4. **이미지 및 설정 확인**
         - 이미지 태그가 올바른지 확인
         - ConfigMap/Secret이 올바른지 확인

  # 2. 정규식 패턴 매칭
  - match:
      issue_name: "High(CPU|Memory)Usage.*"
      source: "prometheus"
    instructions: |
      ## 리소스 사용량 조사 지침

      1. kubectl top으로 현재 사용량 확인
      2. Prometheus에서 시계열 데이터 조회
      3. 최근 배포 이력 확인
      4. HPA 설정 확인

  # 3. 소스별 런북
  - match:
      source: "pagerduty"
    instructions: |
      ## PagerDuty 알림 조사 기본 지침

      1. 알림 상세 내용 확인
      2. 관련 서비스 상태 확인
      3. 최근 변경 사항 확인
```

### 4.6 마크다운 런북 예시

```markdown
<!-- runbooks/troubleshooting-oom.md -->

---
id: troubleshooting-oom
update_date: 2024-01-15
description: Out of Memory 문제 해결 가이드
---

# Out of Memory (OOM) 문제 해결

## 증상
- Pod이 OOMKilled 상태로 종료됨
- 컨테이너가 반복적으로 재시작됨

## 조사 단계

### 1. OOM 발생 확인
```bash
kubectl describe pod <pod-name> -n <namespace>
```
"OOMKilled" 또는 "Memory" 관련 이벤트 확인

### 2. 메모리 사용량 분석
```bash
kubectl top pod <pod-name> -n <namespace>
```

### 3. 메모리 제한 확인
```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'
```

## 해결 방안
1. 메모리 limits 증가
2. 애플리케이션 메모리 누수 수정
3. JVM/런타임 메모리 설정 조정
```

### 4.7 Runbook과 프롬프트 통합

HolmesGPT는 런북을 시스템 프롬프트에 통합합니다:

```python
# holmes/core/investigation.py

def build_investigation_prompt(issue: Issue, runbook_manager: RunbookManager) -> str:
    prompt_parts = []

    # 1. 기본 시스템 프롬프트
    prompt_parts.append(SYSTEM_PROMPT)

    # 2. 이슈 정보
    prompt_parts.append(f"""
    ## Issue to Investigate
    - Name: {issue.name}
    - Source: {issue.source_type}
    - Description: {issue.raw_data}
    """)

    # 3. 매칭된 런북 지침 추가
    instructions = runbook_manager.get_instructions_for_issue(issue)
    if instructions:
        prompt_parts.append("""
        ## Runbook Instructions
        Follow these instructions to investigate the issue:
        """)
        for instruction in instructions:
            prompt_parts.append(instruction)

    return "\n\n".join(prompt_parts)
```

---

## 5. Transformers (변환기)

### 5.1 Transformer 개요

도구 출력을 후처리하여 LLM 컨텍스트 윈도우를 효율적으로 사용:

```python
class Transformer(BaseModel):
    name: str           # 변환기 이름
    config: dict = {}   # 설정

# 레지스트리에서 변환기 생성
transformer_instance = registry.create_transformer(
    transformer.name,
    transformer.config
)
```

### 5.2 LLM Summarize Transformer

**파일 위치**: `holmes/core/transformers/llm_summarize.py`

대용량 출력을 빠른 LLM 모델로 요약:

```python
class LLMSummarizeTransformer:
    def __init__(self, config: dict):
        self.input_threshold = config.get("input_threshold", 5000)
        self.fast_model = config.get("fast_model") or config.get("global_fast_model")
        self.prompt = config.get("prompt", DEFAULT_SUMMARIZE_PROMPT)

    def transform(self, result: StructuredToolResult) -> StructuredToolResult:
        if not result.data:
            return result

        data_str = str(result.data)

        # 임계값 미만이면 변환 안 함
        if len(data_str) < self.input_threshold:
            return result

        # LLM으로 요약
        summary = self._summarize(data_str)

        # 요약이 원본보다 크면 원본 유지
        if len(summary) >= len(data_str):
            return result

        result.data = summary
        return result

    def _summarize(self, content: str) -> str:
        llm = DefaultLLM(model=self.fast_model)
        response = llm.completion(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": content},
            ]
        )
        return response.choices[0].message.content
```

### 5.3 Transformer 설정 예시

```yaml
tools:
  - name: "kubectl_describe"
    command: "kubectl describe {{ kind }} {{ name }}"
    transformers:
      - name: llm_summarize
        config:
          # 1000자 이상일 때만 요약
          input_threshold: 1000

          # 요약용 모델 (기본값: global_fast_model)
          fast_model: "gpt-4o-mini"

          # 커스텀 요약 프롬프트
          prompt: |
            Summarize this kubectl describe output:
            - Focus on errors and warnings
            - Highlight resource status
            - Note any pending or failed conditions
            - Be concise (≤50% of original)
            - Keep grep-friendly keywords
```

### 5.4 Global Fast Model 주입

`--fast-model` CLI 옵션으로 모든 llm_summarize에 적용:

```python
# ToolsetManager._inject_fast_model_into_transformers()

def _inject_fast_model_into_transformers(self, toolsets: List[Toolset]) -> None:
    if not self.global_fast_model:
        return

    for toolset in toolsets:
        # 도구셋 레벨 변환기
        if toolset.transformers:
            for transformer in toolset.transformers:
                if transformer.name == "llm_summarize" and "fast_model" not in transformer.config:
                    transformer.config["global_fast_model"] = self.global_fast_model

        # 도구 레벨 변환기
        for tool in toolset.tools:
            if tool.transformers:
                for transformer in tool.transformers:
                    if transformer.name == "llm_summarize" and "fast_model" not in transformer.config:
                        transformer.config["global_fast_model"] = self.global_fast_model
```

---

## 6. 실제 사용 예시

### 6.1 CLI에서 커스텀 Toolset 사용

```bash
# 커스텀 도구셋 파일
cat > my_toolset.yaml << 'EOF'
toolsets:
  my-service:
    enabled: true
    description: "My custom service monitoring"
    prerequisites:
      - env: [MY_SERVICE_URL]
    tools:
      - name: "check_health"
        description: "Check service health"
        command: "curl -s ${MY_SERVICE_URL}/health | jq ."
      - name: "get_metrics"
        description: "Get service metrics"
        command: "curl -s ${MY_SERVICE_URL}/metrics"
EOF

# 환경변수 설정
export MY_SERVICE_URL="http://localhost:8080"

# 사용
holmes ask "check my service health" -t ./my_toolset.yaml
```

### 6.2 CLI에서 커스텀 Runbook 사용

```bash
# 커스텀 런북 파일
cat > my_runbooks.yaml << 'EOF'
runbooks:
  - match:
      issue_name: "ServiceDown"
    instructions: |
      1. Check if the service process is running
      2. Check service logs for errors
      3. Verify network connectivity
      4. Check dependent services
EOF

# 알림 조사 시 런북 사용
holmes investigate alertmanager \
  --alertmanager-url http://localhost:9093 \
  -r ./my_runbooks.yaml
```

### 6.3 설정 파일로 통합 관리

```yaml
# ~/.holmes/config.yaml

model: gpt-4o
api_key: ${OPENAI_API_KEY}
max_steps: 50
fast_model: gpt-4o-mini

# 빌트인 도구셋 설정
toolsets:
  prometheus:
    enabled: true
    config:
      url: http://prometheus:9090

  grafana/loki:
    enabled: true
    config:
      url: http://loki:3100

  kubernetes/core:
    enabled: true
    llm_instructions: |
      When investigating pods, always check events first.
      Use jq queries for large-scale analysis.

# 커스텀 도구셋 파일
custom_toolsets:
  - /etc/holmes/my_toolset.yaml

# 커스텀 런북 파일
custom_runbooks:
  - /etc/holmes/my_runbooks.yaml
```

### 6.4 서버 모드에서 API 사용

```python
import requests

# 이슈 조사 API
response = requests.post(
    "http://localhost:8080/api/investigate",
    json={
        "issue": {
            "name": "KubePodCrashLooping",
            "source_type": "prometheus",
            "raw_data": "Pod my-app-xxx is crash looping",
        },
        "model": "gpt-4o",
    }
)

result = response.json()
print(result["analysis"])
```

---

## 7. 확장 가이드

### 7.1 새 Python Toolset 추가

```python
# holmes/plugins/toolsets/my_toolset/my_toolset.py

from holmes.core.tools import Tool, Toolset, ToolParameter, StructuredToolResult
from holmes.core.tools import ToolsetEnvironmentPrerequisite, ToolsetTag

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_custom_query",
            description="Query my custom service",
            parameters={
                "query": ToolParameter(
                    description="Query string",
                    type="string",
                    required=True,
                ),
            },
        )

    def _invoke(self, params, context):
        query = params["query"]
        # 커스텀 로직 구현
        result = self._execute_query(query)
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=result,
        )

class MyCustomToolset(Toolset):
    def __init__(self):
        tools = [MyCustomTool()]
        super().__init__(
            name="my-custom",
            description="My custom monitoring toolset",
            tools=tools,
            prerequisites=[
                ToolsetEnvironmentPrerequisite(env=["MY_API_KEY"]),
            ],
            tags=[ToolsetTag.CORE],
        )

# holmes/plugins/toolsets/__init__.py에 등록
# from holmes.plugins.toolsets.my_toolset.my_toolset import MyCustomToolset
# toolsets.append(MyCustomToolset())
```

### 7.2 새 Transformer 추가

```python
# holmes/core/transformers/my_transformer.py

class MyCustomTransformer:
    def __init__(self, config: dict):
        self.threshold = config.get("threshold", 100)

    def transform(self, result: StructuredToolResult) -> StructuredToolResult:
        if not result.data:
            return result

        # 커스텀 변환 로직
        transformed_data = self._process(result.data)
        result.data = transformed_data
        return result

# 레지스트리에 등록
# registry.register("my_transformer", MyCustomTransformer)
```

### 7.3 MCP 서버 연동

```yaml
# config.yaml
mcp_servers:
  my-mcp-server:
    url: "http://localhost:8080/mcp"
    enabled: true
    description: "My MCP server for custom tools"
```

---

## 결론

HolmesGPT의 Toolset과 Runbook 시스템은:

1. **유연한 확장성**: YAML/Python으로 새 도구 쉽게 추가
2. **선언적 정의**: 복잡한 조사 절차를 런북으로 문서화
3. **자동화된 조사**: LLM이 런북 지침에 따라 도구를 자동 호출
4. **효율적인 컨텍스트 관리**: Transformer로 대용량 출력 요약
5. **보안**: 승인 시스템으로 민감한 명령 실행 제어

이 아키텍처는 Kubernetes 운영 자동화에 효과적인 참고 모델이 됩니다.
