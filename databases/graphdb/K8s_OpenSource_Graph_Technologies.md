# Kubernetes 환경 오픈소스 그래프 기술 총정리

## 개요

본 문서는 **특별한 인프라 없이 Kubernetes 환경에서 사용할 수 있는 오픈소스 그래프 관련 기술**들을 총망라하여 정리한 가이드입니다. 그래프 데이터베이스, 처리 프레임워크, GraphRAG 도구, 시각화 도구 등을 포함합니다.

**문서 작성일**: 2026-01-24

---

## 목차

1. [그래프 데이터베이스 (Graph Databases)](#1-그래프-데이터베이스-graph-databases)
2. [그래프 처리 프레임워크](#2-그래프-처리-프레임워크)
3. [GraphRAG / Knowledge Graph 도구](#3-graphrag--knowledge-graph-도구)
4. [그래프 시각화 도구](#4-그래프-시각화-도구)
5. [기술 선택 가이드](#5-기술-선택-가이드)
6. [참고 자료](#6-참고-자료)

---

## 1. 그래프 데이터베이스 (Graph Databases)

### 1.1 분산 그래프 데이터베이스

#### JanusGraph

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/JanusGraph/janusgraph |
| **Stars** | 5,000+ |
| **라이선스** | Apache 2.0 |
| **쿼리 언어** | Gremlin (Apache TinkerPop) |
| **K8s 지원** | Helm Chart, Operator |

**특징:**
- Linux Foundation 프로젝트로 대규모 그래프 처리에 특화
- **수천억 개의 노드와 엣지** 지원
- 다양한 스토리지 백엔드 지원: Cassandra, HBase, Google Cloud Bigtable, BerkeleyDB
- 인덱싱: Elasticsearch, Apache Solr, Apache Lucene

**K8s 배포:**
```bash
# Google Kubernetes Engine 예시
helm repo add janusgraph https://janusgraph.github.io/charts
helm install janusgraph janusgraph/janusgraph
```

**오픈소스 클러스터링**: Neo4j와 달리 오픈소스 버전에서도 클러스터링 완전 지원

**적합한 사용 사례:**
- 대규모 소셜 네트워크 분석
- IoT 데이터 연결 분석
- 빅데이터 환경의 그래프 분석

---

#### NebulaGraph

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/vesoft-inc/nebula |
| **Stars** | 10,000+ |
| **라이선스** | Apache 2.0 |
| **쿼리 언어** | nGQL (Cypher 유사) |
| **K8s 지원** | Helm Chart, Kubernetes Operator |

**아키텍처:**
```
┌─────────────────────────────────────────────────────────┐
│                    NebulaGraph Cluster                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ nebula-     │  │ nebula-     │  │ nebula-         │ │
│  │ graphd      │  │ metad       │  │ storaged        │ │
│  │ (Query)     │  │ (Metadata)  │  │ (Storage)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │               │                  │            │
│         └───────────────┴──────────────────┘            │
│                    Kubernetes Service                    │
└─────────────────────────────────────────────────────────┘
```

**특징:**
- **스토리지-컴퓨트 분리 아키텍처**: 독립적인 확장 가능
- Shared-nothing 설계로 고가용성
- Raft 합의 알고리즘 사용
- 밀리초 단위 지연 시간

**K8s 배포:**
```bash
# NebulaGraph Operator 사용
kubectl create -f https://raw.githubusercontent.com/vesoft-inc/nebula-operator/master/config/crd/bases/nebula.graph.io_nebulaclusters.yaml
helm install nebula-operator nebula-operator/nebula-operator
```

**K8s 장점:**
- 자동 로드 밸런싱
- 장애 자동 복구
- 탄력적 스케일링
- 롤링 업그레이드 지원

---

#### Dgraph

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/dgraph-io/dgraph |
| **Stars** | 20,000+ |
| **라이선스** | Apache 2.0 (v25부터 전 기능) |
| **쿼리 언어** | GraphQL, DQL |
| **K8s 지원** | Helm Chart, Kubernetes Operator |

**특징:**
- **네이티브 GraphQL 지원** - GraphQL 스키마로 데이터베이스 자동 생성
- ACID 트랜잭션 지원
- 수평 확장 가능한 분산 아키텍처
- 내장 검색 및 인덱싱

**K8s 배포:**
```bash
helm repo add dgraph https://charts.dgraph.io
helm install dgraph dgraph/dgraph
```

**GraphQL 예시:**
```graphql
type Person {
  id: ID!
  name: String! @search(by: [hash, fulltext])
  friends: [Person] @hasInverse(field: friends)
}
```

**v25 업데이트 (2025):**
- Apache 2.0 라이선스 하에 모든 기능 오픈소스화 예정

---

### 1.2 인메모리 그래프 데이터베이스

#### Memgraph

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/memgraph/memgraph |
| **Stars** | 3,000+ |
| **라이선스** | BSL 1.1 (3년 후 Apache 2.0) |
| **쿼리 언어** | Cypher |
| **K8s 지원** | 공식 Helm Chart |

**특징:**
- **인메모리 그래프 데이터베이스**로 초저지연
- Neo4j와 Cypher 호환
- 실시간 스트리밍 데이터 처리 지원
- MAGE (Memgraph Advanced Graph Extensions) 알고리즘 라이브러리

**K8s 배포:**
```bash
helm repo add memgraph https://memgraph.github.io/helm-charts
helm install memgraph memgraph/memgraph

# 고가용성 클러스터 (2 data + 3 coordinators)
helm install memgraph-ha memgraph/memgraph-high-availability
```

**Helm Chart 기능:**
- StatefulSet으로 배포
- PersistentVolumeClaim 자동 생성
- Kubernetes Secrets 지원
- Startup/Readiness/Liveness Probe 설정
- K8s Operator 출시 예정 (2025)

**LangChain/LlamaIndex 통합:**
```python
from langchain.graphs import MemgraphGraph
graph = MemgraphGraph(url="bolt://localhost:7687")
```

---

#### FalkorDB

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/FalkorDB/FalkorDB |
| **Stars** | 3,200+ |
| **라이선스** | SSPLv1 |
| **쿼리 언어** | OpenCypher |
| **K8s 지원** | Helm Chart, KubeBlocks Operator |

**특징:**
- **GraphBLAS 기반** 희소 행렬 연산으로 초고속 쿼리
- Redis 모듈로 구현 - 기존 Redis 인프라 활용
- **GraphRAG/GenAI 특화** 설계
- 선형대수 기반 쿼리 실행

**K8s 배포 - Helm:**
```bash
# Redis Sentinel 구성
helm install -f values.yaml falkordb oci://registry-1.docker.io/bitnamicharts/redis

# Redis Cluster 구성 (6 nodes)
helm install falkordb-cluster oci://registry-1.docker.io/bitnamicharts/redis-cluster
```

**K8s 배포 - KubeBlocks:**
```bash
helm repo add kubeblocks-addons https://apecloud.github.io/helm-charts
helm install falkordb-addon kubeblocks-addons/falkordb
```

**KubeBlocks 장점:**
- 자동화된 Day-2 운영
- 고가용성 설정
- 백업/복구 솔루션

---

### 1.3 멀티모델 데이터베이스

#### ArangoDB

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/arangodb/arangodb |
| **Stars** | 13,000+ |
| **라이선스** | BSL 1.1 (v3.12부터) |
| **쿼리 언어** | AQL (ArangoDB Query Language) |
| **K8s 지원** | Kubernetes Operator |

**특징:**
- **그래프 + 문서 + 키-값** 멀티모델
- 단일 쿼리 언어(AQL)로 모든 모델 지원
- 네이티브 JSON 지원
- 분산 아키텍처 및 클러스터링

**K8s 배포:**
```bash
# ArangoDB Kubernetes Operator
helm install arango-crd https://github.com/arangodb/kube-arangodb/releases/download/1.2.42/kube-arangodb-crd-1.2.42.tgz
helm install arango-operator https://github.com/arangodb/kube-arangodb/releases/download/1.2.42/kube-arangodb-1.2.42.tgz
```

**주의사항:**
- v3.12부터 BSL 1.1 라이선스로 변경
- 비프로덕션 환경에서는 자유롭게 사용 가능

---

#### ArcadeDB

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/ArcadeData/arcadedb |
| **Stars** | 3,000+ |
| **라이선스** | Apache 2.0 |
| **쿼리 언어** | SQL, Cypher, Gremlin, GraphQL |
| **K8s 지원** | 공식 Helm Chart |

**특징:**
- **그래프 + 문서 + 키-값 + 벡터 + 시계열** 멀티모델
- OrientDB의 개념적 포크
- 다중 쿼리 언어 지원 (SQL, Cypher, Gremlin, MongoDB, Redis)
- **Java 21 기반** 고성능 엔진
- 벡터 임베딩 지원

**K8s 배포:**
```bash
helm repo add arcadedb https://arcadedata.github.io/arcadedb-charts
helm install arcadedb arcadedb/arcadedb

# HA 클러스터 설정
helm install arcadedb arcadedb/arcadedb \
  --set replicaCount=3 \
  --set persistence.enabled=true
```

**2025 업데이트 (v25.x):**
- Helm Chart 동적 PV 할당 지원
- HA 클러스터 Kubernetes 안정성 개선
- Java 21 완전 지원

---

### 1.4 PostgreSQL 확장

#### Apache AGE

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/apache/age |
| **Stars** | 3,000+ |
| **라이선스** | Apache 2.0 |
| **쿼리 언어** | SQL + openCypher |
| **K8s 지원** | CloudNativePG + 커스텀 이미지 |

**특징:**
- PostgreSQL 확장으로 그래프 기능 추가
- **기존 PostgreSQL 인프라 활용** 가능
- ANSI SQL과 openCypher 동시 사용
- ACID 트랜잭션, MVCC, 트리거 지원
- PostgreSQL 11-18 지원

**K8s 배포 (CloudNativePG):**
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: age-cluster
spec:
  instances: 3
  imageName: apache/age:latest
  postgresql:
    shared_preload_libraries:
      - age
```

**사용 예시:**
```sql
-- 그래프 생성
SELECT create_graph('social_network');

-- Cypher 쿼리 실행
SELECT * FROM cypher('social_network', $$
  CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
  RETURN a, b
$$) AS (a agtype, b agtype);
```

**장점:**
- PostgreSQL 생태계 활용 (pg_vector, PostGIS 등)
- Azure Database for PostgreSQL에서 공식 지원
- 기존 RDB 기반 시스템에 그래프 기능 추가 용이

---

### 1.5 특수 목적 그래프 데이터베이스

#### TypeDB

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/typedb/typedb |
| **Stars** | 4,000+ |
| **라이선스** | AGPL-3.0 (Community) |
| **쿼리 언어** | TypeQL |
| **K8s 지원** | 공식 문서 제공 |

**특징:**
- **타입 시스템 기반** 지식 그래프
- 온톨로지와 추론 지원 (RDF/OWL 불필요)
- 상속, 다형성 등 OOP 개념 적용
- 패턴 매칭 및 규칙 기반 추론

**TypeDB 3.0 (2024.12):**
- Rust로 완전 재작성
- TypeQL 개선
- 성능 대폭 향상

**K8s 배포:**
공식 문서 참조: https://docs.vaticle.com/docs/running-typedb-cluster/kubernetes

**적합한 사용 사례:**
- 사이버 위협 인텔리전스 (STIX)
- 생명과학 데이터 모델링
- 복잡한 도메인 지식 모델링

---

### 1.6 그래프 데이터베이스 비교표

| 데이터베이스 | 라이선스 | 쿼리 언어 | 분산 지원 | K8s 성숙도 | 특화 분야 |
|-------------|----------|-----------|-----------|------------|-----------|
| **JanusGraph** | Apache 2.0 | Gremlin | O (완전) | ★★★★☆ | 대규모 분산 그래프 |
| **NebulaGraph** | Apache 2.0 | nGQL | O (완전) | ★★★★★ | 고성능 분산 처리 |
| **Dgraph** | Apache 2.0 | GraphQL/DQL | O (완전) | ★★★★☆ | GraphQL 네이티브 |
| **Memgraph** | BSL 1.1 | Cypher | O (HA) | ★★★★☆ | 실시간 분석 |
| **FalkorDB** | SSPLv1 | Cypher | O (Redis) | ★★★☆☆ | GraphRAG/GenAI |
| **ArangoDB** | BSL 1.1 | AQL | O (완전) | ★★★★☆ | 멀티모델 |
| **ArcadeDB** | Apache 2.0 | 다중 | O (완전) | ★★★☆☆ | 멀티모델/벡터 |
| **Apache AGE** | Apache 2.0 | Cypher+SQL | O (PG) | ★★★☆☆ | PostgreSQL 통합 |
| **TypeDB** | AGPL-3.0 | TypeQL | O (클러스터) | ★★★☆☆ | 지식 표현/추론 |

---

## 2. 그래프 처리 프레임워크

### Apache TinkerPop / Gremlin

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/apache/tinkerpop |
| **라이선스** | Apache 2.0 |
| **쿼리 언어** | Gremlin |
| **K8s 지원** | Docker 이미지 제공 |

**개요:**
Apache TinkerPop은 **그래프 컴퓨팅 프레임워크**로, OLTP(그래프 데이터베이스)와 OLAP(그래프 분석 시스템) 모두를 지원합니다.

**Gremlin 쿼리 언어:**
```groovy
// 친구의 친구 찾기
g.V().has('name', 'Alice')
  .out('knows')
  .out('knows')
  .dedup()
  .values('name')
```

**TinkerPop 3.8.0 (2025.11):**
- 새로운 타입 변환 스텝: `asBool`, `asNumber`
- `typeOf` 조건자 추가
- `none` → `discard` 이름 변경

**지원 그래프 DB:**
- JanusGraph
- Amazon Neptune
- Azure Cosmos DB
- OrientDB
- Neo4j (Gremlin 플러그인)

**K8s 배포:**
```bash
# Gremlin Server Docker 이미지
docker pull tinkerpop/gremlin-server

# K8s Deployment
kubectl create deployment gremlin-server \
  --image=tinkerpop/gremlin-server:3.8.0
```

**프로덕션 사용 사례:**
- Netflix: 데이터 리니지 서비스
- Amundsen: 데이터 디스커버리 플랫폼
- Altimeter: 클라우드 보안 분석

---

## 3. GraphRAG / Knowledge Graph 도구

### 3.1 GraphRAG 프레임워크

#### Microsoft GraphRAG

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/microsoft/graphrag |
| **라이선스** | MIT |
| **언어** | Python |
| **K8s 지원** | Azure Kubernetes Service (Accelerator) |

**개요:**
Microsoft GraphRAG는 **텍스트 추출, 네트워크 분석, LLM 프롬프팅을 결합**한 end-to-end RAG 시스템입니다.

**작동 원리:**
```
비정형 텍스트 → 엔티티 추출 → 커뮤니티 탐지 → 계층적 요약 → LLM 응답
```

**특징:**
- 전체 데이터셋에 대한 이해가 필요한 질문 해결
- 커뮤니티 탐지 기반 접근법
- LazyGraphRAG: 경량화 버전

**설치 및 사용:**
```bash
pip install graphrag

# 초기화
graphrag init --root ./ragtest

# 인덱싱
graphrag index --root ./ragtest

# 쿼리
graphrag query --root ./ragtest --method global \
  --query "What are the main themes?"
```

**K8s 배포 (GraphRAG Accelerator):**
- Azure Kubernetes Service 기반
- Helm Chart 배포
- 자동 스케일링 지원
- ⚠️ 상당한 Azure 비용 발생 가능

---

#### nano-graphrag

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/gusye1234/nano-graphrag |
| **라이선스** | MIT |
| **언어** | Python |
| **K8s 지원** | 컨테이너화 가능 |

**개요:**
Microsoft GraphRAG의 **경량화 오픈소스 구현**으로, 약 1,100줄의 코드로 핵심 기능을 제공합니다.

**특징:**
- 간결하고 해킹하기 쉬운 구조
- 비동기 및 완전한 타입 힌팅
- 다양한 백엔드 지원: faiss, neo4j, ollama

**쿼리 모드:**
- **Naive**: 기본 사실 검색
- **Local**: 지역 관계 탐색
- **Global**: 전체 커뮤니티 기반 분석

**설치:**
```bash
pip install nano-graphrag
```

**사용 예시:**
```python
from nano_graphrag import GraphRAG

# 초기화
rag = GraphRAG(working_dir="./nano_graphrag_cache")

# 문서 삽입
with open("document.txt") as f:
    rag.insert(f.read())

# 쿼리
result = rag.query("What are the main topics?", param=QueryParam(mode="global"))
```

**커스터마이징:**
- 스토리지 백엔드 교체 가능
- 벡터 DB: nano-vectordb (기본), faiss 등
- 그래프 DB: networkx (기본), neo4j 등

---

### 3.2 RAG/에이전트 프레임워크

#### LlamaIndex

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/run-llama/llama_index |
| **Stars** | 43,000+ |
| **라이선스** | MIT |
| **K8s 지원** | Docker 컨테이너화 |

**Knowledge Graph 통합:**
```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.memgraph import MemgraphGraphStore

# 그래프 스토어 설정
graph_store = MemgraphGraphStore(
    url="bolt://localhost:7687",
    username="memgraph",
    password="password"
)

# Knowledge Graph 인덱스 생성
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    graph_store=graph_store,
    max_triplets_per_chunk=10
)
```

**특징:**
- 다양한 그래프 DB 백엔드 지원
- 문서에서 자동 Knowledge Graph 생성
- GraphRAG 쿼리 모드 지원

---

#### LangChain

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/langchain-ai/langchain |
| **Stars** | 95,000+ |
| **라이선스** | MIT |
| **K8s 지원** | Docker, LangSmith 자체 호스팅 |

**그래프 DB 통합:**
```python
from langchain_community.graphs import MemgraphGraph
from langchain.chains import GraphCypherQAChain

# 그래프 연결
graph = MemgraphGraph(url="bolt://localhost:7687")

# 자연어 → Cypher 쿼리 체인
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

result = chain.run("Who knows Alice?")
```

**지원 그래프 DB:**
- Neo4j
- Memgraph
- FalkorDB
- NebulaGraph
- ArangoDB

---

#### Cognee

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/topoteretes/cognee |
| **라이선스** | Apache 2.0 |
| **특징** | Graph + Vector 하이브리드 메모리 |

**특징:**
- AI 에이전트를 위한 **인지 메모리 레이어**
- 그래프 구조 + 벡터 임베딩 통합
- 세션 간 정보 기억 및 추론
- 모듈식 파이프라인 아키텍처

**지원 그래프 백엔드:**
- NetworkX
- FalkorDB
- Neo4j

---

### 3.3 Knowledge Graph 관리 플랫폼

#### WhyHow Knowledge Graph Studio

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/whyhow-ai/knowledge-graph-studio |
| **라이선스** | MIT |
| **특징** | Low-code KG 관리 |

**특징:**
- 오픈소스 Knowledge Graph 관리 플랫폼
- 커스텀 보안, 모니터링 통합 가능
- 스키마 라이브러리 제공
- 자체 환경에 설치 가능

---

## 4. 그래프 시각화 도구

### Gephi / Gephi Lite

| 항목 | 내용 |
|------|------|
| **웹사이트** | https://gephi.org/ |
| **라이선스** | CDDL 1.0 + GPL 3 |
| **형태** | 데스크톱 (Java) / 웹 (Lite) |

**특징:**
- 10,000+ 학술 논문에서 인용
- 다양한 레이아웃 알고리즘
- 통계 분석 기능 내장
- **Gephi Lite**: 웹 기반 경량 버전 (2025)

**사용 사례:**
- 소셜 네트워크 분석
- 생물학적 네트워크 시각화
- 인용 네트워크 분석

---

### Cytoscape

| 항목 | 내용 |
|------|------|
| **웹사이트** | https://cytoscape.org/ |
| **라이선스** | LGPL |
| **형태** | 데스크톱 (Java) |

**특징:**
- **생명과학 분야** 표준 도구
- 250+ 플러그인 생태계
- 다양한 네트워크 타입 지원
- 속성 데이터 통합 가능

---

### 웹 기반 시각화 라이브러리

#### JavaScript 라이브러리

| 라이브러리 | 특징 | GitHub |
|-----------|------|--------|
| **Cytoscape.js** | Cytoscape의 JS 버전, 모바일 지원 | 10,000+ stars |
| **Sigma.js** | 대규모 그래프 렌더링, WebGL | 11,000+ stars |
| **D3.js** | 범용 시각화, 커스터마이징 | 108,000+ stars |
| **Graphology** | 경량, 모던 그래프 라이브러리 | 1,000+ stars |
| **Cosmos** | WebGL, 100,000+ 노드 렌더링 | 신규 |

**K8s 웹앱 배포 예시:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-visualizer
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: visualizer
        image: graph-viz:latest
        ports:
        - containerPort: 3000
```

---

## 5. 기술 선택 가이드

### 5.1 사용 사례별 추천

```
┌─────────────────────────────────────────────────────────────────┐
│                     그래프 기술 선택 플로우차트                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Q: 주요 사용 사례는?                                             │
│  │                                                               │
│  ├─▶ GraphRAG/GenAI                                              │
│  │   └─▶ FalkorDB, nano-graphrag, Microsoft GraphRAG             │
│  │                                                               │
│  ├─▶ 대규모 분산 처리 (수십억 노드)                                │
│  │   └─▶ JanusGraph, NebulaGraph                                 │
│  │                                                               │
│  ├─▶ 실시간 분석/스트리밍                                         │
│  │   └─▶ Memgraph                                                │
│  │                                                               │
│  ├─▶ GraphQL API                                                 │
│  │   └─▶ Dgraph                                                  │
│  │                                                               │
│  ├─▶ 기존 PostgreSQL 활용                                        │
│  │   └─▶ Apache AGE                                              │
│  │                                                               │
│  ├─▶ 멀티모델 (그래프+문서+...)                                   │
│  │   └─▶ ArangoDB, ArcadeDB                                      │
│  │                                                               │
│  └─▶ 지식 표현/온톨로지/추론                                      │
│      └─▶ TypeDB                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 라이선스별 분류

#### 완전 오픈소스 (Apache 2.0 / MIT)

| 기술 | 라이선스 | 상용 사용 |
|------|----------|-----------|
| JanusGraph | Apache 2.0 | 제한 없음 |
| NebulaGraph | Apache 2.0 | 제한 없음 |
| Dgraph (v25+) | Apache 2.0 | 제한 없음 |
| ArcadeDB | Apache 2.0 | 제한 없음 |
| Apache AGE | Apache 2.0 | 제한 없음 |
| Apache TinkerPop | Apache 2.0 | 제한 없음 |
| nano-graphrag | MIT | 제한 없음 |
| LlamaIndex | MIT | 제한 없음 |
| LangChain | MIT | 제한 없음 |

#### 조건부 오픈소스

| 기술 | 라이선스 | 제한사항 |
|------|----------|----------|
| Memgraph | BSL 1.1 | 3년 후 Apache 2.0 전환 |
| ArangoDB | BSL 1.1 | 프로덕션 사용 제한 |
| FalkorDB | SSPLv1 | SaaS 제공 시 소스 공개 |
| TypeDB | AGPL-3.0 | 수정 시 소스 공개 |

### 5.3 K8s 성숙도별 분류

#### Tier 1: 프로덕션 Ready (공식 Operator/Helm)

- NebulaGraph (Operator + Helm)
- Memgraph (Helm + HA)
- Dgraph (Operator + Helm)
- ArangoDB (Operator)

#### Tier 2: 안정적 배포 가능 (Helm Chart)

- JanusGraph (Helm)
- FalkorDB (Helm + KubeBlocks)
- ArcadeDB (Helm)

#### Tier 3: 수동 구성 필요

- TypeDB
- Apache AGE (CloudNativePG)
- Kuzu (임베디드)

### 5.4 조합 추천

#### GraphRAG 풀스택

```
┌─────────────────────────────────────────┐
│         GraphRAG 풀스택 아키텍처          │
├─────────────────────────────────────────┤
│                                          │
│   [LlamaIndex / LangChain]              │
│            │                             │
│            ▼                             │
│   [nano-graphrag / MS GraphRAG]         │
│            │                             │
│            ▼                             │
│   [FalkorDB / Memgraph / Neo4j]         │
│            │                             │
│            ▼                             │
│   [Kubernetes Cluster]                   │
│                                          │
└─────────────────────────────────────────┘
```

**추천 조합:**
1. **경량 GraphRAG**: nano-graphrag + FalkorDB + LlamaIndex
2. **엔터프라이즈 GraphRAG**: MS GraphRAG + NebulaGraph + LangChain
3. **실시간 분석**: Memgraph + LangChain + Cytoscape.js

---

## 6. 참고 자료

### 공식 문서

- [JanusGraph Documentation](https://docs.janusgraph.org/)
- [NebulaGraph Documentation](https://docs.nebula-graph.io/)
- [Dgraph Documentation](https://dgraph.io/docs/)
- [Memgraph Documentation](https://memgraph.com/docs/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [ArangoDB Documentation](https://docs.arangodb.com/)
- [ArcadeDB Documentation](https://docs.arcadedb.com/)
- [Apache AGE Documentation](https://age.apache.org/docs/)
- [Apache TinkerPop Documentation](https://tinkerpop.apache.org/docs/current/)
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)

### GitHub 저장소

- [JanusGraph](https://github.com/JanusGraph/janusgraph)
- [NebulaGraph](https://github.com/vesoft-inc/nebula)
- [Dgraph](https://github.com/dgraph-io/dgraph)
- [Memgraph Helm Charts](https://github.com/memgraph/helm-charts)
- [FalkorDB](https://github.com/FalkorDB/FalkorDB)
- [ArcadeDB](https://github.com/ArcadeData/arcadedb)
- [Apache AGE](https://github.com/apache/age)
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [LangChain](https://github.com/langchain-ai/langchain)

### 추가 리소스

- [Awesome GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
- [Awesome Knowledge Graph](https://github.com/totogo/awesome-knowledge-graph)
- [GeeksforGeeks - Top 10 Open Source Graph Databases](https://www.geeksforgeeks.org/blogs/open-source-graph-databases/)
- [PuppyGraph - 7 Best Open Source Graph Databases](https://www.puppygraph.com/blog/open-source-graph-databases)

---

*문서 작성: 2026-01-24*
*버전: 1.0*
