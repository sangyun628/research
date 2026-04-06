# LlamaIndex 개발자 가이드 및 Agno 연동 방안

> **작성일:** 2026년 1월 26일
> **목적:** LlamaIndex를 활용한 RAG 시스템 구축 실무 가이드 및 Agno 프레임워크 연동 전략

---

## 목차

1. [LlamaIndex 개요](#1-llamaindex-개요)
2. [설치 및 환경 설정](#2-설치-및-환경-설정)
3. [핵심 컴포넌트 상세](#3-핵심-컴포넌트-상세)
4. [실전 RAG 파이프라인 구축](#4-실전-rag-파이프라인-구축)
5. [Agent 시스템 구축](#5-agent-시스템-구축)
6. [Agno 프레임워크 소개](#6-agno-프레임워크-소개)
7. [LlamaIndex + Agno 연동 전략](#7-llamaindex--agno-연동-전략)
8. [프로덕션 배포 가이드](#8-프로덕션-배포-가이드)
9. [참고 자료](#9-참고-자료)

---

## 1. LlamaIndex 개요

### 1.1 LlamaIndex란?

[LlamaIndex](https://github.com/run-llama/llama_index)는 **LLM 기반 애플리케이션을 위한 데이터 프레임워크**입니다. 문서, 데이터베이스, API 등 다양한 데이터 소스를 LLM과 연결하여 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 데 특화되어 있습니다.

**주요 통계 (2025년 기준):**
- GitHub Stars: 44,600+
- 라이선스: MIT
- Python 버전: >=3.9, <4.0
- 최신 버전: 0.14.7

### 1.2 핵심 철학

```
┌─────────────────────────────────────────────────────────────┐
│                    LlamaIndex Philosophy                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   "데이터를 LLM이 이해할 수 있는 형태로 구조화하고,          │
│    효율적으로 검색하여 정확한 응답을 생성한다"               │
│                                                              │
│   Data Ingestion → Indexing → Retrieval → Response          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 아키텍처 개요

```
┌───────────────────────────────────────────────────────────────────┐
│                     LlamaIndex Architecture                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │   Data      │    │   Index     │    │   Query Engine      │   │
│  │  Connectors │───▶│   Types     │───▶│   & Retrievers      │   │
│  │  (Readers)  │    │             │    │                     │   │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘   │
│        │                  │                       │               │
│        │                  │                       ▼               │
│        │                  │            ┌─────────────────────┐   │
│        │                  │            │  Response           │   │
│        │                  │            │  Synthesizer        │   │
│        │                  │            └─────────────────────┘   │
│        │                  │                       │               │
│        ▼                  ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      Workflows / Agents                      │ │
│  │   (Event-driven orchestration for complex AI processes)      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. 설치 및 환경 설정

### 2.1 설치 방법

#### 방법 1: Starter 패키지 (권장 - 빠른 시작)

```bash
pip install llama-index
```

이 패키지에는 핵심 라이브러리와 자주 사용되는 통합 패키지가 포함됩니다.

#### 방법 2: 커스텀 설치 (À la carte)

프로덕션 환경에서는 필요한 패키지만 선택적으로 설치하는 것이 권장됩니다:

```bash
# 핵심 패키지
pip install llama-index-core

# LLM 프로바이더 (필요한 것만 선택)
pip install llama-index-llms-openai          # OpenAI
pip install llama-index-llms-anthropic       # Anthropic
pip install llama-index-llms-ollama          # Ollama (로컬)

# 임베딩 모델
pip install llama-index-embeddings-openai    # OpenAI Embeddings
pip install llama-index-embeddings-huggingface  # HuggingFace (로컬)

# Vector Store
pip install llama-index-vector-stores-chroma    # ChromaDB
pip install llama-index-vector-stores-qdrant    # Qdrant
pip install llama-index-vector-stores-milvus    # Milvus

# 파일 리더
pip install llama-index-readers-file         # 기본 파일 리더
```

#### 방법 3: 전체 설치 (개발/테스트용)

```bash
pip install llama-index[all]
```

### 2.2 환경 변수 설정

```bash
# .env 파일 또는 환경 변수
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUGGINGFACE_TOKEN="hf_..."
```

### 2.3 Settings 객체 구성

LlamaIndex의 `Settings`는 전역 싱글톤 객체로, 애플리케이션 전체에서 사용되는 기본 설정을 관리합니다.

#### OpenAI 기반 설정

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# LLM 설정
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=4096
)

# 임베딩 모델 설정
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    dimensions=1536
)

# 청킹 설정
Settings.chunk_size = 1024
Settings.chunk_overlap = 200
```

#### 로컬 LLM 설정 (Ollama)

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ollama 로컬 LLM
Settings.llm = Ollama(
    model="llama3.1:8b",
    request_timeout=360.0,
    base_url="http://localhost:11434"
)

# HuggingFace 로컬 임베딩
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    cache_folder="./models"
)

# 컨텍스트 윈도우 설정
Settings.context_window = 4096
Settings.num_output = 512
```

#### Anthropic Claude 설정

```python
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = Anthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=4096
)

# Anthropic은 임베딩 모델이 없으므로 OpenAI 사용
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)
```

### 2.4 프로젝트 구조 권장안

```
my-rag-project/
├── config/
│   ├── __init__.py
│   ├── settings.py          # LlamaIndex Settings 구성
│   └── prompts.py           # 커스텀 프롬프트 템플릿
├── data/
│   ├── raw/                  # 원본 문서
│   ├── processed/            # 전처리된 문서
│   └── indexes/              # 저장된 인덱스
├── src/
│   ├── __init__.py
│   ├── ingestion/            # 데이터 수집 모듈
│   │   ├── loaders.py
│   │   └── transformers.py
│   ├── indexing/             # 인덱싱 모듈
│   │   └── index_builder.py
│   ├── retrieval/            # 검색 모듈
│   │   ├── retrievers.py
│   │   └── rerankers.py
│   ├── agents/               # 에이전트 모듈
│   │   └── rag_agent.py
│   └── api/                  # API 서버
│       └── main.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

---

## 3. 핵심 컴포넌트 상세

### 3.1 Data Connectors (Readers)

데이터 커넥터는 다양한 소스에서 문서를 로드합니다.

#### 기본 파일 리더

```python
from llama_index.core import SimpleDirectoryReader

# 디렉토리 전체 로드
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    required_exts=[".pdf", ".docx", ".txt", ".md"]
).load_data()

print(f"Loaded {len(documents)} documents")
```

#### 특정 파일 타입별 리더

```python
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    MarkdownReader,
    HTMLReader
)

# PDF 전용 리더
pdf_reader = PDFReader()
pdf_docs = pdf_reader.load_data(file="./report.pdf")

# 여러 리더 조합
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./documents",
    file_extractor={
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".md": MarkdownReader(),
        ".html": HTMLReader()
    }
)
documents = reader.load_data()
```

#### 웹/API 리더

```python
# 웹 페이지 로드
from llama_index.readers.web import SimpleWebPageReader

web_docs = SimpleWebPageReader(
    html_to_text=True
).load_data(
    urls=["https://docs.example.com/guide"]
)

# Notion 데이터 로드
from llama_index.readers.notion import NotionPageReader

notion_reader = NotionPageReader(integration_token="secret_...")
notion_docs = notion_reader.load_data(page_ids=["page_id_1", "page_id_2"])

# 데이터베이스 로드
from llama_index.readers.database import DatabaseReader

db_reader = DatabaseReader(
    sql_database="postgresql://user:pass@localhost/db"
)
db_docs = db_reader.load_data(query="SELECT * FROM articles")
```

### 3.2 Node Parser (Chunking)

문서를 검색 가능한 단위(Node)로 분할합니다.

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)

# 방법 1: 문장 기반 분할 (가장 일반적)
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    separator=" "
)
nodes = splitter.get_nodes_from_documents(documents)

# 방법 2: 시맨틱 기반 분할 (의미 단위)
from llama_index.embeddings.openai import OpenAIEmbedding

semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding()
)
semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

# 방법 3: 계층적 분할 (다양한 granularity)
hierarchical_splitter = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 대/중/소 청크
)
hierarchical_nodes = hierarchical_splitter.get_nodes_from_documents(documents)
```

### 3.3 Index Types

LlamaIndex는 여러 종류의 인덱스를 제공합니다.

#### VectorStoreIndex (가장 일반적)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ChromaDB 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# Vector Store 생성
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 인덱스 생성
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# 인덱스 저장
index.storage_context.persist(persist_dir="./storage")

# 인덱스 로드
from llama_index.core import load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./storage",
    vector_store=vector_store
)
loaded_index = load_index_from_storage(storage_context)
```

#### SummaryIndex (문서 요약용)

```python
from llama_index.core import SummaryIndex

# 전체 문서에 대한 요약 인덱스
summary_index = SummaryIndex.from_documents(documents)

# 요약 쿼리
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"
)
response = summary_query_engine.query("이 문서들의 주요 내용을 요약해주세요")
```

#### KnowledgeGraphIndex (관계 추론용)

```python
from llama_index.core import KnowledgeGraphIndex

# Knowledge Graph 인덱스 생성
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,
    include_embeddings=True
)

# 그래프 시각화
from pyvis.network import Network

g = kg_index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line")
net.from_nx(g)
net.show("knowledge_graph.html")
```

### 3.4 Query Engine

쿼리 엔진은 사용자 질문을 처리하고 응답을 생성합니다.

#### 기본 Query Engine

```python
# 가장 간단한 형태
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

#### 고급 Query Engine 설정

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor
)
from llama_index.postprocessor.cohere_rerank import CohereRerank

# Retriever 설정
retriever = index.as_retriever(
    similarity_top_k=10,  # 상위 10개 검색
    vector_store_query_mode="hybrid"  # 하이브리드 검색
)

# Post-processor 설정 (Reranking)
cohere_rerank = CohereRerank(
    api_key="...",
    top_n=5
)

similarity_processor = SimilarityPostprocessor(
    similarity_cutoff=0.7
)

# Query Engine 조합
query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[similarity_processor, cohere_rerank],
    response_mode="compact",  # compact, refine, tree_summarize
    streaming=True
)
```

#### Router Query Engine (다중 인덱스)

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# 여러 인덱스에 대한 Query Engine Tool 생성
tools = [
    QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(),
        description="기술 문서 검색에 사용. 구체적인 기술 질문에 적합"
    ),
    QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(),
        description="문서 요약에 사용. 전체적인 개요 파악에 적합"
    ),
    QueryEngineTool.from_defaults(
        query_engine=kg_index.as_query_engine(),
        description="관계 추론에 사용. 엔티티 간 관계 파악에 적합"
    )
]

# Router Query Engine
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools
)

response = router_engine.query("회사의 조직 구조와 각 부서의 관계는?")
```

### 3.5 Chat Engine

대화형 인터페이스를 제공합니다.

```python
from llama_index.core.memory import ChatMemoryBuffer

# 메모리 설정
memory = ChatMemoryBuffer.from_defaults(
    token_limit=40000
)

# Chat Engine 생성
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",  # condense_question, context, react
    memory=memory,
    system_prompt="""
    당신은 기술 문서 전문가입니다.
    사용자의 질문에 친절하고 정확하게 답변해주세요.
    답변은 한국어로 해주세요.
    """
)

# 대화
response = chat_engine.chat("LlamaIndex의 주요 기능은 무엇인가요?")
print(response)

# 후속 질문 (컨텍스트 유지)
response = chat_engine.chat("그 중에서 가장 중요한 것은?")
print(response)

# 대화 기록 확인
print(chat_engine.chat_history)

# 대화 초기화
chat_engine.reset()
```

---

## 4. 실전 RAG 파이프라인 구축

### 4.1 기본 RAG 파이프라인

```python
"""
기본 RAG 파이프라인 구현
"""
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class BasicRAGPipeline:
    def __init__(
        self,
        data_dir: str = "./data",
        collection_name: str = "documents",
        qdrant_url: str = "http://localhost:6333"
    ):
        # Settings 구성
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 200

        self.data_dir = data_dir
        self.collection_name = collection_name

        # Qdrant 클라이언트
        self.qdrant_client = QdrantClient(url=qdrant_url)

        self.index = None
        self.query_engine = None

    def ingest(self) -> int:
        """문서 수집 및 인덱싱"""
        # 문서 로드
        documents = SimpleDirectoryReader(
            input_dir=self.data_dir,
            recursive=True
        ).load_data()

        # Vector Store 설정
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # 인덱스 생성
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        # Query Engine 생성
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )

        return len(documents)

    def query(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        if self.query_engine is None:
            raise ValueError("먼저 ingest()를 실행하세요")

        response = self.query_engine.query(question)
        return str(response)

    def query_with_sources(self, question: str) -> dict:
        """답변과 출처 함께 반환"""
        if self.query_engine is None:
            raise ValueError("먼저 ingest()를 실행하세요")

        response = self.query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.text[:200] + "...",
                "score": node.score,
                "metadata": node.metadata
            })

        return {
            "answer": str(response),
            "sources": sources
        }


# 사용 예시
if __name__ == "__main__":
    pipeline = BasicRAGPipeline(data_dir="./documents")

    # 문서 수집
    doc_count = pipeline.ingest()
    print(f"Indexed {doc_count} documents")

    # 질문
    result = pipeline.query_with_sources("프로젝트의 주요 목표는 무엇인가요?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents")
```

### 4.2 Hybrid Search RAG 파이프라인

```python
"""
Hybrid Search (Keyword + Vector) RAG 파이프라인
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

class HybridRAGPipeline:
    def __init__(self, documents, similarity_top_k: int = 10):
        self.documents = documents
        self.similarity_top_k = similarity_top_k

        # 인덱스 생성
        self.index = VectorStoreIndex.from_documents(documents)

        # Retrievers 생성
        self._setup_retrievers()

    def _setup_retrievers(self):
        """Hybrid Retriever 설정"""
        # Vector Retriever
        vector_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        # BM25 Retriever (Keyword 기반)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.index.docstore.docs.values(),
            similarity_top_k=self.similarity_top_k
        )

        # Fusion Retriever (Reciprocal Rank Fusion)
        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.6, 0.4],  # Vector 60%, BM25 40%
            num_queries=1,  # 원본 쿼리만 사용
            similarity_top_k=self.similarity_top_k
        )

        # Response Synthesizer
        self.response_synthesizer = get_response_synthesizer(
            response_mode="compact"
        )

        # Query Engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer
        )

    def query(self, question: str) -> str:
        """Hybrid Search 기반 질문 응답"""
        response = self.query_engine.query(question)
        return str(response)


# 사용 예시
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
hybrid_rag = HybridRAGPipeline(documents)
answer = hybrid_rag.query("시스템 아키텍처에 대해 설명해주세요")
```

### 4.3 Advanced RAG with Reranking

```python
"""
Reranking을 포함한 고급 RAG 파이프라인
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    MetadataReplacementPostProcessor
)
from llama_index.core.query_engine import RetrieverQueryEngine

class AdvancedRAGPipeline:
    def __init__(
        self,
        documents,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_top_k: int = 20,
        final_top_k: int = 5
    ):
        self.index = VectorStoreIndex.from_documents(documents)
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

        # Reranker 설정
        self.reranker = SentenceTransformerRerank(
            model=rerank_model,
            top_n=final_top_k
        )

        self._setup_query_engine()

    def _setup_query_engine(self):
        """Query Engine with Reranking 설정"""
        # 초기 검색: 많은 후보 (top_k=20)
        retriever = self.index.as_retriever(
            similarity_top_k=self.initial_top_k
        )

        # Query Engine with Reranking
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[self.reranker],
            response_mode="compact"
        )

    def query(self, question: str) -> dict:
        """Reranking을 적용한 질문 응답"""
        response = self.query_engine.query(question)

        return {
            "answer": str(response),
            "source_nodes": [
                {
                    "text": node.text[:300],
                    "score": node.score,
                    "file_name": node.metadata.get("file_name", "unknown")
                }
                for node in response.source_nodes
            ]
        }


# 사용 예시
advanced_rag = AdvancedRAGPipeline(
    documents,
    initial_top_k=20,  # 초기 검색
    final_top_k=5      # Reranking 후
)
result = advanced_rag.query("데이터베이스 스키마 설계 원칙은?")
```

---

## 5. Agent 시스템 구축

### 5.1 ReAct Agent 기본

```python
"""
LlamaIndex ReAct Agent 구현
"""
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

# 커스텀 함수를 Tool로 변환
def calculate_sum(a: int, b: int) -> int:
    """두 숫자의 합을 계산합니다."""
    return a + b

def get_current_weather(city: str) -> str:
    """지정된 도시의 현재 날씨를 반환합니다."""
    # 실제로는 API 호출
    return f"{city}의 현재 날씨: 맑음, 15°C"

# Function Tools 생성
calc_tool = FunctionTool.from_defaults(
    fn=calculate_sum,
    name="calculator",
    description="두 숫자를 더합니다"
)

weather_tool = FunctionTool.from_defaults(
    fn=get_current_weather,
    name="weather",
    description="도시의 현재 날씨를 조회합니다"
)

# RAG Query Engine Tool
rag_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="회사 내부 문서를 검색합니다. 정책, 가이드라인, 프로세스에 대한 질문에 사용하세요."
)

# ReAct Agent 생성
agent = ReActAgent.from_tools(
    tools=[calc_tool, weather_tool, rag_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=10
)

# Agent 실행
response = agent.chat("서울의 날씨를 알려주고, 15와 27을 더해줘")
print(response)
```

### 5.2 FunctionAgent (Function Calling 지원 LLM용)

```python
"""
Function Calling 기반 Agent
"""
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from typing import List

def search_database(query: str, filters: dict = None) -> List[dict]:
    """
    데이터베이스를 검색합니다.

    Args:
        query: 검색 쿼리
        filters: 필터 조건 (optional)

    Returns:
        검색 결과 리스트
    """
    # 실제 DB 검색 로직
    return [{"id": 1, "title": "Result 1", "content": "..."}]

def send_email(to: str, subject: str, body: str) -> bool:
    """
    이메일을 전송합니다.

    Args:
        to: 수신자 이메일
        subject: 제목
        body: 본문

    Returns:
        전송 성공 여부
    """
    # 실제 이메일 전송 로직
    print(f"Email sent to {to}")
    return True

# Tools 생성
tools = [
    FunctionTool.from_defaults(fn=search_database),
    FunctionTool.from_defaults(fn=send_email),
    QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(),
        name="docs_search",
        description="내부 문서 검색"
    )
]

# Function Calling Agent
agent = FunctionCallingAgent.from_tools(
    tools=tools,
    llm=Settings.llm,
    verbose=True,
    system_prompt="""
    당신은 업무 자동화 에이전트입니다.
    사용자의 요청을 분석하고 적절한 도구를 사용하여 작업을 수행하세요.
    """
)

response = agent.chat(
    "내부 문서에서 '휴가 정책'을 검색하고, 그 내용을 hr@company.com으로 보내줘"
)
```

### 5.3 Multi-Agent Workflow

```python
"""
LlamaIndex AgentWorkflow를 사용한 멀티 에이전트 시스템
"""
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool

# 전문가 에이전트 1: 연구원
def search_papers(topic: str) -> str:
    """학술 논문을 검색합니다."""
    return f"'{topic}'에 관한 최신 논문 3편을 찾았습니다..."

researcher_tools = [FunctionTool.from_defaults(fn=search_papers)]
researcher_agent = FunctionAgent(
    name="researcher",
    description="학술 연구 및 논문 검색 전문가",
    tools=researcher_tools,
    system_prompt="당신은 학술 연구 전문가입니다. 논문을 검색하고 분석합니다."
)

# 전문가 에이전트 2: 분석가
def analyze_data(data: str) -> str:
    """데이터를 분석합니다."""
    return f"데이터 분석 결과: {data}에 대한 인사이트..."

analyst_tools = [FunctionTool.from_defaults(fn=analyze_data)]
analyst_agent = FunctionAgent(
    name="analyst",
    description="데이터 분석 및 인사이트 도출 전문가",
    tools=analyst_tools,
    system_prompt="당신은 데이터 분석 전문가입니다."
)

# 전문가 에이전트 3: 작성자
def write_report(content: str, format: str = "markdown") -> str:
    """보고서를 작성합니다."""
    return f"# 분석 보고서\n\n{content}"

writer_tools = [FunctionTool.from_defaults(fn=write_report)]
writer_agent = FunctionAgent(
    name="writer",
    description="보고서 및 문서 작성 전문가",
    tools=writer_tools,
    system_prompt="당신은 기술 문서 작성 전문가입니다."
)

# AgentWorkflow 생성
workflow = AgentWorkflow(
    agents=[researcher_agent, analyst_agent, writer_agent],
    root_agent="researcher",  # 시작 에이전트
    initial_state={}
)

# 워크플로우 실행
async def run_workflow():
    result = await workflow.run(
        "AI 트렌드에 대해 조사하고 분석한 후 보고서를 작성해줘"
    )
    return result

import asyncio
result = asyncio.run(run_workflow())
print(result)
```

### 5.4 Agent Memory 관리

```python
"""
에이전트 메모리 관리
"""
from llama_index.core.memory import (
    ChatMemoryBuffer,
    VectorMemory,
    SimpleComposableMemory
)
from llama_index.core.agent import ReActAgent

# 방법 1: 기본 Chat Memory
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=40000
)

# 방법 2: Vector Memory (장기 기억)
vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # 기본 in-memory
    embed_model=Settings.embed_model,
    retriever_kwargs={"similarity_top_k": 5}
)

# 방법 3: Composable Memory (단기 + 장기)
composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory,
    secondary_memory_sources=[vector_memory]
)

# Agent with Memory
agent = ReActAgent.from_tools(
    tools=tools,
    llm=Settings.llm,
    memory=composable_memory,
    verbose=True
)

# 대화 (메모리 축적)
agent.chat("내 이름은 김철수야")
agent.chat("나는 AI 엔지니어야")

# 메모리 기반 응답
response = agent.chat("내 이름과 직업이 뭐라고 했지?")
print(response)  # 이전 대화 내용 기억
```

---

## 6. Agno 프레임워크 소개

### 6.1 Agno 개요

[Agno](https://github.com/agno-agi/agno)는 2025년 1월에 출시된 **고성능 멀티 에이전트 프레임워크**입니다. 이전에 Phidata로 알려졌으며, "순수함(pure)"을 뜻하는 그리스어에서 이름을 따왔습니다.

**핵심 특징:**
- **성능:** LangGraph 대비 529배 빠른 인스턴스화, 24배 낮은 메모리 사용
- **철학:** 그래프, 체인 없이 순수 Python만으로 구현
- **통합:** 40+ 모델, 20+ 프로바이더, 100+ 도구 지원

### 6.2 Agno 설치 및 기본 사용

```bash
pip install agno
```

```python
"""
Agno 기본 Agent 생성
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# 기본 Agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="당신은 친절한 AI 어시스턴트입니다.",
    instructions=["한국어로 답변하세요", "간결하게 답변하세요"],
    markdown=True
)

# 실행
agent.print_response("LlamaIndex와 Agno의 차이점은?", stream=True)
```

### 6.3 Agno의 Knowledge (RAG) 기능

```python
"""
Agno 내장 RAG 기능
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder

# Knowledge Base 설정
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://example.com/document.pdf"],
    vector_db=LanceDb(
        uri="./lancedb",
        table_name="documents",
        embedder=OpenAIEmbedder(id="text-embedding-3-small")
    )
)

# Agent with Knowledge
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    search_knowledge=True,  # RAG 검색 활성화
    add_knowledge_to_context=True,
    markdown=True
)

# Knowledge 로드 (최초 1회)
if agent.knowledge:
    agent.knowledge.load()

# 질문
agent.print_response("문서의 주요 내용은?", stream=True)
```

### 6.4 Agno Custom Tools

```python
"""
Agno 커스텀 도구 생성
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from typing import Any, Callable, Dict

# 훅 함수 (로깅용)
def logger_hook(
    function_name: str,
    function_call: Callable,
    arguments: Dict[str, Any]
):
    """도구 실행 전후 로깅"""
    print(f"[LOG] Calling {function_name} with {arguments}")
    result = function_call(**arguments)
    print(f"[LOG] Result: {result}")
    return result

# 커스텀 도구 1: 기본 함수
def get_stock_price(symbol: str) -> str:
    """주식 가격을 조회합니다."""
    # 실제로는 API 호출
    prices = {"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 378.90}
    price = prices.get(symbol.upper(), "N/A")
    return f"{symbol}: ${price}"

# 커스텀 도구 2: 데코레이터 사용
@tool(
    name="search_database",
    description="데이터베이스를 검색합니다",
    tool_hooks=[logger_hook],
    cache_results=True,
    cache_ttl=300  # 5분 캐시
)
def search_db(query: str, limit: int = 10) -> list:
    """데이터베이스 검색"""
    return [{"id": i, "content": f"Result {i}"} for i in range(limit)]

# Agent with Custom Tools
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[get_stock_price, search_db],
    show_tool_calls=True,
    markdown=True
)

agent.print_response("AAPL 주가를 알려주고, DB에서 최근 5개 결과를 검색해줘")
```

### 6.5 Agno Multi-Agent Team

```python
"""
Agno 멀티 에이전트 팀
"""
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat

# 전문가 에이전트들
researcher = Agent(
    name="Researcher",
    model=OpenAIChat(id="gpt-4o"),
    role="연구 전문가",
    instructions=["학술 자료를 검색하고 분석합니다"]
)

analyst = Agent(
    name="Analyst",
    model=OpenAIChat(id="gpt-4o"),
    role="데이터 분석가",
    instructions=["데이터를 분석하고 인사이트를 도출합니다"]
)

writer = Agent(
    name="Writer",
    model=OpenAIChat(id="gpt-4o"),
    role="보고서 작성자",
    instructions=["분석 결과를 바탕으로 보고서를 작성합니다"]
)

# 팀 구성
team = Team(
    name="Research Team",
    agents=[researcher, analyst, writer],
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "1. Researcher가 먼저 자료를 조사합니다",
        "2. Analyst가 데이터를 분석합니다",
        "3. Writer가 최종 보고서를 작성합니다"
    ]
)

# 팀 실행
team.print_response(
    "2025년 AI 기술 트렌드에 대한 분석 보고서를 작성해주세요",
    stream=True
)
```

---

## 7. LlamaIndex + Agno 연동 전략

### 7.1 연동 아키텍처 개요

Agno는 자체 RAG 기능을 제공하지만, LlamaIndex의 강력한 인덱싱 및 검색 기능을 활용하면 더 정교한 RAG 파이프라인을 구축할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                 LlamaIndex + Agno Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Agno Agent Layer                      │   │
│  │  • Agent Orchestration                                   │   │
│  │  • Multi-Agent Teams                                     │   │
│  │  • Tool Management                                       │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                          │                                      │
│                    Custom Tools                                 │
│                          │                                      │
│  ┌───────────────────────▼─────────────────────────────────┐   │
│  │               LlamaIndex RAG Layer                       │   │
│  │  • Advanced Indexing (Vector, Graph, Summary)            │   │
│  │  • Hybrid Search + Reranking                            │   │
│  │  • Query Engine + Response Synthesis                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 LlamaIndex Query Engine을 Agno Tool로 래핑

```python
"""
LlamaIndex Query Engine을 Agno Tool로 사용
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# LlamaIndex 설정
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# LlamaIndex 인덱스 및 Query Engine 생성
documents = SimpleDirectoryReader("./data/docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)


# LlamaIndex Query Engine을 Agno Tool로 래핑
@tool(
    name="search_documents",
    description="내부 문서를 검색합니다. 정책, 가이드라인, 기술 문서에 대한 질문에 사용하세요."
)
def search_documents(query: str) -> str:
    """
    LlamaIndex를 사용하여 문서를 검색합니다.

    Args:
        query: 검색 질문

    Returns:
        검색 결과 및 답변
    """
    response = query_engine.query(query)

    # 출처 정보 포함
    sources = []
    for node in response.source_nodes:
        source_info = node.metadata.get("file_name", "unknown")
        sources.append(source_info)

    result = f"답변: {response}\n\n출처: {', '.join(set(sources))}"
    return result


# Agno Agent with LlamaIndex Tool
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[search_documents],
    instructions=[
        "사용자의 질문에 답변할 때 먼저 문서를 검색하세요",
        "검색 결과를 바탕으로 정확하게 답변하세요",
        "출처를 명시하세요"
    ],
    show_tool_calls=True,
    markdown=True
)

# 실행
agent.print_response("휴가 정책에 대해 알려주세요", stream=True)
```

### 7.3 고급 연동: Hybrid RAG Tool

```python
"""
LlamaIndex Hybrid RAG를 Agno Tool로 사용
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from typing import List, Dict, Any

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank


class LlamaIndexRAGTool:
    """LlamaIndex 기반 고급 RAG Tool 클래스"""

    def __init__(self, documents, rerank_top_n: int = 5):
        # 인덱스 생성
        self.index = VectorStoreIndex.from_documents(documents)

        # Hybrid Retriever 설정
        vector_retriever = self.index.as_retriever(similarity_top_k=20)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=list(self.index.docstore.docs.values()),
            similarity_top_k=20
        )

        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.6, 0.4],
            num_queries=1,
            similarity_top_k=20
        )

        # Reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=rerank_top_n
        )

        # Query Engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            node_postprocessors=[self.reranker]
        )

    def search(self, query: str) -> Dict[str, Any]:
        """검색 수행"""
        response = self.query_engine.query(query)

        return {
            "answer": str(response),
            "sources": [
                {
                    "text": node.text[:200],
                    "score": node.score,
                    "metadata": node.metadata
                }
                for node in response.source_nodes
            ]
        }


# RAG Tool 인스턴스 생성
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()
rag_tool_instance = LlamaIndexRAGTool(documents)


# Agno Tool로 래핑
@tool(
    name="advanced_search",
    description="고급 하이브리드 검색을 수행합니다. 복잡한 질문이나 정밀한 검색이 필요할 때 사용하세요."
)
def advanced_search(query: str, include_sources: bool = True) -> str:
    """
    LlamaIndex Hybrid RAG를 사용한 고급 검색

    Args:
        query: 검색 질문
        include_sources: 출처 포함 여부

    Returns:
        검색 결과
    """
    result = rag_tool_instance.search(query)

    output = f"답변: {result['answer']}"

    if include_sources and result['sources']:
        output += "\n\n### 출처:"
        for i, source in enumerate(result['sources'], 1):
            file_name = source['metadata'].get('file_name', 'unknown')
            output += f"\n{i}. {file_name} (관련도: {source['score']:.2f})"

    return output


# Multi-Tool Agent
@tool(name="web_search", description="웹에서 최신 정보를 검색합니다")
def web_search(query: str) -> str:
    """웹 검색 (실제로는 API 호출)"""
    return f"웹 검색 결과: '{query}'에 대한 최신 정보..."

@tool(name="calculator", description="수학 계산을 수행합니다")
def calculator(expression: str) -> str:
    """계산기"""
    try:
        result = eval(expression)
        return f"계산 결과: {expression} = {result}"
    except:
        return "계산 오류"


# Agno Agent with Multiple Tools
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[advanced_search, web_search, calculator],
    instructions=[
        "내부 문서 질문은 advanced_search를 사용하세요",
        "최신 정보는 web_search를 사용하세요",
        "계산이 필요하면 calculator를 사용하세요",
        "항상 출처를 명시하세요"
    ],
    show_tool_calls=True,
    markdown=True
)

# 실행
agent.print_response(
    "내부 문서에서 보안 정책을 찾고, 최신 보안 트렌드와 비교해주세요",
    stream=True
)
```

### 7.4 LlamaIndex Agent를 Agno Tool로 사용

```python
"""
LlamaIndex Agent를 Agno의 하위 에이전트로 사용
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool


# LlamaIndex 내부 도구들
def search_code(query: str) -> str:
    """코드베이스를 검색합니다."""
    return f"코드 검색 결과: {query}에 관련된 파일들..."

def analyze_logs(log_path: str) -> str:
    """로그 파일을 분석합니다."""
    return f"{log_path} 분석 결과: 에러 3건, 경고 15건..."

# LlamaIndex Agent 생성
llamaindex_agent = ReActAgent.from_tools(
    tools=[
        FunctionTool.from_defaults(fn=search_code),
        FunctionTool.from_defaults(fn=analyze_logs),
        QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(),
            name="docs",
            description="기술 문서 검색"
        )
    ],
    llm=Settings.llm,
    verbose=False
)


# LlamaIndex Agent를 Agno Tool로 래핑
@tool(
    name="technical_assistant",
    description="기술적인 질문에 답변합니다. 코드 검색, 로그 분석, 기술 문서 조회가 가능합니다."
)
def technical_assistant(question: str) -> str:
    """
    LlamaIndex Agent를 사용한 기술 지원

    Args:
        question: 기술 관련 질문

    Returns:
        답변
    """
    response = llamaindex_agent.chat(question)
    return str(response)


# Agno 메인 Agent (Orchestrator)
main_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[technical_assistant],
    instructions=[
        "기술적인 질문은 technical_assistant에게 위임하세요",
        "결과를 종합하여 사용자에게 명확하게 전달하세요"
    ],
    show_tool_calls=True,
    markdown=True
)

# 실행
main_agent.print_response(
    "최근 에러 로그를 분석하고, 관련된 코드와 문서를 찾아주세요",
    stream=True
)
```

### 7.5 실전 통합 아키텍처: Enterprise RAG Agent

```python
"""
엔터프라이즈급 LlamaIndex + Agno 통합 에이전트
"""
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools import tool, Toolkit
from typing import Optional, List, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    Settings,
    StorageContext
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class EnterpriseRAGToolkit(Toolkit):
    """
    LlamaIndex 기반 엔터프라이즈 RAG Toolkit
    Agno의 Toolkit 클래스를 상속하여 구현
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "enterprise_docs"
    ):
        super().__init__(name="enterprise_rag")

        # Qdrant 클라이언트
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # 인덱스들
        self.vector_index = None
        self.summary_index = None
        self.router_engine = None

        # 도구 등록
        self.register(self.search_documents)
        self.register(self.summarize_documents)
        self.register(self.smart_search)

    def initialize(self, documents):
        """인덱스 초기화"""
        # Vector Store 설정
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Vector Index (정밀 검색용)
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )

        # Summary Index (요약용)
        self.summary_index = SummaryIndex.from_documents(documents)

        # Router Query Engine (자동 라우팅)
        self.router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                QueryEngineTool.from_defaults(
                    query_engine=self.vector_index.as_query_engine(
                        similarity_top_k=5
                    ),
                    description="구체적인 정보 검색에 사용. 특정 사실, 수치, 절차에 대한 질문"
                ),
                QueryEngineTool.from_defaults(
                    query_engine=self.summary_index.as_query_engine(
                        response_mode="tree_summarize"
                    ),
                    description="문서 요약에 사용. 전체적인 개요, 주제 파악에 대한 질문"
                )
            ]
        )

    def search_documents(self, query: str, top_k: int = 5) -> str:
        """
        문서 검색을 수행합니다.

        Args:
            query: 검색 질문
            top_k: 반환할 결과 수

        Returns:
            검색 결과 및 출처
        """
        if self.vector_index is None:
            return "Error: 인덱스가 초기화되지 않았습니다"

        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=top_k
        )
        response = query_engine.query(query)

        sources = [
            node.metadata.get("file_name", "unknown")
            for node in response.source_nodes
        ]

        return f"답변: {response}\n출처: {', '.join(set(sources))}"

    def summarize_documents(self, topic: str) -> str:
        """
        특정 주제에 대한 문서를 요약합니다.

        Args:
            topic: 요약할 주제

        Returns:
            요약 결과
        """
        if self.summary_index is None:
            return "Error: 인덱스가 초기화되지 않았습니다"

        query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize"
        )
        response = query_engine.query(f"{topic}에 대해 요약해주세요")
        return f"요약: {response}"

    def smart_search(self, query: str) -> str:
        """
        질문 유형에 따라 자동으로 최적의 검색 방법을 선택합니다.

        Args:
            query: 질문

        Returns:
            검색 결과
        """
        if self.router_engine is None:
            return "Error: 인덱스가 초기화되지 않았습니다"

        response = self.router_engine.query(query)
        return str(response)


# Toolkit 인스턴스 생성 및 초기화
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/enterprise").load_data()

rag_toolkit = EnterpriseRAGToolkit()
rag_toolkit.initialize(documents)


# 전문가 에이전트 1: 문서 검색 전문가
doc_searcher = Agent(
    name="DocumentSearcher",
    model=OpenAIChat(id="gpt-4o"),
    role="문서 검색 전문가",
    tools=[rag_toolkit],
    instructions=[
        "사용자의 질문에 맞는 문서를 정확하게 검색합니다",
        "검색 결과의 출처를 명확히 밝힙니다"
    ]
)

# 전문가 에이전트 2: 분석가
analyzer = Agent(
    name="Analyzer",
    model=OpenAIChat(id="gpt-4o"),
    role="데이터 분석가",
    instructions=[
        "검색된 정보를 분석하고 인사이트를 도출합니다",
        "데이터 기반의 객관적인 분석을 제공합니다"
    ]
)

# 전문가 에이전트 3: 응답 작성자
responder = Agent(
    name="Responder",
    model=OpenAIChat(id="gpt-4o"),
    role="응답 작성 전문가",
    instructions=[
        "분석 결과를 사용자가 이해하기 쉽게 정리합니다",
        "핵심 내용을 명확하게 전달합니다"
    ]
)

# 팀 구성
enterprise_team = Team(
    name="Enterprise RAG Team",
    agents=[doc_searcher, analyzer, responder],
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "1. DocumentSearcher가 관련 문서를 검색합니다",
        "2. Analyzer가 검색 결과를 분석합니다",
        "3. Responder가 최종 응답을 작성합니다"
    ]
)

# 실행
enterprise_team.print_response(
    "우리 회사의 보안 정책 중 데이터 보호에 관한 내용을 분석하고 개선점을 제안해주세요",
    stream=True
)
```

### 7.6 연동 전략 비교 및 권장사항

| 연동 방식 | 복잡도 | 유연성 | 권장 사용처 |
|----------|--------|--------|------------|
| **기본 Tool 래핑** | 낮음 | 중간 | 빠른 프로토타이핑, 단일 인덱스 |
| **Hybrid RAG Tool** | 중간 | 높음 | 정밀 검색이 필요한 프로덕션 |
| **Agent-as-Tool** | 중간 | 높음 | 복잡한 RAG 로직이 필요할 때 |
| **Toolkit 클래스** | 높음 | 매우 높음 | 엔터프라이즈급 시스템 |
| **Multi-Agent Team** | 높음 | 매우 높음 | 복잡한 분석 워크플로우 |

---

## 8. 프로덕션 배포 가이드

### 8.1 FastAPI 서버 구현

```python
"""
LlamaIndex + Agno 기반 RAG API 서버
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from llama_index.core import VectorStoreIndex, Settings

app = FastAPI(title="Enterprise RAG API")

# 전역 에이전트 인스턴스
agent: Optional[Agent] = None
index: Optional[VectorStoreIndex] = None


class QueryRequest(BaseModel):
    question: str
    stream: bool = False
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] = []


@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    global agent, index

    # LlamaIndex 설정
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # 인덱스 로드 (사전에 생성된 인덱스)
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    # Query Engine을 Tool로 래핑
    query_engine = index.as_query_engine(similarity_top_k=5)

    def search_docs(query: str) -> str:
        response = query_engine.query(query)
        return str(response)

    # Agno Agent 초기화
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[search_docs],
        markdown=True
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """질문에 대한 답변 생성"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        response = agent.run(request.question)

        # 출처 추출
        sources = []
        if request.include_sources and hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "content": node.text[:200],
                    "metadata": node.metadata
                })

        return QueryResponse(
            answer=str(response),
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """스트리밍 응답"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def generate():
        response = agent.run(request.question, stream=True)
        for chunk in response:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "healthy", "agent_ready": agent is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8.2 Docker 배포 설정

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 환경 변수
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

### 8.3 모니터링 설정

```python
"""
RAG 시스템 모니터링
"""
from llama_index.core import set_global_handler
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# LlamaIndex 디버그 핸들러
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

# Arize Phoenix 통합 (옵션)
# pip install arize-phoenix
import phoenix as px

px.launch_app()
set_global_handler("arize_phoenix")
```

---

## 9. 참고 자료

### 9.1 공식 문서

- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Agno Documentation](https://docs.agno.com)
- [Agno GitHub](https://github.com/agno-agi/agno)

### 9.2 유용한 리소스

- [LlamaIndex for Beginners 2025](https://medium.com/@gautsoni/llamaindex-for-beginners-2025-a-complete-guide-to-building-rag-apps-from-zero-to-production-cb15ad290fe0)
- [Real Python LlamaIndex Guide](https://realpython.com/llamaindex-examples/)
- [Agno Framework Introduction](https://www.agno.com/blog/introducing-agno)
- [Agno + Groq Integration](https://console.groq.com/docs/agno)

### 9.3 관련 프로젝트

- [LlamaHub](https://llamahub.ai/) - LlamaIndex 통합 패키지 허브
- [LlamaCloud](https://cloud.llamaindex.ai/) - 관리형 LlamaIndex 서비스
- [AgentOS](https://www.agno.com) - Agno 관리형 런타임

---

> **문서 버전:** 1.0
> **최종 업데이트:** 2026-01-26
> **작성자:** AI Architecture Research Team
