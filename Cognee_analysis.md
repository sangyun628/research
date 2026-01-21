# Cognee Analysis Report

> Repository: https://github.com/topoteretes/cognee
> Analysis Date: 2026-01-21

---

## 1. Overview

Cognee is an open-source memory and knowledge platform that transforms raw data into persistent and dynamic AI memory using a combination of vector search and graph databases. It replaces traditional RAG systems with ECL (Extract, Cognify, Load) pipelines.

### Key Differentiators

- **ECL Pipelines**: Modular, customizable data processing pipelines (not just RAG)
- **Graph + Vector Hybrid**: Combines vector similarity with relationship-based knowledge graphs
- **Ontology Integration**: Validates entities against predefined vocabularies
- **Multiple Search Types**: GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, CODE, CYPHER, etc.
- **Triplet Embeddings**: Embeds relationships (S-P-O) for relationship-aware search
- **30+ Data Sources**: Native connectors for documents, audio, images, etc.

### Research Paper

- [Optimizing the Interface Between Knowledge Graphs and LLMs](https://arxiv.org/abs/2505.24478)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Cognee                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Data Sources                          │   │
│  │   (Documents, Audio, Images, CSV, Web, GitHub, etc.)    │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              cognee.add() - Data Ingestion               │   │
│  │        (Document processing, chunking, embedding)        │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           cognee.cognify() - Knowledge Graph             │   │
│  │     (Entity extraction, relationship detection,          │   │
│  │      graph construction, summarization)                  │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           cognee.memify() - Memory Enrichment            │   │
│  │     (Rule associations, graph expansion, coding rules)   │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────┬───────────────────┬──────────────────┐   │
│  │   Vector Store   │    Graph Store    │   Relational DB  │   │
│  │  (embeddings)    │  (relationships)  │   (metadata)     │   │
│  └──────────────────┴───────────────────┴──────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              cognee.search() - Retrieval                 │   │
│  │   (GRAPH_COMPLETION, RAG, CHUNKS, CODE, CYPHER, etc.)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core API

### 3.1 Basic Usage

```python
import cognee
import asyncio

async def main():
    # 1. Add text/documents to cognee
    await cognee.add("Your documents or text here")

    # 2. Generate knowledge graph
    await cognee.cognify()

    # 3. Add memory algorithms (optional)
    await cognee.memify()

    # 4. Search the knowledge
    results = await cognee.search(
        "What does this document say?",
        query_type=SearchType.GRAPH_COMPLETION
    )

if __name__ == '__main__':
    asyncio.run(main())
```

### 3.2 CLI Interface

```bash
# Add data
cognee-cli add "Your text here"

# Process into knowledge graph
cognee-cli cognify

# Search
cognee-cli search "What is this about?"

# Delete all
cognee-cli delete --all

# Start local UI
cognee-cli -ui
```

---

## 4. ECL Pipeline: cognify()

The `cognify()` function is the core processing step that transforms raw data into a structured knowledge graph.

### 4.1 Default Pipeline Tasks

```python
# /cognee/api/v1/cognify/cognify.py

async def get_default_tasks(user, graph_model, chunker, chunk_size, config, ...):
    default_tasks = [
        # 1. Document Classification
        Task(classify_documents),

        # 2. Text Chunking
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),

        # 3. Knowledge Graph Extraction
        Task(
            extract_graph_from_data,
            graph_model=graph_model,  # Default: KnowledgeGraph
            config=config,
            custom_prompt=custom_prompt,
        ),

        # 4. Summarization
        Task(summarize_text),

        # 5. Store Data Points
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
        ),
    ]
    return default_tasks
```

### 4.2 Graph Extraction

```python
# /cognee/tasks/graph/extract_graph_from_data.py

async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Config = None,
    custom_prompt: Optional[str] = None,
) -> List[DocumentChunk]:
    """Extract knowledge graph from document chunks."""

    # Extract graph from each chunk using LLM
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=custom_prompt)
            for chunk in data_chunks
        ]
    )

    # Validate edges - remove those with missing nodes
    for graph in chunk_graphs:
        valid_node_ids = {node.id for node in graph.nodes}
        graph.edges = [
            edge for edge in graph.edges
            if edge.source_node_id in valid_node_ids
            and edge.target_node_id in valid_node_ids
        ]

    # Integrate with ontology validation
    return await integrate_chunk_graphs(
        data_chunks, chunk_graphs, graph_model, ontology_resolver
    )
```

### 4.3 Temporal Graph Processing

```python
# Temporal cognify for time-aware knowledge

async def get_temporal_tasks(user, chunker, chunk_size, chunks_per_batch=10):
    temporal_tasks = [
        Task(classify_documents),
        Task(extract_chunks_from_documents, ...),
        Task(extract_events_and_timestamps),  # Extract temporal events
        Task(extract_knowledge_graph_from_events),  # Build temporal KG
        Task(add_data_points),
    ]
    return temporal_tasks
```

---

## 5. memify() - Memory Enrichment

The `memify()` function adds memory algorithms to the existing knowledge graph.

```python
# /cognee/modules/memify/memify.py

async def memify(
    extraction_tasks: List[Task] = None,
    enrichment_tasks: List[Task] = None,
    data: Optional[Any] = None,
    dataset: str = "main_dataset",
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    ...
):
    """
    Enrichment pipeline that works with already-built graphs.
    Can add custom extraction and enrichment tasks.
    """

    # Default tasks if not specified
    if not extraction_tasks:
        extraction_tasks = [Task(extract_subgraph_chunks)]
    if not enrichment_tasks:
        enrichment_tasks = [
            Task(
                add_rule_associations,
                rules_nodeset_name="coding_agent_rules",
            )
        ]

    # If no data provided, use existing graph as input
    if not data:
        memory_fragment = await get_memory_fragment(
            node_type=node_type,
            node_name=node_name
        )
        data = [memory_fragment]

    # Run enrichment pipeline
    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=[*extraction_tasks, *enrichment_tasks],
        ...
    )
```

---

## 6. Search System

### 6.1 Search Types

```python
# /cognee/modules/search/types.py

class SearchType(str, Enum):
    GRAPH_COMPLETION = "GRAPH_COMPLETION"  # LLM + full graph context
    RAG_COMPLETION = "RAG_COMPLETION"      # LLM + document chunks
    CHUNKS = "CHUNKS"                       # Raw text segments
    SUMMARIES = "SUMMARIES"                # Pre-generated summaries
    CODE = "CODE"                          # Code-specific search
    CYPHER = "CYPHER"                      # Direct Cypher queries
    FEELING_LUCKY = "FEELING_LUCKY"        # Auto-select best type
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"      # Lexical search (Jaccard)
```

### 6.2 Search Function

```python
# /cognee/api/v1/search/search.py

async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: Optional[User] = None,
    datasets: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    verbose: bool = False,
) -> Union[List[SearchResult], CombinedSearchResult]:
    """
    Search Types & Use Cases:

    GRAPH_COMPLETION (Default):
        - Natural language Q&A with full graph context
        - Best for: Complex questions, analysis, insights
        - Returns: Conversational AI responses

    RAG_COMPLETION:
        - Traditional RAG without graph structure
        - Best for: Direct document retrieval
        - Returns: LLM responses from text chunks

    CHUNKS:
        - Raw text segments matching query
        - Best for: Finding specific passages
        - Returns: Ranked text chunks

    CODE:
        - Code-specific search
        - Best for: Finding functions, classes
        - Returns: Structured code information

    CYPHER:
        - Direct graph database queries
        - Best for: Advanced graph traversals
        - Returns: Raw query results
    """
```

### 6.3 Search Performance

| Search Type | Speed | Intelligence | Use Case |
|-------------|-------|--------------|----------|
| GRAPH_COMPLETION | Slow | High | Complex reasoning |
| RAG_COMPLETION | Medium | Medium | Document Q&A |
| CHUNKS | Fast | Low | Passage retrieval |
| SUMMARIES | Fast | Low | Quick overviews |
| CODE | Medium | High | Code understanding |
| FEELING_LUCKY | Variable | High | General queries |

---

## 7. Data Models

### 7.1 KnowledgeGraph (Default Graph Model)

```python
# /cognee/shared/data_models.py

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class Node(BaseModel):
    id: str
    name: str
    type: str  # Entity type (Person, Organization, Concept, etc.)
    properties: Dict[str, Any]

class Edge(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
```

### 7.2 Document Chunk

```python
# /cognee/modules/chunking/models/DocumentChunk.py

class DocumentChunk(BaseModel):
    id: UUID
    text: str
    document_id: UUID
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    contains: Optional[Any]  # Linked graph/entities
```

### 7.3 Triplet (Relationship)

```python
# /cognee/modules/engine/models/Triplet.py

class Triplet(BaseModel):
    subject: Entity
    predicate: str
    object: Entity
    source_chunk_id: UUID
    confidence: float
```

---

## 8. Ontology Integration

Cognee supports ontology-based entity validation and classification.

### 8.1 Ontology Resolver

```python
# /cognee/modules/ontology/base_ontology_resolver.py

class BaseOntologyResolver:
    def get_subgraph(self, entity_type: str) -> Graph:
        """Get ontology subgraph for entity type."""
        pass

    def validate_entity(self, entity: Entity) -> bool:
        """Validate entity against ontology."""
        pass

    def resolve_entity_type(self, entity_name: str) -> str:
        """Map entity to ontology class."""
        pass
```

### 8.2 Configuration

```python
# Environment variables for ontology

ONTOLOGY_FILE_PATH=/path/to/ontology.owl
ONTOLOGY_RESOLVER=default  # or custom resolver class
MATCHING_STRATEGY=fuzzy  # exact, fuzzy, semantic
```

---

## 9. Supported Integrations

### 9.1 LLM Providers

- OpenAI
- Anthropic
- Ollama (local)
- Azure OpenAI
- Google Gemini
- AWS Bedrock

### 9.2 Vector Databases

- Qdrant
- Weaviate
- Pinecone
- Milvus
- PostgreSQL (pgvector)
- In-memory

### 9.3 Graph Databases

- Neo4j
- FalkorDB
- NetworkX (in-memory)

### 9.4 Data Sources

- Text files (txt, md)
- Documents (PDF, DOCX)
- Data files (CSV, JSON)
- Images (with OCR)
- Audio (with transcription)
- Code repositories
- Web pages
- Notion
- GitHub
- Google Drive
- 20+ more connectors

---

## 10. Triplet Embeddings

Cognee supports embedding relationships (triplets) for relationship-aware search.

### 10.1 Triplet Embedding Configuration

```python
# /cognee/modules/cognify/config.py

cognify_config = get_cognify_config()
embed_triplets = cognify_config.triplet_embedding  # True/False
```

### 10.2 How Triplet Embeddings Work

```
Traditional: Embed nodes only
┌────────┐         ┌────────┐
│ Node A │───────→│ Node B │
│ [vec]  │ "knows" │ [vec]  │
└────────┘         └────────┘

With Triplet Embeddings:
┌────────┐         ┌────────┐
│ Node A │───────→│ Node B │
│ [vec]  │ "knows" │ [vec]  │
└────────┘ [vec]   └────────┘
              ↑
    Relationship also embedded!

Query: "Who does A know?"
→ Can search by relationship similarity, not just node similarity
```

---

## 11. Custom Pipelines

Cognee allows building custom ECL pipelines with custom tasks.

### 11.1 Custom Task Definition

```python
from cognee.modules.pipelines.tasks.task import Task

# Define custom task
async def my_custom_extraction(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Custom extraction logic."""
    for chunk in data_chunks:
        # Custom processing
        chunk.metadata["custom_field"] = extract_custom_info(chunk.text)
    return data_chunks

# Use in cognify
await cognee.cognify(
    datasets=["my_data"],
    custom_tasks=[
        Task(classify_documents),
        Task(extract_chunks_from_documents),
        Task(my_custom_extraction),  # Custom task
        Task(extract_graph_from_data),
        Task(add_data_points),
    ]
)
```

### 11.2 Custom Graph Model

```python
from pydantic import BaseModel
from cognee.shared.DataPoint import DataPoint

class ScientificPaper(DataPoint):
    title: str
    authors: List[str]
    abstract: str
    methodology: str
    findings: List[str]
    citations: List[str]

await cognee.cognify(
    datasets=["research_papers"],
    graph_model=ScientificPaper,
)
```

---

## 12. Comparison with Other Memory Systems

| Feature | Cognee | Mem0 | MemU | OpenMemory | Second-Me |
|---------|--------|------|------|------------|-----------|
| **Architecture** | Graph + Vector | Vector + Graph | File + Vector | HSG + Vector | Fine-tuned LLM |
| **Memory Model** | Knowledge Graph | Multi-level | 5 Types + Categories | 5 Sectors | Model Weights |
| **Ontology Support** | Yes | No | No | No | No |
| **Triplet Embedding** | Yes | No | No | No | N/A |
| **Search Types** | 7+ types | 1 type | 1 type | 1 type | N/A |
| **Temporal Support** | Yes | Basic | Basic | Yes | No |
| **Custom Pipelines** | Yes | No | No | No | No |
| **Data Sources** | 30+ | 10+ | Limited | Limited | Limited |

---

## 13. Strengths and Weaknesses

### Strengths

1. **Hybrid Architecture**: Combines vector search with graph relationships
2. **Modular Pipelines**: Fully customizable ECL pipelines
3. **Multiple Search Types**: Optimized for different use cases
4. **Ontology Integration**: Validates against predefined vocabularies
5. **Triplet Embeddings**: Relationship-aware search
6. **Temporal Support**: Time-aware knowledge graphs
7. **Extensive Integrations**: 30+ data sources, multiple DBs
8. **Research Backed**: Based on KG-LLM optimization research
9. **Production Ready**: Cloud platform available

### Weaknesses

1. **Complexity**: Many concepts to learn (cognify, memify, search types)
2. **LLM Dependency**: Requires LLM for graph extraction
3. **No Decay Mechanism**: No automatic memory decay/consolidation
4. **Limited Real-time**: Batch processing, not real-time memory
5. **No Identity Modeling**: No user profile/identity features

---

## 14. Use Cases

### 14.1 Document Q&A

```python
# Add documents
await cognee.add("path/to/documents/")

# Process into knowledge graph
await cognee.cognify()

# Query
results = await cognee.search(
    "What are the key findings?",
    query_type=SearchType.GRAPH_COMPLETION
)
```

### 14.2 Code Understanding

```python
# Add codebase
await cognee.add("path/to/codebase/")

# Process
await cognee.cognify()

# Search code
results = await cognee.search(
    "How does authentication work?",
    query_type=SearchType.CODE
)
```

### 14.3 Persistent Agent Memory

```python
# Store agent interactions
await cognee.add(conversation_history)
await cognee.cognify()
await cognee.memify()

# Retrieve relevant context
context = await cognee.search(
    current_query,
    query_type=SearchType.GRAPH_COMPLETION,
    session_id="agent_session"
)
```

---

## 15. Conclusion

Cognee provides a comprehensive knowledge management platform that goes beyond traditional RAG by combining vector search with graph-based knowledge representation. Its modular ECL pipelines and multiple search types make it highly adaptable to various use cases.

**Best suited for:**
- Knowledge-intensive applications
- Document analysis and Q&A
- Code understanding and analysis
- Applications requiring relationship-aware search
- Custom knowledge graph construction

**Not ideal for:**
- Real-time memory updates
- Identity/persona modeling
- Automatic memory consolidation
- Simple RAG without graph requirements

---

*Analysis completed on 2026-01-21*
