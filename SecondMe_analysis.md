# Second-Me (Mindverse) Analysis Report

> Repository: https://github.com/mindverse/Second-Me
> Analysis Date: 2026-01-21

---

## 1. Overview

Second Me is an open-source framework for creating a personalized "AI Self" - a digital twin that preserves your identity, understands your context, and represents you authentically. It uses Hierarchical Memory Modeling (HMM) and the Me-Alignment Algorithm to fine-tune local LLMs with user-specific data.

### Key Differentiators

- **Local Fine-Tuning**: Unlike other memory systems that store/retrieve memories at runtime, Second Me actually fine-tunes a local LLM with your data
- **AI Self Concept**: Creates a persistent digital identity that can represent you
- **Decentralized Network**: AI selves can connect and collaborate with permission
- **Complete Privacy**: 100% local training and hosting - your data never leaves your machine

### Research Papers

- [AI-Native Memory v1](https://arxiv.org/abs/2406.18312)
- [AI-Native Memory v2](https://arxiv.org/abs/2503.08102)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Second Me                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    User Data Input                       │   │
│  │    (Notes, Documents, Images, Audio, Chats, Todos)      │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L0: Insight Generation                   │   │
│  │         (Document parsing, Image analysis, Audio)       │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L1: Identity Modeling                    │   │
│  │    (Biography, Shades, Topics, Status, Clustering)      │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 L2: Model Training                       │   │
│  │    (Data Synthesis, Fine-tuning, DPO Alignment)         │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Trained AI Self                       │   │
│  │           (Local LLM personalized to YOU)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Second Me Network                       │   │
│  │        (Decentralized AI self collaboration)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Hierarchical Memory Model (L0/L1/L2)

Second Me implements a three-layer hierarchy for processing user data into a personalized AI model.

### 3.1 L0: Low-Level Insight Generation

L0 handles raw data parsing and initial insight extraction.

```python
# /lpm_kernel/L0/l0_generator.py

class L0Generator:
    def insighter(self, inputs: InsighterInput) -> Dict[str, str]:
        """Generate insights from document inputs."""
        datatype = DataType(inputs.file_info.data_type)

        if datatype == DataType.IMAGE:
            insight, title = self._insighter_image(bio, content, ...)
        elif datatype == DataType.AUDIO:
            insight, title = self._insighter_audio(bio, content, ...)
        else:  # DOCUMENT
            insight, title = self._insighter_doc(bio, content, ...)

        return {"title": title, "insight": insight}
```

**Supported Data Types:**
- TEXT: Plain text documents
- MARKDOWN: Markdown files
- PDF: PDF documents
- LINK: Web links
- IMAGE: Images (with vision models)
- AUDIO: Audio files (with Whisper transcription)

### 3.2 L1: Identity Modeling Layer

L1 creates a structured representation of user identity through Biography and Shades.

```python
# /lpm_kernel/L1/l1_generator.py

class L1Generator:
    def gen_global_biography(self, old_profile: Bio, cluster_list: List[Cluster]) -> Bio:
        """Generate global biography from user memories."""
        global_bio = self._global_bio_generate(global_bio)
        return global_bio

    def gen_shade_for_cluster(self, old_memory_list, new_memory_list, shade_info_list):
        """Generate a Shade (personality facet) for a memory cluster."""
        shade_generator = ShadeGenerator()
        return shade_generator.generate_shade(...)

    def gen_topics_for_shades(self, old_cluster_list, old_outlier_memory_list,
                               new_memory_list, cophenetic_distance=1.0):
        """Cluster memories into topics using hierarchical clustering."""
        topics_generator = TopicsGenerator()
        return topics_generator.generate_topics_for_shades(...)
```

**Key L1 Concepts:**

| Concept | Description |
|---------|-------------|
| **Bio** | Global biography - comprehensive user profile |
| **Shade** | Personality facet representing interests/expertise areas |
| **ShadeTimeline** | Temporal evolution of a shade with referenced memories |
| **Cluster** | Group of semantically similar memories |
| **AttributeInfo** | Identity attributes with confidence levels |

### 3.3 L2: Training Data Synthesis & Model Fine-tuning

L2 generates training data and fine-tunes the base model to create a personalized AI.

```python
# /lpm_kernel/L2/l2_generator.py

class L2Generator:
    def gen_subjective_data(self, note_list, basic_info, ...):
        """Generate subjective training data for personalization."""

        # 1. Generate preference Q&A data
        self.gen_preference_data(...)

        # 2. Generate diversity data
        self.gen_diversity_data(...)

        # 3. Generate self Q&A data
        self.gen_selfqa_data(...)

        # 4. Merge all training data
        self.merge_json_files(data_output_base_dir)
```

**Training Data Types:**

1. **Preference Q&A**: Questions about user preferences, opinions, and values
2. **Diversity Data**: Varied responses demonstrating user's communication style
3. **Self Q&A**: First-person responses about user's identity and experiences

---

## 4. Data Models

### 4.1 Note (Memory Unit)

```python
# /lpm_kernel/L1/bio.py

class Note:
    def __init__(
        self,
        noteId: int = None,
        content: str = "",
        createTime: str = "",
        memoryType: str = "",  # TEXT, MARKDOWN, PDF, LINK
        embedding: Optional[List[float]] = None,
        chunks: List[Chunk] = None,
        title: str = "",
        summary: str = "",
        insight: str = "",
        tags: List[str] = None,
        topic: str = None,
    ):
        ...
```

### 4.2 Memory & Cluster

```python
class Memory:
    def __init__(self, memoryId: int, embedding: List[float] = None):
        self.memory_id = memoryId
        self.embedding = np.array(embedding).squeeze() if embedding else None

class Cluster:
    def __init__(
        self,
        clusterId: int,
        memoryList: List[Memory] = [],
        centerEmbedding: List[float] = None,
        is_new=False,
    ):
        self.cluster_id = clusterId
        self.memory_list = memory_list
        self.cluster_center = np.array(centerEmbedding) if centerEmbedding else np.zeros(1536)

    def get_cluster_center(self):
        """Calculate centroid of memory embeddings."""
        self.cluster_center = np.mean(
            [memory.embedding for memory in self.memory_list], axis=0
        )

    def prune_outliers_from_cluster(self):
        """Remove outliers based on distance from cluster center."""
        memory_list = sorted(
            self.memory_list,
            key=lambda x: np.linalg.norm(x.embedding - self.cluster_center),
        )
        # Keep 80% closest memories
        self.memory_list = memory_list[: max(int(self.size * 0.8), 1)]
```

### 4.3 Shade (Personality Facet)

```python
class ShadeInfo:
    def __init__(
        self,
        id: int = None,
        name: str = "",           # e.g., "Software Engineering Enthusiast"
        aspect: str = "",         # e.g., "Professional Skills"
        icon: str = "",           # Visual representation
        descThirdView: str = "",  # Third-person description
        descSecondView: str = "", # Second-person description
        contentThirdView: str = "",
        contentSecondView: str = "",
        timelines: List[ShadeTimeline] = [],
        confidenceLevel: str = None,  # VERY_LOW to VERY_HIGH
    ):
        ...
```

### 4.4 Bio (Global Identity)

```python
class Bio:
    def __init__(
        self,
        contentThirdView: str = "",
        content: str = "",
        summaryThirdView: str = "",
        summary: str = "",
        attributeList: List[AttributeInfo] = [],
        shadesList: List[ShadeInfo] = [],
    ):
        self.shades_list = sorted(
            [ShadeInfo(**shade) for shade in shadesList],
            key=lambda x: len(x.timelines),  # Sort by evidence count
            reverse=True,
        )

    def complete_content(self, second_view: bool = False) -> str:
        """Generate complete biography content."""
        interests = "\n### User's Interests and Preferences ###\n"
        interests += "\n".join([shade._preview_(second_view) for shade in self.shades_list])
        conclusion = "\n### Conclusion ###\n" + self.summary
        return f"## Comprehensive Analysis Report ##\n{interests}\n{conclusion}"
```

---

## 5. Training Pipeline

### 5.1 Data Synthesis Flow

```
User Data → L0 Insight → L1 Identity → L2 Training Data → Fine-tuned Model

1. Preference Q&A Generation
   - Uses topics/clusters from L1
   - Generates questions about user preferences
   - Creates consistent preference responses

2. Diversity Data Generation
   - Uses entities and knowledge graph
   - Generates varied Q&A pairs
   - Ensures communication style diversity

3. Self Q&A Generation
   - First-person identity questions
   - Uses global bio and user intro
   - "Who am I?" type responses
```

### 5.2 Training with DPO (Direct Preference Optimization)

```python
# /lpm_kernel/L2/dpo/dpo_train.py

def train_dpo_model(
    model,
    train_dataset,
    eval_dataset,
    beta=0.1,  # KL penalty coefficient
    learning_rate=5e-5,
    num_train_epochs=3,
):
    """Fine-tune model using DPO for preference alignment."""
    # DPO trains on chosen vs rejected response pairs
    # Optimizes: maximize log P(chosen) - log P(rejected)
```

### 5.3 Me-Alignment Algorithm

The Me-Alignment algorithm ensures the AI self authentically represents the user:

1. **Identity Grounding**: Anchors responses to user's biography and shades
2. **Preference Consistency**: Maintains consistent opinions across interactions
3. **Style Preservation**: Keeps user's communication patterns
4. **Memory Integration**: References relevant user experiences

---

## 6. Memory Management

### 6.1 GPU Memory Optimization

```python
# /lpm_kernel/L2/memory_manager.py

class MemoryManager:
    def get_optimal_training_config(self) -> Dict[str, Any]:
        """Get recommended configs based on hardware."""
        config = {
            "device_map": "auto",
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
        }

        if self.cuda_available:
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere+
                config["bf16"] = True
            elif capability[0] >= 7:  # Volta+
                config["fp16"] = True

            # Adjust for VRAM
            vram_gb = self.get_memory_info()["vram_total_gb"]
            if vram_gb < 8:
                config["gradient_accumulation_steps"] = 4
            elif vram_gb < 16:
                config["gradient_accumulation_steps"] = 2

        return config
```

### 6.2 Model Size Recommendations

| Memory (GB) | Docker (Windows/Linux) | Docker (Mac) | Integrated |
|-------------|------------------------|--------------|------------|
| 8           | ~0.8B                  | ~0.4B        | ~1.0B      |
| 16          | ~1.5B                  | ~0.5B        | ~2.0B      |
| 32          | ~2.8B                  | ~1.2B        | ~3.5B      |

---

## 7. Knowledge Graph Integration (GraphRAG)

Second Me uses Microsoft GraphRAG for knowledge graph construction:

```python
# /lpm_kernel/L2/data_pipeline/graphrag_indexing/

# GraphRAG extracts:
# - Entities (people, places, concepts)
# - Relationships between entities
# - Community summaries (hierarchical clustering)
# - Claims and facts
```

**Integration Points:**
- Entity extraction for diversity data generation
- Topic modeling for preference generation
- Knowledge retrieval during inference

---

## 8. Second Me Network

### 8.1 Decentralized AI Self Sharing

```
┌─────────────────────────────────────────────────────────────────┐
│                    Second Me Network                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │ AI Self │    │ AI Self │    │ AI Self │    │ AI Self │    │
│   │  (You)  │◄──►│ (Alice) │◄──►│  (Bob)  │◄──►│(Charlie)│    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│        │              │              │              │          │
│        │              │              │              │          │
│   ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    │
│   │  Local  │    │  Local  │    │  Local  │    │  Local  │    │
│   │ Machine │    │ Machine │    │ Machine │    │ Machine │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                 │
│   Features:                                                     │
│   • Permission-based sharing                                    │
│   • AI-to-AI collaboration                                      │
│   • Roleplay scenarios                                          │
│   • Collective brainstorming                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Use Cases

- **Roleplay**: Your AI self can represent you in different scenarios
- **AI Space**: Multiple AI selves collaborate to solve problems
- **Icebreaking**: AI selves can introduce their owners to each other
- **Brainstorming**: Collective idea generation from multiple perspectives

---

## 9. Comparison with Other Memory Systems

| Feature | Second Me | Mem0 | MemU | OpenMemory | Cognee |
|---------|-----------|------|------|------------|--------|
| **Approach** | Local Fine-tuning | RAG Memory | RAG Memory | HSG Memory | Knowledge Graph |
| **Memory Type** | Model Weights | Vector Store | File-based | Multi-sector | Graph + Vector |
| **Personalization** | Model-level | Retrieval | Retrieval | Retrieval | Graph Structure |
| **Privacy** | 100% Local | Cloud/Local | Local | Local | Cloud/Local |
| **Training Required** | Yes | No | No | No | No |
| **Identity Modeling** | Shades/Bio | User metadata | Categories | Sectors | Ontology |
| **Inference Speed** | Fast (no RAG) | Variable | Variable | Variable | Variable |

---

## 10. Strengths and Weaknesses

### Strengths

1. **True Personalization**: Fine-tuning creates deeper personalization than RAG
2. **No Runtime Latency**: No retrieval step during inference
3. **Complete Privacy**: All training and inference is local
4. **Identity Persistence**: Model weights preserve identity across sessions
5. **Novel "AI Self" Paradigm**: First-of-kind digital twin approach
6. **Network Effect**: Decentralized collaboration between AI selves
7. **Research Backed**: Based on peer-reviewed AI-Native Memory papers

### Weaknesses

1. **Resource Intensive**: Requires GPU for training
2. **Training Time**: Hours to create initial AI self
3. **Update Cost**: New memories require retraining
4. **Model Size Limits**: Consumer hardware limits model capabilities
5. **No Real-time Memory**: Can't add memories without retraining
6. **Fixed Knowledge**: Model can't access external knowledge post-training

---

## 11. Usage Example

```bash
# 1. Clone and start
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me
make docker-up

# 2. Access web interface
# Open http://localhost:3000

# 3. Upload your memories (notes, documents, etc.)
# 4. Wait for L0/L1/L2 processing
# 5. Start chatting with your AI self
```

**Training Pipeline:**
```
User uploads data
    ↓
L0: Parse documents, extract insights
    ↓
L1: Generate biography, identify shades, cluster topics
    ↓
L2: Synthesize training data (preference, diversity, self-QA)
    ↓
Fine-tune base model (Qwen2.5) with LoRA
    ↓
Convert to GGUF for efficient inference
    ↓
AI Self is ready!
```

---

## 12. Conclusion

Second Me represents a paradigm shift from retrieval-based memory systems to model-embedded memory. By fine-tuning a local LLM with user data, it creates a persistent "AI Self" that truly understands and represents the user.

**Best suited for:**
- Users wanting deep personalization
- Privacy-conscious users
- Those with sufficient local compute resources
- Applications requiring consistent AI identity
- Multi-agent collaboration scenarios

**Not ideal for:**
- Real-time memory updates
- Low-resource environments
- Users with limited training data
- Applications requiring external knowledge access

---

*Analysis completed on 2026-01-21*
