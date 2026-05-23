# Agno Database 시스템 및 커스터마이징 가이드

## 개요

이 문서는 Agno 프레임워크의 데이터베이스 시스템 구조와 커스터마이징 방법을 설명합니다.

---

## 1. DB 저장 아키텍처

### 1.1 저장 담당 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Agent (중앙 컨트롤러)                       │
│                                                                      │
│   agent.py가 직접 담당:                                              │
│   ├─ Session 저장        → self.db.upsert_session()                 │
│   └─ Run 후 매니저 호출   → memory_manager, culture_manager 등       │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐
│   │                        Managers                                 │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │  │ MemoryManager   │  │ CultureManager  │  │ LearningMachine │ │
│   │  │                 │  │                 │  │                 │ │
│   │  │ db.upsert_user_ │  │ db.upsert_      │  │ db.upsert_      │ │
│   │  │ memory()        │  │ culture()       │  │ learning()      │ │
│   │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│   └─────────────────────────────────────────────────────────────────┘
│                                    │                                 │
│                                    ▼                                 │
│                            ┌──────────────┐                         │
│                            │   BaseDb     │                         │
│                            │ (PostgresDb, │                         │
│                            │  SqliteDb등) │                         │
│                            └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 저장 담당 주체별 정리

| 데이터 종류 | 저장 담당 | 호출 시점 | 메서드 |
|------------|----------|----------|--------|
| **Session** | Agent 직접 | Run 완료 후 | `agent._upsert_session()` → `db.upsert_session()` |
| **User Memory** | MemoryManager | Run 완료 후 (백그라운드) | `memory_manager.create_user_memories()` → `db.upsert_user_memory()` |
| **Culture** | CultureManager | Run 완료 후 | `culture_manager.create_culture()` → `db.upsert_culture()` |
| **Learning** | LearningMachine | 명시적 호출 시 | `learning_machine.learn()` → `db.upsert_learning()` |

### 1.3 DB 인스턴스 공유 구조

```python
# agent.py - Agent의 db를 각 Manager에 주입
def _set_memory_manager(self) -> None:
    if self.memory_manager is None:
        self.memory_manager = MemoryManager(model=self.model, db=self.db)
    else:
        if self.memory_manager.db is None:
            self.memory_manager.db = self.db  # db 공유
```

---

## 2. 기본 테이블 구조

### 2.1 BaseDb 클래스 (db/base.py)

Agno는 `BaseDb` 추상 클래스를 통해 13개 테이블을 정의합니다:

```python
class BaseDb(ABC):
    def __init__(
        self,
        session_table: Optional[str] = None,      # agno_sessions
        memory_table: Optional[str] = None,       # agno_memories
        culture_table: Optional[str] = None,      # agno_culture
        learnings_table: Optional[str] = None,    # agno_learnings
        knowledge_table: Optional[str] = None,    # agno_knowledge
        traces_table: Optional[str] = None,       # agno_traces
        spans_table: Optional[str] = None,        # agno_spans
        metrics_table: Optional[str] = None,      # agno_metrics
        eval_table: Optional[str] = None,         # agno_eval_runs
        versions_table: Optional[str] = None,     # agno_schema_versions
        components_table: Optional[str] = None,   # agno_components
        component_configs_table: Optional[str] = None,  # agno_component_configs
        component_links_table: Optional[str] = None,    # agno_component_links
    ):
```

### 2.2 UserMemory 스키마 (db/schemas/memory.py)

```python
@dataclass
class UserMemory:
    """User Memory 스키마 - 고정된 필드"""
    memory: str                          # 필수: 메모리 내용
    memory_id: Optional[str] = None      # UUID
    topics: Optional[List[str]] = None   # 태그/토픽 리스트
    user_id: Optional[str] = None        # 사용자 ID
    input: Optional[str] = None          # 원본 입력 텍스트
    created_at: Optional[int] = None     # 생성 시간 (epoch)
    updated_at: Optional[int] = None     # 수정 시간 (epoch)
    feedback: Optional[str] = None       # 피드백
    agent_id: Optional[str] = None       # 에이전트 ID
    team_id: Optional[str] = None        # 팀 ID
```

---

## 3. 커스터마이징 방법

### 3.1 방법 1: 테이블 이름 변경 (가장 간단)

```python
from agno.db.postgres import PostgresDb

db = PostgresDb(
    connection_string="postgresql://user:pass@localhost/mydb",
    # 테이블 이름 커스터마이징
    memory_table="my_custom_memories",
    session_table="my_custom_sessions",
    culture_table="my_custom_culture",
    learnings_table="my_custom_learnings",
)

agent = Agent(
    model="gpt-4o",
    db=db,
    update_memory_on_run=True,
)
```

### 3.2 방법 2: 기존 DB 클래스 확장 (권장)

기존 PostgresDb를 상속하고 필요한 메서드만 오버라이드:

```python
from agno.db.postgres import PostgresDb
from agno.db.schemas import UserMemory
from typing import Optional, List, Dict, Any
import json

class ExtendedPostgresDb(PostgresDb):
    """PostgresDb를 확장하여 커스텀 필드 지원"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_extension_tables()

    def _create_extension_tables(self):
        """확장 테이블 생성 (기존 테이블과 JOIN 용도)"""
        with self.engine.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_extensions (
                    memory_id VARCHAR(255) PRIMARY KEY,
                    sentiment VARCHAR(50),
                    importance_score FLOAT,
                    embedding VECTOR(1536),
                    custom_metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

    def upsert_user_memory(
        self,
        memory: UserMemory,
        deserialize: Optional[bool] = True,
        # 커스텀 파라미터
        sentiment: Optional[str] = None,
        importance_score: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[UserMemory]:
        """기본 저장 + 확장 데이터 저장"""

        # 1. 기본 메모리 저장 (부모 클래스 호출)
        result = super().upsert_user_memory(memory, deserialize)

        # 2. 확장 데이터 저장
        if result and memory.memory_id:
            self._save_memory_extension(
                memory_id=memory.memory_id,
                sentiment=sentiment or self._analyze_sentiment(memory.memory),
                importance_score=importance_score,
                embedding=embedding or self._generate_embedding(memory.memory),
            )

        return result

    def _save_memory_extension(
        self,
        memory_id: str,
        sentiment: Optional[str] = None,
        importance_score: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ):
        """확장 데이터 저장"""
        with self.engine.connect() as conn:
            conn.execute("""
                INSERT INTO memory_extensions
                    (memory_id, sentiment, importance_score, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET
                    sentiment = EXCLUDED.sentiment,
                    importance_score = EXCLUDED.importance_score,
                    embedding = EXCLUDED.embedding
            """, (memory_id, sentiment, importance_score, embedding))
            conn.commit()

    def _analyze_sentiment(self, text: str) -> str:
        """감정 분석 (예시)"""
        # 실제 구현에서는 LLM 또는 감정 분석 라이브러리 사용
        return "neutral"

    def _generate_embedding(self, text: str) -> List[float]:
        """임베딩 생성 (예시)"""
        # 실제 구현에서는 OpenAI Embeddings 등 사용
        return [0.0] * 1536

    def get_memories_with_extensions(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """확장 데이터와 함께 메모리 조회"""
        with self.engine.connect() as conn:
            result = conn.execute("""
                SELECT m.*, e.sentiment, e.importance_score, e.embedding
                FROM agno_memories m
                LEFT JOIN memory_extensions e ON m.memory_id = e.memory_id
                WHERE m.user_id = %s
                ORDER BY m.updated_at DESC
            """, (user_id,))
            return [dict(row) for row in result]
```

### 3.3 방법 3: 완전 커스텀 DB 구현

`BaseDb`를 상속하여 완전히 새로운 DB 구현:

```python
from agno.db.base import BaseDb, SessionType
from agno.db.schemas import UserMemory
from agno.db.schemas.culture import CulturalKnowledge
from agno.session import Session
from typing import Optional, List, Dict, Any, Tuple, Union

class CustomMongoDb(BaseDb):
    """MongoDB를 사용하는 완전 커스텀 DB 구현"""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "agno_db",
        **kwargs
    ):
        super().__init__(**kwargs)
        from pymongo import MongoClient
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.db.list_collection_names()

    # === Session 관련 (필수 구현) ===
    def get_session(
        self,
        session_id: str,
        session_type: SessionType,
        user_id: Optional[str] = None,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[Session, Dict[str, Any]]]:
        doc = self.db[self.session_table_name].find_one({
            "session_id": session_id,
            "session_type": session_type.value,
        })
        if doc is None:
            return None
        if deserialize:
            return Session.from_dict(doc)
        return doc

    def upsert_session(
        self,
        session: Session,
        deserialize: Optional[bool] = True
    ) -> Optional[Union[Session, Dict[str, Any]]]:
        session_dict = session.to_dict()
        self.db[self.session_table_name].update_one(
            {"session_id": session.session_id},
            {"$set": session_dict},
            upsert=True
        )
        if deserialize:
            return session
        return session_dict

    def delete_session(self, session_id: str) -> bool:
        result = self.db[self.session_table_name].delete_one({
            "session_id": session_id
        })
        return result.deleted_count > 0

    def delete_sessions(self, session_ids: List[str]) -> None:
        self.db[self.session_table_name].delete_many({
            "session_id": {"$in": session_ids}
        })

    def get_sessions(
        self,
        session_type: SessionType,
        user_id: Optional[str] = None,
        component_id: Optional[str] = None,
        session_name: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: Optional[bool] = True,
    ) -> Union[List[Session], Tuple[List[Dict[str, Any]], int]]:
        query = {"session_type": session_type.value}
        if user_id:
            query["user_id"] = user_id

        cursor = self.db[self.session_table_name].find(query)
        if limit:
            cursor = cursor.limit(limit)

        docs = list(cursor)
        if deserialize:
            return [Session.from_dict(doc) for doc in docs]
        return docs, len(docs)

    def rename_session(
        self,
        session_id: str,
        session_type: SessionType,
        session_name: str,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[Session, Dict[str, Any]]]:
        self.db[self.session_table_name].update_one(
            {"session_id": session_id},
            {"$set": {"session_name": session_name}}
        )
        return self.get_session(session_id, session_type, deserialize=deserialize)

    def upsert_sessions(
        self,
        sessions: List[Session],
        deserialize: Optional[bool] = True,
        preserve_updated_at: bool = False,
    ) -> List[Union[Session, Dict[str, Any]]]:
        results = []
        for session in sessions:
            result = self.upsert_session(session, deserialize)
            if result:
                results.append(result)
        return results

    # === Memory 관련 (필수 구현) ===
    def upsert_user_memory(
        self,
        memory: UserMemory,
        deserialize: Optional[bool] = True
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        memory_dict = memory.to_dict()

        # 커스텀 필드 추가 가능
        memory_dict["custom_field"] = "custom_value"

        self.db[self.memory_table_name].update_one(
            {"memory_id": memory.memory_id},
            {"$set": memory_dict},
            upsert=True
        )
        if deserialize:
            return memory
        return memory_dict

    def get_user_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
        search_content: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: Optional[bool] = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        query = {}
        if user_id:
            query["user_id"] = user_id
        if agent_id:
            query["agent_id"] = agent_id
        if topics:
            query["topics"] = {"$in": topics}

        cursor = self.db[self.memory_table_name].find(query)
        if limit:
            cursor = cursor.limit(limit)

        docs = list(cursor)
        if deserialize:
            return [UserMemory.from_dict(doc) for doc in docs]
        return docs, len(docs)

    def get_user_memory(
        self,
        memory_id: str,
        deserialize: Optional[bool] = True,
        user_id: Optional[str] = None,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        query = {"memory_id": memory_id}
        if user_id:
            query["user_id"] = user_id

        doc = self.db[self.memory_table_name].find_one(query)
        if doc is None:
            return None
        if deserialize:
            return UserMemory.from_dict(doc)
        return doc

    def delete_user_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None
    ) -> None:
        query = {"memory_id": memory_id}
        if user_id:
            query["user_id"] = user_id
        self.db[self.memory_table_name].delete_one(query)

    def delete_user_memories(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None
    ) -> None:
        query = {"memory_id": {"$in": memory_ids}}
        if user_id:
            query["user_id"] = user_id
        self.db[self.memory_table_name].delete_many(query)

    def clear_memories(self) -> None:
        self.db[self.memory_table_name].delete_many({})

    def get_all_memory_topics(
        self,
        user_id: Optional[str] = None
    ) -> List[str]:
        pipeline = [
            {"$unwind": "$topics"},
            {"$group": {"_id": "$topics"}},
        ]
        if user_id:
            pipeline.insert(0, {"$match": {"user_id": user_id}})

        result = self.db[self.memory_table_name].aggregate(pipeline)
        return [doc["_id"] for doc in result]

    def get_user_memory_stats(
        self,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        pipeline = [
            {"$group": {
                "_id": "$user_id",
                "count": {"$sum": 1},
            }}
        ]
        result = list(self.db[self.memory_table_name].aggregate(pipeline))
        return result, len(result)

    def upsert_memories(
        self,
        memories: List[UserMemory],
        deserialize: Optional[bool] = True,
        preserve_updated_at: bool = False,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        results = []
        for memory in memories:
            result = self.upsert_user_memory(memory, deserialize)
            if result:
                results.append(result)
        return results

    # === 나머지 필수 메서드들 (약 30개 이상) ===
    # Culture, Learning, Knowledge, Trace, Span, Eval, Metrics 등
    # 각각 구현 필요...

    def get_latest_schema_version(self, table_name: str):
        doc = self.db[self.versions_table_name].find_one({"table_name": table_name})
        return doc.get("version") if doc else self.default_schema_version

    def upsert_schema_version(self, table_name: str, version: str):
        self.db[self.versions_table_name].update_one(
            {"table_name": table_name},
            {"$set": {"version": version}},
            upsert=True
        )

    # ... (나머지 abstractmethod 구현 필요)
    # Cultural Knowledge, Learning, Knowledge, Trace, Span, Eval, Metrics 등
```

### 3.4 방법 4: 별도 저장소 병행 사용

Agno DB는 그대로 두고, 추가 데이터는 별도 저장소에 저장:

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
import redis

class HybridStorageAgent:
    """Agno DB + Redis를 병행 사용하는 에이전트"""

    def __init__(self):
        # Agno 기본 DB
        self.agno_db = PostgresDb(
            connection_string="postgresql://..."
        )

        # 추가 저장소 (Redis for embeddings/cache)
        self.redis = redis.Redis(host='localhost', port=6379)

        # Agent 생성
        self.agent = Agent(
            model="gpt-4o",
            db=self.agno_db,
            update_memory_on_run=True,
        )

    def save_memory_with_embedding(
        self,
        memory_id: str,
        embedding: List[float]
    ):
        """메모리 임베딩을 Redis에 저장"""
        import json
        self.redis.set(
            f"memory_embedding:{memory_id}",
            json.dumps(embedding)
        )

    def get_similar_memories(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[str]:
        """유사 메모리 검색 (Redis에서)"""
        # 실제 구현에서는 벡터 검색 라이브러리 사용
        pass
```

---

## 4. 커스터마이징 전략 비교

| 방법 | 난이도 | 유연성 | 유지보수 | 권장 상황 |
|------|--------|--------|----------|----------|
| **테이블 이름 변경** | ⭐ | 낮음 | 쉬움 | 기존 스키마로 충분할 때 |
| **기존 DB 확장** | ⭐⭐ | 중간 | 보통 | 일부 필드 추가가 필요할 때 |
| **완전 커스텀 DB** | ⭐⭐⭐⭐ | 높음 | 어려움 | 완전히 다른 DB 사용 시 |
| **별도 저장소 병행** | ⭐⭐ | 높음 | 보통 | 특수 기능(벡터 검색 등) 필요 시 |

---

## 5. 주의사항

### 5.1 BaseDb 필수 구현 메서드

`BaseDb`를 상속할 경우 약 40개 이상의 `@abstractmethod`를 구현해야 합니다:

- **Session**: 6개 (get, upsert, delete, rename, list, bulk)
- **Memory**: 8개 (get, upsert, delete, clear, stats, topics, search, bulk)
- **Culture**: 4개 (get, upsert, delete, clear)
- **Learning**: 4개 (get, upsert, delete, list)
- **Knowledge**: 4개 (get, upsert, delete, list)
- **Trace/Span**: 8개
- **Eval**: 5개
- **Metrics**: 2개
- **Schema Version**: 2개

### 5.2 UserMemory 스키마 제약

현재 `UserMemory` 스키마는 고정되어 있어 직접 필드를 추가할 수 없습니다.
확장이 필요한 경우:

1. 별도 테이블 생성 후 JOIN
2. `custom_metadata` JSONB 컬럼 활용 (직접 추가 시)
3. 별도 저장소 병행 사용

### 5.3 MemoryManager와의 호환성

커스텀 DB를 사용해도 `MemoryManager`는 `UserMemory` 스키마를 기준으로 동작합니다.
커스텀 필드를 활용하려면:

1. `MemoryManager`도 함께 커스터마이징
2. 또는 DB 레이어에서 자동 변환 처리

---

## 6. 실전 예제: 임베딩 + 벡터 검색 지원

```python
from agno.db.postgres import PostgresDb
from agno.db.schemas import UserMemory
from agno.memory.manager import MemoryManager
from agno.agent import Agent
from typing import List, Optional
import openai

class VectorEnabledPostgresDb(PostgresDb):
    """벡터 검색을 지원하는 PostgreSQL DB"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_vector_extension()

    def _setup_vector_extension(self):
        """pgvector 확장 및 테이블 설정"""
        with self.engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id VARCHAR(255) PRIMARY KEY,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS memory_embedding_idx
                ON memory_embeddings
                USING ivfflat (embedding vector_cosine_ops)
            """)
            conn.commit()

    def upsert_user_memory(
        self,
        memory: UserMemory,
        deserialize: Optional[bool] = True
    ) -> Optional[UserMemory]:
        # 1. 기본 저장
        result = super().upsert_user_memory(memory, deserialize)

        # 2. 임베딩 생성 및 저장
        if result and memory.memory_id:
            embedding = self._generate_embedding(memory.memory)
            self._save_embedding(memory.memory_id, embedding)

        return result

    def _generate_embedding(self, text: str) -> List[float]:
        """OpenAI 임베딩 생성"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _save_embedding(self, memory_id: str, embedding: List[float]):
        """임베딩 저장"""
        with self.engine.connect() as conn:
            conn.execute("""
                INSERT INTO memory_embeddings (memory_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding
            """, (memory_id, embedding))
            conn.commit()

    def search_similar_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[UserMemory]:
        """유사 메모리 벡터 검색"""
        query_embedding = self._generate_embedding(query)

        with self.engine.connect() as conn:
            result = conn.execute("""
                SELECT m.*, e.embedding <=> %s AS distance
                FROM agno_memories m
                JOIN memory_embeddings e ON m.memory_id = e.memory_id
                WHERE m.user_id = %s
                ORDER BY distance
                LIMIT %s
            """, (query_embedding, user_id, limit))

            return [UserMemory.from_dict(dict(row)) for row in result]


# 사용 예시
db = VectorEnabledPostgresDb(
    connection_string="postgresql://user:pass@localhost/mydb"
)

agent = Agent(
    model="gpt-4o",
    db=db,
    update_memory_on_run=True,
)

# 유사 메모리 검색
similar_memories = db.search_similar_memories(
    query="사용자의 취미는?",
    user_id="user123",
    limit=5
)
```

---

## 7. 요약

| 질문 | 답변 |
|------|------|
| 스키마 변경 가능? | 부분적 - UserMemory 필드 추가는 불가, 별도 테이블로 확장 필요 |
| 공식 인터페이스? | `BaseDb` ABC 제공 (40+ 메서드 구현 필요) |
| 현실적 방법? | 기존 DB(PostgresDb 등) 상속 + 확장 테이블 병행 |
| 완전 커스텀? | 가능하나 구현량 많음 |
| 권장 방식? | 기존 DB 상속 후 필요한 메서드만 오버라이드 + 확장 테이블 |

---

*문서 작성일: 2025년 1월*
*Agno 버전: 분석 기준*
