# NVIDIA Auto-Ontology

DB 메타데이터·샘플 값·쿼리 로그를 LLM으로 컴파일해 비즈니스 용어(Term)↔컬럼, 검토된 SQL 식(SqlAttribute), 조인 관계(SEMANTIC_FK)를 만들고, 이를 MCP·REST·Agent Skills로 에이전트에 노출하는 "에이전트용 컨텍스트 레이어". illumex GSF의 후신이다(NVIDIA 2026-02 인수, 2026-10 개명).

| 문서 | 내용 |
|---|---|
| [코드 레벨 분석](auto-ontology-analysis.md) | 계보, 아키텍처·데이터 모델, 시맨틱 컴파일 파이프라인, LangGraph text-to-SQL, 답변 가능성 검사, MCP, 성능·운영, 경쟁 비교, 종합 평가 |
| [활용 가이드](use-cases.md) | 활용 패턴 6종(MCP 컨텍스트 서버, 레거시 모델 초안, Q&A, 답변 가능성 게이트, 정확도 하네스, 컴포넌트 차용), 도입 로드맵, 보안·한국어·비용 고려사항 |
| [vs Sutura (Deutsche Telekom)](auto-ontology-vs-sutura.md) | LLM이 의미와 SQL을 만드는 Auto-Ontology와, LLM 없이 인증된 지표만 타입 질의로 실행하는 Rust 런타임 Sutura의 비교 |

분석 대상: `NVIDIA/auto-ontology` `caced11` (v1.1.0, 2026-11-03).
