# Jevlike

문맥과 여러 선택지를 받아 각 선택지에 점수를 주는 학습형 모델. 원본 저장소는 `.repos/jevlike/`에 클론되어 있으며 Git 추적에서 제외한다.

| 문서 | 내용 |
|---|---|
| [로컬 데모 실행](local-demo.md) | 브라우저에서 문맥·선택지 입력, 점수·처리시간 확인, 체크포인트 교체 |
| [소스코드 분석](jevlike-analysis.md) | 구조, Attention head, Tiny·HF 차이, 기능과 제약 |
| [자체 학습·파인튜닝](training-and-finetuning.md) | JSONL 데이터, Tiny 학습, 동결 인코더 head 학습, LoRA 확장 |
| [금융 의도 분류 학습](finance-intent-training.md) | 질문·의도 라벨 형식, CPU 학습·추론 명령, snapshot 보완 실행 파일 |
| [금융 플레이북 라우팅 적합성](../decision-model-comparison/financial-playbook-routing.md) | 최근 3질문·21개 의도·CPU 조건에서 mmBERT와 Jevlike 선택 |
| [OpenJev와 비교](../decision-model-comparison/jevlike-vs-openjev.md) | 학습형 선택 모델과 기존 LLM 토큰 점수 방식의 차이 |
| [mmBERT와 비교](../decision-model-comparison/jevlike-vs-mmbert.md) | Tiny와 다국어 사전학습 모델의 차이, 학습·장단점·결합 가능성 |
| [속도·인프라 비용](../decision-model-comparison/performance-and-cost.md) | 로컬 실측, 공개 벤치마크와 비용 가정 |

실행 파일: [서버](demo_server.py), [화면](demo.html), [시작 스크립트](start-demo.sh), [학습 실행 파일](train_local.py).
