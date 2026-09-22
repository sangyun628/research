# 의사결정 모델 비교

| 문서 | 내용 |
|---|---|
| [Kev vs Laya](kev-vs-laya.md) | Qwen LoRA·mmBERT 기반 의사결정 모델의 학습 코드, 한국어 성능 근거, CPU 금융 라우팅 적합성 |
| [Laya 공식 사이트 분석](laya-official-site-analysis.md) | 사이트 주장과 코드·논문 대조, Router 실제 기본값, 속도·확률 보정·다운로드 조건 |
| [Kev 추론 하드웨어](kev-inference-hardware.md) | 모델별 CPU·GPU 시작 사양, BF16 메모리 근거와 FP32 로딩 제약 |
| [Jevlike vs OpenJev](jevlike-vs-openjev.md) | 별도 선택 모델 학습과 기존 LLM의 후보 토큰 점수 추출 비교 |
| [Jevlike vs mmBERT](jevlike-vs-mmbert.md) | Tiny와 다국어 사전학습 인코더의 차이, 학습·장단점·결합 가능성 |
| [금융 플레이북 라우팅](financial-playbook-routing.md) | 최근 3질문·21개 클래스·CPU 배포 조건에 맞춘 모델 선택과 성능 판단 |
| [속도·인프라 비용](performance-and-cost.md) | Jevlike 로컬 측정, OpenJev 공개 측정, 비용 계산의 가정 |

Jevlike 구현과 로컬 실행은 [Jevlike 문서 모음](../jevlike/README.md)을 참고한다.
