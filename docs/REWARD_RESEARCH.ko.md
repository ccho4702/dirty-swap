# 검토 중인 후속 후보: 학습된 범용 판정기로 교체 선택

상태: 문헌 검토와 설계만 진행했다. 현재 residual 실험에 적용한 기능이 아니다.
성능 향상이나 판정기 검증 완료를 주장하지 않는다.

## 새 문헌 목록

Residual 실험의 C2C/CODI/AdaptThink/LSC/Recurrent Depth 5편 및 이전 soft
실험 문헌과 중복하지 않는 아래 5편을 확인했다. 모두 최초 공개가 2025년이며,
표에는 확인한 정식 학회 연도를 적었다. 모델의 preference benchmark 성적이
우리 KV 교체의 QA 성능 향상을 보장하지는 않는다.

| 논문 / 학회 | 확인한 내용 | 우리 설정의 적용 가능성과 한계 |
| --- | --- | --- |
| [VersaPRM](https://proceedings.mlr.press/v267/zeng25h.html), ICML 2025 | 수학 PRM의 다른 분야 전이 한계, 합성된 여러 분야 reasoning으로 학습한 PRM | 수학 전용 판정기를 기본 선택하지 않을 근거. 일반 QA에서의 검증도 필요 |
| [GenPRM](https://ojs.aaai.org/index.php/AAAI/article/view/40797), AAAI 2026; [2025 preprint](https://arxiv.org/abs/2504.00891) | 추론과 코드 검증을 거친 단계 평가, relative progress estimation으로 학습 데이터 생성 | 판정 전 추가 추론의 근거지만 학습·평가가 주로 MATH이므로 범용 기본 판정기로 우선 채택하지 않음 |
| [Skywork-Reward-V2](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8836d0294651a867c2ca51012923997-Abstract-Conference.html), ICLR 2026; [2025 preprint](https://arxiv.org/abs/2507.01352) | 정제된 26M preference pair, 0.6B~8B scalar reward models, 16K 학습 context | [Qwen3-1.7B 공개 모델](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B)은 비용 면에서 첫 검토 대상. 주로 완성된 응답을 평가하므로 짧은 partial probe에 바로 적용하면 분포 차이가 있음 |
| [RM-R1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8e3b8de251afd887fb4589c1e3a3c793-Abstract-Conference.html), ICLR 2026; [2025 preprint](https://arxiv.org/abs/2505.02387) | reasoning distillation과 RL, 입력에 맞춘 평가 기준을 생성하고 후보 비교 | 데이터셋별 수동 기준을 늘리는 대신 learned evaluation을 검토할 근거. 큰 모델과 추가 judge reasoning의 비용 때문에 우선순위는 낮음 |
| [Reward Reasoning Model](https://proceedings.neurips.cc/paper_files/paper/2025/file/dd35bb9efff094897fb6688a57675212-Paper-Conference.pdf), NeurIPS 2025 | RL로 reasoning 후 두 응답의 선호를 판정. 일반 지식·수학 등에서 평가 | 비교 대상은 donor의 겉보기 품질보다 실제 edited-cache 미래여야 한다는 설계에 참고. 추가 추론 비용과 강제 양자택일의 회귀 위험을 측정해야 함 |

추가로 확인한 관련 문헌도 중복 집계하지 않도록 기록한다:
[Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
([공식 repo의 TMLR 표기](https://github.com/mukhal/ThinkPRM)),
[Generative Verifiers](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html)
(ICLR 2025, 최초 preprint는 2024이므로 위 2025 최초 공개 5편에 넣지 않았다).

## 구현 전에 확인할 조건

1. 후보 생성만으로 정답으로 바뀐 사례가 있는지 확인한다. 후보 집합에 정답 경로가
   없으면 판정기를 추가해도 정확도를 높일 수 없다. Gold로 하는 이 분석은 offline
   upper-bound 진단이며 inference 정책에 넣지 않는다.
2. 짧은 partial probe보다 가능한 한 완료된 두 미래를 비교한다. RM에는 질문과
   실제 최종 응답을 주며, 정답 label이나 dataset 이름은 주지 않는다. 길이·표현 선호가
   정오 판정을 대신하지 않는지 실제 correct/incorrect 쌍에서 먼저 확인한다.
3. 선택된 edited future를 채택할 때도 이미 생성된 텍스트와 suffix KV는 유지한다.
   원래 미래와 같거나 judge가 신뢰할 수 없는 경우에는 원래 경로를 유지한다.
   기준은 개발 데이터에서 고정하고 전 task에 동일하게 적용한다.
4. 순정, 무조건 교체, learned selection의 정확도·완료율·전체 시간을 비교한다.
   추가 future 생성과 RM 호출 비용을 모두 포함하고, 일반적인 추가 branch 탐색과도
   비교해야 KV 교체 자체의 장점을 주장할 수 있다.

실제 구현, 모델 다운로드, 판정기 calibration, 신규 cohort 성능 실험은 아직 하지 않았다.
