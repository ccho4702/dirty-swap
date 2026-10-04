# 다음 실험: 후보 KV의 제한된 주입

## 직전 결과와 가설

Soft donor는 GSM8K 4/4→4/4, MATH-500 1/4→1/4, GPQA 1/2→1/2였다.
개선 증거가 없으며 MATH는 4,096토큰 상한에서 3/4가 답을 끝내지 못했다.
이번에는 기존 main 정보가 완전히 사라지는 것을 줄이기 위해 선택된 과거
구간에 `C_new = 0.5 * C_main + 0.5 * C_donor`를 적용한다.
후보는 generic Instruct draft를 main 모델로 다시 인코딩해서 만든다.
기존 text와 교체 구간 뒤의 모든 KV는 그대로 둔다.

이것은 학습된 방법의 재현이나 성능 보장 없이 검증하는 고정 계수 가설이다.
학습은 아직 하지 않았다. 전 데이터셋에서 같은 prompt·계수·분기 규칙을 쓴다.

## 이번 회차에 새로 검토한 논문 5편

직전 soft 회차의 Soft Thinking / Heart of Stone / Soft Tokens, Hard Truths /
Multiplex Thinking / KV Cache Steering과 중복하지 않는다.
아래 논문의 결과는 우리의 mid-reasoning stale-suffix 설정의 성능 증거가 아니다.

| 논문 / 확인된 학회 | 확인한 접근 | 이번 실험에 반영하거나 보류한 이유 |
| --- | --- | --- |
| [Cache-to-Cache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html), ICLR 2026 main | frozen 두 모델 사이 projection, residual cache fusion, 학습된 layer gate. Fuser에 next-token loss 적용 | 잔차 융합을 우선 시험. 원 논문의 prompt cache·모델 간 projection과 달리 우리는 main이 donor text를 재인코딩. 고정 0.5는 논문의 학습 모듈을 재현하지 않음 |
| [CODI](https://aclanthology.org/2025.emnlp-main.36/), EMNLP 2025 main | explicit teacher/implicit student를 공동 학습하고 답변 위치 hidden state를 정렬 | latent 변환에 학습·정렬이 필요할 가능성. frozen 모델의 단순 soft 입력만으로 같은 효과를 기대하지 않음. 전체 모델 훈련은 우선 보류 |
| [AdaptThink](https://aclanthology.org/2025.emnlp-main.184/), EMNLP 2025 main | 품질 제약을 둔 RL로 thinking/no-thinking 선택, 두 모드의 importance sampling 균형 | QA 품질 유지와 비용을 같이 확인하는 기준 채택. 강제로 think를 닫는 heuristic은 추가하지 않음 |
| [Efficient Latent Semantic Clustering](https://aclanthology.org/2025.findings-emnlp.1310/), Findings of EMNLP 2025 | 생성 모델 hidden state의 유사도로 spectral clustering, 중복 reasoning 경로 병합 | 향후 후보가 많을 때 중복 탐색 제거에 유용. 지금은 후보 하나이므로 clustering 시스템 추가하지 않음. main conference로 잘못 표기하지 않음 |
| [Scaling up Test-Time Compute with Latent Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html), NeurIPS 2025 main | recurrent block의 반복 깊이를 늘리는 구조; 3.5B 모델을 처음부터 학습 | 기존 Qwen에 inference 패치로 적용 가능한 방법이 아니므로 보류 |

## 순서와 판정

1. CPU에서 강도 0/1/중간값, BF16, 잘못된 설정, donor 보존을 확인한다.
2. 실제 GPU에서 모든 layer의 prefix/suffix와 token ID 보존, 강도 0 identity,
   순정 `generate` 일치를 확인한다. 완료 결과 재개도 확인한다.
3. 성능은 새 MATH-500 4개와 GPQA 2개를 결과 확인 전에 고정한다.
   기존 소규모 실행 계획/설정 및 완료된 결과에 들어간 ID를 제외하고 기존 seed
   순서에서 고른다. 실행되지 않은 full-dataset 계획의 전체 ID 목록은 제외 대상이 아니다.
   config는 `configs/residual-draft-transfer6.json`이다.
4. 양쪽 모두 greedy, 출력 8,192토큰. 기존 4,096토큰 결과와 직접 점수 비교하지
   않는다. 같은 이번 cohort의 baseline과 비교한다. 상한에 걸린 답은 그대로
   실패 처리하고 종료 비율을 별도로 공개한다.
5. 수학에서 wins > losses이면 별도 새 cohort에서 같은 설정으로 확인한다.
   GPQA 2문항만으로 일반 QA 유지가 확립됐다고 주장하지 않는다.
   성능 변화가 없으면 새로운 조합 grid를 무작정 확장하지 않는다.

비용은 draft 생성, 재인코딩, 복사, 최종 decode를 포함한 end-to-end 시간이다.
inference에는 gold를 넘기지 않는다. coding 실행 평가는 아직 지원하지 않으며
수학 결과를 코딩 성능으로 일반화하지 않는다.

```bash
uv run --frozen python scripts/verify_draft_cache.py \
  --config configs/residual-draft-dev4.json --run-name residual-cache-check
uv run --frozen dirty-swapping run --config configs/residual-draft-transfer6.json \
  --datasets math500 gpqa --split evaluation --run-name residual-transfer
uv run --frozen dirty-swapping resume --run outputs/residual-transfer
```

QA는 완료된 문항/arm마다 원자적 체크포인트를 저장한다. 중단된 문항 하나는
동일 seed의 prompt부터 재생하며 완료 문항은 건너뛴다. 모델·optimizer를 갱신하는
훈련은 없다. source/config/runtime이 바뀌면 기존 실행의 재개를 거부한다.

## 기능 점검 중 수정

첫 GPU 점검은 `draft_span_limit`으로 종료됐다. Generic prompt의 완료 draft가
기존 96-token 구간에 들어가지 않았기 때문이다. 성능 cohort를 실행하기 전에
모든 task의 구간 상한을 256으로 올렸다. 실제 교체 길이는 담기는 draft의 token
수로 정하고 main도 동일 길이를 진행한다. 불완전하거나 여전히 긴 후보는 기존처럼
명시적으로 skip한다. 짧은 후보를 padding하지 않는다. 최초 실패 기록은
`outputs/residual-cache-check-20261005`에 보존한다.
