# 다음 방법: 무작위 soft 후보의 KV 재사용

후속 실험은 [잔차 KV 융합 연구 기록](RESIDUAL_RESEARCH.ko.md)에 정리했다.
`swap_strength` 고정 계수는 구현했으며, 아래의 학습 gate는 아직 설계 단계다.

성공 조건은 일반 QA의 회귀를 억제하면서 수학 또는 코딩의 독립 평가를 높이는 것이다. 모든 데이터셋에 같은 생성·교체 정책과 설정을 적용한다. 현재 이 조건을 충족한 성능 향상을 입증한 상태는 아니다.

## 먼저 시험하는 방법

`soft_swap`은 같은 모델의 다음-token 분포에서 Gumbel-Softmax 가중치를 만들고 token embedding들을 합쳐 짧은 대안 cache를 만든다. 별도 답안 모델이나 LLM 판정기를 호출하지 않는다.

```text
p = top-k / top-p로 제한한 다음-token 분포
w = softmax((log p + Gumbel noise) / soft_temperature)
입력 embedding = Σ w[token] × embedding[token]
```

원래 main prefix의 사본에서 이 연산을 64회 이내로 수행한다. main은 원래 greedy 경로를 계속 생성한다. main rollout과 16-token 지연 뒤, 해당 과거 구간의 K/V만 대안으로 교체한다. 기존 텍스트, prefix KV, suffix KV는 유지한다. `candidate_count=2`는 main과 soft donor 하나이며, `soft.top_k=30`은 별도 30개 rollout을 뜻하지 않는다.

`configs/soft-swap-dev4.json`은 분기 256, span 64, 지연 16, top-k 30, top-p 0.95, 분포 temperature 0.6, Gumbel soft temperature 0.5를 고정한다. 모든 task에서 같은 정책이다. EOS나 think 종료가 대안의 지배적인 성분이면 일찍 종료하고 불완전한 span은 교체하지 않는다. 조기 종료시키는 entropy 규칙은 추가하지 않았다.

Soft donor에는 실제로 생성된 단일 token 문자열이 없다. 저장된 `alternative_text`는 가장 큰 혼합 성분의 ID를 연결한 진단용 proxy이며 `alternative_text_is_proxy=true`로 표시한다. 이를 모델이 실제로 생성한 별도 풀이로 평가하면 안 된다. 실제 최종 답은 main의 정상적인 token 출력에서만 채점한다.

## 근거와 적용 범위

[Soft Thinking](https://arxiv.org/abs/2505.15778)은 확률 가중 embedding으로 reasoning을 수행하며 수학·코딩 개선을 보고했다. 하지만 [LLMs Have a Heart of Stone](https://arxiv.org/abs/2508.03440)은 단순 평균의 저하를 관찰했고, Gumbel 무작위화를 사용했다. 후자의 Table 2에서 DeepSeek-R1-Distill-Qwen-32B는 sampling 대비 MATH-500 94.50→96.00, LiveCodeBench 57.35→59.50, GPQA-Diamond 60.60→63.13을 보고했다.

이 수치는 **논문의 32B, 전체 reasoning 설정**이다. 우리의 4B 모델·제한된 donor 구간·고정된 suffix는 다른 설정이며 논문 결과를 그대로 재현하는 구현이 아니다. 혼합 embedding이 모든 경로를 병렬로 실행하거나 정확한 경로 평균을 계산한다고 가정하지 않는다. 실제 성능으로 판단한다.

[Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)도 training-free soft inference의 효과에 반례를 제시했다. 따라서 단순 soft 평균은 우선순위에서 내리고 무작위화한 후보 한 설정만 시험한다.

## 대안 2: 교체 강도를 개발 데이터로 학습

다음 후보는 `C_new = C_main + gate × (C_alternative − C_main)` 형태의 제한된 교체다. gate는 성공/실패 rollout의 최종 QA label로 개발 단계에서 추정하고, inference에서 고정한 채 모든 task에 적용한다. main 모델은 frozen으로 두고 적은 수의 공통 gate만 학습할 수 있다.

[KV Cache Steering](https://arxiv.org/abs/2507.08799)은 one-shot cache intervention으로 여러 QA benchmark의 개선을 보고했다. 다만 원 연구의 prompt-position steering, 데이터셋별 계수 선택, reasoning 유도와 우리의 과거 span 재사용은 다르다. 이를 우리의 성능 증거로 제시하지 않는다. 이 gate 방식은 **설계 후보이며 현재 구현하지 않았다**. 첫 방법의 실제 효과가 약할 때만 다음으로 진행한다.

## 검증과 실행

작은 실제 Qwen으로 단일 성분의 greedy KV 동일성, 전역 RNG를 건드리지 않는 재현, span 밖 KV와 기존 ID 보존, 종료 token과 잘못된 설정을 검사했다. CPU 전체 52개 검사와 실제 GPU smoke가 통과했다. 입력 embedding 경로를 검증한 것이며 이것만으로 QA 개선을 주장하지 않는다.

새 개발 ID는 이전 실행/계획에서 사용하지 않은 GSM8K train의 seed 순서 첫 4개다. 결과를 보기 전에 고정했다. 두 arm 모두 출력 4,096토큰으로 실행했으며 4/4→4/4, 평균 33.85→36.37초(+7.5%)였다. 기준 모델이 모두 맞혀 정확도 상승은 관측되지 않았다.

다음 확인은 아직 완료된 baseline 출력이 없는 MATH-500 4개와 GPQA 2개를 seed 순서로 고정한 `soft-swap-transfer6.json`이다. 설정을 바꾸지 않고 수학과 비수학 QA에 같은 방법을 적용한다. 이는 작은 탐색 평가이며 full-dataset 결과가 아니다. coding adapter와 실행 평가를 아직 추가하지 않았으므로 코딩 개선은 주장하지 않는다.

```bash
uv run --frozen dirty-swapping run --config configs/soft-swap-dev4.json \
  --datasets gsm8k --split development --run-name soft-dev
uv run --frozen dirty-swapping run --config configs/soft-swap-transfer6.json \
  --datasets math500 gpqa --split evaluation --run-name soft-transfer
# 중단한 동일 source/config의 run은 resume --run outputs/<run-name>
```

출력마다 고정 config, source hash, 모델 revision, seed, latency, proxy 여부, swap/skip 이유가 저장된다. 대안의 생성 비용은 전체 latency에 포함한다. 성능 확인 전에 새로운 조합 grid를 확대하지 않는다.

## 평가 조건에 대한 한계

[공식 Qwen3-4B-Thinking-2507 모델 카드](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)는 temperature 0.6 / top-p 0.95 / top-k 20과 일반적인 출력 32,768토큰을 권장한다. 이번 비교의 main은 기존 greedy이며 4,096토큰이다. 따라서 이 실험은 그 제한된 예산 안의 성능이며 모델의 충분한 reasoning budget에서의 정확도 평가가 아니다. 권장 sampling과 더 긴 예산을 도입할 때는 순정·교체 양쪽을 같이 바꾸고, decoding 변경 이득을 KV 교체 이득으로 보고하지 않아야 한다.

## 완료된 실제 결과

| 구간 | 순정 → 교체 | 평균 시간, 초 | 해석 |
| --- | --- | --- | --- |
| 새 GSM8K 개발 4개 | 4/4 → 4/4 | 33.85 → 36.37 | +7.5%, 정확도 상승 없음 |
| MATH-500 확인 4개 | 1/4 → 1/4 | 142.38 → 144.49 | +1.5%, 양쪽 모두 3개는 최종 답 미완료 |
| GPQA 확인 2개 | 1/2 → 1/2 | 142.82 → 161.45 | +13.0%, 단일 실행 시간 변동 포함 |

이번 제한된 확인에서 wins/losses는 모두 0/0이다. 4B 모델의 짧은 구간 교체가 정확도를 높였다고 주장하지 않는다. 출력 상한 때문에 MATH의 충분한 reasoning-budget 성능도 판정할 수 없다. 현재 설정을 검증된 개선책으로 채택하거나 무작정 grid를 넓힐 근거는 없다.

실제 GPU 36개 layer에서 단일 성분의 greedy-cache 일치, main의 기존 token ID, prefix/suffix K/V 및 기존 next logits 보존, 선택 span의 실제 변경을 확인했다. 모델 weights는 frozen이다. 원자적 QA 결과·설정·runtime과 검사를 [공개 JSON](../outputs/soft-swap-summary-20261005/summary.json)에 보존했다.

후속 우선순위는 공통 sampling과 충분한 생성 예산을 양쪽에 적용한 비교, 그다음 개발 성능으로 학습하는 공통 교체 강도다. 앞선 실패 문항에 맞춘 규칙이나 inference gold 사용을 추가하지 않는다.
