# 제한된 시간에서의 연구 순서

목표는 과거 KV 교체로 최종 답 정확도를 높이는 것이다. 현재까지 baseline은 순정과 토큰 단위로 일치했지만, swap의 정확도 이득은 입증되지 않았다. 고정 top-2 교체가 답을 늦춰 출력 예산 안에서 오답이 되는 사례도 있었다. 먼저 이 실패를 줄일 가능성이 있고 기존 코드로 빠르게 시험할 수 있는 변경부터 진행한다.

| 순서 | 진행할 것 | 다음 단계로 가는 조건 |
| --- | --- | --- |
| 1 | 짧은 지연의 단일 교체 | 개발 표본에서 명백한 정확도·완주율 악화가 없고 추가 비용이 작을 것 |
| 2 | 실제 교체 cache에서 짧은 lookahead를 비교하고 no-swap도 선택 | 1에서 긍정적인 신호가 있거나, 실제 교체 결과를 골라야 할 구체적인 실패가 확인될 것 |
| 보류 | 별도 PRM/verifier, 다양한 sampling·레이어 범위·여러 번 교체 | 앞선 방법에서 이득이 보인 후 필요한 요인만 추가할 것 |

## 지금 실행하는 최소 후보

`configs/pilot-short-delay.json`은 같은 prefix의 top-1/top-2 후보만 32토큰씩 생성하고, main rollout 뒤 16토큰을 더 생성한 시점에 top-2의 과거 KV로 교체한다. reasoning 256토큰 분기, 모든 레이어의 K/V, 같은 위치·같은 길이, 교체 1회는 유지한다. 기존 방식은 후보 4개를 계산해도 top-2만 선택하므로, 첫 파일럿에서는 나머지 후보 준비 비용을 쓰지 않는다. 후보가 일찍 종료되는 경우의 skip 조건은 별도로 기록한다.

첫 gate는 고정된 GSM8K **train 개발 split의 첫 2문제**에서 baseline과 짝지어 실행한다. 이는 의미 없는 실행·완주율·비용 회귀를 빠르게 찾는 확인이며, 정확도 이득을 주장할 표본이 아니다. 출력 한도는 두 arm 모두 4,096토큰이고, 미완료 답은 오답이다. 한도에 닿은 문제는 방법 실패와 긴 추론을 구분해 기록한다.

```bash
uv run --frozen dirty-swapping run --datasets gsm8k --split development --limit 2 \
  --config configs/pilot-short-delay.json --run-name short-delay-gate
```

이 gate를 지나고 기존 지연 128토큰보다 유용한 신호가 있을 때만 개발 표본을 늘린다. 정확도/답 완주율 악화가 보이거나 시간 증가가 약 20%를 넘으면 먼저 trace를 보고 중단 여부를 판단한다. 이 20%는 계산 예산을 위한 실용적 gate이며 통계적 기준이 아니다. 설정을 고정한 뒤에 disjoint 평가 cohort를 실행한다. 여러 지연·길이·분기 offset을 한 번에 grid search하지 않는다.

## 2026-10-04 gate 결과와 다음 결정

GPU RTX 3090에서 개발 ID `gsm8k-train-1292`, `gsm8k-train-3522`를 사용했다. 질문은 seed로 정해진 순서의 첫 두 개이며 정답을 보고 고르지 않았다. baseline은 한 번만 실행하고 두 교체 설정에서 동일 cohort와 생성 규칙을 사용했다. 두 swap 설정의 차이는 지연 16/128토큰뿐이다. 각각 후보 2개·rollout 32토큰·분기 256토큰·출력 상한 4,096토큰이다.

| 설정 | 정답 / 문제 | 답 추출 | 평균 시간 | 총 생성 토큰 |
| --- | --- | --- | --- | --- |
| baseline | 2 / 2 | 2 / 2 | 60.63 s | 3,092 |
| 지연 128 (`pilot-reference.json`) | 2 / 2 | 2 / 2 | 59.77 s | 3,061 |
| 지연 16 (`pilot-short-delay.json`) | 2 / 2 | 2 / 2 | 62.33 s | 3,124 |

각 swap은 두 문제에서 모두 적용됐다. 짧은 지연의 시간 차이는 문제마다 방향이 달랐고, 정확도 이득은 관측되지 않았다. 두 문제와 한 번의 실행으로 효과가 없다고 결론 내릴 수는 없지만, **제한된 시간에서 지연 grid를 확대할 근거도 없다**. 짧은 지연 탐색은 여기서 보류한다. 설정과 원자적 결과는 로컬 `outputs/short-delay-gate-20261004`, `outputs/reference-delay-gate-20261004` 및 `outputs/research-gate-20261004-summary.json`에 보존했다. 생성된 결과는 Git에 포함하지 않는다.

다음 개발 우선순위는 실제 교체 상태와 no-swap을 비교하는 후보 3이다. 먼저 작은 개발용 논리·계산 예시에서 scorer가 명백한 오류를 구분하는지 확인하고, 통과하면 하나의 짧은 probe 설정으로 구현한다. verifier 모델 교체나 entropy threshold sweep을 함께 추가하지 않는다.

## 두 번째 후보의 구현 조건

실제 교체 시점의 cache를 복사해 no-swap과 대안 하나를 각각 시험한다. 기존 suffix KV는 그대로 두고 새 미래 토큰만 짧게 생성한다. 첫 미래 토큰은 기존 logits를 쓰므로 probe는 최소 두 토큰 이상이어야 한다. 논리적 진전과 오류를 평가할 신뢰할 scorer가 있을 때만 대안을 선택하고, no-swap보다 나아 보이지 않으면 교체하지 않는다.

단순히 entropy가 줄거나 후보의 log-probability가 높다는 것을 정답에 가까워졌다는 뜻으로 쓰지 않는다. scorer 구축 비용이 첫 후보보다 클 수 있어 지금은 구현하지 않는다. no-swap 선택지가 있어도 scorer 오류로 인한 정확도 하락은 가능하다.

## 이번에 보류하는 탐색

별도 verifier 모델 도입, top-k 전체 순회, K-only/V-only 및 레이어 부분 교체, 다중 swap, entropy threshold sweep, LongBench의 장문 cohort, AIME/HMMT 전체 평가를 동시에 진행하지 않는다. 개발 gate에서 이득이 확인된 후 그 이득을 설명하는 데 꼭 필요한 비교만 수행한다. 최종 비교에는 순정 baseline, 정확도·완주율, 전체 latency, 생성 토큰 수, 실제 swap/skip 이유를 모두 남긴다.
