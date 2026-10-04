# 2026-10-04 연구 결과 — changho

**KV 교체 기능과 순정 비교군은 작동한다. 독립 확인 cohort에서 전체 QA 정확도 향상은 아직 확인되지 않았다.** 개발 개선만으로 공식 baseline을 변경하지 않는다. 공유된 `main` 파일 상태는 연구 이전 버전을 유지한다.

이 문서는 이전 수학 중심 탐색의 기록이다. 현재 공통 정책과 미검증 범위는 [일반 방법론](GENERAL_METHOD.ko.md)에 있다. 아래 수치를 일반 QA 전이 결과로 해석하지 않는다.

## 2026-10-05 추가 확인: 50% KV 혼합

Generic draft를 main으로 재인코딩하고 기존 K/V와 50%씩 혼합했다. 미사용
MATH-500 4개와 GPQA 2개를 사전 고정하고 양쪽 모두 greedy/8,192토큰으로 비교했다.
수학은 3/4→3/4(+3.8% 시간), GPQA는 0/2→0/2(-8.0% 시간)로 정확도 개선은 없었다.
실제 교체는 6개 모두 적용됐다. 작은 표본과 답변 미완료 때문에 QA 보존을 입증한
결과로 해석하지 않는다.

개발용 사후 진단 한 문항에서는 순정 sampling만으로 답변 미완료→정답이 됐다.
**KV 교체 성과가 아니다.** 자세한 조건과 후속 진단은
[잔차 교체 기록](RESIDUAL_RESEARCH.ko.md), 측정값과 검증은
[공개 JSON](../outputs/residual-swap-summary-20261005/summary.json)에 있다.

## 실행 조건

RTX 3090 24 GiB, BF16, Transformers 4.57.1, Torch 2.6.0, seed 18. 측정한 package revision은 `9f65ce8`이며 [공개 결과 JSON](../outputs/research-summary-20261004/summary.json)에 각 run의 source manifest digest, model revision, runtime, 문제별 결과와 원래 보고서의 metrics를 보존했다. 정답 라벨과 문제 원문은 포함하지 않는다.

main은 frozen Qwen3-4B-Thinking-2507이다. `draft_swap`은 별도 frozen Qwen3-4B-Instruct-2507의 짧은 풀이 끝부분을 **원래 main prefix에서 다시 처리해 main 모델의 KV**로 만든다. 다른 모델의 KV를 직접 복사하지 않는다. reasoning 64–512토큰의 첫 줄 경계에서 최대 96토큰의 과거 span을 한 번 교체한다. 실제 span은 선택한 텍스트의 token 수다. 기존 텍스트와 suffix KV는 유지한다.

아래 주요 비교는 양쪽 모두 **출력 최대 2,048토큰**이다. 초안 생성과 교체 준비 비용을 전체 시간에 포함한다. 기준 제공 설정의 32,768토큰 또는 전체 GSM8K test 점수로 해석하면 안 된다.

## 완료된 비교

| 설정 / split | n | 순정 → 교체 정답 수 | 개선 / 악화 | 평균 시간, 초 | 시간 변화 |
| --- | ---: | --- | --- | --- | ---: |
| 초안 KV, 지연 16 / 개발 | 4 | 2 → 3 | 2 / 1 | 70.37 → 58.35 | −17.1% |
| 같은 설정 / 독립 test A | 8 | 5 → 5 | 0 / 0 | 51.19 → 55.23 | +7.9% |
| 초안 KV, 지연 1 / 개발 | 4 | 2 → 3 | 2 / 1 | 66.39 → 55.19 | −16.9% |
| 같은 설정 / 독립 test B | 8 | 6 → 6 | 1 / 1 | 55.35 → 54.55 | −1.4% |

test A와 B는 서로 다른 문항이다. 각각 결과를 보기 전에 seed 순서 첫 4개와 입력 길이 상위 4개를 고정했다. ID는 해당 [지연 16 설정](../configs/draft-swap-heldout8.json)과 [지연 1 설정](../configs/draft-swap-delay1-heldout8.json)에 있다. test B는 A의 8개를 제외한 다음 cohort다. 각 설정의 별도 비교로 보고하며 하나의 개선 점수로 합치지 않는다. 작은 표본과 단일 실행으로 통계적 일반화나 미세한 속도 개선을 주장하지 않는다.

## 의미 있는 사례와 한계

- 개발 1292: 같은 2,048 상한에서 순정은 답을 못 냈고, 지연 1 교체는 1,038토큰으로 정답을 냈다. 개발 3331도 1,370토큰 이하에서 정답을 냈다. 전체 cohort에는 새 실패도 있으므로 이 사례만을 성능 개선 결과로 제시하지 않는다.
- 독립 test B의 640은 새로 맞혔고 1306은 새로 실패했다. 정답 수는 그대로였다.
- 개발 1202는 잘못된 초안의 큰 산술 오류 때문에 추가 검증이 길어졌다. 2,048 상한에서 교체는 실패했고, **같은 설정의 4,096 상한 검사**에서는 순정 2,009토큰, 교체 2,078토큰으로 둘 다 정답이었다. 작은 예산에서의 오류는 여전히 오류이며, 긴 예산 결과로 지우지 않는다.
- 저장된 초안 단독의 사후 정답 수는 개발 3/4, test A 5/8, test B 5/8이다. main이 일부 잘못된 초안을 교정했지만 순정 main의 test 정확도를 넘지는 못했다. 초안 단계 평균 시간은 약 4초이며 별도 standalone 실행 시간 벤치마크는 아니다.
- GPU peak 약 15.4 GiB는 paired 프로세스에 두 모델이 resident인 값이다. baseline은 초안 호출을 하지 않지만 idle 모델도 메모리에 있다. 순정 단일 모델의 최소 메모리로 해석하지 않는다.

## 검증한 실행 의미

CPU 검사 45개, lint, wheel build, 실제 paired GPU 실행을 통과했다. `scripts/verify_draft_cache.py`는 실제 36개 attention layer에서 다음을 확인했고 완료 checkpoint 재개도 확인했다.

- 기존 token ID와 prefix K/V 보존, 모든 layer의 선택 span 변경.
- suffix K/V 보존, 추가 미래 생성 뒤에도 기존 suffix 보존.
- 첫 미래 token은 편집 전 logits를 사용.
- 실제 branch·rollout·delay 후 첫 미래 token까지 baseline이 Hugging Face `model.generate`와 일치.
- 두 모델의 eval/frozen 상태와 gradient 없음.

실제 SIGTERM으로 중단된 이전 연구 run도 완료된 5개를 유지해 재개했으며 해당 checkpoint SHA256이 그대로였다. 학습이나 optimizer 업데이트는 제공 방법에 없다. 모델 가중치의 CPU 전후 동일성도 기존 작은 Qwen 통합 검사로 확인한다.

## 제외한 후보

고정 top-2 probe는 개발 4/4를 유지했지만 실제 교체가 없고 시간 +20.5%였다. 숫자 분기 probe도 교체가 없고 +10.3%였다. 숫자 분기 일부는 2023/2024 같은 예시 연도만 바꿨다. 일반 guidance의 32·128토큰 교체는 개발의 전체 정답 수를 늘리지 못했다.

추가 수학 판단기 gate는 실제 개발 초안 4개에서 2/4만 원하는 결정을 했다. 해로운 산술 초안은 거부했지만 유용한 초안 3개 중 2개도 거부했다. 따라서 이 필터를 추가해 더 큰 QA run을 진행하지 않았다. 합성 gate 통과가 실제 QA 선택기의 신뢰성을 뜻하지 않는다. 과정은 [연구 기록](PROBE_RESEARCH.ko.md)에 있다.

## 이전 버전의 재현

아래 결과의 정확한 재현은 측정한 package revision `9f65ce8`에서 수행한다. 이후 QA 판단 기준과 draft 준비가 일반화됐으므로 최신 code의 결과와 섞지 않는다. 프로젝트 루트에서 새로운 run 이름으로 실행한다. 모델 두 개가 필요하므로 할당된 장치와 여유 메모리를 먼저 확인한다.

```bash
uv sync --frozen
uv run --frozen dirty-swapping setup --datasets gsm8k --config configs/draft-swap-delay1-dev4.json
uv run --frozen dirty-swapping run --datasets gsm8k --split development \
  --config configs/draft-swap-delay1-dev4.json --run-name my-draft-dev4
uv run --frozen dirty-swapping run --datasets gsm8k --split evaluation \
  --config configs/draft-swap-delay1-heldout8.json --run-name my-draft-test8
uv run --frozen python scripts/verify_draft_cache.py \
  --config configs/draft-swap-delay1-dev4.json --run-name my-cache-check
```

출력·checkpoint·log는 `outputs/`와 `logs/`에 저장된다. 중단한 QA는 `dirty-swapping resume --run outputs/<run-name>`, 통합 검사는 `verify_draft_cache.py --resume outputs/<run-name>`으로 재개한다. config와 source를 바꿨다면 새 run을 만든다. 위 확인 cohort는 이미 연구에서 사용된 문항이므로 새 방법의 독립 확인에 재사용했다고 주장하면 안 된다.
