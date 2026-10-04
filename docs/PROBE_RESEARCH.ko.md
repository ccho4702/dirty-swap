# 실제 교체 상태의 probe 선택기 — changho 연구

공식 `main`의 파일 상태는 연구 계획 push 이전 `b63bb2b`와 동일하게 복원했다. 이후 연구는 `changho`에서 진행하며, 아래 방법은 공식 baseline의 검증된 성능을 대체하는 결과가 아니다.

## 방법

현재 probe 선택기는 수학 QA만 지원한다. 같은 prefix에서 top-1과 top-2의 32토큰 rollout을 만들고 top-1으로 주 경로를 이어간다. reasoning 256토큰에서 분기한 뒤 rollout과 128토큰 지연이 끝나면, 현재 cache의 사본 두 개를 만든다. 하나는 no-swap, 다른 하나는 과거 32토큰 KV만 대안으로 교체한다. 기존 텍스트와 그 구간 뒤의 KV는 보존한다.

두 trial에서 새 미래 64토큰을 생성한다. 첫 미래 토큰은 기존 logits를 사용하므로, 실제 교체 효과는 그 뒤 토큰에서 나타날 수 있다. 미래 토큰열이 같으면 교체를 거부하고 judge를 실행하지 않는다. 다르면 질문, 기존 reasoning의 마지막 256토큰, 두 미래 텍스트만 같은 frozen 모델의 judge에 준다. gold answer는 전달하지 않는다.

judge는 최대 128토큰으로 계산·논리적 진전을 비교한 뒤 `A`, `B`, 동률 `T`에 대한 조건부 점수를 낸다. 후보 순서를 뒤집어 두 번 판단하며, 두 판단 모두 대안이 원래 경로와 동률보다 최소 0.15 높은 점수일 때만 교체를 채택한다. 이 점수는 **정답 확률로 보정된 값이 아니다**. 입력 상한 초과, 판단 불일치, 동률은 no-swap을 택한다. 선택된 trial의 probe와 cache를 그대로 이어 써서 미래를 다시 생성하지 않는다.

## scorer gate

실제 QA를 대량 실행하기 전에 간단한 산술·비율 예시를 검사했다. 기대 선택 라벨은 judge 호출 뒤에만 평가에 사용했다.

| 버전 | 올바른 수정 채택 | 해로운 수정 거부 | 동률 거부 | 결정 |
| --- | --- | --- | --- | --- |
| reasoning 없는 직접 점수 | 4/6 | 6/6 | 4/4 | 실패 |
| 일반 프롬프트 + 판단 64토큰 | 2/6 | 6/6 | 4/4 | 실패 |
| 계산부터 확인하는 프롬프트 + 256토큰 | 6/6 | 6/6 | 4/4 | gate 통과, 비용 큼 |
| 새로운 숫자·잘못된 이전 reasoning + 128토큰 | 6/6 | 6/6 | 4/4 | 별도 gate 통과 |
| 패키지 구현의 재현 검사 | 6/6 | 6/6 | 4/4 | gate 통과 |

128토큰 별도 gate의 16개 비교에는 약 165초, 패키지 검사에는 약 167초가 들었다. 한 비교는 두 후보 순서의 판단을 포함한다. 이 작은 합성 gate는 명백한 오류와 위치 편향을 걸러내는 최소 조건이며, QA 정확도 개선의 증거가 아니다.

```bash
uv run --frozen python scripts/verify_probe_judge.py --run-name judge-check
# 중단 후: 같은 config와 source에서
uv run --frozen python scripts/verify_probe_judge.py --resume outputs/judge-checks/judge-check

uv run --frozen dirty-swapping run --datasets gsm8k --split development --limit 4 \
  --config configs/probe-preference.json --run-name probe-dev4
```

## 검사와 비용

단위 검사에서는 선택/거부/동일 probe, 순서 편향, 긴 judge 입력 거부, 원래 cache와 suffix 보존, 선택된 probe의 commit, 출력 예산 검사를 확인한다. 실제 GPU smoke에서는 선택기가 no-swap을 고르는 경로도 실행됐다.

`report.json`의 `mean_probe_s`는 두 trial 생성과 임시 cache 복사 시간을 포함한다. 선택된 trial의 미래 생성은 최종 생성의 일부이므로 전부 추가 overhead인 것은 아니다. `mean_judge_s`와 `mean_alternative_rollout_s`를 별도로 보고 전체 `latency_s`로 비용을 판단한다. raw case에는 두 probe, 판단용 reasoning, 순서별 점수, 최종 선택이 저장된다.

동일 모델 judge가 실제 오류를 놓치거나, 잘린 미래를 평가할 근거를 찾지 못할 수 있다. no-swap 선택지가 있어도 성능 보존은 보장되지 않는다. 고정된 개발 cohort에서 실제 swap 채택 수, 답 완주율, paired wins/losses와 시간을 확인한 후 필요한 변경 하나만 진행한다.

## 고정 분기의 첫 실제 결과

seed로 정해진 GSM8K train 첫 4문제에서 baseline과 선택기 모두 4/4 정답이었다. 그러나 선택기는 4개 모두 교체를 거부했고, 평균 시간은 67.8초에서 81.7초로 약 20.5% 증가했다. 이 설정의 표본을 늘리지 않는다. 기록된 두 probe는 대부분 같은 추론의 재표현이거나 계산을 시작하기 전의 fragment여서, 고정 256토큰 분기가 유용한 대안을 준비하지 못하는 것이 구체적 실패였다.

다음 변경은 `configs/probe-numeric.json`의 `numeric_ambiguity` 분기다. reasoning 64–1,536토큰에서 top-1/top-2가 서로 다른 숫자 문자열이고 상대 확률 비율이 0.05 이상인 최초 지점에서 분기한다. 비율은 후보 plausibility 기준이며 정확도 신호가 아니다. 해당 지점이 없으면 교체하지 않는다. 그 밖의 rollout 32·지연 128·probe 64·judge 128은 유지한다.

`configs/probe-numeric-dev4.json`에는 결과를 보기 전에 고정한 개발 cohort ID가 있다: 기존 두 control `gsm8k-train-1292`, `gsm8k-train-3522`와 train 입력 길이 상위 두 문항 `gsm8k-train-3331`, `gsm8k-train-1202`다. 긴 문항은 gold나 baseline 성공 여부를 보고 선택하지 않았다. `execution.case_ids`는 지정된 split에 없는 ID를 거부하며, 이 부분 cohort가 완료돼도 full-suite primary score를 내지 않는다.

숫자 분기 점검 중 동일한 logits에서 `topk`의 첫 토큰이 `argmax`와 다를 수 있는 재현 가능한 문제가 발견됐다. 연구 backend는 첫 후보를 항상 순정 `argmax`로 고정하고 나머지 후보만 top-k에서 고른다. tied-logit baseline parity 회귀 테스트를 추가했다. 수정 전 숫자 run은 결과가 저장되기 전에 중단하고 새 source identity로 다시 실행했다. 이 수정은 `changho`에만 있으며 공식 `main`에는 아직 반영하지 않았다. 중복 SIGINT/SIGTERM이 원자적 status cleanup을 다시 끊는 상황도 handler를 복원하는 context로 처리한다.

```bash
uv run --frozen dirty-swapping run --datasets gsm8k --split development \
  --config configs/probe-numeric-dev4.json --run-name numeric-dev4
```

## 숫자 분기의 결과와 다음 gate

위 고정 개발 4문제, 4,096 출력 토큰에서 baseline과 numeric probe 모두 4/4였다. 실제 교체는 0회: judge 거부 2회, 숫자 분기 없음 1회, 동일한 미래 1회다. 평균 시간은 74.25초에서 81.86초로 10.3% 늘었다. 이 설정도 확장하지 않는다.

`configs/guided-step-dev4.json`은 다음 좁은 가설을 검사한다. 64–512토큰 안의 첫 줄 경계에서 같은 prefix를 복사해 `Let's compute the quantities directly.\n`를 짧게 teacher-force하고, 나머지를 greedy로 생성해 32토큰 대안을 만든다. 원래 경로를 16토큰 더 이어간 뒤 대안의 과거 32토큰 KV를 한 번 교체한다. 원래 텍스트와 기존 suffix KV는 보존하며, gold·judge·학습을 쓰지 않는다. prefix가 길거나 EOS/think 종료를 포함하면 쓰기 전에 거부한다.

이번 개발 gate는 **양쪽 모두 2,048 출력 토큰**의 비용 제한 설정이다. 긴 추론의 완주율을 높일 수 있는지를 본다. 4,096/32,768 토큰 또는 무제한 추론 대비 정확도 개선으로 해석하면 안 된다. 동일 개발 ID를 유지하며, QA 결과를 보기 전에 설정을 고정했다. 이 묶음은 후보 생성·분기 위치·지연을 함께 바꾼 가설로, 각 요소의 독립 효과를 주장하지 않는다.

```bash
uv run --frozen dirty-swapping run --datasets gsm8k --split development \
  --config configs/guided-step-dev4.json --run-name guided-dev4
```

개발 gate에서 개선이 있으면 설정을 고정해 `configs/guided-step-heldout8.json`의 별도 test 8문제로 확인한다. test 결과를 보지 않고 seed 첫 4개와 입력 길이 상위 4개를 선택했다: 1259, 1038, 524, 379, 1077, 1209, 1199, 1176. 전체 GSM8K test 점수가 아니라 작은 독립 확인 cohort다.

32토큰 guided 개발 gate에서는 baseline 2/4, swap 2/4, wins/losses 0/0이었다. 실제 교체는 4/4 실행됐고 평균 시간은 68.32→66.34초였으나 작은 표본의 시간 변동도 있어 속도 개선을 확정하지 않는다. 더 큰 표본으로 확장하지 않고 **rollout 길이만 128**로 늘린 `guided-step128-dev4.json`을 다음 gate로 고정한다. 비용 상한·질문·분기·지연·guidance는 그대로다. 대안 계산 텍스트도 raw case에 보존한다. heldout ID는 이미 정한 같은 8개를 쓰며 결과를 보고 고르지 않는다.
