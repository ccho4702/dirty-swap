# 데이터셋 구분 없는 KV 교체 방법

연구의 공통 조건은 **질문 유형과 데이터셋을 바꿔도 같은 후보 생성, 교체, 선택 정책과 기준을 사용하는 것**이다. 수학용 지시문, 숫자 분기, 특정 문제를 위한 규칙은 이전 탐색용 대조군이며 일반 방법론의 근거로 삼지 않는다. 원래 목표는 최종 QA 정확도 향상이고 시간은 함께 측정한다.

## 현재 실행 가능한 공통 경로

`configs/general-probe.json` 하나를 모든 지원 QA 데이터셋에 적용한다.

1. 동일한 frozen 모델과 prefix에서 main 및 native 대안 rollout을 만든다.
2. main을 이어간 뒤 현재 상태의 사본 두 개를 만든다. 하나는 **no-swap**, 하나는 과거의 동일 길이 KV span만 대안으로 교체한다. 기존 텍스트와 suffix KV는 보존한다.
3. 두 실제 상태에서 같은 길이의 미래를 생성한다. 교체 전 대안 텍스트 자체의 좋고 나쁨으로 선택하지 않는다.
4. 동일한 QA 판단기가 질문·제공 문맥·이전 reasoning·두 미래를 보고 **근거와의 일치, 사실적 일관성, 추론 타당성, 답으로의 진전**을 비교한다. 질문 종류나 dataset ID에 따른 판단 기준 변경은 없다. 후보 순서를 뒤집어 확인하고, 명확한 개선이 없으면 no-swap을 택한다.
5. 선택한 미래와 cache를 그대로 이어 써서 최종 답을 생성한다.

`runner.py`의 정책 경로에서 math-only 조건을 제거했다. `draft_swap`도 완주·길이·cache 호환 조건으로 후보를 준비하며, 수학 boxed 답의 존재를 채택 조건으로 요구하지 않는다. `configs/general-draft.json`의 초안 지시문도 질문과 제공 문맥에 근거한 짧은 답을 요청하는 공통 문구다. 중심 비교는 별도 초안 모델 없이 native 후보를 사용하는 `general-probe`이다.

| 계층 | 데이터셋별로 처리할 수 있는 것 | 정책에서 고정하는 것 |
| --- | --- | --- |
| 입력·평가 adapter | 원본 필드, 선택지, 요구 출력 형식, 정답 추출·동등성 채점 | 정답은 후보 생성·선택에 제공하지 않음 |
| 후보·교체·선택 | 없음: 동일 config와 구현을 사용 | candidate count, span·delay·probe budget, 판단 기준, no-swap 선택지 |

답 형식에 맞춘 기본 프롬프트와 최종 채점은 평가 계층이다. 이를 수학/과학/장문마다 다른 교체 정책을 쓰는 근거로 삼지 않는다. cache layout 호환성 검사는 모델 adapter의 실행 조건이며 QA 분야에 따른 휴리스틱이 아니다.

## 일반 방법론의 목적함수

질문 `q`, 현재 상태 `s`, 교체 action `a`에서 생성되는 최종 답을 `Y_a`라 두면 목표는 다음과 같다.

```text
maximize    E[QA_correct(q, Y_policy)]
subject to  E[time_policy / time_plain] <= common_budget
            at most one past-cache edit
            existing text and downstream KV preserved
```

선택 정책은 실제 교체 상태에서의 미래로 **최종 답 정답률의 기대 변화**를 추정해야 한다. 낮은 entropy, 높은 token likelihood, 특정 숫자의 등장, 짧은 텍스트 자체를 QA 향상의 정의로 쓰지 않는다. no-swap은 필수 action이다. 검색·판단 비용은 no-swap을 택한 경우에도 전체 시간에 포함한다. 위 목적함수는 연구의 내부 최적화 기준이며 공식 평가에 임의의 합산 점수를 추가하지 않는다.

현재 구현의 frozen QA 판단기는 공통 기준을 실행하는 baseline이다. **잘 보정된 correctness 확률이나 학습된 value function이 구현됐다는 뜻은 아니다.** 앞으로의 우선순위는 공통 value/선택 정책을 개발 데이터의 실제 최종 QA 결과로 학습·보정하는 것이다. gold는 이 offline label 생성과 평가에만 쓰고 inference 정책에는 주지 않는다. main 모델 변경 없이 먼저 후보 선택 하나를 검증하고, 이후 필요할 때만 timing/span 정책을 학습한다.

## 일반성을 확인하는 실험 규칙

- 개발에서 결정한 scorer·정책·threshold·budget을 고정하고 다른 QA 분야에 적용한다. dataset ID, math/choice 여부, 문제별 규칙으로 분기하지 않는다.
- mathematical QA, science MCQA, context-grounded QA에서 같은 방법의 정확도와 전체 시간을 보고한다. 한 수학 데이터셋의 개선을 general QA 개선으로 부르지 않는다.
- 비교군은 순정 inference, 동일한 검색 비용을 쓰되 교체하지 않는 control, 정책이 고른 교체다. 추가 모델을 쓰면 그 모델 단독 결과도 따로 본다.
- 데이터셋별로 나쁜 결과를 제거하거나 정책을 바꿔 하나의 일반 성능으로 합치지 않는다. 이미 사용한 확인 문항은 다음 정책의 독립 확인으로 재사용하지 않는다.
- reward 학습 데이터도 여러 QA 유형이어야 한다. 현재 지원된 train은 GSM8K이며 이것만으로 비수학 일반성을 입증하지 않는다. 비수학 train의 후보는 [HotpotQA 공식 training/development 자료](https://hotpotqa.github.io/)다. **HotpotQA adapter와 critic 학습은 아직 구현하지 않았다.** GPQA·LongBench 같은 기존 evaluation gold를 학습 데이터로 돌려 쓰지 않는다.

## 같은 설정으로 실행

```bash
uv run --frozen dirty-swapping run \
  --config configs/general-probe-smoke.json \
  --datasets aime25 hmmt25 gsm8k gpqa math500 supergpqa longbench_v2 \
  --split evaluation --limit 1 --run-name general-seven-smoke
```

위 128-token smoke는 정책이 모든 데이터셋에서 같은 경로를 실행하는지 확인한다. QA 성능 벤치마크가 아니다. 학습된 일반 정책의 성능과 모델 간 전이는 아직 검증되지 않았다. 이전 수학 중심 탐색의 실제 성능 한계는 [완료 결과](RESULTS.ko.md)에 남겨 두었다.

## 완료된 공통 경로 검사

수학/선택형 질문이 동일 정책을 실행하는 회귀 검사를 포함해 CPU 47개가 통과했다. 같은 `general-probe-smoke.json`으로 지원 데이터셋 7개 각각 baseline/swap, 총 14개 arm 실행을 완료했다. 동일 probe는 동일한 최적화 규칙으로 judge를 생략했고, 차이가 난 probe에는 같은 QA 판단기가 적용됐다. 수학·과학·문맥·다중 추론의 통제 예시는 올바른 수정 4/4, 해로운 수정 거부 4/4, 동률 거부 4/4였다. 이는 실행 의미와 최소 판단 검사이며 QA 정확도 향상의 증거가 아니다.

```bash
uv run --frozen python scripts/verify_probe_judge.py --run-name general-qa-check
```

원자적 결과의 공통 정책 적용 범위는 [공개 JSON](../outputs/research-summary-20261004/summary.json)의 `dataset_independent_update`에 보존했다. 수학 중심의 이전 QA 수치는 해당 revision의 탐색 기록으로 유지한다.
