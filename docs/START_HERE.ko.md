# 처음 시작하기: 버려진 토큰의 재사용

레포를 처음 받았다면 루트에서 `./start.sh`를 실행하세요. 설치 없이 문서·설정·코드의 위치와 첫 실행 명령을 한 번에 보여줍니다.

이 프로젝트의 질문은 **생성 도중 선택하지 않은 토큰과 그 KV cache를 버리지 않고 다시 쓰면 text QA의 최종 답이 좋아지는가?**입니다. 정확도만 높아도 대안 생성에 시간이 지나치게 들면 실용적이지 않습니다. 따라서 모든 실험은 **최종 답 정확도**와 **순정 추론 대비 전체 시간**을 함께 봅니다. [Group 18 제안서](proposal.pdf)의 Dirty Swapping은 이 큰 주제를 시험하는 첫 기준 구현입니다.

## 지금 제공하는 기준 구현

모델은 Qwen3-4B-Thinking-2507이며 가중치를 학습하거나 바꾸지 않습니다. `<think>` 안의 정해진 위치에서 동일한 prefix로부터 다음 토큰 상위 후보 4개의 짧은 경로를 만듭니다. 확률이 가장 높은 경로로 본문을 계속 생성하고, 128토큰 뒤 과거 32토큰 구간의 KV만 두 번째 후보의 KV로 교체합니다. **이미 생성된 텍스트와 교체 구간 뒤의 KV는 그대로 둡니다.** 기본 설정은 trace당 교체 1회입니다.

```text
주 경로:       A [B C] D → 이후 생성
버린 대안:     A [E F]
교체 뒤 텍스트: A  B C  D
교체 뒤 KV:    A [E F] D   (D의 기존 KV는 유지)
```

`baseline`은 같은 모델·프롬프트·greedy 생성 경로에서 대안 rollout과 KV 교체를 하지 않는 비교군입니다. 이 baseline은 순정 autoregressive 생성과 토큰 단위로 일치하는지 검증했습니다. `swap`은 대안 준비 비용까지 포함하는 실험군입니다. 방법의 자세한 조건은 [method.md](method.md)에 있습니다.

## 10분 시작 경로

Linux, NVIDIA GPU와 드라이버, `uv`가 필요합니다. 첫 다운로드에는 인터넷과 모델 저장 공간이 필요합니다. 프로젝트 루트에서 실행합니다.

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen dirty-swapping setup --datasets gsm8k
uv run --frozen dirty-swapping run --datasets gsm8k --limit 1 \
  --config configs/smoke.json --run-name first-smoke
uv run --frozen dirty-swapping report --run outputs/first-smoke
```

`uv sync --frozen`은 레포 내부 `.venv`에 잠금 파일의 의존성을 설치합니다. `setup`은 원본 데이터와 모델을 내려받고 해시를 검증한 뒤 입력을 준비합니다. `run`은 같은 문제를 baseline과 swap으로 실행합니다. 출력·로그·모델은 각각 `outputs/`, `logs/`, `models/` 등에 저장되며 Git에서 제외됩니다. 중단되면 `uv run --frozen dirty-swapping resume --run outputs/first-smoke`로 완료된 문제를 재사용합니다.

위 `smoke.json`은 빠른 동작 확인을 위해 **출력 64토큰, rollout 4토큰, 지연 4토큰**으로 줄인 설정입니다. 답이 나오기 전에 종료될 수 있으므로 이 결과의 정확도를 성능 수치로 쓰지 않습니다. 제안서 설정은 [default.json](../src/dirty_swapping/default.json)의 후보 4개·rollout 32토큰·지연 128토큰·reasoning 256토큰 뒤 분기와 **최대 출력 32,768토큰**입니다. 어려운 문제는 4,096토큰에서도 `<think>` 안에 남는 사례가 확인돼 긴 출력 한도를 기본값으로 둡니다. `uv run --frozen dirty-swapping config > my-config.json`으로 설정을 복사해 수정하고 `--config my-config.json`으로 실험할 수 있습니다. 입력·출력 토큰 상한과 모델 context 상한의 합도 확인하세요.

## 데이터셋 전체 풀

| 키 | 데이터셋 | 현재 상태 | 권장 용도 |
| --- | --- | --- | --- |
| `aime25` | AIME 2025, 30문제 | 기본 평가 지원 | 어려운 수학 추론 |
| `hmmt25` | HMMT February 2025, 30문제 | 기본 평가 지원 | 수학 일반화 확인 |
| `gsm8k` | GSM8K, train 7,473 / test 1,319 | 기본 평가 지원 | train은 개발, test는 최종 평가 |
| `gpqa` | GPQA Main, 448문제 | 기본 평가 지원 | 과학·전문지식 QA |
| `supergpqa` | SuperGPQA, 26,529문제 | 다운로드·어댑터 지원 | 더 큰 규모의 QA 확장 실험 |
| `longbench_v2` | LongBench v2, 503문제 | 다운로드·어댑터 지원 | 긴 문맥에서 오래된 suffix KV 영향 확인 |
| `math500` | MATH-500, 500문제 | 다운로드·어댑터 지원 | 수학 별도 평가 트랙 |

기본 네 데이터셋은 한 명령에서 선택할 수 있습니다: `uv run --frozen dirty-swapping setup` 뒤 `uv run --frozen dirty-swapping run --run-name <새-이름>`. `--datasets`로 나머지 세 트랙을 포함하거나 일부만 고를 수 있고 `--limit`는 작은 확인용입니다. AIME/HMMT/GPQA/MATH-500/SuperGPQA/LongBench는 이 레포에서 평가 데이터로 다룹니다. 최종 평가 문항의 정답을 대안 선택에 사용하지 마세요. 출처·고정 revision·SHA256, 정규화 필드와 split, 긴 문맥 자르기 규칙은 [datasets.md](datasets.md)와 [datasets.json](../src/dirty_swapping/datasets.json)에 있습니다. 현재 기본 입력 상한은 4,608토큰이며 LongBench v2 문맥 503개는 모두 이 상한에 맞춰 잘립니다.

## 반드시 함께 볼 두 성능 축

1. **QA 정확도:** 데이터셋별 `정답 수 / 완료 문제 수`, 기본 네 데이터셋의 macro accuracy, 그리고 같은 문제에서 `swap − baseline`의 percentage point 차이를 봅니다. 답을 끝내지 못한 사례와 교체가 건너뛰어진 사례도 셉니다.
2. **추론 비용:** 문제별 입력 처리 시작부터 최종 답 생성까지의 시간을 잽니다. 대안 rollout과 KV 교체 시간도 포함합니다. `시간 증가율 = (방법 시간 / 순정 시간 − 1) × 100%`로 순정 대비 비용을 표시하고, 평균·중앙값·p95와 피크 GPU 메모리를 함께 봅니다.

현재 실행기는 `baseline`과 `swap`을 같은 문제로 비교합니다. baseline의 토큰 경로는 순정 생성과 일치하지만 제어 코드의 작은 비용은 있을 수 있으므로, 절대적인 순정 시간 비교가 중요하면 같은 모델의 일반 `model.generate`도 별도로 측정하세요. 모델·프롬프트·토큰 상한·seed·GPU 조건을 맞추고 비교해야 시간과 정확도 차이를 해석할 수 있습니다. [competition.md](competition.md)에 제출·비교 규칙을 정리했습니다.

이 레포의 소표본 검증에서는 일곱 데이터셋 각 한 문제의 순정과 baseline이 256토큰 모두 일치했습니다. 제안서 설정에서 swap은 대안 rollout 준비만큼 더 오래 걸렸고, 긴 LongBench 입력도 GPU에서 실행됐습니다. 하지만 **정확도 보존은 아직 확인되지 않았습니다.** 짧은 MATH-500 문항에서 2,048토큰 상한일 때 baseline은 정답을 냈고 swap은 답을 끝내지 못했습니다. 4,096토큰에서는 둘 다 정답이었습니다. 완주율과 시간 비용을 정확도와 함께 보고해야 하는 이유입니다. 자세한 측정은 [검증 기록](implementation.md)을 보세요.

## 실험의 자유도

큰 틀은 **선택되지 않아 버려질 토큰 또는 경로의 정보를 재사용한다**는 것입니다. 그 안에서 분기 기준, 후보 수와 길이, 대안 선택기, 교체 위치와 횟수, KV를 쓰는 방식, 모델·프롬프트·데이터셋 등을 탐색할 수 있습니다. 다만 **현재 v1 실행 코드가 곧바로 지원하는 범위**는 후보 수·rollout 길이·분기 offset·지연·토큰 상한·모델 경로 같은 설정값의 변경입니다. 대안 선택은 top-2, 교체는 trace당 1회, full-attention `DynamicCache`라는 제약이 있습니다. 다른 선택기·여러 번 교체·다른 cache 구조는 `engine.py`, `cache.py`, `backend.py`의 구현과 검증을 바꿔야 합니다.

연구 설정을 자유롭게 바꾸더라도 **그 설정과 정확히 맞는 순정 경로**를 같이 실행해 개선이 재사용 방법에서 왔는지 확인하세요. 모델이나 학습 방식까지 바꾼다면 변화 내용을 따로 공개해 모델 자체의 이득과 재사용의 이득을 구분하세요. 첫 수정은 [default.json](../src/dirty_swapping/default.json)을 복사해 한 변수만 바꾸는 ablation부터 시작하면 결과를 읽기 쉽습니다.
