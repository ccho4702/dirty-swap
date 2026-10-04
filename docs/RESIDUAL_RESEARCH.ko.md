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

## 초반 교체의 수학 결과와 다음 진단

수학 4개는 3/4→3/4, wins/losses 0/0이었다. 교체는 4개 모두 적용됐다.
양쪽 모두 3개는 최종 답을 완성했고, 나머지 1개는 8,192토큰 상한에서 reasoning이
끝나지 않았다. 평균 시간은 173.19→179.73초(+3.8%)다.

GPQA 2개는 0/2→0/2, 평균 시간 325.91→299.83초(-8.0%)였다. 순정은 최종
답을 추출한 사례가 없었고, 교체는 하나의 최종 답을 냈지만 오답이었다.
따라서 정확도 개선도, 일반 QA 성능 보존의 충분한 증거도 얻지 못했다.
6개 모두 실제 교체가 적용됐으며 전체 wins/losses는 0/0이다.

Offline 진단에서 미완료 문항 `math500-test/intermediate_algebra/1422.json`의
generic draft는 정답을 포함했다. 저장된 명시적 final answer를 기존 Math-Verify로
확인했다. 정답 label은 이 사후 진단에만 사용했으며 생성/교체 정책에 제공하지 않았다.
완료되지 않은 main 출력을 draft 답으로 대체해 점수를 올리지 않는다.

추가로 저장된 main reasoning을 확인하니 순정·교체 모두 이미 정답을 언급한 뒤
재검토를 반복하고 있었다. 따라서 지식 전달 실패라고 단정할 수 없으며, 이 사례의
직접적인 실패는 reasoning 종료와 최종 답변 완료다. 늦은 교체가 성공해도 우선
동일 예산에서의 completion 이득으로 보고해야 한다. 잘못된 reasoning을 수정한
증거로 해석하지 않는다. 단순 reasoning-budget 종료 제어와도 비교할 필요가 있다.

다음은 `residual-draft-late-diagnostic.json`으로 **분기 구간만** 64~512에서
4,096~4,608 reasoning tokens로 옮기는 진단이다. 혼합 0.5, generic draft,
구간 상한 256, suffix 16, 전체 출력 8,192는 그대로다. 이는 출력 예산의 절반까지
reasoning이 지속될 때 개입하는 공통 규칙이며 task 이름을 검사하지 않는다.
일찍 완료된 응답은 교체 없이 종료한다.

해당 문항은 결과를 본 뒤 선택했으므로 이후에는 개발용 진단 사례다.
이 한 문항에서 좋아져도 독립 benchmark 향상으로 세지 않는다. 유망할 때만 같은
설정을 미사용 문항에 고정해 평가한다. 기능 구현은 기존 코드의 configuration으로
가능하며 새로운 학습을 추가하지 않는다.

별도의 [범용 reward model 문헌 검토](REWARD_RESEARCH.ko.md)는 후보 집합에 실제
정답 이득이 생기는지 확인한 뒤 선택 기능을 추가할 때 참고한다.

## 순정 sampling 진단

[공식 Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)의
temperature 0.6 / top-p 0.95 / top-k 20으로 같은 개발 진단 문항을 native
`model.generate`에 넣었다. 모델·prompt·seed·출력 한도는 유지했다.
Greedy는 8,192토큰에서 답변 미완료였고, sampling은 8,136토큰/316.09초에
최종 정답으로 완료됐다. **이것은 KV 교체의 성과가 아니다.** 결과를 본 뒤 고른
한 문항의 진단이므로 일반 benchmark 향상으로 보고하지 않는다.

이 결과만으로 greedy가 모든 문제에서 나쁘다고 결론 내릴 수 없지만, 이후 KV
방법의 주장은 권장 sampling을 양쪽에 적용한 순정 비교도 포함해야 한다.
현재 일반 runner의 main은 여전히 greedy다. 이 진단 명령만 native sampling을
사용하며 runner에 sampling을 지원했다고 잘못 설명하지 않는다.

```bash
uv run --frozen python scripts/check_native_sampling.py \
  --config configs/native-sampling-diagnostic.json --run-name native-check
uv run --frozen python scripts/check_native_sampling.py --resume outputs/native-check
```

실제 생성 도중 SIGINT 중단 후 동일 설정·seed로 prompt부터 재생해 완료하는
검사를 통과했다. 완료 checkpoint는 SHA 검증 뒤 모델을 읽지 않고 반환한다.
잘못된 sampling 설정은 run 생성 전에 거부한다. 이 진단의 원본 출력은
`outputs/native-sampling-diagnostic-20261005`에 있으며 공개 요약에는 정답 문자열을
넣지 않는다. 코어 55개 단위 검사, 실제 GPU 36-layer cache 보존 및 native greedy
일치 검사를 통과했다. 늦은 교체 configuration은 별도의 개발 진단이다.
