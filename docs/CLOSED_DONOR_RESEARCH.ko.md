# 새 가설: 완료된 짧은 답변의 KV를 재사용

후반 50% 혼합 진단은 0/1→0/1이었다. 순정 318.76초, 교체 333.55초(+4.6%)였으며
교체는 think를 닫았지만 최종 답을 8,192토큰 안에 완성하지 못했다. 이 문항은
이미 관찰한 개발용 사례이며 독립 평가 성적으로 세지 않는다.

이번 후보는 main의 공유 prefix에서 `</think>`와 짧은 generic draft를 처리해
완료된 답변 단계의 KV를 만든다. 그 후보 KV로 main의 같은 길이 과거 구간을
한 번 교체한다. main의 기존 텍스트, prefix KV, 뒤쪽 KV, 기존 next logits를
보존한다. 강제 종료 tag는 후보에만 사용한다. 후보가 phase 전환과 간결한 답변의
정보를 전달해 불필요한 재검토를 줄일 수 있다는 가설이다. 성능 증거는 아직 없다.

모든 task에 같은 조건을 쓴다: reasoning 4,096~4,608토큰의 첫 줄 경계, 후보
최대 256토큰, suffix 16토큰, 강도 1.0, 출력 한도 8,192토큰. 그 전에 완료된
응답은 그대로 종료한다. 별도 frozen Instruct 모델은 질문만 보고 같은 prompt로
draft를 만든다. 정답 label과 dataset 이름을 정책에 주지 않는다.

## 이번 회차의 새로운 문헌 5편

Soft / residual / reward 문헌 목록과 중복하지 않는 아래 논문을 새로 확인했다.
정식 출판은 모두 2025년 이후다. 원 논문들의 학습·텍스트 제어 방법을 그대로
재현한 것은 아니며, 그 성적을 우리의 cache 교체 성적으로 제시하지 않는다.

| 논문 / 학회 | 확인한 방법 | 반영 또는 보류 |
| --- | --- | --- |
| [s1: Simple test-time scaling](https://aclanthology.org/2025.emnlp-main.1025/), EMNLP 2025 | SFT 뒤 생각 종료/연장 token으로 budget forcing | 완료 신호가 reasoning 길이에 영향을 주는 근거. 우리 main에 visible tag를 넣는 것과는 다르며, 향후 budget forcing과 비교할 수 있음 |
| [Thinkless: LLM Learns When to Think](https://proceedings.neurips.cc/paper_files/paper/2025/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html), NeurIPS 2025 | 짧은/긴 reasoning mode, control loss와 response loss를 나눈 DeGRPO | task 이름 대신 공통 mode 전환을 검토할 근거. 추가 RL 훈련은 현재 구현하지 않음 |
| [Dynamic Early Exit in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html), ICLR 2026; 2025 preprint | transition에서 trial answer confidence로 조기 종료; QA와 coding 등 여러 benchmark | 짧은 trial answer를 중간에 생성하는 방향을 참고. confidence gate는 구현하지 않았으며 우리의 KV 재사용이 같은 효과를 낸다고 가정하지 않음 |
| [ShorterBetter](https://proceedings.neurips.cc/paper_files/paper/2025/file/377ca194437916417e3dbe0dc67254f1-Paper-Conference.pdf), NeurIPS 2025 | sampled response 중 가장 짧은 정답 길이를 dynamic RL reward로 사용 | 짧아지는 것만으로 성공 판정하지 않고 최종 정답과 같이 평가. 추가 모델 훈련은 보류 |
| [How Far Are We from Optimal Reasoning Efficiency?](https://papers.nips.cc/paper_files/paper/2025/hash/1022661f3f43406065641f16ce25eafa-Abstract-Conference.html), NeurIPS 2025 | reasoning efficiency frontier와 REO-RL, quality/length tradeoff | 정확도·완료율·전체 비용을 같이 보고. 추가 후보/모델 비용을 숨기지 않음 |

추가 참고로 [Overthinking 연구](https://proceedings.mlr.press/v267/chen25bx.html)
(ICML 2025)도 확인했다. 위 다섯 편의 중복 집계에 포함하지 않았다.

## 검증 및 판정

`generation.sampling`을 양쪽 main에 적용했다. HF temperature/top-k/top-p
filters와 trace별 private RNG를 사용하며, fork는 RNG 상태도 복사한다.
분기 첫 token도 샘플링하므로 순정에서 argmax를 강제하지 않는다. 후보 생성은
main의 RNG를 진행시키지 않는다. sampling 변경으로 얻은 이득은 KV 효과로 세지 않는다.

코어 단위 검사 62개와 실제 GPU 검사를 통과했다. GPU에서는 sampled baseline의
native generate 일치, 강도 0 identity, 모든 layer의 실제 span 변경, text 및
prefix/suffix/기존 logits 보존을 확인했다. 후보의 EOS, 중복 종료 marker,
잘못된 marker 위치·설정은 거부한다. 학습은 하지 않았고 두 모델 weights는 frozen이다.

개발 진단은 `closed-donor-dev1.json`이다. 여기서 기능이나 시간이 좋아져도 독립
성능 개선으로 보고하지 않는다. 미사용 MATH-500 4개와 GPQA 2개를 결과 확인 전에
`closed-donor-transfer6.json`에 고정한다. 성공 기준은 수학 wins > losses와
일반 QA에서 관측된 회귀 여부다. 작은 표본의 무변화는 QA 보존 증명이 아니다.
답변 미완료→정답과 완료된 오답→정답도 구분해 설명한다.

```bash
uv run --frozen python scripts/verify_draft_cache.py \
  --config configs/closed-donor-smoke.json --run-name closed-cache-check
uv run --frozen dirty-swapping run --config configs/closed-donor-dev1.json \
  --datasets math500 --split evaluation --run-name closed-dev
uv run --frozen dirty-swapping run --config configs/closed-donor-transfer6.json \
  --datasets math500 gpqa --split evaluation --run-name closed-transfer
```

정확도 비교의 baseline은 같은 sampling·seed·출력 예산이다. 모델은 처음에 한 번
읽고, 결과는 case/arm 단위로 원자적으로 저장한다. 중단한 case는 같은 seed의
prompt부터 재생하며 완료된 case는 건너뛴다. source/config/runtime을 바꾸면 재개를
거부한다. main text를 직접 종료시키는 별도 비교군은 아직 구현하지 않았다.

## 완료된 개발 진단

동일 sampling·seed·8,192-token 조건에서 정답은 1/1→1/1이었다.
생성 길이는 8,136→4,288토큰, 전체 시간은 329.05→174.89초(-46.8%)였다.
초안 생성·재인코딩·교체 비용을 포함한 시간이다. 개발 문항의 속도 결과이며,
정확도 상승이나 독립 평가의 개선으로 세지 않는다.

순정의 전체 8,136-token 생성 문자열이 이전 native `generate` 기록과 일치했다.
별도로 GPQA 입력 82토큰과 LongBenchV2 입력 4,577토큰에서 첫 64개의 sampled
출력이 native 생성과 일치했다. 이는 해당 입력의 decoding 검증이며 QA 점수가 아니다.

교체 출력은 마지막에 명시적인 boxed answer와 EOS를 생성했지만 visible
`</think>`를 생성하지 않았다. 기존 scorer는 EOS로 끝난 응답의 마지막 boxed
answer를 채점하며 양쪽에 같은 규칙을 적용한다. 따라서 `eos_ended=true`와
`thinking_complete=false`를 함께 기록했다. closing tag를 전제로 출력 내용을
분리하는 downstream client에서는 이 형식 차이를 처리해야 한다.

측정과 검증은 [공개 개발 요약](../outputs/closed-donor-development-summary-20261005/summary.json)에
보존했다. 원문 reasoning이나 정답 label은 포함하지 않는다. 사전 고정한 새 6문항
평가는 `outputs/closed-donor-transfer6-20261005`에서 진행 중이다.
