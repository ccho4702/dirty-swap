#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

printf 'Dirty Swapping | 시작 지도 / Start map\n'
printf 'Repository: %s\n\n' "$repo_dir"

cat <<'EOF'
목적 / Purpose
  버려질 후보 토큰의 KV를 재사용해 text QA를 개선할 수 있는지 실험합니다.
  함께 볼 두 수치: 최종 답 정확도와 순정 추론 대비 전체 시간.
  현재 기준 방법은 training-free이며, 다른 재사용 방식도 연구할 수 있습니다.

읽을 자료 / What is in this repository
  docs/START_HERE.ko.md             처음 읽는 한국어 안내: 목적, 셋업, 실험 자유도
  README.md                         실행 명령과 저장 경로
  docs/datasets.md                  지원 데이터셋 7개, 원본 형식, split, 전처리
  docs/competition.md               답 추출·채점, 시간 측정, 공정 비교 규칙
  docs/method.md                    현재 KV 교체 방법과 baseline 조건
  docs/implementation.md            실제 검증 결과와 아직 검증되지 않은 범위
  docs/RESEARCH_PLAN.ko.md          시간 제한 아래의 실험 우선순위와 중단 기준
  docs/PROBE_RESEARCH.ko.md         changho의 실험용 probe/no-swap 선택기
  docs/proposal.pdf                 Group 18 프로젝트 제안서
  skills/onboard-dirty-swapping/   코딩 에이전트용 온보딩 SKILL.md

코드와 설정 / Code and configuration
  src/dirty_swapping/default.json   기준 실험 설정
  src/dirty_swapping/datasets.json  데이터 출처·revision·SHA256
  configs/smoke.json                빠른 동작 확인용 설정 (정확도 벤치마크 아님)
  src/dirty_swapping/              데이터 준비, 생성, KV 교체, 평가 코드
  tests/                            CPU 단위·통합 테스트

첫 실행 / First run (repository root에서)
  uv sync --frozen
  uv run --frozen python -m unittest discover -s tests -v
  uv run --frozen dirty-swapping setup --datasets gsm8k
  uv run --frozen dirty-swapping run --datasets gsm8k --limit 1 --config configs/smoke.json --run-name first-smoke
  uv run --frozen dirty-swapping report --run outputs/first-smoke

먼저 docs/START_HERE.ko.md를 읽으세요. 이 명령은 설치·다운로드·GPU 실행을 하지 않습니다.
EOF
