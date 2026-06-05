# Attack-team handoff (2026-06-05)

공격팀 → 방어팀. 텍스트 요약/분석은 여기, 이미지가 든 전체 패키지(zip)는 GitHub Release 참고.

## attacks/  — 공격별 요약
- `*_summary.csv` : 공격 sweep 요약 (성공률, similarity, l2/linf, time)
- `*_attack_summary.csv` : handoff 패키지에 포함된 verification_attack_summary

## analysis/ — 공격-방어 통합 분석 (공격팀 관점)
- `smoothing_bypass_pgd_plain_s20.csv` / `smoothing_bypass_pgd_adaptive_smoothing.csv`
  : plain PGD vs adaptive PGD의 Gaussian smoothing(r=3) 우회율
  : 결론 — adaptive는 eps0.020에서 우회율 96%로 smoothing 방어 무력화 (docs/adaptive_smoothing_bypass_2026-06-05.md)

## 전체 패키지 (adv/source/target/perturbation 이미지 포함) — GitHub Release
- `facenet_pgd_png_package.zip` (PNG, 성공 125) ← DAE/DiffPure 방어 입력용 핵심
- `facenet_fgsm_package.zip`, `facenet_adv_training_package.zip`
