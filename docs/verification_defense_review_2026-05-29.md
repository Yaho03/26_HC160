# FaceNet Verification Defense Review - 2026-05-29

## 1. Received Files

Defense team shared `verification_defense.zip`.

Main contents:

- `verification_defense_summary.csv`
- `attack_handoff_jpeg_index.csv`
- `jpeg/verification_defense_jpeg.csv`
- `bitdepth/verification_defense_bitdepth.csv`
- `smoothing/verification_defense_smoothing.csv`
- `figures/vd_bar_defense_success.png`
- `figures/vd_bar_sim_drop.png`
- `figures/vd_heatmap.png`
- `verification_defense_report.md`

The result covers FaceNet verification attacks generated from the handoff package.

## 2. Important Evaluation Note

The original handoff package selected 212 successful attack samples based on attack-time tensor similarity.

However, after the adversarial images were saved as JPEG and reloaded by the defense team, 41 samples no longer crossed the FaceNet verification threshold.

Therefore:

| Basis | Attack-success samples | Failed after reload |
|---|---:|---:|
| Original attack-time tensor metric | 212 | 0 |
| Reloaded JPEG-file metric | 171 | 41 |

Defense evaluation should be interpreted using the JPEG-file metric because the defense team received image files, not in-memory tensors.

The defense team handled this by rebuilding `attack_handoff_jpeg_index.csv` and evaluating defenses against `accepted_after_attack` recomputed from the saved JPEG files.

## 3. Clean Interpretation Rule

The verification threshold is:

```text
threshold = 0.479662
```

For each sample:

```text
accepted_after_attack  = similarity_after_attack  >= threshold
accepted_after_defense = similarity_after_defense >= threshold

defense_success = accepted_after_attack == True
                  and accepted_after_defense == False
```

In other words, a defense is successful only when an attack image that was accepted as the target identity becomes rejected after defense.

## 4. Summary Results

Overall results based on 212 rows:

| Defense | Attack success before defense | Defense success | Still accepted after defense | Avg similarity drop |
|---|---:|---:|---:|---:|
| JPEG q=75 | 171 / 212 = 80.66% | 0 / 212 = 0.00% | 171 / 212 = 80.66% | 0.0002 |
| Bit-depth 4bit | 171 / 212 = 80.66% | 40 / 212 = 18.87% | 131 / 212 = 61.79% | 0.0491 |
| Gaussian smoothing r=3 | 171 / 212 = 80.66% | 167 / 212 = 78.77% | 4 / 212 = 1.89% | 0.3738 |

Epsilon-specific results:

| Defense | Epsilon | Samples | Attack success before defense | Defense success | Still accepted |
|---|---:|---:|---:|---:|---:|
| JPEG q=75 | 0.005 | 45 | 36 / 45 = 80.00% | 0 / 45 = 0.00% | 36 / 45 = 80.00% |
| JPEG q=75 | 0.010 | 167 | 135 / 167 = 80.84% | 0 / 167 = 0.00% | 135 / 167 = 80.84% |
| Bit-depth 4bit | 0.005 | 45 | 36 / 45 = 80.00% | 12 / 45 = 26.67% | 24 / 45 = 53.33% |
| Bit-depth 4bit | 0.010 | 167 | 135 / 167 = 80.84% | 28 / 167 = 16.77% | 107 / 167 = 64.07% |
| Gaussian smoothing r=3 | 0.005 | 45 | 36 / 45 = 80.00% | 35 / 45 = 77.78% | 1 / 45 = 2.22% |
| Gaussian smoothing r=3 | 0.010 | 167 | 135 / 167 = 80.84% | 132 / 167 = 79.04% | 3 / 167 = 1.80% |

## 5. Key Findings

### Gaussian smoothing is the strongest current defense.

Gaussian smoothing reduced average target similarity by about 0.3738 and dropped still-accepted attacks to 1.89% overall.

This is consistent with the attack mechanism: PGD perturbations are high-frequency pixel-level changes, and smoothing directly suppresses that component.

### JPEG recompression does not defend the current handoff images.

JPEG q=75 defense success was 0.00%.

This does not necessarily mean JPEG preprocessing is always useless. It means the current adversarial samples were already saved as JPEG, so applying JPEG compression again changed little. The average similarity drop was only 0.0002.

For a cleaner JPEG-defense evaluation, the attack team should generate and hand off PNG adversarial images or tensor-preserving arrays.

### Bit-depth reduction has limited but visible effect.

Bit-depth 4bit reduced attack success from 80.66% to 61.79%, but this is much weaker than Gaussian smoothing.

Its behavior is less stable because quantization may remove some perturbations but preserve or even shift others depending on pixel-bin boundaries.

## 6. Issue Found in Defense Artifact

`attack_handoff_jpeg_index.csv` contains:

- `accepted_after_attack`: recomputed using saved JPEG files
- `attack_success_before_defense`: still all `True` from the original handoff selection

These differ for 41 rows.

This is not fatal because the defense CSVs and summary correctly use `accepted_after_attack` for the defense-success calculation. However, in reports and future scripts, use `accepted_after_attack` as the reliable attack-success flag for this defense package.

## 7. Attack-Team Follow-up

Recommended next attack-team tasks:

1. Generate PNG or lossless adversarial handoff samples for FaceNet verification.
2. Re-run the same defense pipeline on PNG-based attacks.
3. Compare:
   - tensor-time attack success
   - JPEG-file attack success
   - PNG-file attack success
   - defense success after JPEG / smoothing / bit-depth
4. Add adaptive attack tests against Gaussian smoothing:
   - attack through smoothing transform
   - or run Expectation over Transformation style attack
5. Prepare final attack-defense curve:
   - epsilon vs attack success
   - epsilon vs defense success
   - similarity drop by defense

## 8. Message Back to Defense Team

Suggested response:

```text
verification 방어 결과 확인했습니다.

핵심 결과는 smoothing(radius=3)이 가장 강했고, JPEG 재압축은 현재 handoff 이미지가 이미 JPEG라 방어 효과가 거의 없었습니다.

다만 attack_handoff_jpeg_index.csv에서 accepted_after_attack은 JPEG 재로드 기준으로 재계산되어 있고,
attack_success_before_defense는 원래 handoff 기준 True가 유지되어 41개 row에서 두 값이 다릅니다.
요약 계산은 accepted_after_attack 기준으로 잘 되어 있어서 결과 해석에는 문제 없습니다.

다음 실험에서는 공격팀에서 PNG/lossless adversarial image를 다시 만들어 전달하면 JPEG 방어 효과를 더 공정하게 볼 수 있을 것 같습니다.
```

