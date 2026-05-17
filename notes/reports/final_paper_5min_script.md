# 5-Minute Presentation Script

## Slide 1: Title

Good morning. This project studies adversarially robust face recognition, with a specific focus on whether training-time recognizer diversity can improve robustness beyond attack diversity alone.

The deployed model is still a single ArcFace-style recognizer. The question is not whether a larger deployed ensemble performs better, but whether extra training pressure from multiple recognition objectives can make one target recognizer more robust.

## Slide 2: Motivation

Face recognition systems are used for verification and identification, so the embedding space has to remain both separable and stable. Clean benchmark accuracy is important, but deployment reliability also depends on appearance variation, noisy or dirty data, and adversarial inputs.

Recent robust training work already shows the value of stronger adversarial pressure. This project asks a narrower follow-up question: after adding multiple attackers, does adding multiple recognizer objectives further harden the target, or does it mainly disturb verification quality?

## Slide 3: Method

The experiment uses a five-condition ladder. The clean baseline gives the reference point. The recognizer-only setting adds ArcFace, CosFace, and CurricularFace pressure without adversarial images. The single-attack setting tests one adversarial source. The attack-ensemble setting uses PGD, BPFA, and DFANet. The full condition combines the same three attackers with the three recognizer objectives.

The key comparison is attack-ensemble-only versus full-ensemble, because both use the same attack set. The difference is whether the target also receives recognizer-side pressure during training. At inference time, all auxiliary recognizers and attack machinery are removed.

## Slide 4: Robustness Results

The robustness result is positive. Attack-ensemble-only reaches 52.32 percent average robust accuracy, while the full ensemble reaches 56.55 percent. Attack success rate drops from 44.32 percent to 39.45 percent.

The improvement is not isolated to one attack. PGD improves from 45.45 to 52.55 percent, BPFA from 66.09 to 69.56 percent, and DFANet from 45.43 to 47.55 percent. This supports the main interpretation that recognizer-objective diversity adds useful pressure beyond attack diversity alone.

## Slide 5: Verification And Cost

The tradeoff is visible in verification. Compared with attack-ensemble-only, the full ensemble slightly reduces verification accuracy from 98.33 to 98.21 percent, and EER increases from 1.78 to 1.93 percent. The effect is small, but it matters because verification is the core face recognition operating mode.

The training cost is much larger. The baseline can run at batch size 128, while adversarial conditions are memory-capped at 48 or 32. Full-ensemble training takes about 880.7 GPU-hours, compared with 604.2 for attack-ensemble-only and 101.9 for the clean baseline. So the result is best viewed as a robustness-verification-cost frontier, not a free improvement.

## Slide 6: Takeaways

The main conclusion is that recognizer diversity can strengthen a single deployed face recognizer when added on top of an attacker ensemble. It improves robust accuracy and reduces attack success while preserving the same inference-time model size.

The limitations are equally important. The evaluation is in-domain, the attack set is not adaptive, and the training cost is high. Future work should test unseen and adaptive attacks, external face benchmarks, and cheaper training schedules such as lower-frequency recognizer pressure, cached adversarial examples, or more efficient adversarial training.

Overall, the contribution is an investigative ablation: multi-recognizer multi-attacker training is promising, but its value depends on whether the robustness gain justifies the verification and training-cost tradeoff.
