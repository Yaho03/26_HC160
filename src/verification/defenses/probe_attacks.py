"""
계측용 공격 종류

단일 공격 종류로 산출한 임계값은 그 공격의 지문을 외운 것과 구별되지 않는다.
촬영은 사람 시간이 들어 되돌리기가 가장 비싸므로 한 세션에서 여러 공격을 번갈아
생성해 함께 기록한다.

모든 종류는 같은 PGD 생성기를 파라미터만 바꿔 호출한다. FGSM은 step_size가 epsilon과
같은 1스텝 PGD다. 별도 구현을 두면 전처리와 정규화가 어긋날 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

ATTACK_KINDS: tuple[str, ...] = ("pgd", "fgsm", "pgd_low_eps")


class UnknownAttackError(ValueError):
    """선언되지 않은 공격 종류. 촬영 시작 전에 거부한다."""


@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 0.03
    steps: int = 40
    step_size: float = 0.002
    low_epsilon_ratio: float = 0.25


def build_attack_params(kinds, config: AttackConfig) -> dict[str, dict]:
    """
    종류별 PGD 파라미터. 촬영 시작 전에 검증해 사람 시간을 버리지 않는다.
    """
    unknown = [kind for kind in kinds if kind not in ATTACK_KINDS]
    if unknown:
        raise UnknownAttackError(
            f"알 수 없는 공격 종류 {unknown}. 사용 가능: {list(ATTACK_KINDS)}"
        )

    table = {
        # 표준 반복 공격
        "pgd": {
            "epsilon": config.epsilon,
            "steps": config.steps,
            "step_size": config.step_size,
        },
        # 1스텝. perturbation 구조가 PGD와 달라 squeeze 반응도 다르다.
        "fgsm": {
            "epsilon": config.epsilon,
            "steps": 1,
            "step_size": config.epsilon,
        },
        # 더 작은 예산. 탐지가 어려운 쪽 경계를 본다.
        "pgd_low_eps": {
            "epsilon": round(config.epsilon * config.low_epsilon_ratio, 6),
            "steps": config.steps,
            "step_size": config.step_size,
        },
    }
    return {kind: table[kind] for kind in kinds}


def attack_for_index(index: int, kinds) -> str:
    """공격 기회마다 종류를 번갈아 쓴다. 한 주기 안에 모든 종류가 나온다."""
    kinds = list(kinds)
    return kinds[index % len(kinds)]


def run_attack(
    kind: str,
    crop: Image.Image,
    target_embedding,
    config: AttackConfig,
    *,
    generator=None,
    device=None,
) -> tuple[Image.Image, float]:
    params = build_attack_params([kind], config)[kind]
    if generator is None:
        from src.verification.defenses.verification_defense_temporal_camera import (
            generate_adversarial,
        )

        generator = generate_adversarial
    return generator(
        crop,
        target_embedding,
        epsilon=params["epsilon"],
        n_steps=params["steps"],
        step_size=params["step_size"],
        device=device,
    )
