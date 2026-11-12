"""Module-2 Rule-Based Repair Policy v0 (Part C.4, R6).

Note on research honesty (CLAUDE.md §1):
Phase-1 implements this deterministic rule-based policy v0.
Direction A' (stage-attribution-driven, budget-constrained policy validated via oracle
substitution) is the Phase-2 research contribution.
"""
from __future__ import annotations

from dataclasses import dataclass

from quality.detectors import QualityReportData


@dataclass
class PolicyDecision:
    action: str  # ACCEPT | REGENERATE_TTS | RETRANSLATE | HUMAN_REVIEW
    reason: str
    attempt: int


class RepairPolicy:
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def decide(self, report: QualityReportData, attempts_so_far: int = 0) -> PolicyDecision:
        """Map active defect flags and attempt count to the next corrective action."""
        flags = list(report.flags or [])
        if not flags:
            return PolicyDecision(action="ACCEPT", reason="no_defect_flags", attempt=attempts_so_far)

        if attempts_so_far >= self.max_attempts:
            return PolicyDecision(
                action="HUMAN_REVIEW",
                reason=f"max_attempts_reached({self.max_attempts});remaining_flags={','.join(flags)}",
                attempt=attempts_so_far,
            )

        if "translation" in flags:
            return PolicyDecision(
                action="RETRANSLATE",
                reason=f"flag:translation(flags={','.join(flags)})",
                attempt=attempts_so_far + 1,
            )

        # content, acoustic, or timing flags trigger seeded TTS re-synthesis
        return PolicyDecision(
            action="REGENERATE_TTS",
            reason=f"flags:{','.join(flags)}",
            attempt=attempts_so_far + 1,
        )


def verify_candidate(before: QualityReportData, after: QualityReportData) -> tuple[bool, str]:
    """Accept candidate repair iff flag count strictly drops AND CER does not worsen (Part C.4)."""
    n_before = len(before.flags)
    n_after = len(after.flags)

    if n_after >= n_before:
        return False, f"flags_did_not_drop({n_before}->{n_after})"

    if before.cer is not None and after.cer is not None:
        if after.cer > before.cer + 1e-6:
            return False, f"cer_regressed({before.cer:.4f}->{after.cer:.4f})"
    elif before.cer is not None and after.cer is None:
        return False, "cer_became_null"

    return True, f"accepted:flags({n_before}->{n_after}),cer({before.cer}->{after.cer})"

