"""Shared execution boundaries for IO/compute consistency and unknown timing."""

from __future__ import annotations


class CaseEvaluationError(ValueError):
    """An expected per-case limitation, not a worker/infrastructure failure."""

    evaluation_status = "not_evaluated"
    code = "not_evaluated"

    def report(self) -> dict[str, object]:
        result: dict[str, object] = {"status": self.evaluation_status, "reason": self.code, "message": str(self)}
        if self.code == "full_hit_finalization_unmodeled":
            result.update(full_hit_finalization_status="not_evaluated",
                          affected_request_count=1,
                          request_id=getattr(self, "request_id", None),
                          model_id=getattr(self, "model_id", None))
        return result


class UnsupportedDSACombination(CaseEvaluationError):
    evaluation_status = "unsupported"
    code = "unsupported_dsa_restore_compute_combination"


class FullHitFinalizationNotEvaluated(CaseEvaluationError):
    code = "full_hit_finalization_unmodeled"


def validate_execution_combination(model, policy=None) -> None:
    """Validate after request policy overrides, before executing a case.

    DSA storage currently contains latent KV. Materialized sparse restoration
    has no shared selected-set expansion plan, so it is conservatively rejected
    as a configuration (including degenerate zero-hit requests).
    """
    formula = model.compute.prefill_formula
    if getattr(formula, "type", None) != "dsa":
        return
    if formula.restored_cache_representation != "latent":
        raise UnsupportedDSACombination(
            f"{model.name}: DSA storage contains latent KV; expanded restore inputs "
            "are not supported. Use restored_cache_representation=latent.")
    sparse = (getattr(policy, "sparse_access_policy", None)
              or getattr(model.attention, "sparse_access_policy", None))
    mode = sparse.restore_mode if sparse is not None else model.attention.read_mode
    if mode == "sparse_on_demand" and formula.mla_algorithm == "materialized":
        raise UnsupportedDSACombination(
            f"{model.name}: sparse_on_demand + materialized DSA is unsupported: "
            "selected KV reads do not match full-prefix latent expansion. "
            "Use full_restore with latent KV; no algorithm or read mode was changed.")


def require_evaluated_full_hit(breakdown, *, model_id: str, request_id: str = "") -> None:
    if getattr(breakdown, "full_hit_finalization_status", "not_applicable") == "not_evaluated":
        error = FullHitFinalizationNotEvaluated(
            f"{model_id}{' request ' + request_id if request_id else ''}: full-hit "
            "first-token finalization is not modeled (final hidden/logits or output "
            "head unavailable). This case cannot advance a completion timeline.")
        error.model_id = model_id
        error.request_id = request_id
        raise error
