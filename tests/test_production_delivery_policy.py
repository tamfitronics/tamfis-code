from tamfis_code.orchestrator.production import (
    build_delivery_policy,
    completion_is_evidence_bound,
)


def test_mutating_complex_work_requires_plan_review_verification_and_resume():
    policy = build_delivery_policy(
        read_only=False, requires_validation=True, complexity="complex"
    )
    assert policy.planning == "required"
    assert policy.mutation_requires_scope is True
    assert policy.review_required is True
    assert policy.verification_required is True
    assert policy.resumable is True
    assert not completion_is_evidence_bound(policy, evidence_count=0, passed=True)
    assert completion_is_evidence_bound(policy, evidence_count=1, passed=True)


def test_read_only_simple_work_can_finish_without_mutation_evidence():
    policy = build_delivery_policy(
        read_only=True, requires_validation=False, complexity="trivial"
    )
    assert policy.planning == "proportional"
    assert policy.mutation_requires_scope is False
    assert completion_is_evidence_bound(policy, evidence_count=0, passed=True)
