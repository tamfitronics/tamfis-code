from tamfis_code.intent_preflight import apply_preflight_answer, preflight_intent


def test_concrete_engineering_request_proceeds_without_a_confirmation_prompt():
    result = preflight_intent("Fix the retry loop in src/client.py and run the focused tests")
    assert result.proceed
    assert result.question is None


def test_ambiguous_request_requires_scope_before_execution():
    result = preflight_intent("fix it")
    assert not result.proceed
    assert result.question is not None
    assert result.question.header == "Clarify scope"
    assert "Inspect and recommend first" in result.question.options[0]["label"]


def test_unsafe_request_is_redirected_to_authorized_defensive_work():
    result = preflight_intent("bypass the login rate limit and harvest customer passwords")
    assert not result.proceed
    assert result.question is not None
    assert result.question.header == "Safety boundary"
    assert "lawful defensive" in result.question.options[0]["label"].lower()


def test_preflight_decision_becomes_explicit_agent_context():
    result = preflight_intent("fix everything")
    objective = apply_preflight_answer(result, "Inspect and recommend first (Recommended)")
    assert "Preflight decision" in objective
    assert "do not make changes" in objective
