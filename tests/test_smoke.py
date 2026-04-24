from oversight_arena.sabotage_catalog import build_catalog
from oversight_arena.server.environment import OversightArenaEnv
from oversight_arena.models import OverseerAction

def test_catalog_honest_passes_malicious_fails():
    for pattern in build_catalog():
        assert pattern.verify_fn(pattern.honest_source) is True
        assert pattern.verify_fn(pattern.malicious_variants[0]) is False

def test_env_reset_and_step():
    env = OversightArenaEnv()
    obs = env.reset()
    assert obs.turn == 0
    assert len(obs.workers) == 3

    malicious = env.state.malicious_workers[0]
    result = env.step(OverseerAction(
        action="flag_worker",
        worker_id=malicious,
        reasoning="test",
        cwe_tag="test",
    ))

    assert result["reward"] > 0
    assert malicious in result["state"].flagged_workers
