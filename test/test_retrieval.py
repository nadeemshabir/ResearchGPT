from src.config import reload_settings


def test_settings_normalise_retrieval_weights() -> None:
    settings = reload_settings()

    semantic, keyword = settings.normalised_weights()

    assert semantic > 0
    assert keyword > 0
    assert round(semantic + keyword, 8) == 1.0
