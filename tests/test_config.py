from gateway.config import Settings


def test_csv_settings_are_normalized() -> None:
    settings = Settings(public_api_keys=" a, b ,,","+" allowed_models="base, tool-calling ")
    assert settings.public_api_key_set == ("a", "b")
    assert settings.allowed_model_set == frozenset({"base", "tool-calling"})
