"""Run with:  python -m pytest -q   (or  python tests/test_app.py)"""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src import facts as F  # noqa: E402
from src import llm as L  # noqa: E402
from src import offline as O  # noqa: E402


def _app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    return at


def test_app_renders_without_exceptions():
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5


def test_every_interactive_widget_survives_a_change():
    at = _app()
    # threshold slider on the graph tab
    sl = [s for s in at.select_slider if s.key == "g_thr"][0]
    sl.set_value(0.5).run()
    assert not at.exception
    # wallet radio
    r = [x for x in at.radio if x.key == "g_wallet"][0]
    r.set_value("wallet_a").run()
    assert not at.exception
    # market window
    m = [x for x in at.radio if x.key == "mk_span"][0]
    m.set_value("All").run()
    assert not at.exception
    # position size
    n = [x for x in at.number_input if x.key == "mk_pos"][0]
    n.set_value(50000).run()
    assert not at.exception
    # sentiment filters
    kw = [x for x in at.text_input if x.key == "s_kw"][0]
    kw.set_value("ETF").run()
    assert not at.exception


def test_chat_suggestion_offline():
    at = _app()
    btn = [b for b in at.button if b.key == "sugg_2"][0]
    btn.click().run()
    assert not at.exception
    msgs = at.chat_message
    assert len(msgs) >= 2
    assert "VaR" in msgs[-1].markdown[0].value
    assert "\\$316" in msgs[-1].markdown[0].value  # dollar signs are escaped so they are not parsed as LaTeX


def test_chat_typed_question_offline():
    at = _app()
    at.chat_input[0].set_value("Why is market risk moderate?").run()
    assert not at.exception
    assert "Moderate" in at.chat_message[-1].markdown[0].value


def test_offline_answers_cover_all_topics():
    S = F.build_state()
    sec = F.build_sections(S)
    qs = ["what is expected shortfall", "how accurate is the graph model", "explain wallet 3KGiq1 shap",
          "recent sentiment", "does sentiment predict returns", "give me an overall summary",
          "what's the weather", "How reliable is the market model?", "Is 97% accuracy good?"]
    for q in qs:
        a = O.offline_answer(q, S, sec)
        assert len(a) > 80, q
        assert "nan" not in a.lower().replace("finance", ""), q


def test_context_is_small_enough_for_free_tier():
    S = F.build_state()
    sec = F.build_sections(S)
    ctx, used = F.build_context("Summarise everything: market, graph and sentiment", sec)
    assert len(ctx) < 9500
    assert "limitations" in used


class _FakeResp:
    def __init__(self, status=200, lines=()):
        self.status_code = status
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def json(self):
        return {"error": {"message": "x"}}


def _sse(*chunks):
    out = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    return out + ["data: [DONE]"]


def test_stream_parses_sse():
    cfg = L.LLMConfig("groq", "gsk_test", "m", L.PROVIDERS["groq"]["url"])
    with mock.patch("requests.post", return_value=_FakeResp(200, _sse("Hel", "lo"))):
        assert "".join(L.stream_answer(cfg, [{"role": "user", "content": "hi"}])) == "Hello"


def test_rate_limit_retries_on_backup_model():
    cfg = L.LLMConfig("groq", "gsk_test", "llama-3.3-70b-versatile", L.PROVIDERS["groq"]["url"])
    seen = []

    def fake_post(url, headers, json, **kw):
        seen.append(json["model"])
        return _FakeResp(429) if len(seen) == 1 else _FakeResp(200, _sse("ok"))

    with mock.patch("requests.post", side_effect=fake_post):
        assert "".join(L.stream_answer(cfg, [])) == "ok"
    assert seen == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_bad_key_gives_friendly_error():
    cfg = L.LLMConfig("groq", "bad", "m", L.PROVIDERS["groq"]["url"])
    with mock.patch("requests.post", return_value=_FakeResp(401)):
        try:
            list(L.stream_answer(cfg, []))
            assert False
        except L.LLMError as e:
            assert "rejected" in str(e)


def test_provider_resolution():
    assert L.resolve_config("offline").provider == "offline"
    assert L.resolve_config("auto", "gsk_abc").provider == "groq"
    assert L.resolve_config("auto", "sk-or-abc").provider == "openrouter"
    assert L.resolve_config("groq", "").provider in ("offline", "groq")  # depends on env secrets


def test_live_chat_falls_back_when_provider_fails():
    at = _app()
    [x for x in at.selectbox if x.key == "llm_choice"][0].set_value("groq").run()
    [x for x in at.text_input if x.key == "llm_key"][0].set_value("gsk_fake").run()
    with mock.patch("requests.post", return_value=_FakeResp(401)):
        at.chat_input[0].set_value("What is VaR?").run()
    assert not at.exception
    assert any("rejected" in w.value for w in at.warning)
    assert "Value at Risk" in at.chat_message[-1].markdown[-1].value


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
