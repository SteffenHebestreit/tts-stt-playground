"""Behavioural tests for benchmarks/run_german_eval.py: run(), compare(), targets.

The runner is driven end to end against a small stub server that speaks the three
wire protocols the stack exposes (the backends' /transcribe, the OpenAI-style
/v1/audio/transcriptions, the gateway's /api/stt). The stub parses the multipart
body with the same framework the real services use, so a request that put the
audio in the wrong field or left out `provider` fails here the way it would fail
against a real container. No audio, no model, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi import FastAPI, File, Form, Header, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"

# The runner always passes a timeout, which is what a real httpx client needs;
# Starlette's TestClient only warns that it ignores it.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_german_eval_runner_tests", BENCHMARKS / "run_german_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_german_eval_runner_tests"] = module
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCHMARKS))
    return module


class FakeClock:
    """A clock the stub advances, so latencies are exact instead of slept."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Stub:
    """Records what arrived and answers with a scripted transcript per clip."""

    def __init__(self, clock, transcripts):
        self.clock = clock
        self.transcripts = transcripts      # clip file name -> hypothesis
        self.calls = []                     # one dict per request that arrived
        self.delay = lambda n: 0.5          # seconds request number n (0-based) takes
        self.duration = None                # stated clip length, or None to omit it
        self.status = lambda n, name: 200   # HTTP status for request n on clip `name`
        self.headers = lambda n, name: {}   # extra response headers

    def answer(self, api, path, audio, fields, auth):
        n = len(self.calls)
        self.calls.append({"api": api, "path": path, "clip": audio.filename,
                           "fields": fields, "auth": auth})
        self.clock.advance(self.delay(n))
        status = self.status(n, audio.filename)
        if status != 200:
            return JSONResponse({"detail": "boom"}, status_code=status)
        body = {"text": self.transcripts[audio.filename]}
        if self.duration is not None:
            body["duration"] = self.duration
        return JSONResponse(body, headers=self.headers(n, audio.filename))


def build_app(stub):
    app = FastAPI()

    @app.post("/transcribe")
    def stt_form(audio: UploadFile = File(...), language: str = Form(None),
                 authorization: Optional[str] = Header(None)):
        return stub.answer("stt-form", "/transcribe", audio, {"language": language}, authorization)

    @app.post("/v1/audio/transcriptions")
    def openai(file: UploadFile = File(...), model: str = Form(None), language: str = Form(None),
               response_format: str = Form("json"), provider: str = Form(None),
               authorization: Optional[str] = Header(None)):
        return stub.answer("openai", "/v1/audio/transcriptions", file,
                           {"model": model, "language": language,
                            "response_format": response_format, "provider": provider},
                           authorization)

    @app.post("/api/stt")
    def gateway(audio: UploadFile = File(...), provider: str = Form(...), language: str = Form("auto"),
                authorization: Optional[str] = Header(None)):
        return stub.answer("gateway", "/api/stt", audio,
                           {"provider": provider, "language": language}, authorization)

    return app


CLIPS = {
    "a.wav": ("Guten Tag zusammen", "Guten Tag zusammen"),
    "b.wav": ("Wie geht es dir heute", "Wie geht es dir heute"),
    "c.wav": ("Das Wetter ist schön", "Das Wetter ist schon"),   # one umlaut error
    "d.wav": ("Ich möchte einen Kaffee", "Ich möchte einen Kaffee"),
}


@pytest.fixture
def workspace(tmp_path, runner, monkeypatch):
    """Clips on disk, a JSONL manifest, a clock and a stub-backed client."""
    clock = FakeClock()
    monkeypatch.setattr(runner, "_clock", clock)

    manifest = tmp_path / "manifest.jsonl"
    lines = []
    for name, (reference, _hyp) in CLIPS.items():
        (tmp_path / name).write_bytes(name.encode())
        lines.append(json.dumps({"audio": name, "text": reference}, ensure_ascii=False))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stub = Stub(clock, {name: hyp for name, (_ref, hyp) in CLIPS.items()})
    with TestClient(build_app(stub)) as client:
        yield SimpleNamespace(dir=tmp_path, manifest=manifest, stub=stub, client=client,
                              clock=clock, out=tmp_path / "out.json")


def run_eval(runner, workspace, *extra, out=None):
    out = out or workspace.out
    code = runner.main(
        ["--manifest", str(workspace.manifest), "--out", str(out), *extra],
        client=workspace.client,
    )
    report = json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).exists() else None
    return code, report


# --- choosing the target -----------------------------------------------------


def test_default_target_is_the_gateways_openai_route(runner):
    target = runner.resolve_target("http://localhost:3000")
    assert (target.api, target.url, target.provider) == (
        "openai", "http://localhost:3000/v1/audio/transcriptions", None)


def test_url_pointing_at_a_backend_selects_the_stt_form_protocol(runner):
    target = runner.resolve_target("http://ignored:3000", url="http://parakeet:5005/transcribe")
    assert (target.api, target.url) == ("stt-form", "http://parakeet:5005/transcribe")
    assert target.base_url == "http://parakeet:5005"


def test_url_on_the_openai_route_and_whisper_cpp_inference_are_openai(runner):
    assert runner.resolve_target("x", url="http://h:1/v1/audio/transcriptions").api == "openai"
    assert runner.resolve_target("x", url="http://h:1/inference").api == "openai"


def test_provider_alone_selects_the_gateway_adapter(runner):
    target = runner.resolve_target("http://gw:3000/", provider="qwen3-asr")
    assert (target.api, target.url, target.provider) == (
        "gateway", "http://gw:3000/api/stt", "qwen3-asr")


def test_a_bare_host_url_is_a_base_url(runner):
    assert runner.resolve_target("x", url="http://h:5001").url == "http://h:5001/v1/audio/transcriptions"
    assert runner.resolve_target("x", url="http://h:5001", api="stt-form").url == "http://h:5001/transcribe"


def test_an_unrecognised_endpoint_needs_an_explicit_api(runner):
    with pytest.raises(SystemExit, match="cannot tell which API"):
        runner.resolve_target("x", url="http://h:1/custom/path")
    assert runner.resolve_target("x", url="http://h:1/custom/path", api="stt-form").api == "stt-form"


def test_a_relative_or_non_http_url_is_refused(runner):
    for bad in ("/transcribe", "localhost:5001/transcribe", "ftp://h/transcribe"):
        with pytest.raises(SystemExit, match="absolute http"):
            runner.resolve_target("x", url=bad)


@pytest.mark.parametrize("kwargs", [
    {"url": "http://h:1/v1/audio/transcriptions", "provider": "parakeet"},
    {"api": "openai", "provider": "parakeet"},
    {"url": "http://h:5005/transcribe", "provider": "parakeet"},
])
def test_provider_is_refused_where_it_would_be_silently_ignored(runner, kwargs):
    """/v1/audio/transcriptions drops unknown form fields, so this would mislabel a run."""
    with pytest.raises(SystemExit, match="--provider cannot be used"):
        runner.resolve_target("http://h:3000", **kwargs)


def test_the_gateway_api_without_a_provider_is_refused(runner):
    with pytest.raises(SystemExit, match="needs --provider"):
        runner.resolve_target("http://h:3000", api="gateway")


# --- run(): the wire contracts ----------------------------------------------


def test_direct_backend_run_sends_the_audio_field_and_scores(runner, workspace):
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                            "--label", "direct")
    assert code == 0
    assert report["api"] == "stt-form" and report["provider"] is None
    assert report["utterances"] == 4 and report["failed"] == 0
    assert {c["api"] for c in workspace.stub.calls} == {"stt-form"}
    assert all(c["fields"] == {"language": "de"} for c in workspace.stub.calls)
    # one wrong umlaut in 4 utterances of 3+5+4+4 = 16 reference words
    assert report["wer_counts"]["substitutions"] == 1
    assert report["wer"] == pytest.approx(1 / 16)


def test_gateway_default_run_uses_the_openai_fields(runner, workspace):
    code, report = run_eval(runner, workspace, "--base-url", "http://testserver")
    assert code == 0
    assert report["api"] == "openai"
    assert {c["path"] for c in workspace.stub.calls} == {"/v1/audio/transcriptions"}
    assert workspace.stub.calls[0]["fields"] == {
        "model": "whisper-1", "language": "de", "response_format": "json", "provider": None}


def test_provider_run_goes_through_api_stt_with_the_provider_field(runner, workspace):
    code, report = run_eval(runner, workspace, "--base-url", "http://testserver",
                            "--provider", "parakeet", "--label", "parakeet")
    assert code == 0
    assert report["api"] == "gateway" and report["provider"] == "parakeet"
    assert {c["path"] for c in workspace.stub.calls} == {"/api/stt"}
    assert all(c["fields"]["provider"] == "parakeet" for c in workspace.stub.calls)


def test_api_key_is_sent_as_a_bearer_token_and_never_written_to_the_report(runner, workspace):
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                            "--api-key", "s3cret-key")
    assert code == 0
    assert {c["auth"] for c in workspace.stub.calls} == {"Bearer s3cret-key"}
    assert "s3cret-key" not in json.dumps(report)


def test_no_api_key_means_no_authorization_header(runner, workspace, monkeypatch):
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert {c["auth"] for c in workspace.stub.calls} == {None}


# --- run(): sampling ---------------------------------------------------------


@pytest.fixture
def many_rows():
    return [(Path(f"clip{i:02d}.wav"), f"satz {i}") for i in range(40)]


def test_limit_is_a_random_sample_not_the_first_n(runner, many_rows):
    """The first N rows of a corpus are a handful of speakers, not a sample."""
    sample = runner.sample_manifest(many_rows, 5, seed=runner.DEFAULT_SEED)
    assert len(sample) == 5
    assert sample != many_rows[:5]


def test_the_same_seed_gives_the_same_sample_and_another_seed_a_different_one(runner, many_rows):
    first = runner.sample_manifest(many_rows, 8, seed=7)
    assert first == runner.sample_manifest(many_rows, 8, seed=7)
    assert any(runner.sample_manifest(many_rows, 8, seed=s) != first for s in range(8, 20))


def test_a_sample_keeps_manifest_order_and_has_no_repeats(runner, many_rows):
    sample = runner.sample_manifest(many_rows, 12, seed=3)
    assert sample == sorted(sample, key=many_rows.index)
    assert len({p for p, _ in sample}) == 12


def test_a_limit_at_or_above_the_size_returns_everything(runner, many_rows):
    assert runner.sample_manifest(many_rows, 40, seed=1) == many_rows
    assert runner.sample_manifest(many_rows, 999, seed=1) == many_rows
    assert runner.sample_manifest(many_rows, None, seed=1) == many_rows


def test_run_records_the_sample_so_two_runs_can_be_paired(runner, workspace):
    _, first = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                        "--limit", "2", "--seed", "5", out=workspace.dir / "one.json")
    _, second = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                         "--limit", "2", "--seed", "5", out=workspace.dir / "two.json")
    assert first["utterances"] == 2 and first["manifest_rows"] == 4
    assert (first["limit"], first["seed"]) == (2, 5)
    assert [r["audio"] for r in first["results"]] == [r["audio"] for r in second["results"]]


# --- run(): warmup and timing ------------------------------------------------


def test_warmup_is_excluded_from_the_timings_and_reported_as_the_cold_start(runner, workspace):
    """The first request after an idle-unload pays a model load; it must not be a data point."""
    workspace.stub.delay = lambda n: 9.0 if n == 0 else 0.5
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert code == 0
    assert len(workspace.stub.calls) == 5, "one warmup request plus four timed ones"
    assert report["utterances"] == 4, "the warmup request is not scored"
    assert report["cold_start_seconds"] == pytest.approx(9.0)
    assert report["wall_seconds"] == pytest.approx(4 * 0.5)
    assert report["latency"]["median"] == pytest.approx(0.5)
    assert report["latency"]["max"] == pytest.approx(0.5)


def test_without_warmup_the_cold_load_lands_in_the_timings(runner, workspace):
    workspace.stub.delay = lambda n: 9.0 if n == 0 else 0.5
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                            "--warmup", "0")
    assert len(workspace.stub.calls) == 4
    assert report["cold_start_seconds"] is None
    assert report["wall_seconds"] == pytest.approx(9.0 + 3 * 0.5)
    assert report["latency"]["max"] == pytest.approx(9.0)


def test_several_warmup_requests_all_go_out_and_only_the_first_is_the_cold_start(runner, workspace):
    workspace.stub.delay = lambda n: [7.0, 1.0, 0.5][n] if n < 3 else 0.5
    _, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                         "--warmup", "3")
    assert len(workspace.stub.calls) == 3 + 4
    assert report["warmup"]["seconds"] == pytest.approx([7.0, 1.0, 0.5])
    assert report["cold_start_seconds"] == pytest.approx(7.0)


def test_a_failed_warmup_does_not_lose_the_run_unless_strict(runner, workspace):
    workspace.stub.status = lambda n, name: 503 if n == 0 else 200
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert code == 0
    assert report["utterances"] == 4
    assert report["cold_start_seconds"] is None

    workspace.stub.calls.clear()
    workspace.stub.status = lambda n, name: 503 if n == 0 else 200
    assert runner.main(
        ["--manifest", str(workspace.manifest), "--url", "http://testserver/transcribe",
         "--strict"], client=workspace.client) == 1


def test_rtf_is_reported_only_when_the_backend_states_the_clip_length(runner, workspace):
    workspace.stub.duration = 2.0
    _, direct = run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert direct["audio_seconds"] == pytest.approx(8.0)
    assert direct["rtf"] == pytest.approx(0.25)          # 0.5 s per 2.0 s clip

    workspace.stub.duration = None
    _, gateway = run_eval(runner, workspace, "--base-url", "http://testserver",
                          out=workspace.dir / "gw.json")
    assert gateway["rtf"] is None and gateway["audio_seconds"] is None


# --- run(): failures ---------------------------------------------------------


def test_a_failed_clip_is_counted_and_recorded_and_the_rest_still_scored(runner, workspace, capsys):
    workspace.stub.status = lambda n, name: 500 if name == "b.wav" else 200
    code, report = run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert code == 0
    assert report["utterances"] == 3 and report["failed"] == 1
    assert report["errors"][0]["audio"].endswith("b.wav")
    assert "HTTP 500" in report["errors"][0]["error"]
    assert "FAILED" in capsys.readouterr().err


def test_strict_aborts_on_the_first_failed_clip(runner, workspace):
    workspace.stub.status = lambda n, name: 500 if name == "a.wav" and n > 0 else 200
    code = runner.main(["--manifest", str(workspace.manifest), "--strict",
                        "--url", "http://testserver/transcribe"], client=workspace.client)
    assert code == 1


def test_every_clip_failing_is_an_error_not_an_empty_report(runner, workspace):
    workspace.stub.status = lambda n, name: 500
    with pytest.raises(SystemExit, match="every utterance failed"):
        run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert not workspace.out.exists()


def test_missing_audio_is_reported_before_any_request_is_sent(runner, workspace):
    (workspace.dir / "c.wav").unlink()
    with pytest.raises(SystemExit, match="audio file.*not found"):
        run_eval(runner, workspace, "--url", "http://testserver/transcribe")
    assert workspace.stub.calls == []


def test_a_response_from_a_fallback_provider_is_not_scored(runner, workspace):
    """The gateway falls back silently when its default is down; that is another model's WER."""
    workspace.stub.headers = lambda n, name: (
        {"X-Provider-Fallback": "whisper->qwen3-asr", "X-Provider": "qwen3-asr"}
        if name == "c.wav" else {})
    code, report = run_eval(runner, workspace, "--base-url", "http://testserver")
    assert code == 0
    assert report["fallback_responses"] == 1
    assert report["utterances"] == 3 and report["failed"] == 1
    # c.wav carried the only error; with it left out the run is clean
    assert report["wer"] == 0.0


def test_a_run_answered_entirely_by_a_fallback_says_so(runner, workspace):
    workspace.stub.headers = lambda n, name: {"X-Provider-Fallback": "whisper->qwen3-asr"}
    with pytest.raises(SystemExit, match="fallback provider"):
        run_eval(runner, workspace, "--base-url", "http://testserver")


def test_the_gateway_adapter_naming_another_provider_is_not_scored(runner, workspace):
    workspace.stub.headers = lambda n, name: {"X-Provider": "whisper"}
    with pytest.raises(SystemExit, match="fallback provider"):
        run_eval(runner, workspace, "--base-url", "http://testserver", "--provider", "parakeet")


def test_ctrl_c_reports_the_utterances_that_finished(runner, tmp_path, monkeypatch):
    """Hours of results must survive an interrupt."""
    clips = []
    for i in range(5):
        (tmp_path / f"{i}.wav").write_bytes(b"x")
        clips.append({"audio": f"{i}.wav", "text": "ein zwei drei"})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(c) for c in clips), encoding="utf-8")

    class Response:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"text": "ein zwei drei"}

    class Interrupting:
        calls = 0

        def post(self, *args, **kwargs):
            Interrupting.calls += 1
            if Interrupting.calls == 4:          # warmup + 2 clips done, third interrupted
                raise KeyboardInterrupt
            return Response()

    out = tmp_path / "r.json"
    code = runner.main(["--manifest", str(manifest), "--out", str(out)], client=Interrupting())
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert report["interrupted"] is True
    assert report["utterances"] == 2 and report["not_run"] == 3 and report["failed"] == 0


# --- compare() ---------------------------------------------------------------


def _counts(c):
    return {"substitutions": c.substitutions, "deletions": c.deletions,
            "insertions": c.insertions, "reference_length": c.reference_length}


@pytest.fixture
def make_report(runner):
    """Build a saved-report dict the way run() does, from (audio, reference, hypothesis)."""

    def build(label, hyps, *, fold=False, extra=None):
        results = [{
            "audio": audio, "reference": reference, "hypothesis": hypothesis,
            "wer": _counts(runner.word_errors(reference, hypothesis)),
            "cer": _counts(runner.character_errors(reference, hypothesis)),
            "seconds": 0.5,
        } for audio, reference, hypothesis in hyps]
        return {"label": label, "fold_umlauts": fold, "results": results, **(extra or {})}

    return build


def write_reports(tmp_path, baseline, candidate):
    b, c = tmp_path / "base.json", tmp_path / "cand.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(candidate), encoding="utf-8")
    return str(b), str(c)


def rows(n, wrong_every=None):
    out = []
    for i in range(n):
        reference = f"das ist satz nummer {i} mit einigen worten"
        wrong = wrong_every and i % wrong_every == 0
        out.append((f"c{i}.wav", reference, reference.replace("einigen", "keinen") if wrong else reference))
    return out


def test_compare_flags_a_real_regression_and_lists_the_worst_utterances(runner, make_report, tmp_path, capsys):
    base, cand = write_reports(tmp_path, make_report("float16", rows(60)),
                               make_report("int8", rows(60, wrong_every=2)))
    assert runner.compare(base, cand, resamples=300, character_level=False, show=3) == 0
    out = capsys.readouterr().out
    assert "baseline  float16" in out and "candidate int8" in out
    assert "SIGNIFICANT" in out and "NO SIGNIFICANT" not in out
    assert "Worst 3 regressions:" in out
    assert "keinen" in out


def test_compare_of_identical_runs_is_not_significant_and_lists_no_regressions(runner, make_report, tmp_path, capsys):
    same = make_report("x", rows(30, wrong_every=5))
    base, cand = write_reports(tmp_path, same, make_report("y", rows(30, wrong_every=5)))
    runner.compare(base, cand, resamples=200, character_level=False, show=5)
    out = capsys.readouterr().out
    assert "NO SIGNIFICANT DIFFERENCE" in out
    assert "Worst" not in out, "a regressions header over an empty list is misleading"


def test_compare_pairs_on_the_audio_path_when_a_clip_failed_in_one_run(runner, make_report, tmp_path, capsys):
    full = rows(20)
    base, cand = write_reports(tmp_path, make_report("a", full), make_report("b", full[:15]))
    runner.compare(base, cand, resamples=100, character_level=False, show=0)
    captured = capsys.readouterr()
    assert "5 utterance(s) present in the baseline but not the candidate" in captured.err
    assert "but not the baseline" not in captured.err

    runner.compare(cand, base, resamples=100, character_level=False, show=0)
    assert "5 utterance(s) present in the candidate but not the baseline" in capsys.readouterr().err


def test_compare_of_disjoint_runs_is_refused(runner, make_report, tmp_path):
    base, cand = write_reports(tmp_path, make_report("a", rows(5)),
                               make_report("b", [(f"other{n}.wav", r, h) for n, (_, r, h) in enumerate(rows(5))]))
    with pytest.raises(SystemExit, match="share no utterances"):
        runner.compare(base, cand, resamples=50, character_level=False, show=0)


def test_compare_keeps_a_clip_listed_twice_as_two_utterances(runner, make_report, tmp_path, capsys):
    """Keyed on the path alone, the second occurrence overwrote the first."""
    twice = [("dup.wav", "eins zwei", "eins zwei"), ("dup.wav", "drei vier", "drei vier"),
             ("solo.wav", "fuenf sechs", "fuenf sechs")]
    bad = [("dup.wav", "eins zwei", "eins zwei"), ("dup.wav", "drei vier", "drei xxxx"),
           ("solo.wav", "fuenf sechs", "fuenf sechs")]
    base, cand = write_reports(tmp_path, make_report("a", twice), make_report("b", bad))
    runner.compare(base, cand, resamples=50, character_level=False, show=5)
    captured = capsys.readouterr()
    assert "WER 0.00%" in captured.out and "WER 16.67%" in captured.out   # 1 error in 6 words
    assert "present in the baseline but not the candidate" not in captured.err


def test_compare_refuses_runs_scored_with_different_umlaut_folding(runner, make_report, tmp_path):
    base, cand = write_reports(tmp_path, make_report("a", rows(5), fold=False),
                               make_report("b", rows(5), fold=True))
    with pytest.raises(SystemExit, match="fold-umlauts"):
        runner.compare(base, cand, resamples=50, character_level=False, show=0)


def test_compare_prints_latency_and_cold_start_when_both_reports_have_them(runner, make_report, tmp_path, capsys):
    base, cand = write_reports(
        tmp_path,
        make_report("a", rows(10), extra={"latency": {"median": 0.40}, "cold_start_seconds": 9.0}),
        make_report("b", rows(10), extra={"latency": {"median": 0.35}, "cold_start_seconds": 4.5}))
    runner.compare(base, cand, resamples=50, character_level=False, show=0)
    out = capsys.readouterr().out
    assert "0.40 s -> 0.35 s" in out and "9.00 s -> 4.50 s" in out


def test_compare_warns_about_fallbacks_and_interrupted_runs(runner, make_report, tmp_path, capsys):
    base, cand = write_reports(
        tmp_path,
        make_report("a", rows(10), extra={"fallback_responses": 3}),
        make_report("b", rows(10), extra={"interrupted": True}))
    runner.compare(base, cand, resamples=50, character_level=False, show=0)
    err = capsys.readouterr().err
    assert "3 baseline response(s) came from a fallback provider" in err
    assert "candidate run was interrupted" in err


def test_compare_notes_reference_text_that_differs_between_runs(runner, make_report, tmp_path, capsys):
    other = [(a, r + " extra", h) for a, r, h in rows(4)]
    base, cand = write_reports(tmp_path, make_report("a", rows(4)), make_report("b", other))
    runner.compare(base, cand, resamples=50, character_level=False, show=0)
    assert "different reference text" in capsys.readouterr().err


def test_a_saved_run_compares_against_itself_end_to_end(runner, workspace, capsys):
    """run() writes what compare() reads: the two halves agree on the format."""
    _, first = run_eval(runner, workspace, "--url", "http://testserver/transcribe",
                        "--label", "one", out=workspace.dir / "one.json")
    _, second = run_eval(runner, workspace, "--base-url", "http://testserver",
                         "--label", "two", out=workspace.dir / "two.json")
    code = runner.main(["--compare", str(workspace.dir / "one.json"), str(workspace.dir / "two.json"),
                        "--resamples", "100"])
    assert code == 0
    out = capsys.readouterr().out
    assert "baseline  one" in out and "candidate two" in out
    assert "NO SIGNIFICANT DIFFERENCE" in out


# --- command line ------------------------------------------------------------


@pytest.mark.parametrize("flags", [["--limit", "0"], ["--limit", "-3"], ["--warmup", "-1"]])
def test_nonsensical_counts_are_rejected_up_front(runner, flags, capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--manifest", "x.tsv", *flags])
    assert exc.value.code == 2


def test_manifest_is_required_unless_comparing(runner):
    with pytest.raises(SystemExit) as exc:
        runner.main([])
    assert exc.value.code == 2


# --- manifests ---------------------------------------------------------------


def test_common_voice_tsv_with_bare_double_quotes_keeps_every_row(runner, tmp_path):
    """A sentence opening with an unbalanced quote used to swallow the rows after it."""
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.mp3").write_bytes(b"")
    manifest = tmp_path / "validated.tsv"
    manifest.write_text(
        "client_id\tpath\tsentence\tup_votes\n"
        "x\ta.mp3\tGuten Tag\t2\n"
        'y\tb.mp3\t"Wie geht es dir\t3\n'
        "z\tc.mp3\tBis morgen\t2\n",
        encoding="utf-8",
    )
    rows_ = runner.load_manifest(manifest)
    assert [text for _, text in rows_] == ['Guten Tag', '"Wie geht es dir', 'Bis morgen']


def test_a_byte_order_mark_does_not_hide_the_first_column(runner, tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    csv_file = tmp_path / "m.csv"
    csv_file.write_bytes("﻿path,text\na.wav,Guten Tag\n".encode("utf-8"))
    assert [t for _, t in runner.load_manifest(csv_file)] == ["Guten Tag"]

    jsonl = tmp_path / "m.jsonl"
    jsonl.write_bytes('﻿{"audio": "a.wav", "text": "Guten Tag"}\n'.encode("utf-8"))
    assert [t for _, t in runner.load_manifest(jsonl)] == ["Guten Tag"]


def test_jsonl_rows_without_a_reference_are_skipped_like_tsv_rows(runner, tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"audio": "a.wav", "text": "Guten Tag"}\n'
        '{"audio": "b.wav", "text": ""}\n'
        '{"audio": "c.wav", "sentence": "  "}\n',
        encoding="utf-8",
    )
    assert [t for _, t in runner.load_manifest(manifest)] == ["Guten Tag"]


def test_jsonl_line_that_is_not_an_object_is_a_clear_error(runner, tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('["a.wav", "Guten Tag"]\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="expected a JSON object"):
        runner.load_manifest(manifest)
