"""End-to-end synthesis tests. All need weights (see conftest.py)."""

import numpy as np
import pytest

pytestmark = pytest.mark.model


def test_synthesize_produces_audio(tts):
    audio = tts.synthesize("Xin chào các bạn.", voice=None)
    assert audio.ndim == 2 and audio.shape[0] == 1
    assert audio.shape[1] > tts.sample_rate * 0.2, "suspiciously short"
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() > 1e-3, "output is silent"


def test_seed_makes_generation_reproducible(tts):
    np.random.seed(42)
    a = tts.synthesize("Hôm nay trời đẹp quá.", voice=None)
    np.random.seed(42)
    b = tts.synthesize("Hôm nay trời đẹp quá.", voice=None)
    assert np.array_equal(a, b)


def test_streaming_matches_batch_length(tts):
    text = "Một đoạn văn bản để kiểm tra chế độ phát trực tuyến."
    np.random.seed(7)
    batch = tts.synthesize(text, voice=None)
    np.random.seed(7)
    streamed = np.concatenate(
        [c.reshape(-1) for c in tts.synthesize_stream(text, voice=None)])
    # The streaming decoder is KV-cached rather than batch-decoded, so samples can
    # differ at chunk seams; the frame count — hence the length — must not.
    assert streamed.shape[0] == batch.shape[1]


def test_voice_changes_the_output(tts):
    voices = tts.list_voices()
    if not voices:
        pytest.skip("no voice packs in this model")
    text = "Xin chào các bạn."
    np.random.seed(3)
    with_voice = tts.synthesize(text, voice=voices[0])
    np.random.seed(3)
    without = tts.synthesize(text, voice=None)
    assert not (with_voice.shape == without.shape and np.array_equal(with_voice, without))


def test_unknown_voice_lists_what_exists(tts):
    with pytest.raises(FileNotFoundError, match="available"):
        tts.synthesize("Xin chào.", voice="definitely-not-a-voice")


def test_voice_query_mismatch_is_rejected(tts, tmp_path):
    """Latents from a different model have the right dtype and rank, so they would
    feed the graph cleanly and produce confident nonsense. They must be refused."""
    from zerotts.voices import load_voice

    vdir = tmp_path / "wrong"
    vdir.mkdir()
    wrong_q = tts.n_voice_queries + 1
    np.savez(vdir / "voice.npz",
             n_voice_queries=np.int64(wrong_q),
             voice_emb=np.zeros((1, wrong_q, tts.d_model), dtype=np.float32))
    with pytest.raises(ValueError, match="different weights"):
        load_voice(tmp_path, "wrong", expect_queries=tts.n_voice_queries)


def test_raw_latents_accepted(tts):
    """A latent array from anywhere should work — that is the documented escape
    hatch for voices obtained outside this package."""
    emb = np.zeros((1, tts.n_voice_queries, tts.d_model), dtype=np.float32)
    audio = tts.synthesize("Xin chào.", voice=emb, max_frames=40)
    assert audio.shape[1] > 0


def test_no_voice_encoder_is_exposed(tts):
    """The cloning path must be absent, not merely disabled."""
    for attr in ("encode_voice", "encode_voice_from_codes", "voice_encoder_sess"):
        assert not hasattr(tts, attr), f"{attr} should not exist in the OSS build"
    assert not hasattr(tts.codec, "encode_reference")
