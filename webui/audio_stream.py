"""Gapless live playback for streamed generation — one continuous WAV response
per run, served straight off the app's FastAPI instance.

WHY THIS EXISTS (the streaming-artifact bug)
--------------------------------------------
Gradio's ``gr.Audio(streaming=True)`` output does NOT hand the browser a
continuous waveform. Every value a generator yields becomes its own **HLS
segment**: ``Blocks.handle_streaming_outputs`` opens a ``playlist.m3u8`` per run
and calls ``Audio.stream_output`` on each chunk, which runs it through
``_convert_to_adts`` — an independent pydub/ffmpeg **AAC** encode per chunk
(gradio/components/audio.py, gradio/route_utils.py::MediaStream).

AAC is lossy and frame-based (1024 samples/frame). An independently encoded
segment therefore carries ~1024-2112 samples of *encoder priming* glued to its
front and is zero-padded out to a frame boundary at its back. Concatenated by
the browser's media source, every chunk boundary becomes a click plus a short
gap — and our first-segment ramp yields chunks of 1, 2, 4 codec frames (0.08 s,
0.16 s, 0.32 s), i.e. segments shorter than the codec's own priming, which is
pure garbage. That is exactly why the saved .wav plays perfectly while the live
player is choppy: the audio is fine, the transport is not. Buffering into bigger
chunks only makes the clicks rarer, never gapless.

So we bypass that path entirely. A run opens a stream here, pushes raw int16 PCM
into it, and the UI shows a plain ``<audio>`` element pointing at
``/zerotts/stream/<id>.wav``. That route answers with a single chunked WAV: one
header, then PCM as it arrives, no re-encode and no per-chunk boundaries at all.
The browser decodes it progressively as one file — the same thing the history
player does with the finished file on disk.

The RIFF/data sizes in the header are the usual 0xFFFFFFFF "unknown length"
sentinel, since we can't know the length up front; browsers play such a stream
progressively and stop at end-of-body. The live element is therefore not
seekable — the completed-file player next to it is.

Chunks served so far are retained for the life of the stream, so a client that
reconnects (or a second listener) gets the audio from the top and then follows
along live. Streams are reaped by TTL.
"""

from __future__ import annotations

import html
import struct
import threading
import time
import uuid

import numpy as np

# Route the player fetches from. Absolute, so this assumes the Gradio app is
# mounted at "/" (webui/app.py mounts it there).
STREAM_ROUTE = "/zerotts/stream"

# Give up on a stream whose producer has gone quiet for this long without ever
# closing it — a cancelled run (Stop button) can drop its generator without
# running our close(), and we don't want the HTTP response hanging forever.
STALL_TIMEOUT_SEC = 30.0

# How long a finished stream stays replayable before it is reaped.
STREAM_TTL_SEC = 900.0

_UNKNOWN = 0xFFFFFFFF

_streams: dict[str, _Stream] = {}
_registry_lock = threading.Lock()


class _Stream:
    """A single run's PCM, readable while it is still being written."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.chunks: list[bytes] = []
        self.done = False
        self.created = time.time()
        self.touched = self.created
        self._cv = threading.Condition()

    def push(self, data: bytes) -> None:
        with self._cv:
            self.chunks.append(data)
            self.touched = time.time()
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self.done = True
            self.touched = time.time()
            self._cv.notify_all()

    def read(self):
        """Yield chunks from the beginning, blocking until more arrive."""
        i = 0
        while True:
            with self._cv:
                while i >= len(self.chunks) and not self.done:
                    if not self._cv.wait(STALL_TIMEOUT_SEC):
                        return  # producer went away without closing
                if i >= len(self.chunks):
                    return  # done and fully drained
                data = self.chunks[i]
                i += 1
            yield data


def _reap() -> None:
    cutoff = time.time() - STREAM_TTL_SEC
    with _registry_lock:
        for sid in [s for s, st in _streams.items() if st.touched < cutoff]:
            del _streams[sid]


def open_stream(sample_rate: int) -> str:
    """Register a new stream and return its id. Call before showing the player
    so the browser's request can't arrive before the stream exists."""
    _reap()
    sid = uuid.uuid4().hex
    with _registry_lock:
        _streams[sid] = _Stream(int(sample_rate))
    return sid


def push(sid: str, pcm_int16: np.ndarray) -> None:
    stream = _streams.get(sid)
    if stream is not None:
        stream.push(np.ascontiguousarray(pcm_int16, dtype=np.int16).tobytes())


def close(sid: str) -> None:
    stream = _streams.get(sid)
    if stream is not None:
        stream.close()


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", _UNKNOWN) + b"WAVE"
        + b"fmt " + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate,
            sample_rate * block_align, block_align, bits,
        )
        + b"data" + struct.pack("<I", _UNKNOWN)
    )


def player_html(sid: str | None = None, autoplay: bool = True) -> str:
    """The live player element (or an empty placeholder when sid is None)."""
    if not sid:
        return (
            "<audio controls style='width:100%' "
            "aria-label='Live output (idle)'></audio>"
        )
    src = html.escape(f"{STREAM_ROUTE}/{sid}.wav")
    # A fresh element per run: pointing an existing <audio> at a new src leaves
    # the old, already-buffered stream half-alive in some browsers.
    return (
        f"<audio id='zerotts-live-{html.escape(sid)}' src='{src}' controls "
        f"{'autoplay ' if autoplay else ''}preload='auto' style='width:100%'></audio>"
    )


def mount(app) -> None:
    """Register the streaming route on a FastAPI app."""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    @app.get(STREAM_ROUTE + "/{sid}.wav")
    def zerotts_audio_stream(sid: str):  # noqa: ANN202
        stream = _streams.get(sid)
        if stream is None:
            raise HTTPException(status_code=404, detail="Unknown or expired stream.")

        def body():
            yield wav_header(stream.sample_rate)
            yield from stream.read()

        return StreamingResponse(
            body(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                # No length up front, so range requests can't be honoured;
                # saying so keeps browsers from probing with one.
                "Accept-Ranges": "none",
                "X-Accel-Buffering": "no",  # don't let a proxy sit on the chunks
            },
        )
