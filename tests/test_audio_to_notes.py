#!/usr/bin/env python3
"""Unit tests for audio-to-notes scripts (no model / network needed).

Run with:  pytest tests/   (or:  python -m pytest tests/)
Core deps only (stdlib); the scripts are imported purely for their
module-level, side-effect-free helper functions.
"""
import importlib.util
import json
import os
import sys
import tempfile

# ---- import the scripts as modules (they live one dir up from tests/) -------
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS_DIR, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


transcribe = _load("transcribe.py")
fetch_input = _load("fetch_input.py")
diarize = _load("diarize.py")


# --------------------------------------------------------------------------- #
# 1. diarize.speaker_at  —— 时间重叠归因
# --------------------------------------------------------------------------- #
def test_speaker_at_full_overlap():
    turns = [(0.0, 10.0, "SPEAKER_00"), (10.0, 20.0, "SPEAKER_01")]
    assert diarize.speaker_at({"start": 12.0, "end": 18.0}, turns) == "SPEAKER_01"


def test_speaker_at_no_overlap():
    turns = [(0.0, 10.0, "SPEAKER_00")]
    assert diarize.speaker_at({"start": 100.0, "end": 110.0}, turns) == "SPEAKER_??"


def test_speaker_at_cross_speaker_low_coverage():
    # 段横跨两人但各自覆盖都很少 -> 不足以归属，标 ??
    turns = [(0.0, 10.0, "SPEAKER_00"), (90.0, 100.0, "SPEAKER_01")]
    assert diarize.speaker_at({"start": 0.0, "end": 100.0}, turns) == "SPEAKER_??"


def test_speaker_at_tie_picks_first_max():
    # 两段等重叠，取第一个达到最大重叠者
    turns = [(0.0, 10.0, "SPEAKER_00"), (10.0, 20.0, "SPEAKER_01")]
    assert diarize.speaker_at({"start": 5.0, "end": 15.0}, turns) == "SPEAKER_00"


# --------------------------------------------------------------------------- #
# 2. transcribe cache manifest  —— 防跨输入静默复用
# --------------------------------------------------------------------------- #
def _args(**kw):
    class A:
        pass
    a = A()
    a.backend = kw.get("backend", "sensevoice")
    a.model = kw.get("model", "small")
    a.language = kw.get("language", "zh")
    a.with_emotion = kw.get("with_emotion", False)
    a.emotion_model = kw.get("emotion_model", "iic/emotion2vec_plus_large")
    return a


def test_manifest_matches_same_input():
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, "probe.txt")
        open(probe, "w").close()
        args = _args()
        entry = transcribe._manifest_entry(args, probe)
        mp = os.path.join(td, "manifest.json")
        json.dump(entry, open(mp, "w"))
        assert transcribe._manifest_matches(mp, args, probe) is True


def test_manifest_mismatch_on_param_change():
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, "probe.txt")
        open(probe, "w").close()
        args = _args()
        entry = transcribe._manifest_entry(args, probe)
        mp = os.path.join(td, "manifest.json")
        json.dump(entry, open(mp, "w"))
        # 改变 language -> 不应匹配
        assert transcribe._manifest_matches(mp, _args(language="en"), probe) is False


def test_manifest_missing_file_returns_false():
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, "probe.txt")
        open(probe, "w").close()
        assert transcribe._manifest_matches(os.path.join(td, "nope.json"), _args(), probe) is False


# --------------------------------------------------------------------------- #
# 3. fetch_input  —— 类型嗅探与协议白名单
# --------------------------------------------------------------------------- #
def test_classify_by_content_type():
    assert fetch_input._classify("audio/mpeg", b"") == "audio"
    assert fetch_input._classify("text/html", b"<html>") == "text"
    assert fetch_input._classify("video/mp4", b"") == "audio"  # 视频容器也可转写


def test_classify_by_magic_bytes():
    assert fetch_input._classify("application/octet-stream", b"ID3xxxx") == "audio"   # mp3
    assert fetch_input._classify("application/octet-stream", b"RIFFxxxx") == "audio"  # wav
    # 无法判定的二进制 -> None（主流程会回退到 URL 后缀 / 当文本）
    assert fetch_input._classify("application/octet-stream", b"hello") is None


def test_is_allowed_scheme():
    assert fetch_input._is_allowed_scheme("https://example.com/a.mp3") is True
    assert fetch_input._is_allowed_scheme("http://localhost:8080/a") is True   # 私网允许
    assert fetch_input._is_allowed_scheme("file:///etc/passwd") is False
    assert fetch_input._is_allowed_scheme("ftp://host/x") is False
