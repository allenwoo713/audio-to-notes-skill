#!/usr/bin/env python3
"""Handle a URL input for audio-to-notes.

Two cases:
  * URL points to an audio/video file  -> download it, return type=audio
  * URL points to a web page           -> fetch HTML, strip to plain text, return type=text

Type detection is based on the **response** (Content-Type header + file magic
bytes + Content-Disposition), NOT the URL suffix -- many media endpoints serve
audio with opaque or missing extensions. The downloaded bytes are treated as
UNTRUSTED DATA: this script never executes or interprets content as instructions.

Output goes to <workdir>/downloads/. Prints a single JSON line to stdout:
  {"type": "audio", "path": "..."}   or   {"type": "text", "path": "..."}
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import urlparse

UA = "Mozilla/5.0 (compatible; audio-to-notes/1.0)"

# 允许的最大下载字节数（约 2GB），防止失控/恶意端点拖垮磁盘。
MAX_BYTES = 2 * 1024 * 1024 * 1024

# 仅接受 http/https。localhost / 私网地址允许（用户已确认信任本地/内网源）。
ALLOWED_SCHEMES = {"http", "https"}

AUDIO_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/aac",
    "audio/ogg", "audio/flac", "audio/webm", "audio/x-wav", "audio/wav",
    "audio/x-ms-wma", "audio/amr", "audio/3gpp",
    "video/mp4", "video/m4v", "video/mov", "video/quicktime", "video/x-matroska",
    "video/webm", "video/x-msvideo", "video/x-flv",
}
TEXT_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}

# 常见音频容器的文件头“魔数”，用于在 Content-Type 缺失/可疑时二次嗅探。
MAGIC_AUDIO = (
    (b"ID3", "mp3"), (b"\xff\xfb", "mp3"), (b"\xff\xf3", "mp3"), (b"\xff\xf2", "mp3"),
    (b"OggS", "ogg"), (b"fLaC", "flac"), (b"RIFF", "wav"),
    (b"\x00\x00\x00\x18ftyp", "mp4"), (b"ftypisom", "mp4"), (b"ftypM4A", "m4a"),
    (b"\x1a\x45\xdf\xa3", "webm"), (b"MK", "mkv"),
    (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11", "wma"),
)

# 作为最后兜底：URL 路径后缀（仅当 Content-Type 与魔数都无法判定时才看它）。
EXT_AUDIO = {"mp3", "m4a", "wav", "ogg", "aac", "flac", "webm", "mp4", "mkv",
             "mov", "avi", "opus", "wma", "amr", "3gp"}


def _classify(content_type, head):
    """Return 'audio' / 'text' / None from response Content-Type + magic bytes."""
    ct = (content_type or "").lower()
    if any(t in ct for t in AUDIO_CONTENT_TYPES) or ct.startswith("video/"):
        return "audio"
    if any(t in ct for t in TEXT_CONTENT_TYPES):
        return "text"
    for magic, _ in MAGIC_AUDIO:
        if head.startswith(magic):
            return "audio"
    return None


def _is_allowed_scheme(url):
    """Only http/https are accepted (localhost / private nets allowed by policy)."""
    return urlparse(url).scheme in ALLOWED_SCHEMES


def _classify_from_file(path):
    with open(path, "rb") as f:
        head = f.read(65536)
    for magic, _ in MAGIC_AUDIO:
        if head.startswith(magic):
            return "audio"
    # 尝试当文本读；失败则归为音频（保守下载）。
    try:
        head.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "audio"


def _dl_with_curl(url, dest):
    """curl fallback (better redirect/range support). Uses --fail so HTTP errors
    don't silently produce an empty/error page, and --max-filesize to honor cap."""
    if not shutil.which("curl"):
        return False
    rc = subprocess.run(
        ["curl", "-L", "-f", "-A", UA, "--max-filesize", str(MAX_BYTES), "-o", dest, url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode
    return rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def _strip_html(html):
    html = re.sub(r"(?is)<script.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?</style>", " ", html)
    html = re.sub(r"(?is)<head.*?</head>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Fetch a URL into audio or text")
    ap.add_argument("--url", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        sys.stderr.write(
            f"[fetch] 不支持的协议：{parsed.scheme}（仅允许 http/https，含 localhost/私网）。\n")
        sys.exit(1)

    dl_dir = os.path.join(args.workdir, "downloads")
    os.makedirs(dl_dir, exist_ok=True)

    # ---- 主路径：urllib 流式抓取，先嗅探类型再分支 ----
    try:
        req = urllib.request.Request(args.url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(65536)
            kind = _classify(r.headers.get("Content-Type"), head)
            if kind is None:
                # 无法从响应判定：看 URL 后缀作最后兜底（音频直链常见）。
                ext = os.path.splitext(parsed.path.split("?")[0])[1].lstrip(".").lower()
                kind = "audio" if ext in EXT_AUDIO else "text"

            if kind == "audio":
                dest = os.path.join(dl_dir, "input_media.bin")
                downloaded = len(head)
                with open(dest, "wb") as w:
                    w.write(head)
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > MAX_BYTES:
                            w.close()
                            os.remove(dest)
                            sys.stderr.write(
                                f"[fetch] 下载超过上限 {MAX_BYTES} 字节，已中止。\n")
                            sys.exit(1)
                        w.write(chunk)
                if os.path.getsize(dest) == 0:
                    sys.stderr.write("[fetch] 音频下载为空，请手动下载后作为输入传入。\n")
                    sys.exit(1)
                print(json.dumps({"type": "audio", "path": os.path.abspath(dest)},
                                 ensure_ascii=False))
                return

            # text：用声明的字符集解码（缺失则 utf-8，替换不可解码字节）。
            charset = r.headers.get_content_charset() or "utf-8"
            try:
                text = head.decode(charset, "replace")
            except (LookupError, UnicodeDecodeError):
                text = head.decode("utf-8", "replace")
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                try:
                    text += chunk.decode(charset, "replace")
                except (LookupError, UnicodeDecodeError):
                    text += chunk.decode("utf-8", "replace")
            text = _strip_html(text)
            dest = os.path.join(dl_dir, "page.txt")
            with open(dest, "w", encoding="utf-8") as f:
                f.write(text)
            print(json.dumps({"type": "text", "path": os.path.abspath(dest)},
                             ensure_ascii=False))
            return
    except Exception as e:
        sys.stderr.write(f"[fetch] urllib 抓取失败，尝试 curl 回退：{e}\n")

    # ---- 回退路径：curl 全量下载到临时文件，再按魔数/编码判定类型 ----
    tmp = os.path.join(dl_dir, "input_media.bin")
    if not _dl_with_curl(args.url, tmp):
        sys.stderr.write("[fetch] 抓取失败，请手动下载后作为输入文件传入。\n")
        sys.exit(1)
    kind = _classify_from_file(tmp)
    if kind == "text":
        with open(tmp, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8", "replace")
        except Exception:
            text = raw.decode("latin-1", "replace")
        text = _strip_html(text)
        dest = os.path.join(dl_dir, "page.txt")
        os.remove(tmp)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        print(json.dumps({"type": "text", "path": os.path.abspath(dest)},
                         ensure_ascii=False))
    else:
        print(json.dumps({"type": "audio", "path": os.path.abspath(tmp)},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
