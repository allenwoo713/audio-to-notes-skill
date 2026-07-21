#!/usr/bin/env python3
"""Handle a URL input for audio-to-notes.

Two cases:
  * URL points to an audio/video file  -> download it, return type=audio
  * URL points to a web page           -> fetch HTML, strip to plain text, return type=text

Output goes to <workdir>/downloads/. Prints a single JSON line to stdout:
  {"type": "audio", "path": "..."}   or   {"type": "text", "path": "..."}

The caller (SKILL.md flow) then either transcribes the audio or reads the text directly.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

AUDIO_EXT_RE = re.compile(r"\.(mp3|m4a|wav|ogg|aac|flac|webm|mp4|mkv|mov|avi|opus)(\?|$)", re.I)
UA = "Mozilla/5.0 (compatible; audio-to-notes/1.0)"


def _dl(url, dest):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as w:
            shutil.copyfileobj(r, w)
        return True
    except Exception:
        # fall back to curl (better redirect/range support)
        if shutil.which("curl"):
            rc = subprocess.run(["curl", "-L", "-A", UA, "-o", dest, url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            return rc == 0
        return False


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

    dl_dir = os.path.join(args.workdir, "downloads")
    os.makedirs(dl_dir, exist_ok=True)

    if AUDIO_EXT_RE.search(args.url.split("?")[0]):
        dest = os.path.join(dl_dir, "input_media" + os.path.splitext(args.url.split("?")[0])[1][:6])
        ok = _dl(args.url, dest)
        if not ok or os.path.getsize(dest) == 0:
            sys.stderr.write("[fetch] 音频下载失败，请手动下载后作为输入文件传入。\n")
            sys.exit(1)
        print(json.dumps({"type": "audio", "path": os.path.abspath(dest)}, ensure_ascii=False))
        return

    # page -> text
    try:
        req = urllib.request.Request(args.url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", "ignore")
    except Exception as e:
        sys.stderr.write(f"[fetch] 网页抓取失败：{e}\n")
        sys.exit(1)
    text = _strip_html(raw)
    dest = os.path.join(dl_dir, "page.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(json.dumps({"type": "text", "path": os.path.abspath(dest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
