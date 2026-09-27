#!/usr/bin/env python3
"""Загрузчик английского корпуса «In Search of Lost Time» для RAG-демо.

Скачивает три тома в переводе C. K. Scott Moncrieff (public domain) с
Project Gutenberg, вырезает служебный текст, делит книгу на части по
заголовкам и нарезает главы в demo_docs/proust/texts/ (~45 КБ на файл,
суммарно ~500 КБ — как у прежнего синтетического корпуса). Запуск:

    .venv/bin/python scripts/download_corpus.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests


class Volume:
    def __init__(self, vol_id: int, title: str, eid: int, start_marker: str, heading_re: str):
        self.vol_id = vol_id
        self.title = title
        self.eid = eid
        self.start_marker = start_marker
        self.heading_re = re.compile(heading_re)

# start_marker — уникальная строка первой содержательной строки текста:
# всё до неё (титул, переводчик, оглавление) выбрасывается.
VOLUMES = [
    Volume(1, "Swann's Way", 7178,
           "For a long time I used to go to bed early",
           r"^(?:OVERTURE|COMBRAY|SWANN IN LOVE|PLACE-NAMES: THE NAME)$"),
    Volume(2, "Within a Budding Grove", 63532,
           "My mother, when it was a question of our having M. de Norpois to dinner",
           r"^PART [IVX]+$"),
    Volume(3, "The Guermantes Way", 73425,
           "We made our way back along the Avenue Gabriel",
           r"^_CHAPTER (?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)_$"),
]
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt"
USER_AGENT = "rag-kb-mcp-demo/0.1 (corpus download for RAG homework)"

MAX_VOLUME_KB = 165   # бюджет одного тома в итоговом корпусе
CHUNK_TARGET_KB = 45  # примерный размер одного файла-главы
SLUG_RE = re.compile(r"[^a-z0-9]+")

OUT_DIR = Path(__file__).resolve().parent.parent / "demo_docs" / "proust" / "texts"


def fetch(eid: int) -> str:
    """Скачивает plain-text версию книги с Project Gutenberg."""
    resp = requests.get(GUTENBERG_URL.format(eid=eid), headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def clean_text(text: str, volume: Volume) -> str:
    """Убирает служебный текст Gutenberg и front matter издательства."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = end = None
    for i, line in enumerate(lines):
        if start is None and "*** START OF THE PROJECT GUTENBERG EBOOK" in line:
            start = i + 1
        elif "*** END OF THE PROJECT GUTENBERG EBOOK" in line:
            end = i
            break
    if start is None or end is None:
        raise RuntimeError(f"Gutenberg #{volume.eid}: START/END markers not found")
    body = lines[start:end]
    for i, line in enumerate(body):
        if volume.start_marker in line:
            return "\n".join(body[i:])
    raise RuntimeError(f"Gutenberg #{volume.eid}: start marker not found")


def segment(volume: Volume, text: str) -> list[tuple[str, str]]:
    """Делит текст на части по заголовкам; первая часть — до заголовка."""
    lines = text.split("\n")
    bounds = [i for i, line in enumerate(lines) if volume.heading_re.match(line.strip())]
    if not bounds or bounds[0] != 0:
        bounds = [0] + [b for b in bounds if b > 0]
    segments: list[tuple[str, str]] = []
    for pos, nxt in zip(bounds, bounds[1:] + [len(lines)], strict=False):
        heading = lines[pos].strip()
        if not volume.heading_re.match(heading):
            segments.append(("Opening", "\n".join(lines[pos:nxt]).strip()))
            continue
        subtitle = ""
        if heading.startswith("_CHAPTER"):
            seg_title = heading.strip("_").title().replace("Chapter", "Chapter")
        else:
            # у «PART I» подзаголовок идёт следующей непустой строкой в _..._
            for probe in lines[pos + 1 : pos + 5]:
                if probe.strip():
                    if probe.strip().startswith("_") and probe.strip().endswith("_"):
                        subtitle = probe.strip().strip("_")
                    break
            seg_title = f"Part {heading.split()[-1]}" + (f": {subtitle}" if subtitle else "")
        body = "\n".join(lines[pos:nxt]).strip()
        if body:
            segments.append((seg_title, body))
    return segments


def chunk_text(text: str, target_kb: int) -> list[str]:
    """Режет длинную часть на файлы по границам абзацев."""
    limit = target_kb * 1024
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        buf.append(para)
        size += len(para) + 2
        if size >= limit:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def slugify(title: str) -> str:
    return SLUG_RE.sub("-", title.lower()).strip("-")[:40] or "part"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.md"):
        old.unlink()

    total_files, total_bytes = 0, 0
    for volume in VOLUMES:
        print(f"[vol {volume.vol_id}] {volume.title} (Gutenberg #{volume.eid})...")
        segments = segment(volume, clean_text(fetch(volume.eid), volume))
        time.sleep(2)

        written = 0
        seq = 0
        for seg_title, seg_text in segments:
            if written >= MAX_VOLUME_KB * 1024:
                break
            chunks = chunk_text(seg_text, CHUNK_TARGET_KB)
            for k, chunk in enumerate(chunks, 1):
                if written >= MAX_VOLUME_KB * 1024:
                    break
                seq += 1
                counter = f" ({k}/{len(chunks)})" if len(chunks) > 1 else ""
                header = (
                    f"# {volume.title} — {seg_title}{counter}\n\n"
                    f"*Marcel Proust, trans. C. K. Scott Moncrieff. "
                    f"Source: Project Gutenberg #{volume.eid}, public domain. "
                    f"RAG demo corpus excerpt.*\n\n"
                )
                content = header + chunk + "\n"
                path = OUT_DIR / f"vol{volume.vol_id}_{seq:02d}_{slugify(seg_title)}.md"
                path.write_text(content, encoding="utf-8")
                written += len(content)
        available = sum(len(t) for _, t in segments)
        total_files += seq
        total_bytes += written
        print(f"[vol {volume.vol_id}] files: {seq}, ~{written // 1024} KB (available ~{available // 1024} KB)")

    print(f"done: {total_files} chapter files, ~{total_bytes // 1024} KB total -> {OUT_DIR}")


if __name__ == "__main__":
    main()
