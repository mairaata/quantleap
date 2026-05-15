"""Convert Distill-style draft to Hugo PaperMod markdown."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "hallucination" / "2026-04-27-hallucination-cs.md"
DST = ROOT / "content" / "scribble" / "2026" / "hallucination-open-loop" / "index.md"

FIGURE_MAP = {
    "image 13.png": "image-13.png",
    "image 0.png": "image-0.png",
    "image 7.png": "image-7.png",
    "image 8.png": "image-8.png",
    "image 1.png": "image-1.png",
    "image 12.png": "image-12.png",
    "image 2.png": "image-2.png",
    "image 10.png": "image-10.png",
    "image 3.png": "image-3.png",
    "image 5.png": "image-5.png",
    "image 6.png": "image-6.png",
    "image 11.png": "image-11.png",
    "image 4.png": "image-4.png",
    "image 9.png": "image-9.png",
}


def slugify_image(path: str) -> str:
    name = path.split("/")[-1]
    return FIGURE_MAP.get(name, name.replace(" ", "-"))


def figure_repl(match: re.Match) -> str:
    img = slugify_image(match.group(1))
    caption = match.group(2).strip()
    return (
        f"![{caption}](/images/2026/hallucination/{img})\n\n"
        f"*{caption}*"
    )


def takeaway_repl(match: re.Match) -> str:
    inner = match.group(1)
    inner = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n", inner, flags=re.DOTALL)
    inner = re.sub(r"<li[^>]*>", "- ", inner)
    inner = re.sub(r"</li>", "\n", inner)
    inner = re.sub(r"<[^>]+>", "", inner)
    lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
    quoted = "\n".join(f"> {ln}" for ln in lines)
    return f"> **Key takeaways**\n>\n{quoted}\n"


def convert_body(text: str) -> str:
    if "</script>" in text:
        text = text.split("</script>", 1)[1].lstrip("\n")

    text = re.sub(r"<d-cite\s+key=\"[^\"]*\"\s*></d-cite>", "", text)

    figure_pattern = (
        r"\{%\s*include figure\.liquid\s+loading=\"eager\"\s+"
        r"path=\"assets/img/2026-04-27-hallucination-cs/([^\"]+)\"[^%]+%\}\s*\n*"
        r"<div class=\"caption\">\s*\n?\s*([^<]+?)\s*\n?</div>"
    )
    text = re.sub(figure_pattern, figure_repl, text, flags=re.DOTALL)
    text = re.sub(r"\{%\s*include figure\.liquid[^%]+%\}", "", text)

    text = re.sub(
        r'<motion.div class="l-body"[^>]*>(.*?)</div>',
        takeaway_repl,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'<div class="l-body"[^>]*>(.*?)</div>',
        takeaway_repl,
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r'<abbr title="[^"]*">([^<]*)</abbr>', r"\1", text)

    text = re.sub(
        r"\*\*An Observer\*\* \(Error Detection\)\*\*:\*\*",
        "**An Observer (Error Detection):**",
        text,
    )
    text = re.sub(
        r"\*\*An Actuator\*\* \(Dynamic Control\)\*\*:\*\*",
        "**An Actuator (Dynamic Control):**",
        text,
    )
    text = re.sub(
        r"\*\*A Feedback Loop\*\* \(Iterative Correction\)\.\*\*:\*\*",
        "**A Feedback Loop (Iterative Correction):**",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    raw = SRC.read_text(encoding="utf-8")
    body = convert_body(raw)
    front = """+++
title = "From Memorization to Divergence: A Systems-Control Perspective on LLM Hallucination"
date = 2026-05-15T00:00:00Z
draft = false
description = "Hallucinations in LLMs are predictable outcomes of open-loop architecture. This post reframes them through systems and control theory—from memorization and drift to closed-loop correction."
tags = ["LLMs", "Hallucination", "AI Safety", "Intelligent Systems"]
categories = ["AI"]
showToc = true
TocOpen = true
+++

"""
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(front + body, encoding="utf-8")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
