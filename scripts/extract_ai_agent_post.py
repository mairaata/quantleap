"""Extract AI agent post from gh-pages RSS into Hugo markdown."""
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSS = ROOT / "scripts" / "ai-agent-rss.xml"
OUT = ROOT / "content" / "scribble" / "2025" / "ai-agent-series-1" / "index.md"
NS = {"content": "http://purl.org/rss/1.0/modules/content/"}


def html_to_markdown(body: str) -> str:
    body = html.unescape(body)
    body = body.replace("\u2014", "—").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    body = body.replace("ÔÇö", "—").replace("ÔÇÖ", "'").replace("ÔÇ£", '"').replace("ÔÇ¥", '"')

    body = re.sub(r"<figure>\s*<img src=\"([^\"]+)\"[^>]*>\s*<figcaption>.*?</figcaption>\s*</figure>", r"![](\1)", body, flags=re.DOTALL)
    body = re.sub(r"!<figure>.*?</figure>\s*</p>", "", body, flags=re.DOTALL)
    body = re.sub(r"<p>!<figure>.*?</figure>\s*</p>", "", body, flags=re.DOTALL)
    body = re.sub(r"!<figure>.*?</figure>", "", body, flags=re.DOTALL)

    body = re.sub(r"<h2 id=\"[^\"]*\">(.*?)</h2>", r"\n## \1\n", body, flags=re.DOTALL)
    body = re.sub(r"<h3 id=\"[^\"]*\">(.*?)</h3>", r"\n### \1\n", body, flags=re.DOTALL)
    body = re.sub(r"<p>(.*?)</p>", r"\1\n\n", body, flags=re.DOTALL)
    body = re.sub(r"<strong>(.*?)</strong>", r"**\1**", body, flags=re.DOTALL)
    body = re.sub(r"<em>(.*?)</em>", r"*\1*", body, flags=re.DOTALL)
    body = re.sub(r"<li><em>(.*?)</em></li>", r"- *\1*", body, flags=re.DOTALL)
    body = re.sub(r"<li>(.*?)</li>", r"- \1", body, flags=re.DOTALL)
    body = re.sub(r"<ul>\s*", "", body)
    body = re.sub(r"</ul>\s*", "\n", body)
    body = re.sub(r"<ol>\s*", "", body)
    body = re.sub(r"</ol>\s*", "\n", body)
    body = re.sub(r"<blockquote>\s*<p>(.*?)</p>\s*</blockquote>", r"> \1\n", body, flags=re.DOTALL)
    body = re.sub(r'<a href="([^"]+)">([^<]+)</a>', r"[\2](\1)", body)
    body = re.sub(r"<div class=\"highlight\">.*?</motion.div>", "", body, flags=re.DOTALL)
    body = re.sub(r"<div class=\"highlight\">.*?</div>", "", body, flags=re.DOTALL)
    body = re.sub(r"<span[^>]*>|</span>|</code></pre></div>", "", body)
    body = re.sub(r"<code[^>]*>|</code>|<pre[^>]*>|</pre>", "", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main() -> None:
    tree = ET.parse(RSS)
    item = tree.find(".//item")
    encoded = item.find("content:encoded", NS).text
    start = encoded.find("<![CDATA[") + 9
    end = encoded.rfind("]]>")
    body = encoded[start:end]
    md = html_to_markdown(body)

    front = """+++
title = "AI Agents and LLMs: Redefining the Future of Intelligent Systems"
date = 2025-01-01T00:00:00Z
draft = false
description = "Exploring the role of AI agents and Large Language Models (LLMs) in redefining the future of intelligent systems."
tags = ["AI Agents", "LLMs", "Intelligent Systems"]
categories = ["AI", "Technology"]
showToc = true
TocOpen = false
+++

"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(front + md + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(md)} chars)")


if __name__ == "__main__":
    main()
