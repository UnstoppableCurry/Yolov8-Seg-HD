#!/usr/bin/env python3
"""Checks for the static docs/ GitHub Pages site."""

from __future__ import annotations

import http.server
import re
import sys
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CANONICAL_PREFIX = "https://unstoppablecurry.github.io/Yolov8-Seg-HD/"
HTML_FILES = ["index.html", "change.html", "usage.html", "notes.html", "404.html"]

# Claims this overview must not invent.
FORBIDDEN = [
    r"mAP\s*[:=]\s*\d",
    r"准确率\s*\d",
    r"达到\s*\d+\s*%",
    r"SOTA",
    r"客户案例",
    r"生产环境已验证",
    r"本仓库训练结果",
    r"本项目推理输出",
]


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.lang = None
        self.title = ""
        self._in_title = False
        self.meta: dict[str, str] = {}
        self.canonical = None
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.alts: list[tuple[str, str | None]] = []
        self.aria_current: list[str] = []
        self.has_skip = False
        self.has_main = False
        self.nav_labels: list[str] = []
        self.h1 = 0
        self.figcaptions: list[str] = []
        self._in_figcaption = False
        self._figbuf: list[str] = []
        self.text_bits: list[str] = []

    def handle_starttag(self, tag, attrs):
        ad = dict(attrs)
        self.tags.append(tag)
        if tag == "html":
            self.lang = ad.get("lang")
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = ad.get("name") or ad.get("property")
            if key and "content" in ad:
                self.meta[key] = ad["content"]
        if tag == "link" and ad.get("rel") == "canonical":
            self.canonical = ad.get("href")
        if tag == "a":
            href = ad.get("href")
            if href:
                self.hrefs.append(href)
            if href == "#main" or "skip" in (ad.get("class") or ""):
                self.has_skip = True
            if ad.get("aria-current"):
                self.aria_current.append(ad["aria-current"])
        if tag == "img":
            src = ad.get("src")
            if src:
                self.srcs.append(src)
            self.alts.append((src or "", ad.get("alt")))
        if tag in {"script", "link"} and ad.get("href"):
            self.hrefs.append(ad["href"])
        if tag == "link" and ad.get("href"):
            self.hrefs.append(ad["href"])
        if tag == "main" or ad.get("id") == "main":
            self.has_main = True
        if tag == "nav":
            self.nav_labels.append(ad.get("aria-label") or "")
        if tag == "h1":
            self.h1 += 1
        if tag == "figcaption":
            self._in_figcaption = True
            self._figbuf = []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "figcaption":
            self._in_figcaption = False
            self.figcaptions.append("".join(self._figbuf))

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_figcaption:
            self._figbuf.append(data)
        self.text_bits.append(data)


def parse_file(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def local_target(href: str, page: Path) -> Path | None:
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https", "mailto"}:
        return None
    if href.startswith("#"):
        return page
    if href.startswith("/Yolov8-Seg-HD/"):
        rel = href[len("/Yolov8-Seg-HD/") :]
        if not rel or rel.endswith("/"):
            rel = rel + "index.html" if rel else "index.html"
        return DOCS / rel
    if href.startswith("/"):
        raise AssertionError(f"{page.name}: site-root path {href!r} will break on project Pages")
    return (page.parent / parsed.path).resolve()


def test_required_pages_exist() -> None:
    for name in HTML_FILES:
        path = DOCS / name
        assert path.is_file(), f"missing {path}"
    assert (DOCS / "assets/css/style.css").is_file()
    assert (DOCS / "robots.txt").is_file()
    assert (DOCS / "sitemap.xml").is_file()
    assert (DOCS / ".nojekyll").is_file()
    assert (DOCS / "assets/upstream/bus.jpg").is_file()
    assert (DOCS / "assets/upstream/ATTRIBUTION.txt").is_file()


def test_html_quality() -> None:
    for name in HTML_FILES:
        path = DOCS / name
        p = parse_file(path)
        assert p.lang in {"zh-Hans", "zh-CN", "zh"}, f"{name} lang={p.lang}"
        assert p.title.strip(), f"{name} empty title"
        assert p.has_skip, f"{name} missing skip link"
        assert p.has_main, f"{name} missing main"
        assert p.h1 == 1, f"{name} h1 count={p.h1}"
        assert any(p.nav_labels), f"{name} nav missing aria-label"
        if name != "404.html":
            assert p.canonical and p.canonical.startswith(CANONICAL_PREFIX), name
            assert "description" in p.meta, name
            assert p.meta.get("og:locale") == "zh_CN", name
            assert "viewport" in path.read_text(encoding="utf-8")
        for src, alt in p.alts:
            assert alt and alt.strip(), f"{name} img {src} missing alt"
            if "bus.jpg" in src:
                assert "不是" in alt or "非本" in alt, f"sample image alt must disclaim results: {alt}"
        text = "".join(p.text_bits)
        for pat in FORBIDDEN:
            assert not re.search(pat, text), f"{name} matched forbidden claim {pat}"
        if name == "index.html":
            assert "Upstream / sample" in text
            assert any("不是" in c and ("结果" in c or "预测" in c) for c in p.figcaptions), p.figcaptions


def test_links_resolve() -> None:
    for name in HTML_FILES:
        page = DOCS / name
        p = parse_file(page)
        for href in p.hrefs + p.srcs:
            target = local_target(href, page)
            if target is None:
                continue
            assert target.exists(), f"{name} broken local link {href} -> {target}"


def test_sitemap_and_robots() -> None:
    sm = (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    for loc in [
        CANONICAL_PREFIX,
        f"{CANONICAL_PREFIX}change.html",
        f"{CANONICAL_PREFIX}usage.html",
        f"{CANONICAL_PREFIX}notes.html",
    ]:
        assert loc in sm, loc
    robots = (DOCS / "robots.txt").read_text(encoding="utf-8")
    assert f"{CANONICAL_PREFIX}sitemap.xml" in robots


def test_no_repo_secrets_in_docs() -> None:
    blob = ""
    for path in DOCS.rglob("*"):
        if path.suffix.lower() in {".jpg", ".png", ".webp"}:
            continue
        if path.is_file():
            blob += path.read_text(encoding="utf-8", errors="ignore")
    assert "x-access-token" not in blob
    assert "BEGIN PRIVATE KEY" not in blob


def test_http_render() -> None:
    handler = http.server.SimpleHTTPRequestHandler
    # Serve docs/ as site root, matching Pages artifact contents.
    class DocsHandler(handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DOCS), **kwargs)

        def log_message(self, fmt, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DocsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}/"
    try:
        for name in ["index.html", "change.html", "usage.html", "notes.html", "assets/css/style.css", "assets/upstream/bus.jpg", "robots.txt", "sitemap.xml"]:
            url = urljoin(base, name)
            req = Request(url, headers={"User-Agent": "pages-test"})
            with urlopen(req, timeout=5) as resp:
                body = resp.read()
                assert resp.status == 200, name
                assert len(body) > 0, name
        with urlopen(urljoin(base, "index.html"), timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert 'href="assets/css/style.css"' in html
        css_url = urljoin(base, "assets/css/style.css")
        with urlopen(css_url, timeout=5) as resp:
            css = resp.read().decode("utf-8")
        assert "skip-link" in css
        assert "@media (max-width: 40rem)" in css
        # Project-page 404 uses absolute prefix; confirm those files exist so GitHub can serve them.
        assert (DOCS / "assets/css/style.css").is_file()
    except (HTTPError, URLError) as exc:
        raise AssertionError(exc) from exc
    finally:
        server.shutdown()
        server.server_close()


def test_readme_points_to_pages() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://unstoppablecurry.github.io/Yolov8-Seg-HD/" in readme


def main() -> int:
    tests = [
        test_required_pages_exist,
        test_html_quality,
        test_links_resolve,
        test_sitemap_and_robots,
        test_no_repo_secrets_in_docs,
        test_http_render,
        test_readme_points_to_pages,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
