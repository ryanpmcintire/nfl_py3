import argparse
import pathlib
import sys
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (research archive reader)"
FIELDS = "timestamp,original,statuscode,mimetype,length"


def get(url, delay, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:
            print(f"ERR {url} {e}", file=sys.stderr)
            time.sleep(2 * delay * (k + 1))
    return b""


def cdx(prefix, collapse):
    q = {"url": prefix, "matchType": "prefix", "fl": FIELDS, "limit": "20000"}
    if collapse:
        q["collapse"] = collapse
    return "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(q)


def main():
    p = argparse.ArgumentParser(description="Wayback CDX prefix queries and id_ snapshot fetches.")
    p.add_argument("--cdx", nargs="*", default=[], help="name=url_prefix pairs")
    p.add_argument("--collapse", default="urlkey")
    p.add_argument("--outdir", required=True)
    p.add_argument("--fetch", default="", help="TSV of timestamp, url, relative output path")
    p.add_argument("--delay", type=float, default=4.0)
    a = p.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    first = True
    for spec in a.cdx:
        name, prefix = spec.split("=", 1)
        if not first:
            time.sleep(a.delay)
        first = False
        body = get(cdx(prefix, a.collapse), a.delay)
        (out / f"cdx_{name}.txt").write_bytes(body)
        print(name, len(body.splitlines()))
    if not a.fetch:
        return
    for line in pathlib.Path(a.fetch).read_text().splitlines():
        if not line.strip():
            continue
        ts, url, dest = line.split("\t")
        target = out / dest
        if target.exists() and target.stat().st_size > 0:
            continue
        if not first:
            time.sleep(a.delay)
        first = False
        body = get(f"https://web.archive.org/web/{ts}id_/{url}", a.delay)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        print(dest, len(body))


if __name__ == "__main__":
    main()
