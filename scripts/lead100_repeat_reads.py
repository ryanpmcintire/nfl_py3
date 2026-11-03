import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_llm_read as L


def main():
    ap = argparse.ArgumentParser(description="LEAD-100 repeat reads over a stored LEAD-99 packet")
    ap.add_argument("--source", required=True)
    ap.add_argument("--k", type=int, required=True)
    a = ap.parse_args()
    src = Path(a.source)
    d = json.loads(src.read_text(encoding="utf-8"))
    fz = json.loads(L.FROZEN.read_text())
    if fz["prompt_sha256"] != L.prompt_sha() or fz["model"] != L.MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    if d["prompt_sha256"] != L.prompt_sha():
        raise SystemExit("stored run used a different prompt")
    packet_text = json.dumps(d["packet"], ensure_ascii=False, sort_keys=True)
    if L.sha256_bytes(packet_text.encode("utf-8")) != d["packet_sha256"]:
        raise SystemExit("stored packet hash mismatch")
    ids = [g["game_id"] for g in d["packet"]["games"]]
    prompt_text = L.PROMPT.read_text(encoding="utf-8")
    cwd = L.ART / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    out_dir = L.ART / "repeats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(a.k):
        env, raw = L.call_claude(prompt_text, packet_text, str(cwd))
        rows = None
        err = None
        try:
            rows = L.validate(L.extract_json(raw), ids)
        except Exception as e:
            err = repr(e)
        t = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = out_dir / f"{src.stem}-repeat-{t}.json"
        path.write_text(
            json.dumps(
                {
                    "source_run": src.name,
                    "packet_sha256": d["packet_sha256"],
                    "prompt_sha256": L.prompt_sha(),
                    "model": L.MODEL,
                    "raw_response": raw,
                    "envelope": env,
                    "valid": rows is not None,
                    "error": err,
                    "parsed_rows": rows,
                    "recorded_at_utc": t,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(path, rows is not None, flush=True)


main()
