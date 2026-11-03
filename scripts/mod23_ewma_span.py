import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "mod23_unit8_span.py"
WIDE_SPANS = "(8, 16, 24, 32, 48, 64)"
text = SOURCE.read_text()
text = text.replace("SPANS = (3, 4, 6, 8, 12, 16, 24)", f"SPANS = {WIDE_SPANS}")
text = text.replace("mod23_unit8_span", "mod23_ewma_span_wide")
text = text.replace("grid of 3 to 24 games", "grid of 8 to 64 games")
wide = WIDE_SPANS[1:-1].split(", ")
text = text.replace("in (3, 24)", f"in ({wide[0]}, {wide[-1]})")
namespace = {"__file__": str(SOURCE), "__name__": "__main__", "sys": sys}
exec(compile(text, str(SOURCE), "exec"), namespace)
