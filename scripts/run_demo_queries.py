"""Run representative queries against the configured product catalog and write a verification report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engine import load_engine

engine = load_engine()
records = engine.catalog.records
rows = []


def check(label, query, predicate, expectation):
    result = engine.retriever.search(query)
    ok = bool(result.products) and all(predicate(p) for p in result.products)
    rows.append((label, query, expectation, result.method, result.total_matches, len(result.products),
                 "PASS" if ok else "FAIL"))
    return ok


first = records[0]
check("Product ID", f"product id {first['id']}", lambda p: p["id"] == first["id"], "returns exact source record")
check("Exact product name", first["productDisplayName"], lambda p: p["id"] == first["id"],
      "returns the corresponding source product")
check("Men's jeans under INR 1,500",
      "Find men jeans under 1500",
      lambda p: p.get("gender") == "Men" and p.get("subCategory") == "Jeans" and p.get("price", 999999) <= 1500,
      "gender/category/price filters all match the source record")
check("White women's T-shirts", "Find white women tshirts",
      lambda p: p.get("gender") == "Women" and p.get("baseColour") == "White" and p.get("subCategory") == "Tshirts",
      "description-extracted colour, gender, and category filters match")
check("Rating above 4", "shirts rating above 4.0",
      lambda p: p.get("rating") is not None and p["rating"] >= 4.0, "every result has rating >= 4.0")
check("Discount above 40%", "shirts discount above 40%",
      lambda p: p.get("discountPercentage") is not None and p["discountPercentage"] >= 40,
      "every result has discount >= 40%")
check("Size M", "show size M", lambda p: "M" in (p.get("sizeValues") or []),
      "every result lists size M")
check("Myntra links and images", f"product id {first['id']}",
      lambda p: p.get("productUrl", "").startswith("https://www.myntra.com/")
      and p.get("imageUrl", "").startswith("https://"), "uses the same product record's URLs")

lines = ["# Demo query results (generated from the configured catalogue)", "",
         f"Dataset records: {len(records)} · Embedder: `{engine.embedder.name}`", "",
         "| Demo | Query | Expected behaviour | Method | Total matches | Returned | Result |",
         "|---|---|---|---|---:|---:|---|"]
lines += [f"| {a} | `{b}` | {c} | {d} | {f} | {g} | **{h}** |"
          for a, b, c, d, f, g, h in rows]
Path(__file__).resolve().parent.parent.joinpath("docs", "demo_results.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(0 if all(r[-1] == "PASS" for r in rows) else 1)
