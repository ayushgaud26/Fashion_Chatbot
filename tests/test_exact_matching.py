def test_exact_product_id(retriever, records):
    for rec in records[:5]:
        res = retriever.search(str(rec["id"]))
        assert res.method == "exact_id" and res.ids == [rec["id"]]
    res = retriever.search(f"show me product ID {records[3]['id']}")
    assert res.method == "exact_id" and res.ids == [records[3]["id"]]


def test_unknown_explicit_id_is_no_match(retriever):
    res = retriever.search("product id 99999999")
    assert res.method == "no_match" and not res.products


def test_exact_name_takes_priority_over_semantic(retriever, records):
    rec = records[0]
    res = retriever.search(rec["productDisplayName"])
    assert res.method == "exact_name" and rec["id"] in res.ids
    assert all(p["productDisplayName"] == rec["productDisplayName"] for p in res.products)


def test_exact_name_is_case_and_punctuation_insensitive(retriever, records):
    rec = records[0]
    res = retriever.search("  " + rec["productDisplayName"].upper() + "!! ")
    assert res.method == "exact_name" and rec["id"] in res.ids


def test_partial_name_is_not_an_exact_match(retriever, records):
    partial = " ".join(records[0]["productDisplayName"].split()[:2])
    res = retriever.search(partial)
    assert res.method != "exact_name"


def test_duplicate_names_return_all_and_note(retriever, records):
    from collections import Counter
    name, n = Counter(r["productDisplayName"] for r in records).most_common(1)[0]
    if n < 2:
        return
    res = retriever.search(name)
    assert res.method == "exact_name" and len(res.ids) == n and res.notes
