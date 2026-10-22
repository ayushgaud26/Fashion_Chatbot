import pytest

from src.filters import parse_query


def ids_for(retriever, q):
    res = retriever.search(q)
    return res, res.products


def test_gender_filter(retriever, records):
    res, prods = ids_for(retriever, "Show women's shirts")
    assert prods and all(p["gender"] == "Women" for p in prods)
    expected = sum(1 for r in records if r["gender"] == "Women" and r["subCategory"] == "Shirts")
    assert res.total_matches == expected


def test_mens_does_not_match_women(retriever):
    _, prods = ids_for(retriever, "men's shirts")
    assert prods and all(p["gender"] == "Men" for p in prods)


def test_category_and_subcategory_filter(retriever, records):
    res, prods = ids_for(retriever, "Show products in the footwear category")
    assert res.total_matches == sum(r["masterCategory"] == "Footwear" for r in records)
    assert all(p["masterCategory"] == "Footwear" for p in prods)
    res, prods = ids_for(retriever, "jeans")
    assert prods and all(p["subCategory"] == "Jeans" for p in prods)


def test_article_type_filter_handles_plurals(retriever, records):
    for q in ("jeans", "show me a jean"):
        res, prods = ids_for(retriever, q)
        assert res.total_matches == sum(r["subCategory"] == "Jeans" for r in records)


def test_colour_filter_multiword(retriever, records):
    res, prods = ids_for(retriever, "navy blue shirts")
    assert prods and all("Navy Blue" in [colour.strip() for colour in p["baseColour"].split(",")]
                         and p["subCategory"] == "Shirts" for p in prods)


def test_combined_filters_all_apply(retriever, records):
    res, prods = ids_for(retriever, "blue men's jeans for casual wear")
    expected = [r for r in records if r["baseColour"] == "Blue" and r["gender"] == "Men"
                and r["articleType"] == "Jeans" and r["usage"] == "Casual"]
    assert res.total_matches == len(expected) and {p["id"] for p in prods} == {r["id"] for r in expected}


def test_all_matches_returned_up_to_limit(retriever, records):
    res = retriever.search("black formal shoes")
    expected = sum(r["baseColour"] == "Black" and r["articleType"] == "Formal Shoes" for r in records)
    assert res.total_matches == expected and len(res.products) == min(expected, retriever.max_results)
    big = retriever.search("footwear")
    assert len(big.products) == min(retriever.max_results, big.total_matches)
    assert big.truncated == (big.total_matches > retriever.max_results)


def test_synonyms_only_use_dataset_values(engine):
    pq = parse_query("sneakers", engine.catalog.vocab)
    assert set(pq.filters["articleType"]) <= set(engine.catalog.vocab.values["articleType"])
    pq = parse_query("gray t shirts", engine.catalog.vocab)
    assert pq.filters["baseColour"] == ["Grey"] and pq.filters["articleType"] == ["Tshirts"]


def test_semantic_ranking_never_breaks_filters(retriever):
    res = retriever.search("comfortable casual clothing for men")
    assert res.products
    assert all(p["gender"] == "Men" and p["usage"] == "Casual"
               and p["masterCategory"] == "Clothing" for p in res.products)


def test_casual_party_wear_does_not_treat_wear_as_product_type(retriever):
    res = retriever.search("casual party wear")
    assert res.products
    assert all(p["usage"] in {"Casual", "Party"} for p in res.products)
