# Demo query results (generated from the configured catalogue)

Dataset records: 1000 · Embedder: `hashing-384 (offline fallback, not semantic)`

| Demo | Query | Expected behaviour | Method | Total matches | Returned | Result |
|---|---|---|---|---:|---:|---|
| Product ID | `product id 20319780` | returns exact source record | exact_id | 1 | 1 | **PASS** |
| Exact product name | `RARE RABBIT Men Maroon Striped Crop Denim Jacket with Patchwork` | returns the corresponding source product | exact_name | 1 | 1 | **PASS** |
| Men's jeans under INR 1,500 | `Find men jeans under 1500` | gender/category/price filters all match the source record | structured | 3 | 3 | **PASS** |
| White women's T-shirts | `Find white women tshirts` | description-extracted colour, gender, and category filters match | structured | 5 | 5 | **PASS** |
| Rating above 4 | `shirts rating above 4.0` | every result has rating >= 4.0 | structured | 79 | 50 | **PASS** |
| Discount above 40% | `shirts discount above 40%` | every result has discount >= 40% | structured | 113 | 50 | **PASS** |
| Size M | `show size M` | every result lists size M | structured | 315 | 50 | **PASS** |
| Myntra links and images | `product id 20319780` | uses the same product record's URLs | exact_id | 1 | 1 | **PASS** |
