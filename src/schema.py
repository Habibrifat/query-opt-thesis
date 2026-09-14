TABLES = ["region", "nation", "part", "supplier",
          "partsupp", "customer", "orders", "lineitem"]

# 9 foreign-key join edges of the TPC-H schema
JOIN_EDGES = [
    ("nation",   "n_regionkey", "region",   "r_regionkey"),
    ("supplier", "s_nationkey", "nation",   "n_nationkey"),
    ("customer", "c_nationkey", "nation",   "n_nationkey"),
    ("orders",   "o_custkey",   "customer", "c_custkey"),
    ("lineitem", "l_orderkey",  "orders",   "o_orderkey"),
    ("lineitem", "l_partkey",   "part",     "p_partkey"),
    ("lineitem", "l_suppkey",   "supplier", "s_suppkey"),
    ("partsupp", "ps_partkey",  "part",     "p_partkey"),
    ("partsupp", "ps_suppkey",  "supplier", "s_suppkey"),
]

# Columns the generator may put predicates on: (table, column, kind)
PRED_COLS = [
    ("customer",   "c_acctbal",     "num"),
    ("customer",   "c_mktsegment",  "cat"),
    ("orders",     "o_totalprice",  "num"),
    ("orders",     "o_orderstatus", "cat"),
    ("orders",     "o_orderdate",   "date"),
    ("lineitem",   "l_quantity",    "num"),
    ("lineitem",   "l_extendedprice", "num"),
    ("lineitem",   "l_discount",    "num"),
    ("lineitem",   "l_tax",         "num"),
    ("lineitem",   "l_shipdate",    "date"),
    ("lineitem",   "l_returnflag",  "cat"),
    ("lineitem",   "l_linestatus",  "cat"),
    ("lineitem",   "l_shipmode",    "cat"),
    ("part",       "p_size",        "num"),
    ("part",       "p_retailprice", "num"),
    ("supplier",   "s_acctbal",     "num"),
]