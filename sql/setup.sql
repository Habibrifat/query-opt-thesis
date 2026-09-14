-- ============================================================
-- TPC-H Schema + Data Load (Scale Factor 1)
-- ============================================================

CREATE TABLE region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name      CHAR(25),
    r_comment   VARCHAR(152)
);

CREATE TABLE nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name      CHAR(25),
    n_regionkey INTEGER,
    n_comment   VARCHAR(152)
);

CREATE TABLE part (
    p_partkey     INTEGER PRIMARY KEY,
    p_name        VARCHAR(55),
    p_mfgr        CHAR(25),
    p_brand       CHAR(10),
    p_type        VARCHAR(25),
    p_size        INTEGER,
    p_container   CHAR(10),
    p_retailprice DECIMAL(15,2),
    p_comment     VARCHAR(23)
);

CREATE TABLE supplier (
    s_suppkey   INTEGER PRIMARY KEY,
    s_name      CHAR(25),
    s_address   VARCHAR(40),
    s_nationkey INTEGER,
    s_phone     CHAR(15),
    s_acctbal   DECIMAL(15,2),
    s_comment   VARCHAR(101)
);

CREATE TABLE partsupp (
    ps_partkey    INTEGER,
    ps_suppkey    INTEGER,
    ps_availqty   INTEGER,
    ps_supplycost DECIMAL(15,2),
    ps_comment    VARCHAR(199),
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
    c_custkey    INTEGER PRIMARY KEY,
    c_name       VARCHAR(25),
    c_address    VARCHAR(40),
    c_nationkey  INTEGER,
    c_phone      CHAR(15),
    c_acctbal    DECIMAL(15,2),
    c_mktsegment CHAR(10),
    c_comment    VARCHAR(117)
);

CREATE TABLE orders (
    o_orderkey      INTEGER PRIMARY KEY,
    o_custkey       INTEGER,
    o_orderstatus   CHAR(1),
    o_totalprice    DECIMAL(15,2),
    o_orderdate     DATE,
    o_orderpriority CHAR(15),
    o_clerk         CHAR(15),
    o_shippriority  INTEGER,
    o_comment       VARCHAR(79)
);

CREATE TABLE lineitem (
    l_orderkey      INTEGER,
    l_partkey       INTEGER,
    l_suppkey       INTEGER,
    l_linenumber    INTEGER,
    l_quantity      DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount      DECIMAL(15,2),
    l_tax           DECIMAL(15,2),
    l_returnflag    CHAR(1),
    l_linestatus    CHAR(1),
    l_shipdate      DATE,
    l_commitdate    DATE,
    l_receiptdate   DATE,
    l_shipinstruct  CHAR(25),
    l_shipmode      CHAR(15),
    l_comment       VARCHAR(44),
    PRIMARY KEY (l_orderkey, l_linenumber)
);

-- NOTE: Run strip_trailing.py first if your .tbl files end each line with '|'
COPY region   FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/region.tbl'   WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY nation   FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/nation.tbl'   WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY part     FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/part.tbl'     WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY supplier FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/supplier.tbl' WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY partsupp FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/partsupp.tbl' WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY customer FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/customer.tbl' WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY orders   FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/orders.tbl'   WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');
COPY lineitem FROM 'C:/Users/Shah Newaz Habib/Desktop/query-opt-thesis/data/lineitem.tbl' WITH (FORMAT csv, DELIMITER '|', QUOTE E'\b');

SELECT 'region' AS t, COUNT(*) FROM region
UNION ALL SELECT 'nation', COUNT(*) FROM nation
UNION ALL SELECT 'part', COUNT(*) FROM part
UNION ALL SELECT 'supplier', COUNT(*) FROM supplier
UNION ALL SELECT 'partsupp', COUNT(*) FROM partsupp
UNION ALL SELECT 'customer', COUNT(*) FROM customer
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'lineitem', COUNT(*) FROM lineitem;

CREATE INDEX idx_nation_regionkey    ON nation(n_regionkey);
CREATE INDEX idx_supplier_nationkey  ON supplier(s_nationkey);
CREATE INDEX idx_customer_nationkey  ON customer(c_nationkey);
CREATE INDEX idx_partsupp_partkey    ON partsupp(ps_partkey);
CREATE INDEX idx_partsupp_suppkey    ON partsupp(ps_suppkey);
CREATE INDEX idx_orders_custkey      ON orders(o_custkey);
CREATE INDEX idx_orders_orderdate    ON orders(o_orderdate);
CREATE INDEX idx_lineitem_orderkey   ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_partkey    ON lineitem(l_partkey);
CREATE INDEX idx_lineitem_suppkey    ON lineitem(l_suppkey);
CREATE INDEX idx_lineitem_shipdate   ON lineitem(l_shipdate);

ANALYZE;