import psycopg2
import pandas as pd

# Test connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="tpch",
    user="postgres",
    password="password"
)
print("Connected!")