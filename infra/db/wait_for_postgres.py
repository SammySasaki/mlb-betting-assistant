
import time
import psycopg2
import os

DATABASE_URL = os.environ["DATABASE_URL"]

for i in range(30):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        print("Database is ready!")
        break
    except psycopg2.OperationalError:
        print("Waiting for database...")
        time.sleep(1)
else:
    raise TimeoutError("Database never became available")