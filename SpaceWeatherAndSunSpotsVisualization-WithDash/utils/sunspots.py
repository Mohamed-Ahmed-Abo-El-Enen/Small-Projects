import pandas as pd
from datetime import datetime as dt
import urllib.request
import sqlite3


def chunks(data, rows=10000):
    for i in range(0, len(data), rows):
        yield data[i:i + rows]


fileobject = urllib.request.urlopen("http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt")
df = pd.DataFrame(columns=["Date", "Sunspot_count", "Sunspot_sd", "Observ_No"])

conn = sqlite3.connect("space.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sunspots'")
result = cur.fetchone()
if result is not None and result[0] == "sunspots":
    cur.execute("drop table sunspots")

cur.execute('''
    CREATE TABLE sunspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE,
    sunspot_count INTEGER,
    sunspot_sd REAL,
    sunspot_obs_no INTEGER
    );
    ''')

sunspots = [line for line in fileobject]

divData = chunks(sunspots)

for chunk in divData:
    cur.execute('BEGIN TRANSACTION')

    for line in chunk:
        row_bytes = line.split()
        date = row_bytes[0].decode("utf-8") + "-" + row_bytes[1].decode("utf-8") + "-" + row_bytes[2].decode("utf-8")
        row_txt = [date, row_bytes[4].decode("utf-8"), row_bytes[5].decode("utf-8"), row_bytes[6].decode("utf-8")]
        row_txt[0] = dt.strptime(row_txt[0], "%Y-%m-%d").strftime("%Y-%m-%d")
        a_series = pd.Series(row_txt, index=df.columns)

        query = 'INSERT INTO sunspots (date, sunspot_count, sunspot_sd, sunspot_obs_no) VALUES ("%s", "%s", "%s", "%s")' % (
            a_series["Date"], a_series["Sunspot_count"], a_series["Sunspot_sd"], a_series["Observ_No"])
        cur.execute(query)

    cur.execute('COMMIT')

conn.commit()

conn.close()