import ftplib
import sqlite3
import datetime
import pandas as pd


def chunks(data, rows=10000):
    for i in range(0, len(data), rows):
        yield data[i:i + rows]


ftp = ftplib.FTP('ftp.seismo.nrcan.gc.ca', 'anonymous',
                 'user')

print("File List:")
files = ftp.dir()


now = datetime.datetime.now()
ftp.cwd("intermagnet/minute/provisional/IAGA2002/" + str(now.year) + "/" + str(now.strftime("%m")))

ftp.pwd()

file_list = ftp.nlst()


df = pd.DataFrame(columns=["Date_time", "Bx", "By", "Bz", "Bf"])


conn = sqlite3.connect("space.db", isolation_level=None)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='geo_mag'")

result = cur.fetchone()

if result is not None and result[0] == "geo_mag":
    cur.execute("drop table geo_mag")

cur.execute('''
    CREATE TABLE geo_mag (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station CHARACTER(3),
    lat REAL,
    long REAL,
    date_time DATETIME,
    bx REAL,
    by REAL,
    bz REAL,
    bf REAL
    );
    ''')

for file in file_list:
    station = ''
    lat = 0
    long = 0

    date_today = str(now.year) + str(now.strftime("%m")) + str(now.strftime("%d"))

    if date_today in file:
        ftp.retrbinary("RETR " + file, open(file, 'wb').write)
        temp = open(file, 'rb')

        data_rows = 0

        geo_mag = [line for line in temp]

        divData = chunks(geo_mag)  # divide into 10000 rows each

        for chunk in divData:
            cur.execute('BEGIN TRANSACTION')

            for line in chunk:
                if data_rows == 1:
                    row_bytes = line.split()
                    date_time = row_bytes[0].decode("utf-8") + " " + row_bytes[1].decode("utf-8")[:8]
                    row_txt = [date_time, row_bytes[3].decode("utf-8"), row_bytes[4].decode("utf-8"),
                               row_bytes[5].decode("utf-8"), row_bytes[6].decode("utf-8")]

                    a_series = pd.Series(row_txt, index=df.columns)

                    query = 'INSERT INTO geo_mag (station, lat, long, date_time, bx, by, bz, bf) VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' % (
                        station, lat, long, a_series["Date_time"], a_series["Bx"], a_series["By"], a_series["Bz"],
                        a_series["Bf"])
                    cur.execute(query)
                else:
                    if 'IAGA Code' in line.decode("utf-8") or 'IAGA CODE' in line.decode("utf-8"):
                        station = line.decode('utf-8').split()[2]
                        print(station)
                    elif 'Latitude' in line.decode("utf-8"):
                        lat = line.decode('utf-8').split()[2]
                    elif 'Longitude' in line.decode("utf-8"):
                        long = line.decode('utf-8').split()[2]
                    elif 'DATE       TIME' in line.decode("utf-8"):
                        data_rows = 1

            cur.execute('COMMIT')

conn.commit()

conn.close()