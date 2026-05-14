from pathlib import Path
import sqlite3

ROOT = Path(r"C:\Users\cprad\Desktop\VS coding projects\ECE 484\ece484_rosbags")

for bag_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    db3_files = list(bag_dir.glob("*.db3"))

    if not db3_files:
        print(f"\n{bag_dir.name}: no .db3 files found")
        continue

    print(f"\n==============================")
    print(f"Bag: {bag_dir.name}")

    for db_path in db3_files:
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  DB: {db_path.name}")
        print(f"  Size: {size_mb:.2f} MB")

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            cur.execute("PRAGMA integrity_check;")
            integrity = cur.fetchone()[0]
            print(f"  Integrity: {integrity}")

            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cur.fetchall()]
            print(f"  Tables: {tables}")

            if "messages" in tables:
                cur.execute("SELECT COUNT(*) FROM messages;")
                msg_count = cur.fetchone()[0]
                print(f"  messages rows: {msg_count}")

            conn.close()

        except Exception as e:
            print(f"  ERROR opening sqlite db: {e}")