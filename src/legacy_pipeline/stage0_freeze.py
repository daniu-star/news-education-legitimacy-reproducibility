# -*- coding: utf-8 -*-
"""
Stage 0: 数据库冻结与版本锁定
- 复制 crawler.db 为冻结版本
- 计算 SHA256 哈希
- 导出完整 schema
- 记录环境版本
"""
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DB = os.path.join(BASE, "xhs_crawler", "data", "crawler.db")
FROZEN_DIR = os.path.join(BASE, "analysis", "00_raw")
FROZEN_DB = os.path.join(FROZEN_DIR, "crawler_frozen.db")
SCHEMA_DIR = os.path.join(BASE, "analysis", "01_schema")

os.makedirs(FROZEN_DIR, exist_ok=True)
os.makedirs(SCHEMA_DIR, exist_ok=True)

print("=" * 60)
print("Stage 0: 数据库冻结")
print("=" * 60)

# 1. 计算源库哈希
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

src_hash = sha256_file(SRC_DB)
print(f"源库 SHA256: {src_hash}")
print(f"源库大小: {os.path.getsize(SRC_DB)} bytes")

# 2. 复制为冻结版本
shutil.copy2(SRC_DB, FROZEN_DB)
frozen_hash = sha256_file(FROZEN_DB)
print(f"冻结库 SHA256: {frozen_hash}")
print(f"哈希一致: {src_hash == frozen_hash}")

# 3. 导出完整 schema
conn = sqlite3.connect(SRC_DB)
cursor = conn.cursor()

schema_doc = []
schema_doc.append("# crawler.db Schema 文档")
schema_doc.append("")
schema_doc.append(f"- 数据库路径: {SRC_DB}")
schema_doc.append(f"- 冻结时间: {datetime.datetime.now().isoformat()}")
schema_doc.append(f"- SHA256: {src_hash}")
schema_doc.append("")

tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
).fetchall()

schema_json = {}
for (tname,) in tables:
    schema_doc.append(f"## 表: {tname}")
    schema_doc.append("")
    create_sql = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tname,)
    ).fetchone()[0]
    schema_doc.append(f"```sql\n{create_sql}\n```")
    schema_doc.append("")

    # 获取列信息
    cols = cursor.execute(f"PRAGMA table_info({tname})").fetchall()
    col_list = []
    for col in cols:
        col_list.append({
            "name": col[1],
            "type": col[2],
            "notnull": col[3],
            "default": col[4],
            "pk": col[5],
        })
    schema_json[tname] = {
        "columns": col_list,
        "create_sql": create_sql,
    }

    # 行数
    count = cursor.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
    schema_doc.append(f"- 行数: {count}")
    schema_doc.append("")

# 写 schema 文档
with open(os.path.join(SCHEMA_DIR, "schema.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(schema_doc))

# 写 schema JSON
with open(os.path.join(SCHEMA_DIR, "schema.json"), "w", encoding="utf-8") as f:
    json.dump(schema_json, f, ensure_ascii=False, indent=2)

# 4. 冻结清单
freeze_record = {
    "frozen_at": datetime.datetime.now().isoformat(),
    "source_db": SRC_DB,
    "source_sha256": src_hash,
    "frozen_db": FROZEN_DB,
    "frozen_sha256": frozen_hash,
    "hash_match": src_hash == frozen_hash,
    "python_version": sys.version,
    "sqlite_version": sqlite3.sqlite_version,
    "tables": schema_json,
}

with open(os.path.join(FROZEN_DIR, "freeze_record.json"), "w", encoding="utf-8") as f:
    json.dump(freeze_record, f, ensure_ascii=False, indent=2)

conn.close()
print("\n完成: 冻结版本已保存")
print(f"- 冻结库: {FROZEN_DB}")
print(f"- schema文档: {SCHEMA_DIR}/schema.md")
print(f"- 冻结清单: {FROZEN_DIR}/freeze_record.json")
