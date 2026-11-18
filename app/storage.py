import os
import sqlite3
from pathlib import Path
from uuid import uuid4

def connection():
    path = Path(os.getenv("RAG_DB_PATH", "storage/rag.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connection() as con:
        con.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, filename TEXT, page_count INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, document_id TEXT, page INTEGER, content TEXT)")

def save_document(filename, pages):
    did = str(uuid4())
    with connection() as con:
        con.execute("INSERT INTO documents VALUES (?, ?, ?)", (did, filename, len(pages)))
        for page_number, page_text in enumerate(pages, start=1):
            # character chunks with modest overlap; simple on purpose for educational clarity
            step, size = 800, 1000
            for start in range(0, len(page_text), step):
                chunk = page_text[start:start + size].strip()
                if chunk:
                    con.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (str(uuid4()), did, page_number, chunk))
    return {"id": did, "filename": filename, "pages": len(pages)}

def all_chunks(document_id):
    with connection() as con:
        rows = con.execute("SELECT id, page, content FROM chunks WHERE document_id=?", (document_id,)).fetchall()
    return [dict(r) for r in rows]

def list_documents():
    with connection() as con:
        rows = con.execute("SELECT * FROM documents ORDER BY rowid DESC").fetchall()
    return [dict(r) for r in rows]
