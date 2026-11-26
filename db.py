"""SQLite storage: tickets, dynamic field definitions, and field values (EAV)."""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("TICKET_DB", "tickets.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT DEFAULT '',
    priority TEXT DEFAULT 'Medium',
    status TEXT DEFAULT 'Open',
    predicted_solution TEXT DEFAULT '',
    prediction_confidence REAL,
    ai_response TEXT DEFAULT '',
    final_response TEXT DEFAULT '',
    ai_error TEXT DEFAULT '',
    is_active INTEGER DEFAULT 1,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS field_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('text','dropdown','number','date','checkbox')),
    required INTEGER DEFAULT 0,
    options TEXT DEFAULT '',          -- comma separated, dropdown only
    is_active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ticket_field_values (
    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
    field_id INTEGER NOT NULL REFERENCES field_definitions(id),
    value TEXT,
    PRIMARY KEY (ticket_id, field_id)
);
"""


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def now():
    return datetime.now().isoformat(timespec="seconds")


def active_fields():
    with conn() as c:
        return c.execute("SELECT * FROM field_definitions WHERE is_active=1 ORDER BY id").fetchall()


def add_field(label, type_, required, options):
    with conn() as c:
        c.execute("INSERT INTO field_definitions (label,type,required,options) VALUES (?,?,?,?)",
                  (label, type_, int(required), options))


def deactivate_field(fid):
    with conn() as c:
        c.execute("UPDATE field_definitions SET is_active=0 WHERE id=?", (fid,))


def list_tickets():
    with conn() as c:
        return c.execute("SELECT * FROM tickets WHERE is_active=1 ORDER BY id DESC").fetchall()


def get_ticket(tid):
    with conn() as c:
        t = c.execute("SELECT * FROM tickets WHERE id=? AND is_active=1", (tid,)).fetchone()
        if not t:
            return None, {}
        vals = {r["field_id"]: r["value"] for r in
                c.execute("SELECT field_id, value FROM ticket_field_values WHERE ticket_id=?", (tid,))}
        return t, vals


def create_ticket(data, extra):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO tickets (customer_name,subject,description,category,priority,status,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (data["customer_name"], data["subject"], data["description"], data["category"],
             data["priority"], "Open", now(), now()))
        tid = cur.lastrowid
        save_values(c, tid, extra)
        return tid


def update_ticket(tid, data, extra):
    with conn() as c:
        c.execute(
            "UPDATE tickets SET customer_name=?,subject=?,description=?,category=?,priority=?,status=?,"
            "final_response=?,updated_at=? WHERE id=?",
            (data["customer_name"], data["subject"], data["description"], data["category"],
             data["priority"], data["status"], data["final_response"], now(), tid))
        save_values(c, tid, extra)


def save_values(c, tid, extra):
    for fid, v in extra.items():
        c.execute("INSERT OR REPLACE INTO ticket_field_values VALUES (?,?,?)", (tid, fid, v))


def save_ai(tid, **kw):
    """Save pipeline output (predicted_solution, prediction_confidence, ai_response, ai_error)."""
    cols = ", ".join(f"{k}=?" for k in kw)
    with conn() as c:
        c.execute(f"UPDATE tickets SET {cols}, updated_at=? WHERE id=?", (*kw.values(), now(), tid))


def soft_delete(tid):
    with conn() as c:
        c.execute("UPDATE tickets SET is_active=0, updated_at=? WHERE id=?", (now(), tid))
