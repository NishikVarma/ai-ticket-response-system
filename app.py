"""FastAPI app: ticket CRUD, dynamic fields admin, AI pipeline trigger.

Run: uv run uvicorn app:app
"""
from datetime import date

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

import db
import pipeline

app = FastAPI(title="AI Ticket Response System")
templates = Jinja2Templates(directory="templates")

PRIORITIES = ["Low", "Medium", "High", "Urgent"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
CATEGORIES = ["General", "Account", "Billing", "Shipping", "Technical"]
FIELD_TYPES = ["text", "dropdown", "number", "date", "checkbox"]


def render(request, name, **ctx):
    return templates.TemplateResponse(request, name, ctx)


def redirect(url):
    return RedirectResponse(url, status_code=303)


async def read_form(request):
    """Return (core fields, dynamic field values as {field_id: str}, errors)."""
    form = await request.form()
    core = {k: (form.get(k) or "").strip() for k in
            ["customer_name", "subject", "description", "category", "priority", "status", "final_response"]}
    errors, extra = {}, {}
    for k in ("customer_name", "subject", "description"):
        if not core[k]:
            errors[k] = "Required"
    for f in db.active_fields():
        raw = (form.get(f"field_{f['id']}") or "").strip()
        err = validate_field(f, raw)
        if err:
            errors[f"field_{f['id']}"] = err
        extra[f["id"]] = raw
    return core, extra, errors


def validate_field(f, raw):
    if f["type"] == "checkbox":
        if f["required"] and raw != "1":
            return "Must be checked"
        return None
    if not raw:
        return "Required" if f["required"] else None
    try:
        if f["type"] == "number":
            float(raw)
        elif f["type"] == "date":
            date.fromisoformat(raw)
        elif f["type"] == "dropdown" and raw not in [o.strip() for o in f["options"].split(",")]:
            return "Choose a listed option"
    except ValueError:
        return "Invalid " + f["type"]
    return None


def form_ctx(ticket, values, errors, new):
    return dict(ticket=ticket, values=values, errors=errors, new=new, fields=db.active_fields(),
                priorities=PRIORITIES, statuses=STATUSES, categories=CATEGORIES)


@app.get("/")
def index(request: Request):
    return render(request, "list.html", tickets=db.list_tickets())


@app.get("/tickets/new")
def new_ticket(request: Request):
    return render(request, "ticket.html", **form_ctx({"priority": "Medium", "category": "General"}, {}, {}, True))


@app.post("/tickets")
async def create_ticket(request: Request):
    core, extra, errors = await read_form(request)
    if errors:
        return render(request, "ticket.html", **form_ctx(core, {str(k): v for k, v in extra.items()}, errors, True))
    tid = db.create_ticket(core, extra)
    pipeline.run(tid)
    return redirect(f"/tickets/{tid}")


@app.get("/tickets/{tid}")
def ticket_detail(request: Request, tid: int):
    ticket, vals = db.get_ticket(tid)
    if not ticket:
        return render(request, "list.html", tickets=db.list_tickets(), message="Ticket not found.")
    return render(request, "ticket.html", **form_ctx(ticket, {str(k): v for k, v in vals.items()}, {}, False))


@app.post("/tickets/{tid}")
async def update_ticket(request: Request, tid: int):
    ticket, _ = db.get_ticket(tid)
    if not ticket:
        return redirect("/")
    core, extra, errors = await read_form(request)
    if errors:
        core = {**dict(ticket), **core}
        return render(request, "ticket.html", **form_ctx(core, {str(k): v for k, v in extra.items()}, errors, False))
    db.update_ticket(tid, core, extra)
    return redirect(f"/tickets/{tid}")


@app.post("/tickets/{tid}/regenerate")
def regenerate(tid: int):
    if db.get_ticket(tid)[0]:
        pipeline.run(tid)
    return redirect(f"/tickets/{tid}")


@app.post("/tickets/{tid}/delete")
def delete_ticket(tid: int):
    db.soft_delete(tid)
    return redirect("/")


@app.get("/fields")
def fields_page(request: Request):
    return render(request, "fields.html", fields=db.active_fields(), types=FIELD_TYPES, error="")


@app.post("/fields")
def add_field(request: Request, label: str = Form(""), type: str = Form("text"),
              required: str = Form(""), options: str = Form("")):
    label, options = label.strip(), options.strip()
    error = ""
    if not label:
        error = "Label is required."
    elif type not in FIELD_TYPES:
        error = "Unknown field type."
    elif type == "dropdown" and not options:
        error = "Dropdown needs comma-separated options."
    if error:
        return render(request, "fields.html", fields=db.active_fields(), types=FIELD_TYPES, error=error)
    db.add_field(label, type, bool(required), options if type == "dropdown" else "")
    return redirect("/fields")


@app.post("/fields/{fid}/delete")
def delete_field(fid: int):
    db.deactivate_field(fid)
    return redirect("/fields")
