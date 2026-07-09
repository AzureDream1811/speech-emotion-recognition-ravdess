from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(redirect_slashes=False)

items = []

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ItemRequest(BaseModel):
    item: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"items": items}
    )


@app.post("/add")
def create_item(req: ItemRequest):
    items.append(req.item)
    return req


@app.post("/add/batch/")
def create_items_batch(new_items: list):
    items.extend(new_items)
    return {"items": new_items}


@app.get("/items/")
def read_items():
    return {"items": items}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id < 0 or item_id >= len(items):
        return {"error": "Item not found"}
    return {"item": items[item_id]}
