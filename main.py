import uvicorn
import datetime
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
from dsg_oop import Dsg

app = FastAPI()
# origins = ["http://localhost:8000"]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/now")
async def now():
    return {"now": datetime.datetime.now().strftime('%Y-%m-%d')}


@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
'''
@app.post("/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./data"
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)
        print(file.filename)
    fname= [file.filename for file in files][0]
    Dsg(fname).image_grid_generate()
    return FileResponse(f"./data/{fname}_grid.png")
'''


@app.post("/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./data"
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)
        print(file.filename)
    fname= [file.filename for file in files][0]
    Dsg(fname).image_grid_generate()
    return FileResponse(f"./data/{fname}_align.png")
if __name__ == '__main__':
    uvicorn.run(app)
