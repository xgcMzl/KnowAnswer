from fastapi import FastAPI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
