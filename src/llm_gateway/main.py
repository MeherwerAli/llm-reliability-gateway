import uvicorn

from .api import create_app

app = create_app()


def run() -> None:
    uvicorn.run("llm_gateway.main:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    run()
