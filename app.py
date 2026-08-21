from fastapi import FastAPI

__version__ = "0.1.0"

app = FastAPI(title="PeopleIn", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
