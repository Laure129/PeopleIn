from fastapi import FastAPI

from . import __version__

app = FastAPI(title="PeopleIn", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
