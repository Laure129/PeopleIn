from fastapi import FastAPI

app = FastAPI(title="PeopleIn")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
