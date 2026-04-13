from __future__ import annotations

import uvicorn

from app.app_factory import app


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)