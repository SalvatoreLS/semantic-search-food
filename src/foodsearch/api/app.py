from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from foodsearch.api.backend import DemoBackend, DemoError, logger
from foodsearch.api.schemas import (
    CompareResponse,
    LabelIn,
    LabelProgress,
    LabelQueue,
    QueriesResponse,
    SearchResponse,
    Summary,
)
from foodsearch.images import media_type
from foodsearch.llm import LLMClient
from foodsearch.systems import COMPARE_SYSTEM, HEADLINE_SYSTEM

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
API_PREFIXES = ("/api/", "/images/")


def create_app(
    backend: DemoBackend | None = None,
    config: Path | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    demo = backend or DemoBackend.from_artifacts(config, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ok, report = demo.image_report()
        (logger.info if ok else logger.warning)(report)
        yield

    app = FastAPI(title="FoodSearch demo", lifespan=lifespan)
    app.state.backend = demo

    @app.middleware("http")
    async def revalidate_frontend(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        if not request.url.path.startswith(API_PREFIXES):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(DemoError)
    async def demo_error(request: Request, error: DemoError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/api/queries", response_model=QueriesResponse)
    def queries() -> QueriesResponse:
        return demo.queries_response()

    @app.get("/api/search", response_model=SearchResponse)
    def search(q: str = Query(min_length=1), system: str = HEADLINE_SYSTEM) -> SearchResponse:
        return demo.search(q, system)

    @app.get("/api/compare", response_model=CompareResponse)
    def compare(
        q: str = Query(min_length=1),
        left: str = COMPARE_SYSTEM,
        right: str = HEADLINE_SYSTEM,
    ) -> CompareResponse:
        return demo.compare(q, left, right)

    @app.get("/api/label/queue", response_model=LabelQueue)
    def label_queue() -> LabelQueue:
        return demo.label_queue()

    @app.post("/api/label", response_model=LabelProgress)
    def label(body: LabelIn) -> LabelProgress:
        return demo.record_label(body.pair_id, body.grade)

    @app.get("/api/label/export")
    def label_export() -> FileResponse:
        path = demo.labels_file()
        if path is None:
            raise HTTPException(404, "No human labels yet")
        return FileResponse(path, media_type="text/csv", filename="human.csv")

    @app.get("/api/summary", response_model=Summary)
    def summary() -> Summary:
        return demo.summary()

    @app.get("/images/{item_id}")
    def image(item_id: str) -> FileResponse:
        path = demo.images.path(item_id)
        if path is None:
            raise HTTPException(404, "No local image for this item")
        return FileResponse(path, media_type=media_type(path))

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
