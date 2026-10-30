"""Per-invocation wiring: settings, consoles, logging, and factories for the engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from nextchap.cli.ui import PlainReporter, RichReporter, RichReview
from nextchap.config import Settings, resolve_api_key
from nextchap.llm.client import OpenRouterClient
from nextchap.llm.cost import CostMeter
from nextchap.models import ModelRegistry
from nextchap.pipeline.book import Book
from nextchap.pipeline.engine import Engine
from nextchap.retrieval.embed import make_embedder
from nextchap.retrieval.store import Reranker, VectorStore


@dataclass
class AppContext:
    data_dir: Path | None = None
    verbose: int = 0
    plain: bool = False
    out: Console = field(default_factory=Console)
    err: Console = field(default_factory=lambda: Console(stderr=True))
    _settings: Settings | None = None
    _registry: ModelRegistry | None = None
    log_file: Path | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            overrides = {"data_dir": self.data_dir} if self.data_dir else {}
            self._settings = Settings(**overrides)  # type: ignore[arg-type]
        return self._settings

    @property
    def interactive(self) -> bool:
        return not self.plain and self.err.is_terminal

    def setup_logging(self) -> None:
        level = (
            logging.WARNING
            if self.verbose == 0
            else logging.INFO
            if self.verbose == 1
            else logging.DEBUG
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.DEBUG)
        console_handler = RichHandler(
            console=self.err,
            show_path=False,
            show_time=False,
            rich_tracebacks=self.verbose > 0,
            markup=False,
        )
        console_handler.setLevel(level)
        root.addHandler(console_handler)
        try:
            log_dir = self.settings.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / "nextchap.log"
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)
        except OSError:
            self.log_file = None
        for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(
                logging.DEBUG if self.verbose > 1 else logging.WARNING
            )

    def registry(self, refresh: bool = False) -> ModelRegistry:
        if self._registry is None or refresh:
            self._registry = ModelRegistry.load(self.settings, refresh=refresh)
        return self._registry

    def client(self, meter: CostMeter) -> OpenRouterClient:
        return OpenRouterClient(
            self.settings, resolve_api_key(self.settings) or "", meter, self.registry()
        )

    def reporter(self, meter: CostMeter, preview: bool = True) -> RichReporter | PlainReporter:
        if self.interactive:
            return RichReporter(self.err, meter, self.settings, show_preview=preview)
        return PlainReporter(self.err)

    def engine(
        self,
        book: Book,
        meter: CostMeter,
        reporter: RichReporter | PlainReporter,
        review: bool = False,
    ) -> Engine:
        client = self.client(meter)
        settings = self.settings

        def store_factory() -> VectorStore:
            return VectorStore(
                book.paths.index,
                make_embedder(settings, client),
                settings.retrieval,
                self._reranker(),
            )

        hooks = RichReview(self.err, reporter) if review else None
        return Engine(book, client, meter, store_factory, reporter, hooks)

    def store_factory(self, book: Book) -> VectorStore:
        meter = CostMeter()
        client = None
        if self.settings.retrieval.embed_backend == "remote":
            client = self.client(meter)
        return VectorStore(
            book.paths.index,
            make_embedder(self.settings, client),
            self.settings.retrieval,
            self._reranker(),
        )

    def _reranker(self) -> Reranker | None:
        if not self.settings.retrieval.rerank:
            return None
        from nextchap.retrieval.rerank import QwenReranker

        return QwenReranker(self.settings.retrieval.rerank_model)
