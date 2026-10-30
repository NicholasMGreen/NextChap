"""Error types. Every user-facing failure carries a hint the CLI can show."""

from __future__ import annotations


class NextChapError(Exception):
    """Base class for expected, user-actionable failures."""

    hint: str = ""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        if hint:
            self.hint = hint


class ConfigError(NextChapError):
    hint = "Run `nextchap config show` to inspect the resolved settings."


class InvalidBookId(NextChapError):
    hint = "Book IDs use lowercase letters, digits, '_' and '-', and start with a letter or digit."


class BookNotFound(NextChapError):
    hint = "Run `nextchap books list` to see available books."


class BookExists(NextChapError):
    hint = "Pass --force to re-import from scratch, or choose another --book-id."


class BookLocked(NextChapError):
    hint = "Another nextchap process is working on this book. Wait for it to finish."


class IndexMismatch(NextChapError):
    hint = "Run `nextchap index rebuild <book-id>` (local only, no API cost)."


class BudgetExceeded(NextChapError):
    hint = "Raise --max-cost or set generation.max_cost in nextchap.toml."


class ParseError(NextChapError):
    hint = "Check that chapter headings look like 'Chapter 12: Title' on their own line."


class GenerationAborted(NextChapError):
    hint = "Progress is saved. Re-run the same command to resume where it stopped."


class LLMError(NextChapError):
    """A model call failed after retries."""


class FatalLLMError(LLMError):
    """A model call failed in a way retrying cannot fix (auth, credits, bad request)."""


class RetryableLLMError(LLMError):
    """Transient failure (rate limit, 5xx, timeout, empty answer)."""
