"""Thesis binder: sidebar tree + section dropdown (single source of truth for paths)."""

from __future__ import annotations

from typing import Any, Iterator
from urllib.parse import quote

# Value format: "Paper>Section>..." joined with ">" for storage and API.

SECTION_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Paper",
        [
            ("Paper>Abstract", "Abstract"),
            ("Paper>Introduction", "Introduction"),
            ("Paper>Literature review", "Literature review"),
            ("Paper>Main content>Method>Participants", "Main content › Method › Participants"),
            ("Paper>Main content>Method>Research design", "Main content › Method › Research design"),
            ("Paper>Main content>Method>Data collection", "Main content › Method › Data collection"),
            ("Paper>Main content>Method>Instruments", "Main content › Method › Instruments"),
            ("Paper>Main content>Method>Data analysis", "Main content › Method › Data analysis"),
            ("Paper>Main content>Method>Procedure", "Main content › Method › Procedure"),
            ("Paper>Main content>Results>Theme or RQ 1", "Main content › Results › Theme or RQ 1"),
            ("Paper>Main content>Results>Theme or RQ 2", "Main content › Results › Theme or RQ 2"),
            ("Paper>Main content>Discussion", "Main content › Discussion"),
            ("Paper>Main content>Conclusion", "Main content › Conclusion"),
            ("Paper>References", "References"),
            ("Paper>Appendices>Interview guide", "Appendices › Interview guide"),
            ("Paper>Appendices>Questionnaire / QAs", "Appendices › Questionnaire / QAs"),
            ("Paper>Appendices>Ethics / info sheets", "Appendices › Ethics / info sheets"),
        ],
    ),
    (
        "Front matter",
        [
            ("Front matter>Title page", "Title page"),
            ("Front matter>Other required pages", "Other required pages"),
        ],
    ),
    (
        "Research (not for compile)",
        [
            ("Research>Transcripts", "Transcripts"),
            ("Research>Coding memos", "Coding memos"),
            ("Research>PDFs / notes", "PDFs / notes"),
        ],
    ),
]

# Nested structure for sidebar <details> / lists.
SIDEBAR_TREE: list[dict[str, Any]] = [
    {
        "label": "Paper",
        "compiles": True,
        "children": [
            {"label": "Abstract", "path": "Paper>Abstract"},
            {"label": "Introduction", "path": "Paper>Introduction"},
            {"label": "Literature review", "path": "Paper>Literature review"},
            {
                "label": "Main content",
                "children": [
                    {
                        "label": "Method",
                        "children": [
                            {"label": "Participants", "path": "Paper>Main content>Method>Participants"},
                            {"label": "Research design", "path": "Paper>Main content>Method>Research design"},
                            {"label": "Data collection", "path": "Paper>Main content>Method>Data collection"},
                            {"label": "Instruments", "path": "Paper>Main content>Method>Instruments"},
                            {"label": "Data analysis", "path": "Paper>Main content>Method>Data analysis"},
                            {"label": "Procedure", "path": "Paper>Main content>Method>Procedure"},
                        ],
                    },
                    {
                        "label": "Results",
                        "children": [
                            {"label": "Theme or RQ 1", "path": "Paper>Main content>Results>Theme or RQ 1"},
                            {"label": "Theme or RQ 2", "path": "Paper>Main content>Results>Theme or RQ 2"},
                        ],
                    },
                    {"label": "Discussion", "path": "Paper>Main content>Discussion"},
                    {"label": "Conclusion", "path": "Paper>Main content>Conclusion"},
                ],
            },
            {"label": "References", "path": "Paper>References"},
            {
                "label": "Appendices",
                "children": [
                    {"label": "Interview guide", "path": "Paper>Appendices>Interview guide"},
                    {"label": "Questionnaire / QAs", "path": "Paper>Appendices>Questionnaire / QAs"},
                    {"label": "Ethics / info sheets", "path": "Paper>Appendices>Ethics / info sheets"},
                ],
            },
        ],
    },
    {
        "label": "Front matter",
        "compiles": True,
        "children": [
            {"label": "Title page", "path": "Front matter>Title page"},
            {"label": "Other required pages", "path": "Front matter>Other required pages"},
        ],
    },
    {
        "label": "Research",
        "compiles": False,
        "note": "Scrivener: do not compile into thesis unless appendix",
        "children": [
            {"label": "Transcripts", "path": "Research>Transcripts"},
            {"label": "Coding memos", "path": "Research>Coding memos"},
            {"label": "PDFs / notes", "path": "Research>PDFs / notes"},
        ],
    },
]


def section_path_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    for _, opts in SECTION_GROUPS:
        for value, label in opts:
            out[value] = label
    return out


def iter_leaf_paths(nodes: list[dict[str, Any]]) -> Iterator[str]:
    for n in nodes:
        p = n.get("path")
        if isinstance(p, str) and p.strip():
            yield p
        ch = n.get("children")
        if isinstance(ch, list):
            yield from iter_leaf_paths(ch)


def all_section_paths_ordered() -> list[str]:
    """Binder order: depth-firstwalk of SIDEBAR_TREE leaves."""
    return list(iter_leaf_paths(SIDEBAR_TREE))


VALID_SECTION_PATHS: frozenset[str] = frozenset(section_path_labels().keys())


def section_href(path: str) -> str:
    return "/section/" + quote(path, safe="")


def section_label(section_path: str) -> str:
    return section_path_labels().get(section_path, section_path)
