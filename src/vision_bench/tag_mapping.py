from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from .dataset import normalize_tag
from .errors import BenchmarkError
from .io import identifier


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise BenchmarkError("Tag mapping JSON contains a duplicate key.")
    return obj


def _tag(value: str) -> str:
    tag = normalize_tag(value)
    if not tag:
        raise BenchmarkError("Tag mapping names must normalize to nonempty tags.")
    return tag


def _ordered(graph: dict, category: str, kind: str) -> list[str]:
    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise BenchmarkError(f"Tag mapping {kind} contains a cycle in {category!r}.") from exc


@dataclass(frozen=True)
class TagRules:
    aliases: dict[str, str]
    ancestors: dict[str, set[str]]

    def canonical(self, tag: str) -> str:
        return self.aliases.get(tag, tag)

    def expand(self, tags: set[str]) -> set[str]:
        canonical = {self.canonical(tag) for tag in tags}
        return canonical.union(*(self.ancestors[tag] for tag in canonical if tag in self.ancestors))


@dataclass(frozen=True)
class TagMapping:
    rules: dict[str, TagRules]
    sha256: str
    include_implications: bool

    def describe(self) -> dict:
        return {
            "sha256": self.sha256,
            "mode": "aliases_and_implications" if self.include_implications else "aliases",
            "score_rule": "maximum contributing native score",
            "decision_rule": "any contributing native tag at its own operating threshold",
        }


def _category_rules(category: str, value: dict, include_implications: bool) -> TagRules:
    if not isinstance(value, dict) or value.keys() - {"aliases", "implications"}:
        raise BenchmarkError(f"Invalid tag mapping rules for {category!r}.")
    aliases = value.get("aliases", {})
    implications = value.get("implications", {})
    if not isinstance(aliases, dict) or not isinstance(implications, dict):
        raise BenchmarkError(f"Aliases and implications must be objects in {category!r}.")
    alias_edges = {_tag(old): _tag(new) for old, new in aliases.items()}
    if len(alias_edges) != len(aliases):
        raise BenchmarkError(f"Duplicate normalized alias in {category!r}.")
    canonical = {}
    for tag in _ordered({old: [new] for old, new in alias_edges.items()}, category, "aliases"):
        if tag in alias_edges:
            target = alias_edges[tag]
            canonical[tag] = canonical.get(target, target)

    edges = defaultdict(set)
    for source, targets in implications.items():
        if not isinstance(targets, list) or not targets:
            raise BenchmarkError(f"Implication targets must be a nonempty array in {category!r}.")
        child = _tag(source)
        for target in targets:
            parent = _tag(target)
            edges[canonical.get(child, child)].add(canonical.get(parent, parent))
    ancestors = {}
    for tag in _ordered(dict(edges), category, "implications"):
        parents = edges.get(tag, set())
        if parents and include_implications:
            ancestors[tag] = parents.union(*(ancestors[parent] for parent in parents if parent in ancestors))
    return TagRules(canonical, ancestors)


def load_tag_mapping(path: Path, *, include_implications: bool = False) -> TagMapping:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise BenchmarkError("Tag mapping schema_version must be 1.")
    categories = value.get("categories")
    if not isinstance(categories, dict):
        raise BenchmarkError("Tag mapping needs a categories object.")
    rules = {
        identifier(category): _category_rules(category, entries, include_implications)
        for category, entries in categories.items()
    }
    return TagMapping(rules, hashlib.sha256(raw).hexdigest(), include_implications)
