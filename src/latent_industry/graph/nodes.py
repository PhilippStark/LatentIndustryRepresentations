"""
nodes.py — the node index shared by every view.

Three SNI versions (SNI92, SNI02, SNI07) × four granularities: 4-digit codes
are the embedding targets; 3-digit, 2-digit and section codes are hierarchy
hubs. Descriptions are the SCB labels from metadata/mapping_Sni_*_{1992,2002,2007}.json.
"""

import json

import pandas as pd

from latent_industry.common.paths import METADATA_DIR
from latent_industry.common.sni_codes import SNI_2digits_to_sections


def _load_4d(year: str) -> dict[str, str]:
    with open(METADATA_DIR / f"mapping_Sni_4digits_{year}.json") as f:
        d = json.load(f)
    d.pop("0000", None)
    return d


def _load_3d(year: str) -> dict[str, str]:
    with open(METADATA_DIR / f"mapping_Sni_3digits_{year}.json") as f:
        d = json.load(f)
    d.pop("000", None)
    return d


def _load_2d(year: str) -> dict[str, str]:
    with open(METADATA_DIR / f"mapping_Sni_2digits_{year}.json") as f:
        d = json.load(f)
    d.pop("00", None)
    return d


def _load_section(year: str) -> dict[str, str]:
    with open(METADATA_DIR / f"mapping_Sni_section_{year}.json") as f:
        d = json.load(f)
    d.pop("Unknown", None)
    return d


_YEAR_BY_VERSION = {"SNI07": "2007", "SNI02": "2002", "SNI92": "1992"}


class NodeIndex:
    """
    nodes: DataFrame(node_id, code, node_type, version, description, is_target).
    node_idx: (node_type, version, code) -> node_id.
    target_ids: node ids of the 4d embedding targets, in table order.

    Node id order is fixed and all cached artifacts depend on it: all 4d
    (SNI07, SNI02, SNI92), then all 3d (same version order), then all 2d,
    then all sections.
    """

    def __init__(self):
        sni_4d = {ver: _load_4d(_YEAR_BY_VERSION[ver]) for ver in ("SNI07", "SNI02", "SNI92")}
        hub_3d = {ver: _load_3d(_YEAR_BY_VERSION[ver]) for ver in ("SNI07", "SNI02", "SNI92")}
        hub_2d = {ver: _load_2d(_YEAR_BY_VERSION[ver]) for ver in ("SNI07", "SNI02", "SNI92")}
        hub_section = {ver: _load_section(_YEAR_BY_VERSION[ver]) for ver in ("SNI07", "SNI02", "SNI92")}

        self.sni_4d = sni_4d
        self.hub_3d, self.hub_2d, self.hub_section = hub_3d, hub_2d, hub_section
        self.sections = {
            "SNI07": SNI_2digits_to_sections("SNI07"),
            "SNI02": SNI_2digits_to_sections("SNI02"),
            "SNI92": SNI_2digits_to_sections("SNI92"),
        }

        node_rows = []
        for version in ("SNI07", "SNI02", "SNI92"):
            for code, desc in sorted(sni_4d[version].items()):
                node_rows.append({"code": code, "node_type": "4d", "version": version,
                                   "description": desc, "is_target": True})
        for version in ("SNI07", "SNI02", "SNI92"):
            for code, desc in sorted(hub_3d[version].items()):
                node_rows.append({"code": code, "node_type": "3d", "version": version,
                                   "description": desc, "is_target": False})
        for version in ("SNI07", "SNI02", "SNI92"):
            for code, desc in sorted(hub_2d[version].items()):
                node_rows.append({"code": code, "node_type": "2d", "version": version,
                                   "description": desc, "is_target": False})
        for version in ("SNI07", "SNI02", "SNI92"):
            for letter, desc in sorted(hub_section[version].items()):
                node_rows.append({"code": letter, "node_type": "section", "version": version,
                                   "description": desc, "is_target": False})

        self.nodes = pd.DataFrame(node_rows).reset_index(drop=True)
        self.nodes.index.name = "node_id"
        self.N = len(self.nodes)

        self.node_idx: dict[tuple, int] = {
            (row.node_type, row.version, row.code): nid
            for nid, row in self.nodes.iterrows()
        }

        target_mask = self.nodes["is_target"].values
        self.target_ids = self.nodes[target_mask].index.tolist()
        self.target_pos = {nid: i for i, nid in enumerate(self.target_ids)}

    def get(self, node_type: str, version: str, code: str) -> int | None:
        return self.node_idx.get((node_type, version, code))

    def summary(self) -> str:
        n = self.nodes
        counts = n.groupby(["node_type", "version"]).size()
        lines = [f"Nodes: {self.N} total  ({len(self.target_ids)} targets)"]
        for (t, v), c in counts.items():
            lines.append(f"  {t:<8} {v:<6}: {c}")
        return "\n".join(lines)
