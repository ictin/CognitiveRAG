from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Literal

from .schemas import GraphEdge, GraphNode


class GraphMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT,
                    properties_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_relation ON graph_edges(relation_type);
                """
            )

    @staticmethod
    def _dump(payload: dict) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _load(payload: str | None) -> dict:
        if not payload:
            return {}
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def upsert_node(self, node: GraphNode) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_nodes(node_id, node_type, label, properties_json, provenance_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_type=excluded.node_type,
                    label=excluded.label,
                    properties_json=excluded.properties_json,
                    provenance_json=excluded.provenance_json,
                    updated_at=excluded.updated_at
                """,
                (
                    node.node_id,
                    node.node_type,
                    node.label,
                    self._dump(node.properties),
                    self._dump(node.provenance),
                    node.created_at,
                    node.updated_at,
                ),
            )

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_edges(edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    source_node_id=excluded.source_node_id,
                    relation_type=excluded.relation_type,
                    target_node_id=excluded.target_node_id,
                    properties_json=excluded.properties_json,
                    provenance_json=excluded.provenance_json,
                    updated_at=excluded.updated_at
                """,
                (
                    edge.edge_id,
                    edge.source_node_id,
                    edge.relation_type,
                    edge.target_node_id,
                    self._dump(edge.properties),
                    self._dump(edge.provenance),
                    edge.created_at,
                    edge.updated_at,
                ),
            )

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT node_id, node_type, label, properties_json, provenance_json, created_at, updated_at
                FROM graph_nodes WHERE node_id=?
                """,
                (node_id,),
            ).fetchone()
        if not row:
            return None
        return GraphNode(
            node_id=row['node_id'],
            node_type=row['node_type'],
            label=row['label'],
            properties=self._load(row['properties_json']),
            provenance=self._load(row['provenance_json']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at
                FROM graph_edges WHERE edge_id=?
                """,
                (edge_id,),
            ).fetchone()
        if not row:
            return None
        return GraphEdge(
            edge_id=row['edge_id'],
            source_node_id=row['source_node_id'],
            relation_type=row['relation_type'],
            target_node_id=row['target_node_id'],
            properties=self._load(row['properties_json']),
            provenance=self._load(row['provenance_json']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def get_edges_for_node(self, node_id: str, *, direction: Literal['outgoing', 'incoming', 'both'] = 'both') -> List[GraphEdge]:
        if direction not in {'outgoing', 'incoming', 'both'}:
            raise ValueError(f'unsupported direction: {direction}')

        if direction == 'outgoing':
            query = (
                "SELECT edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at "
                "FROM graph_edges WHERE source_node_id=? ORDER BY relation_type, edge_id"
            )
            params = (node_id,)
        elif direction == 'incoming':
            query = (
                "SELECT edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at "
                "FROM graph_edges WHERE target_node_id=? ORDER BY relation_type, edge_id"
            )
            params = (node_id,)
        else:
            query = (
                "SELECT edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at "
                "FROM graph_edges WHERE source_node_id=? OR target_node_id=? ORDER BY relation_type, edge_id"
            )
            params = (node_id, node_id)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            GraphEdge(
                edge_id=r['edge_id'],
                source_node_id=r['source_node_id'],
                relation_type=r['relation_type'],
                target_node_id=r['target_node_id'],
                properties=self._load(r['properties_json']),
                provenance=self._load(r['provenance_json']),
                created_at=r['created_at'],
                updated_at=r['updated_at'],
            )
            for r in rows
        ]

    def get_edges_by_relation(self, relation_type: str, *, limit: int | None = None) -> List[GraphEdge]:
        q = (
            "SELECT edge_id, source_node_id, relation_type, target_node_id, properties_json, provenance_json, created_at, updated_at "
            "FROM graph_edges WHERE relation_type=? ORDER BY edge_id"
        )
        params: tuple = (relation_type,)
        if limit is not None:
            q += " LIMIT ?"
            params = (relation_type, int(limit))

        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()

        return [
            GraphEdge(
                edge_id=r['edge_id'],
                source_node_id=r['source_node_id'],
                relation_type=r['relation_type'],
                target_node_id=r['target_node_id'],
                properties=self._load(r['properties_json']),
                provenance=self._load(r['provenance_json']),
                created_at=r['created_at'],
                updated_at=r['updated_at'],
            )
            for r in rows
        ]

    def get_neighbors(
        self,
        node_id: str,
        *,
        relation_type: str | None = None,
        direction: Literal['outgoing', 'incoming', 'both'] = 'both',
    ) -> List[GraphNode]:
        edges = self.get_edges_for_node(node_id, direction=direction)
        if relation_type:
            edges = [e for e in edges if e.relation_type == relation_type]

        neighbor_ids: list[str] = []
        for edge in edges:
            if edge.source_node_id == node_id:
                neighbor_ids.append(edge.target_node_id)
            else:
                neighbor_ids.append(edge.source_node_id)

        dedup = sorted({nid for nid in neighbor_ids if nid and nid != node_id})
        nodes: List[GraphNode] = []
        for nid in dedup:
            node = self.get_node(nid)
            if node:
                nodes.append(node)
        return nodes

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self.upsert_node(node)

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            self.upsert_edge(edge)
