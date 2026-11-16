"""Bi-Temporal Knowledge Graph Construction and Point-in-Time Query Engine (Phase 2).

Implements the Graphiti/Zep-inspired Bi-Temporal Knowledge Graph architecture:
- Every decision edge carries four temporal coordinates:
  1. valid_from (float: 1.0=Meeting a, 2.0=Meeting b, 3.0=Meeting c, 4.0=Meeting d)
  2. invalid_at (Optional[float]: None if active on the temporal frontier; set to t_supersede when replaced)
  3. created_at (ISO-8601 system ingestion timestamp)
  4. deleted_at (Optional[str]: None unless rolled back)
- Connects superseded decisions via explicit `SUPERSEDES` and `SUPERSEDED_BY` relationships.
- Supports both Neo4j Bolt Cypher execution (`neo4j` driver + `.cypher` export) and a fast,
  zero-dependency embedded `networkx.MultiDiGraph` + JSON store (`data/bitemporal_graph.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from src.models.schemas import (
    ActionItemNode,
    DecisionEdge,
    DisagreementNode,
    TeamExtractionBundle,
)

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None  # type: ignore


CANONICAL_TOPIC_ALIASES: Dict[str, str] = {
    "lcd": "display_screen",
    "lcd_display": "display_screen",
    "lcd_screen": "display_screen",
    "display": "display_screen",
    "screen": "display_screen",
    "touch_screen": "display_screen",
    "touchscreen": "display_screen",
    "speech": "speech_recognition",
    "voice_recognition": "speech_recognition",
    "voice_control": "speech_recognition",
    "speech_recognition_feature": "speech_recognition",
    "material": "case_material",
    "casing_material": "case_material",
    "back_material": "case_material",
    "remote_material": "case_material",
    "energy": "energy_source",
    "power": "energy_source",
    "power_source": "energy_source",
    "battery": "energy_source",
    "battery_options": "energy_source",
    "locator": "locator_feature",
    "lost_remote_locator": "locator_feature",
    "anti_loss_feature": "locator_feature",
    "finder_function": "locator_feature",
    "navigation": "navigation_control",
    "directional_control": "navigation_control",
    "scrolling_mechanism": "navigation_control",
    "cost": "production_cost_budget",
    "budget": "production_cost_budget",
    "final_costing": "production_cost_budget",
    "production_cost": "production_cost_budget",
    "price": "selling_price",
    "retail_price": "selling_price",
    "sales_price": "selling_price",
    "target_group": "target_age_group",
    "age_group": "target_age_group",
    "target_demographic": "target_age_group",
    "colour": "case_colour",
    "color": "case_colour",
    "case_color": "case_colour",
}


def canonicalize_topic(raw_topic: str) -> str:
    """Normalize topic strings while preserving specific sub-topics when distinct."""
    cleaned = raw_topic.strip().lower().replace("-", "_").replace(" ", "_")
    return CANONICAL_TOPIC_ALIASES.get(cleaned, cleaned)


class BiTemporalKnowledgeGraph:
    """Bi-Temporal Knowledge Graph managing evolving decisions, supersession chains, actions, and disagreements."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.teams: Dict[str, TeamExtractionBundle] = {}
        self.decision_index: Dict[str, DecisionEdge] = {}
        self.supersession_chains: Dict[str, List[List[str]]] = {}

    def ingest_bundle(self, bundle: TeamExtractionBundle) -> None:
        """Ingest a single team's extraction bundle and execute bi-temporal supersession mutations."""
        team_id = bundle.team_id
        self.graph.add_node(
            f"Team:{team_id}",
            node_type="Team",
            team_id=team_id,
            meetings_processed=bundle.meetings_processed,
            speaker_roles=bundle.speaker_roles,
            speaker_names=bundle.speaker_names,
        )

        for spk_id, role in bundle.speaker_roles.items():
            spk_node = f"Speaker:{team_id}:{spk_id}"
            self.graph.add_node(
                spk_node,
                node_type="Speaker",
                team_id=team_id,
                speaker_id=spk_id,
                role=role,
                name=bundle.speaker_names.get(spk_id, ""),
            )
            self.graph.add_edge(f"Team:{team_id}", spk_node, rel_type="HAS_MEMBER")

        # Sort decisions chronologically by valid_from while preserving original list order for ties
        indexed_decisions = list(enumerate(bundle.decisions))
        indexed_decisions.sort(key=lambda pair: (pair[1].valid_from, pair[0]))
        ordered_decisions = [d for _, d in indexed_decisions]

        by_id: Dict[str, DecisionEdge] = {d.edge_id: d for d in ordered_decisions}
        active_by_topic: Dict[str, DecisionEdge] = {}

        for dec in ordered_decisions:
            self.decision_index[dec.edge_id] = dec
            c_topic = canonicalize_topic(dec.topic)

            # 1. Explicit supersession link from subagent extraction
            if dec.supersedes_edge_id and dec.supersedes_edge_id in by_id:
                prev_edge = by_id[dec.supersedes_edge_id]
                if prev_edge.invalid_at is None or prev_edge.invalid_at > dec.valid_from:
                    prev_edge.invalid_at = dec.valid_from
            # 2. Automatic chronological supersession on matching canonical topic across later meetings
            elif c_topic in active_by_topic:
                prev_edge = active_by_topic[c_topic]
                if prev_edge.edge_id != dec.edge_id and prev_edge.valid_from < dec.valid_from:
                    # If the new edge modifies, replaces, drops, or updates the topic in a subsequent meeting
                    if prev_edge.invalid_at is None:
                        prev_edge.invalid_at = dec.valid_from
                    if not dec.supersedes_edge_id:
                        dec.supersedes_edge_id = prev_edge.edge_id

            active_by_topic[c_topic] = dec

        # Populate graph nodes and bi-temporal edges
        for dec in ordered_decisions:
            dec_node = f"Decision:{dec.edge_id}"
            c_topic = canonicalize_topic(dec.topic)
            topic_node = f"Topic:{team_id}:{c_topic}"

            self.graph.add_node(
                topic_node,
                node_type="Topic",
                team_id=team_id,
                topic=c_topic,
            )
            self.graph.add_node(
                dec_node,
                node_type="DecisionState",
                **dec.model_dump(),
                canonical_topic=c_topic,
            )

            # Bi-Temporal DECIDED_ON edge from Team to DecisionState
            self.graph.add_edge(
                f"Team:{team_id}",
                dec_node,
                key=dec.edge_id,
                rel_type="DECIDED_ON",
                topic=dec.topic,
                canonical_topic=c_topic,
                status=dec.status,
                valid_from=dec.valid_from,
                invalid_at=dec.invalid_at,
                created_at=dec.created_at,
                deleted_at=dec.deleted_at,
            )
            self.graph.add_edge(dec_node, topic_node, rel_type="ABOUT_TOPIC")

            if dec.supersedes_edge_id and dec.supersedes_edge_id in by_id:
                prev_node = f"Decision:{dec.supersedes_edge_id}"
                self.graph.add_edge(
                    dec_node,
                    prev_node,
                    rel_type="SUPERSEDES",
                    at_time=dec.valid_from,
                )
                self.graph.add_edge(
                    prev_node,
                    dec_node,
                    rel_type="SUPERSEDED_BY",
                    at_time=dec.valid_from,
                )

        # Build evolution chains per team
        chains: List[List[str]] = []
        superseded_targets = {
            d.supersedes_edge_id for d in ordered_decisions if d.supersedes_edge_id in by_id
        }
        # Find chain heads (earliest edges that are superseded by something, or trace backward from terminal edges)
        terminal_edges = [
            d for d in ordered_decisions if d.supersedes_edge_id in by_id and d.edge_id not in superseded_targets
        ]
        for term in terminal_edges:
            chain: List[str] = [term.edge_id]
            curr = term
            visited = {curr.edge_id}
            while curr.supersedes_edge_id and curr.supersedes_edge_id in by_id:
                prev_id = curr.supersedes_edge_id
                if prev_id in visited:
                    break
                visited.add(prev_id)
                chain.append(prev_id)
                curr = by_id[prev_id]
            chain.reverse()
            if len(chain) >= 2:
                chains.append(chain)

        self.supersession_chains[team_id] = chains

        # Add Action Items
        for act in bundle.action_items:
            if act.resolution_meeting_id and act.invalid_at is None:
                # Set invalid_at on open status if resolved in a subsequent meeting
                res_letter = act.resolution_meeting_id[-1].lower()
                res_order = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}.get(res_letter)
                if res_order and res_order > act.valid_from:
                    act.invalid_at = res_order
            act_node = f"Action:{act.action_id}"
            self.graph.add_node(act_node, node_type="ActionItem", **act.model_dump())
            self.graph.add_edge(
                f"Team:{team_id}",
                act_node,
                rel_type="HAS_ACTION_ITEM",
                valid_from=act.valid_from,
                invalid_at=act.invalid_at,
                status=act.status,
            )

        # Add Disagreements
        for dis in bundle.disagreements:
            dis_node = f"Disagreement:{dis.disagreement_id}"
            self.graph.add_node(dis_node, node_type="Disagreement", **dis.model_dump())
            self.graph.add_edge(
                f"Team:{team_id}",
                dis_node,
                rel_type="HAD_DISAGREEMENT",
                valid_from=dis.valid_from,
                invalid_at=dis.invalid_at,
                is_open=dis.is_open,
            )

        bundle.decisions = ordered_decisions
        self.teams[team_id] = bundle

    def build_from_directory(self, extracted_dir: Path | str = "data/extracted") -> Dict[str, Any]:
        """Load all 35 team extraction bundles and construct the full Bi-Temporal Knowledge Graph."""
        base = Path(extracted_dir)
        for json_file in sorted(base.glob("*.json")):
            bundle = TeamExtractionBundle.model_validate_json(json_file.read_text(encoding="utf-8"))
            self.ingest_bundle(bundle)
            # Save mutated bi-temporal timestamps back to extracted bundle
            json_file.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

        return self.summary_stats()

    def get_active_frontier(self, team_id: str, as_of: float = 4.0) -> List[DecisionEdge]:
        """Return all active (non-superseded) decisions for `team_id` at time `as_of` (`invalid_at is None` or `> as_of`)."""
        bundle = self.teams.get(team_id)
        if not bundle:
            return []
        active = [
            d
            for d in bundle.decisions
            if d.valid_from <= as_of and (d.invalid_at is None or d.invalid_at > as_of) and d.deleted_at is None
        ]
        return sorted(active, key=lambda d: (d.valid_from, d.edge_id))

    def get_superseded_history(self, team_id: str) -> List[DecisionEdge]:
        """Return all historical decisions for `team_id` where `invalid_at` is populated."""
        bundle = self.teams.get(team_id)
        if not bundle:
            return []
        return [d for d in bundle.decisions if d.invalid_at is not None]

    def get_evolution_chains(self, team_id: str) -> List[List[DecisionEdge]]:
        """Return ordered chains of `DecisionEdge` objects connected by `SUPERSEDES` links."""
        chain_ids = self.supersession_chains.get(team_id, [])
        result: List[List[DecisionEdge]] = []
        for cid_list in chain_ids:
            edges = [self.decision_index[eid] for eid in cid_list if eid in self.decision_index]
            if len(edges) >= 2:
                result.append(edges)
        return result

    def get_action_items(self, team_id: str, status_filter: Optional[str] = None) -> List[ActionItemNode]:
        """Return tracked action items for `team_id`."""
        bundle = self.teams.get(team_id)
        if not bundle:
            return []
        items = bundle.action_items
        if status_filter:
            items = [a for a in items if a.status.upper() == status_filter.upper()]
        return items

    def get_disagreements(self, team_id: str, only_open: bool = False) -> List[DisagreementNode]:
        """Return tracked disagreements and objections for `team_id`."""
        bundle = self.teams.get(team_id)
        if not bundle:
            return []
        items = bundle.disagreements
        if only_open:
            items = [d for d in items if d.is_open]
        return items

    def summary_stats(self) -> Dict[str, Any]:
        """Return global graph telemetry."""
        total_decisions = sum(len(b.decisions) for b in self.teams.values())
        active_decisions = sum(len(self.get_active_frontier(tid)) for tid in self.teams)
        superseded_decisions = sum(len(self.get_superseded_history(tid)) for tid in self.teams)
        total_chains = sum(len(c) for c in self.supersession_chains.values())
        total_actions = sum(len(b.action_items) for b in self.teams.values())
        total_disagreements = sum(len(b.disagreements) for b in self.teams.values())
        return {
            "teams_in_graph": len(self.teams),
            "total_graph_nodes": self.graph.number_of_nodes(),
            "total_graph_edges": self.graph.number_of_edges(),
            "total_decision_edges": total_decisions,
            "active_frontier_decisions": active_decisions,
            "superseded_historical_decisions": superseded_decisions,
            "supersession_chains": total_chains,
            "total_action_items": total_actions,
            "total_disagreements": total_disagreements,
        }

    def save_to_disk(
        self,
        json_path: Path | str = "data/bitemporal_graph.json",
        cypher_path: Path | str = "data/bitemporal_graph.cypher",
    ) -> None:
        """Persist the Bi-Temporal Knowledge Graph to JSON and generate an executable Neo4j Cypher dump."""
        jpath = Path(json_path)
        jpath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "stats": self.summary_stats(),
            "supersession_chains": self.supersession_chains,
            "teams": {tid: b.model_dump() for tid, b in sorted(self.teams.items())},
        }
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Export Neo4j Cypher statements for bi-temporal graph creation
        cpath = Path(cypher_path)
        cypher_lines: List[str] = [
            "// Auto-generated Bi-Temporal Knowledge Graph Cypher Dump",
            "CREATE CONSTRAINT team_id_unique IF NOT EXISTS FOR (t:Team) REQUIRE t.team_id IS UNIQUE;",
            "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:DecisionState) REQUIRE d.edge_id IS UNIQUE;",
        ]
        for tid, bundle in sorted(self.teams.items()):
            cypher_lines.append(
                f"MERGE (t:Team {{team_id: {json.dumps(tid)}}}) "
                f"SET t.speaker_roles = {json.dumps(json.dumps(bundle.speaker_roles))};"
            )
            for dec in bundle.decisions:
                inv_val = "null" if dec.invalid_at is None else str(dec.invalid_at)
                cypher_lines.append(
                    f"MERGE (d:DecisionState {{edge_id: {json.dumps(dec.edge_id)}}}) "
                    f"SET d.team_id = {json.dumps(dec.team_id)}, "
                    f"d.meeting_id = {json.dumps(dec.meeting_id)}, "
                    f"d.topic = {json.dumps(dec.topic)}, "
                    f"d.decision_value = {json.dumps(dec.decision_value)}, "
                    f"d.status = {json.dumps(dec.status)}, "
                    f"d.speaker_id = {json.dumps(dec.speaker_id)}, "
                    f"d.valid_from = {dec.valid_from}, "
                    f"d.invalid_at = {inv_val}, "
                    f"d.created_at = {json.dumps(dec.created_at)}, "
                    f"d.semantic_target_quote = {json.dumps(dec.semantic_target_quote)};"
                )
                cypher_lines.append(
                    f"MATCH (t:Team {{team_id: {json.dumps(tid)}}}), (d:DecisionState {{edge_id: {json.dumps(dec.edge_id)}}}) "
                    f"MERGE (t)-[r:DECIDED_ON {{edge_id: {json.dumps(dec.edge_id)}}}]->(d) "
                    f"SET r.valid_from = {dec.valid_from}, r.invalid_at = {inv_val}, r.status = {json.dumps(dec.status)};"
                )
                if dec.supersedes_edge_id:
                    cypher_lines.append(
                        f"MATCH (d_new:DecisionState {{edge_id: {json.dumps(dec.edge_id)}}}), "
                        f"(d_old:DecisionState {{edge_id: {json.dumps(dec.supersedes_edge_id)}}}) "
                        f"MERGE (d_new)-[:SUPERSEDES {{at_time: {dec.valid_from}}}]->(d_old);"
                    )
        cpath.write_text("\n".join(cypher_lines), encoding="utf-8")

    @classmethod
    def load_from_disk(cls, json_path: Path | str = "data/bitemporal_graph.json") -> "BiTemporalKnowledgeGraph":
        """Load the persisted Bi-Temporal Knowledge Graph from disk."""
        jpath = Path(json_path)
        kg = cls()
        if not jpath.exists():
            kg.build_from_directory()
            kg.save_to_disk(json_path=jpath)
            return kg
        raw = json.loads(jpath.read_text(encoding="utf-8"))
        for _, bundle_dict in sorted(raw.get("teams", {}).items()):
            kg.ingest_bundle(TeamExtractionBundle.model_validate(bundle_dict))
        return kg

    def sync_to_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ) -> bool:
        """Attempt live synchronization to a Neo4j Bolt instance if available."""
        if GraphDatabase is None:
            return False
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=2.0)
            with driver.session() as session:
                for tid, bundle in self.teams.items():
                    session.run(
                        "MERGE (t:Team {team_id: $team_id}) SET t.speaker_roles = $roles",
                        team_id=tid,
                        roles=json.dumps(bundle.speaker_roles),
                    )
                    for dec in bundle.decisions:
                        session.run(
                            """
                            MATCH (t:Team {team_id: $team_id})
                            MERGE (d:DecisionState {edge_id: $edge_id})
                            SET d.meeting_id = $meeting_id,
                                d.topic = $topic,
                                d.decision_value = $decision_value,
                                d.status = $status,
                                d.speaker_id = $speaker_id,
                                d.valid_from = $valid_from,
                                d.invalid_at = $invalid_at,
                                d.semantic_target_quote = $quote
                            MERGE (t)-[r:DECIDED_ON {edge_id: $edge_id}]->(d)
                            SET r.valid_from = $valid_from, r.invalid_at = $invalid_at
                            """,
                            team_id=tid,
                            edge_id=dec.edge_id,
                            meeting_id=dec.meeting_id,
                            topic=dec.topic,
                            decision_value=dec.decision_value,
                            status=dec.status,
                            speaker_id=dec.speaker_id,
                            valid_from=dec.valid_from,
                            invalid_at=dec.invalid_at,
                            quote=dec.semantic_target_quote,
                        )
            driver.close()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    kg = BiTemporalKnowledgeGraph()
    stats = kg.build_from_directory("data/extracted")
    kg.save_to_disk()
    print("Bi-Temporal Knowledge Graph Built:")
    print(json.dumps(stats, indent=2))

    print("\n--- Team TS3012 Active Frontier (invalid_at is None) ---")
    for edge in kg.get_active_frontier("TS3012"):
        print(f" [{edge.meeting_id} | t={edge.valid_from}] {edge.topic} ({edge.status}): {edge.decision_value}")

    print("\n--- Team TS3012 Evolution Chains (SUPERSEDES) ---")
    for chain in kg.get_evolution_chains("TS3012"):
        arrow_str = " -> ".join(
            f"[{e.meeting_id} t={e.valid_from}..{e.invalid_at}: {e.topic}={e.status}]" for e in chain
        )
        print(" ", arrow_str)
