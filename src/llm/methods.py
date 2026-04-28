"""LLM evaluation method identifiers."""

from __future__ import annotations


APPROACH_RADIUS_1_METHOD = "centralized_rule_with_approach_radius_1"

CENTRALIZED_RULE_LLM_METHODS = {
    "centralized_rule_multi_bt",
    "centralized_without_repair",
    APPROACH_RADIUS_1_METHOD,
    "llm_dag_with_dependency_prompt_and_centralized_rules",
}

APPROACH_RADIUS_1_COORDINATION = {
    "enable_cell_reservation": True,
    "enable_edge_reservation": True,
    "enable_resource_lock": True,
    "enable_recovery_lock": True,
    "enable_approach_zone_lock": True,
    "approach_radius": 1,
    "max_wait_ticks": 3,
    "reassignment_wait_ticks": 8,
}
