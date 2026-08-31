"""
order_tool.py

Deterministic Python tool -- NO LLM involved in this file. The agent
calls this function when it needs to look up a real order; the raw
orders.json is never loaded into the LLM's prompt.

Schema (confirmed against real data/orders-data-dictionary.md):
  orders.json = {"dataset_name": ..., "snapshot_at": ..., "orders": [...]}
  orders is a LIST of records, each keyed by "order_id" (not a dict).

Design: ALLOW-LIST, not block-list. SAFE_FIELDS/SAFE_ITEM_FIELDS name
exactly which fields may reach the LLM; everything else (customer.*,
internal.*, or any field not listed) is dropped by default, even if
present in the raw record. Fail closed, not open.

Important per the data dictionary: tool output is itself UNTRUSTED
DATA. A field like customer_safe_message is customer-facing but still
just text -- agent.py must treat it as content to relay, never as an
instruction to follow (same discipline as KB chunk text).
"""

import json
from pathlib import Path

# Top-level order fields safe to return (from orders-data-dictionary.md).
SAFE_FIELDS = {
    "order_id", "membership_tier", "placed_at", "status",
    "status_updated_at", "shipped_at", "delivered_at", "carrier",
    "tracking_number", "estimated_delivery", "customer_safe_message",
}

# Per-item fields safe to return (sku is deliberately excluded --
# not on the data dictionary's safe list).
SAFE_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# Statuses where an old carrier estimate would misleadingly imply the
# order is still in motion.
NO_ETA_STATUSES = {"cancelled", "returned"}

# Statuses that require a human handoff rather than an agent answer.
HANDOFF_STATUSES = {"exception"}


def normalize_order_id(raw_id: str) -> str:
    """Trims whitespace, uppercases -- 'ord-1004' and ' ORD-1004 ' both resolve."""
    return raw_id.strip().upper()


def load_orders_data(data_path: str = "data/orders.json") -> dict:
    """Loads the full raw file (dataset_name, snapshot_at, orders list)."""
    with open(Path(data_path), encoding="utf-8") as f:
        return json.load(f)


def get_snapshot_at(data_path: str = "data/orders.json") -> str:
    """
    Returns the dataset's snapshot_at timestamp -- used as "now" for
    deterministic time-based eval checks (e.g. the 30-minute
    cancellation window), per the data dictionary's instruction.
    """
    return load_orders_data(data_path).get("snapshot_at")


def _find_order(order_id: str, orders_list: list[dict]) -> dict | None:
    """orders is a LIST, not a dict -- linear scan by order_id."""
    for record in orders_list:
        if record.get("order_id") == order_id:
            return record
    return None


def lookup_order(order_id: str, data_path: str = "data/orders.json") -> dict:
    """
    Takes a raw order ID (any case/whitespace), returns a dict with
    ONLY safe fields, or a structured not-found result. Never raises
    on a bad ID -- always returns something predictable for the agent
    to hand to the LLM.
    """
    if not order_id or not order_id.strip():
        return {"found": False, "reason": "no_order_id_provided"}

    normalized_id = normalize_order_id(order_id)

    try:
        data = load_orders_data(data_path)
    except FileNotFoundError:
        return {"found": False, "reason": "orders_data_unavailable"}

    record = _find_order(normalized_id, data.get("orders", []))
    if record is None:
        # Do NOT guess a similar ID -- data dictionary explicitly
        # forbids this. Return not-found, exactly as given.
        return {"found": False, "reason": "order_not_found", "order_id": normalized_id}

    safe_record = {
        field: record.get(field)
        for field in SAFE_FIELDS
        if field in record
    }

    # Items: allow-list per-item fields too (drops sku).
    safe_record["items"] = [
        {k: item.get(k) for k in SAFE_ITEM_FIELDS if k in item}
        for item in record.get("items", [])
    ]

    # Cancelled/returned: an old carrier estimate would misleadingly
    # imply the order is still arriving. Null it out; do not invent
    # a new one.
    if safe_record.get("status") in NO_ETA_STATUSES:
        safe_record["estimated_delivery"] = None

    if safe_record.get("estimated_delivery") is None:
        safe_record["estimated_delivery"] = "unavailable"

    # Flag statuses that need a human handoff -- the agent decides
    # HOW to say this, but the tool marks the fact deterministically
    # so that decision isn't left to the LLM's judgment alone.
    safe_record["needs_handoff"] = safe_record.get("status") in HANDOFF_STATUSES

    return {"found": True, **safe_record}


if __name__ == "__main__":
    import sys
    test_id = sys.argv[1] if len(sys.argv) > 1 else "ORD-1001"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/orders.json"
    print(json.dumps(lookup_order(test_id, data_path), indent=2))