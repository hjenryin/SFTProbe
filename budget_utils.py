"""
Shared budget allocation utility.

Allocates a total budget across named items proportionally to their weights,
with size-based adjustment to handle items that have fewer candidates than
their proportional share.
"""


def budget_allocation(weights, sizes, total_budget):
    """
    Allocate total_budget across items proportionally to weights,
    capped by available sizes.

    Args:
        weights: Dict mapping names to proportional weights (e.g. probabilities).
                 Does NOT need to sum to 1; will be normalized internally.
        sizes:   Dict mapping names to available capacity (int).
        total_budget: Total number to allocate (int).

    Returns:
        Dict mapping names to allocated amounts (int), summing to total_budget
        (or less if total capacity is insufficient).
    """
    names = list(weights.keys())
    if not names:
        return {}

    # Normalize weights to probabilities
    total_weight = sum(weights[n] for n in names)
    if total_weight == 0:
        # Equal distribution if all weights are zero
        probs = {n: 1.0 / len(names) for n in names}
    else:
        probs = {n: weights[n] / total_weight for n in names}

    # Sort by size ascending so we process smallest (most constrained) first
    sorted_names = sorted(names, key=lambda n: sizes[n])

    final_allocation = {}
    remaining_budget = total_budget
    remaining_probs = dict(probs)

    for name in sorted_names:
        available = sizes[name]

        # Compute this item's share of the remaining budget
        if remaining_probs:
            total_remaining_prob = sum(remaining_probs.values())
            if total_remaining_prob > 0:
                proportion = remaining_probs[name] / total_remaining_prob
            else:
                proportion = 0
            requested = int(proportion * remaining_budget)
        else:
            requested = 0

        if available < requested:
            # Constrained: take all available, surplus goes to others
            final_allocation[name] = available
            remaining_budget -= available
        else:
            allocated = min(requested, available)
            final_allocation[name] = allocated
            remaining_budget -= allocated

        del remaining_probs[name]

    # Handle rounding leftovers: give to largest items with remaining capacity
    if remaining_budget > 0:
        for name in reversed(sorted_names):
            if final_allocation[name] < sizes[name]:
                can_add = min(remaining_budget, sizes[name] - final_allocation[name])
                final_allocation[name] += can_add
                remaining_budget -= can_add
                if remaining_budget == 0:
                    break

    return final_allocation
