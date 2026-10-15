

def test_goal_slider_weights():
    from kruszyk.rl import KEEP_T, goal_weights, signals_objective

    assert goal_weights(0.0) == (0.0, 0.0)  # only the cost per tonne
    assert goal_weights(0.5) == (KEEP_T, 0.0)  # as before
    keep, gain = goal_weights(1.0)
    assert keep > KEEP_T and gain == keep  # more tonnes are worth money
    base = [{"hauled_t": 1000.0, "cost_total": 20000.0}]
    cheaper_fewer = [{"hauled_t": 950.0, "cost_total": 18000.0}]  # 18.95 zł/t, 5 % tonnes fewer
    more = [{"hauled_t": 1050.0, "cost_total": 22000.0}]  # 20.95 zł/t, 5 % more
    assert signals_objective(cheaper_fewer, base, *goal_weights(0.0)) < signals_objective(more, base, *goal_weights(0.0))
    assert signals_objective(more, base, *goal_weights(1.0)) < signals_objective(cheaper_fewer, base, *goal_weights(1.0))
