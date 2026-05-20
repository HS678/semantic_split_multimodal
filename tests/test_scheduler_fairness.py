from trainers.schedulers import FairRandomFullModalityScheduler


def test_fair_scheduler_no_replacement_until_pool_exhausted():
    pools = {
        0: ["m0_c0", "m0_c1", "m0_c2"],
        1: ["m1_c0", "m1_c1", "m1_c2"],
    }
    sch = FairRandomFullModalityScheduler(pools, seed=7)

    seen = {0: set(), 1: set()}
    for _ in range(3):
        sel = sch.select()
        seen[0].add(sel[0])
        seen[1].add(sel[1])

    # First cycle should visit all clients in each cluster exactly once.
    assert seen[0] == set(pools[0])
    assert seen[1] == set(pools[1])

    # Next selection should still return valid ids after refresh.
    nxt = sch.select()
    assert nxt[0] in pools[0]
    assert nxt[1] in pools[1]
