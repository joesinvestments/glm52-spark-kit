import sys
p=sys.argv[1]; s=open(p).read()
old = '''    if os.environ.get("VLLM_ONE_GPU_PER_NODE", "0") == "1":
        if isinstance(pg, ProcessGroup):
            world_size = torch.distributed.get_world_size(group=pg)
        else:
            world_size = pg.world_size
        return [i == source_rank for i in range(world_size)]
'''
new = '''    if os.environ.get("VLLM_ONE_GPU_PER_NODE", "0") == "1":
        if isinstance(pg, ProcessGroup):
            world_size = torch.distributed.get_world_size(group=pg)
            _rk = torch.distributed.get_rank(group=pg)
        else:
            world_size = pg.world_size
            _rk = pg.rank
        # gloo-probe: this is called immediately before the first collective on
        # each new group (shm_broadcast handle broadcast). Log wall-clock arrival
        # per rank so inter-rank skew can be measured instead of guessed.
        import time as _t
        logger.info("[gloo-probe] rank=%s world=%s src=%s arrived_pre_first_collective at %.3f",
                    _rk, world_size, source_rank, _t.time())
        return [i == source_rank for i in range(world_size)]
'''
assert s.count(old)==1; open(p,"w").write(s.replace(old,new,1)); print("gloo probe added")
