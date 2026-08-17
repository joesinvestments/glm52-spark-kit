import sys, re
p=sys.argv[1]; s=open(p).read()
# instrument create_from_process_group: timestamp each phase for reader and writer
old = '''        status = in_the_same_node_as(pg, source_rank=writer_rank)
        if group_rank == writer_rank:'''
new = '''        status = in_the_same_node_as(pg, source_rank=writer_rank)
        import time as _t
        _t0 = _t.time()
        logger.info("[shm-probe] rank=%s writer=%s world=%s phase=after_same_node t=%.3f",
                    group_rank, writer_rank, group_world_size, _t0)
        if group_rank == writer_rank:'''
assert s.count(old)==1; s=s.replace(old,new,1)
old = '''            handle = buffer_io.export_handle()
            if isinstance(pg, ProcessGroup):
                dist.broadcast_object_list(
                    [handle], src=global_ranks[writer_rank], group=pg
                )
            else:
                pg.broadcast_obj(handle, writer_rank)'''
new = '''            handle = buffer_io.export_handle()
            logger.info("[shm-probe] rank=%s phase=writer_mq_created dt=%.3f handle_addr=%s",
                        group_rank, _t.time()-_t0, getattr(handle, "connect_ip", None) or getattr(handle, "remote_subscribe_addr", None))
            if isinstance(pg, ProcessGroup):
                dist.broadcast_object_list(
                    [handle], src=global_ranks[writer_rank], group=pg
                )
            else:
                pg.broadcast_obj(handle, writer_rank)
            logger.info("[shm-probe] rank=%s phase=writer_broadcast_done dt=%.3f", group_rank, _t.time()-_t0)'''
assert s.count(old)==1; s=s.replace(old,new,1)
old = '''            if isinstance(pg, ProcessGroup):
                recv = [None]
                dist.broadcast_object_list(
                    recv, src=global_ranks[writer_rank], group=pg
                )
                handle = recv[0]  # type: ignore
            else:
                handle = pg.broadcast_obj(None, writer_rank)
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io'''
new = '''            if isinstance(pg, ProcessGroup):
                recv = [None]
                dist.broadcast_object_list(
                    recv, src=global_ranks[writer_rank], group=pg
                )
                handle = recv[0]  # type: ignore
            else:
                handle = pg.broadcast_obj(None, writer_rank)
            logger.info("[shm-probe] rank=%s phase=reader_got_handle dt=%.3f handle_addr=%s",
                        group_rank, _t.time()-_t0, getattr(handle, "connect_ip", None) or getattr(handle, "remote_subscribe_addr", None))
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
            logger.info("[shm-probe] rank=%s phase=reader_connected dt=%.3f", group_rank, _t.time()-_t0)
        if blocking:
            buffer_io.wait_until_ready()
            logger.info("[shm-probe] rank=%s phase=ready dt=%.3f", group_rank, _t.time()-_t0)
        return buffer_io'''
assert s.count(old)==1; s=s.replace(old,new,1)
open(p,"w").write(s); print("shm probe added")
