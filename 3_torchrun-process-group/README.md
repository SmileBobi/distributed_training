# 多进程启动及进程间通讯

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from datetime import timedelta
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

STORE_HOST = "127.0.0.1"
STORE_PORT = 45679

def run_main_store():
    return dist.TCPStore(
    STORE_HOST,
    STORE_PORT,
    is_master=True,
    wait_for_workers=False,
    timeout=datetime.timedelta(seconds=36000000))


def all_reduce(prefix, data, rank, world_size):
    timeout = datetime.timedelta(seconds=10)

    is_master = True if rank == 0 else False

    devices = torch.cuda.device_count()
    for i in range(devices):
        name = torch.cuda.get_device_name(i)
        logging.info(f"==========group: 2, device id: {i}, device name: {name}============")

    store = dist.TCPStore(
            STORE_HOST,
            STORE_PORT,
            is_master=False,
            wait_for_workers=False,
            timeout=timeout,
        )

    store = dist.PrefixStore(prefix, store)

    pg = dist.ProcessGroupNCCL(store, rank, world_size, timedelta(seconds=1000))
    work = pg.allreduce(data)
    work.wait()

    logging.info("all_reduce done")

if __name__ == "__main__":
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    replica_id = int(os.getenv("REPLICA_ID", 0))
    replica_num = int(os.getenv("REPLICA_NUM", 1))
    device_offset = int(os.getenv("DEVICE_OFFSET", 0))

    if rank == 1 and replica_id == 1:
        main_store = run_main_store()

    prefix = os.getenv("STORE_PREFIX", "abcdefg")
    prefix = prefix + "_" + str(rank)

    torch.cuda.set_device(rank + device_offset)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    data = torch.randn(4,4).cuda()
    # dist.all_reduce(data, group=None)

    logging.info(f"rank: {rank}, replica_id: {replica_id}, data: {data}")

    logging.info(f"prefix: {prefix}, replica_id: {replica_id}, replica_num: {replica_num}, rank: {rank}, world_size: {world_size}")

    if rank == 1:
        ctx = mp.get_context("spawn")
        worker = ctx.Process(target = all_reduce, args=(prefix, data, replica_id, replica_num), daemon=True)
        worker.start()
        worker.join()

        logging.info(f"allreduce finished and output data: {data}")
        logging.info(f"run test_process_group.py successfully !!!")

    if rank == 0 and replica_id == 0:
        import time
        time.sleep(3)
```

# 运行脚本

```bash
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 REPLICA_ID=0 REPLICA_NUM=2 torchrun --standalone --nnodes=1 --nproc-per-node=2 test_process_group.py &

CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_OFFSET=2 REPLICA_ID=1 REPLICA_NUM=2 torchrun --nnodes=1 --nproc-per-node=2 --rdzv-endpoint=localhost:29400 test_process_group.py &
```

# torchrun 启动流程

![torchrun impl](./images/torchrun.png)