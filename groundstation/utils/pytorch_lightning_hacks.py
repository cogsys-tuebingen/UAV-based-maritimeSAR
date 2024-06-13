import os
import torch.distributed as torch_distrib
from pytorch_lightning import _logger as log
import datetime
import torch
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule
from torch.nn.parallel.distributed import DistributedDataParallel

import resource


def increase_filedesc_limit(n=4096):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (n, rlimit[1]))


class MyDDP(DDPPlugin):
    """
     Added find_unused_parameters=True
     and enable timeout
    """
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def init_ddp_connection(self, global_rank: int = None, world_size: int = None) -> None:
        global_rank = global_rank if global_rank is not None else 0
        world_size = world_size if world_size is not None else 1
        # TODO: From where to get cluster environment?
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        os.environ['NCCL_BLOCKING_WAIT'] = '1'

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size,
                                             timeout=datetime.timedelta(minutes=10))

