# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from monai.data import DataLoader


import argparse
from launch import launch_dist

from monai.utils import set_determinism
from models.resample import SequentialDistributedSampler, distributed_concat
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./checkpoints/",
                 master_ip='localhost',
                 master_port=17750,
                 training_script="train.py",
                 dataloader_num_workers: int = 0,
                 pin_memory: bool = False,
                 ):
        assert env_type in ["pytorch", "ddp", "DDP"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.scheduler = None 
        self.model = None
        self.auto_optim = True
        # safer dataloader settings
        self.loader_num_workers = max(int(dataloader_num_workers), 0)
        self.loader_pin_memory = bool(pin_memory)
        self.train_dataset = None  # keep reference for retries

        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True

        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            print("GPU count mismatch")
            os._exit(0)

        if env_type == "DDP" or env_type == "ddp":
            self.ddp = True
            self.get_dist_args()
            if not self.not_call_launch:
                launch_dist(env_type=env_type,
                            num_nodes=1,
                            gpus_per_node=num_gpus,
                            master_addr=master_ip,
                            master_port=master_port,
                            training_script=training_script,
                            )
                os._exit(1)
            self.initialize_distributed()


    def print_rank_0(self, msg: str):
        if getattr(self, 'local_rank', 0) == 0:
            print(msg)

    def initialize_distributed(self):
        """Initialize torch.distributed."""
        if self.env_type == 'pytorch':
            self.print_rank_0('No need to initialize')
            return
        if self.env_type == 'DDP' or "deepspeed" in self.env_type:
            device = self.local_rank if self.local_rank is not None else 0
            torch.cuda.set_device(device)
            init_method = 'env://'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method)
            self.world_size = torch.distributed.get_world_size()
            self.print_rank_0(f"world size is {self.world_size}")

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None:
            return None
        num_workers = self.loader_num_workers
        pin_memory = self.loader_pin_memory
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              prefetch_factor=2 if num_workers > 0 else None,
                              persistent_workers=False if num_workers == 0 else False)
        else:
            if not train:
                sampler = SequentialDistributedSampler(dataset, batch_size=batch_size)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              sampler=sampler,
                              drop_last=False,
                              pin_memory=pin_memory,
                              prefetch_factor=2 if num_workers > 0 else None,
                              persistent_workers=False if num_workers == 0 else False)

    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default = 0, help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="do not call launch")
        ds_args = parser.parse_args()
        self.rank = int(os.environ.get('RANK',0))
        # self.local_rank = int(os.environ["LOCAL_RANK"])

        self.local_rank = ds_args.local_rank
        self.not_call_launch = ds_args.not_call_launch
        self.device = self.local_rank
    
        # self.master_addr = os.environ.get('MASTER_ADDR','127.0.0.1')
        # self.master_port = os.environ.get('MASTER_PORT','17500')

    def validation_single_gpu(self, val_dataset,):
        if self.ddp:
            print(f"single-gpu mode does not support DDP")
            exit(0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.model.to(self.device)
        val_outputs = []
        self.model.eval()
        return_list = False  # initialize
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list) :
                batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            else :
                print("unsupported data type")
                exit(0)

            with torch.no_grad():
                val_out = self.validation_step(batch)
                assert val_out is not None 

            if isinstance(val_out, (list, tuple)):
                return_list = True
            val_outputs.append(val_out)
        if isinstance(val_out, list) or isinstance(val_out, tuple):
            return_list = True

        val_outputs = torch.tensor(val_outputs)
        if not return_list:
            # single metric
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v):
                    v_sum += v
                    length += 1

            if length == 0:
                v_sum = 0
            else :
                v_sum = v_sum / length             
        else :
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]

            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]):
                        v_sum[i] += v[i]
                        length[i] += 1

            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else :
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def train(self,
                train_dataset,
                optimizer=None,
                model=None,
                val_dataset=None,
                scheduler=None,
              ):
        # keep dataset reference for fallback
        self.train_dataset = train_dataset
        if scheduler is not None:
            self.scheduler = scheduler

        set_determinism(1234 + self.local_rank)
        if self.model is not None:
            pass
        self.global_step = 0
        if self.env_type == "pytorch":
            if self.model is not None:
                self.model.to(self.device)
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)

        elif self.ddp:
            if self.local_rank == 0:
                os.makedirs(self.logdir, exist_ok=True)
                self.writer = SummaryWriter(self.logdir)
            else:
                self.writer = None
            if self.model is not None:
                self.model.cuda(self.local_rank)
                # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[self.local_rank],
                                                                    output_device=self.local_rank,
                                                                    find_unused_parameters=True)
         
        else :
            print("not support env_type")
            exit(0)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        if val_dataset is not None:
            val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
        else :
            val_loader = None 
            
        for epoch in range(0, self.max_epochs):
            self.epoch = epoch 
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            # retry wrapper for worker crashes
            try:
                self.train_epoch(train_loader, epoch)
            except RuntimeError as e:
                if 'DataLoader worker' in str(e):
                    print("[WARN] DataLoader worker crashed. Retrying with num_workers=0...")
                    self.loader_num_workers = 0
                    train_loader = self.get_dataloader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
                    self.train_epoch(train_loader, epoch)
                else:
                    raise

            val_outputs = []
            if (epoch+1) % self.val_every == 0 \
                    and val_loader is not None :
                if self.model is not None:
                    self.model.eval()
                if self.ddp:
                    torch.distributed.barrier()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else :
                        print("unsupported data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None 

                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                # aggregate across ranks
                if self.ddp:
                    val_outputs = torch.tensor(val_outputs).cuda(self.local_rank)
                    torch.distributed.barrier()

                    val_outputs = distributed_concat(val_outputs, num_total_examples=len(val_loader.sampler.dataset))
                else :
                    val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
                    if not return_list:
                        # single metric
                        length = 0
                        v_sum = 0.0
                        for v in val_outputs:
                            if not torch.isnan(v):
                                v_sum += v
                                length += 1

                        if length == 0:
                            v_sum = 0
                        else :
                            v_sum = v_sum / length 
                        self.validation_end(mean_val_outputs=v_sum)
                    
                    else :
                        num_val = len(val_outputs[0])
                        length = [0.0 for i in range(num_val)]
                        v_sum = [0.0 for i in range(num_val)]

                        for v in val_outputs:
                            for i in range(num_val):
                                if not torch.isnan(v[i]):
                                    v_sum[i] += v[i]
                                    length[i] += 1

                        for i in range(num_val):
                            if length[i] == 0:
                                v_sum[i] = 0
                            else :
                                v_sum[i] = v_sum[i] / length[i]

                        self.validation_end(mean_val_outputs=v_sum)

            if self.scheduler is not None:
                self.scheduler.step()
            if self.model is not None:
                self.model.train()


    def train_epoch(self, 
                    loader,
                    epoch,
                    ):
        if self.model is not None:
            self.model.train()
        if self.local_rank == 0:
            with tqdm(total=len(loader)) as t:

                for idx, batch in enumerate(loader):
                    self.global_step += 1
                    t.set_description('Epoch %i' % epoch)
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].contiguous().to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else :
                        print("unsupported data type")
                        exit(0)
                    
                    if self.model is not None:
                        for param in self.model.parameters(): param.grad = None
                    loss = self.training_step(batch)

                    if self.auto_optim and hasattr(self, 'optimizer') and self.optimizer is not None:
                        loss.backward()
                        self.optimizer.step()
                        # step-level scheduler per batch
                        if self.scheduler is not None:
                            self.scheduler.step()
                            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            t.set_postfix(loss=loss.item(), lr=lr)
                    t.update(1)
        else :
            for idx, batch in enumerate(loader):
                self.global_step += 1
                if isinstance(batch, dict):
                        batch = {
                            x: batch[x].contiguous().to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                elif isinstance(batch, list) :
                    batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                else :
                    print("unsupported data type")
                    exit(0)

                for param in self.model.parameters(): param.grad = None

                loss = self.training_step(batch)
                if self.auto_optim and hasattr(self, 'optimizer') and self.optimizer is not None:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

            for param in self.model.parameters() : param.grad = None

    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs):
        pass 


    def log(self, k, v, step):
        if self.env_type == "pytorch":
            self.writer.add_scalar(k, scalar_value=v, global_step=step)

        else :
            if self.local_rank == 0:
                self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 

        self.model.load_state_dict(new_sd, strict=strict)
        
        print(f"model parameters are loaded successfully.")
