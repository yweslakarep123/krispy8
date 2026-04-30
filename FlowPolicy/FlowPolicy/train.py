if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os

# Force IPv4 for DNS resolution so Minari dataset downloads don't hang on
# unreachable IPv6 CloudFront endpoints. No-op when FLOWP_FORCE_IPV4=0.
def _force_ipv4():
    import socket
    if os.environ.get("FLOWP_FORCE_IPV4", "1") == "0":
        return
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, *args, **kwargs):
        results = _orig_getaddrinfo(host, *args, **kwargs)
        ipv4 = [r for r in results if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else results

    socket.getaddrinfo = _ipv4_getaddrinfo


_force_ipv4()
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from flow_policy_3d.policy.flowpolicy_lowdim import FlowPolicyLowdim as FlowPolicy
from flow_policy_3d.dataset.base_dataset import BaseDataset
from flow_policy_3d.env_runner.base_runner import BaseRunner
from flow_policy_3d.common.checkpoint_util import TopKCheckpointManager
from flow_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from flow_policy_3d.model.flow.ema_model import EMAModel
from flow_policy_3d.model.common.lr_scheduler import get_scheduler
import warnings
warnings.filterwarnings("ignore")

OmegaConf.register_new_resolver("eval", eval, replace=True)

# #region agent log
def _agent_debug_log(run_id, hypothesis_id, location, message, data):
    import json
    payload = {
        "sessionId": "04e3ae",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/home/daffa/Documents/krispy8/.cursor/debug-04e3ae.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion

class TrainFlowPolicyWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: FlowPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FlowPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0



    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # #region agent log
                _agent_debug_log(
                    run_id="step-debug",
                    hypothesis_id="S1_S2_S3_S4",
                    location="train.py:resume-checkpoint:after-load",
                    message="loaded checkpoint counters",
                    data={
                        "checkpoint_path": str(lastest_ckpt_path),
                        "global_step_after_load": int(self.global_step),
                        "epoch_after_load": int(self.epoch),
                    },
                )
                # #endregion

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        # #region agent log
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1_H2_H3_H4_H5",
            location="train.py:wandb.init:before",
            message="before wandb.init",
            data={
                "output_dir": str(self.output_dir),
                "logging_group": str(cfg.logging.group),
                "logging_name": str(cfg.logging.name),
                "logging_id": str(getattr(cfg.logging, "id", None)),
                "logging_resume": str(getattr(cfg.logging, "resume", None)),
                "hydra_output_dir": str(HydraConfig.get().runtime.output_dir),
            },
        )
        # #endregion
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        # #region agent log
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1_H2_H3_H4_H5",
            location="train.py:wandb.init:after",
            message="after wandb.init",
            data={
                "run_id": str(getattr(wandb_run, "id", None)),
                "run_name": str(getattr(wandb_run, "name", None)),
                "run_resumed": bool(getattr(wandb_run, "resumed", False)),
                "run_dir": str(getattr(wandb_run, "dir", None)),
            },
        )
        # #endregion
        try:
            existing_output_dir = wandb.config.get("output_dir", None)
            new_output_dir = str(self.output_dir)
            should_update_output_dir = (
                existing_output_dir is None or str(existing_output_dir) == new_output_dir
            )
            # #region agent log
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="H1_H2_H3_H4_H5",
                location="train.py:wandb.config.update:decision",
                message="decide whether to update output_dir",
                data={
                    "existing_output_dir": str(existing_output_dir),
                    "new_output_dir": new_output_dir,
                    "run_resumed": bool(getattr(wandb_run, "resumed", False)),
                    "should_update_output_dir": should_update_output_dir,
                },
            )
            # #endregion
            if should_update_output_dir:
                wandb.config.update(
                    {
                        "output_dir": new_output_dir,
                    }
                )
                # #region agent log
                _agent_debug_log(
                    run_id="post-fix",
                    hypothesis_id="H1_H2_H3_H4_H5",
                    location="train.py:wandb.config.update:applied",
                    message="output_dir update applied",
                    data={"output_dir": new_output_dir},
                )
                # #endregion
            else:
                # #region agent log
                _agent_debug_log(
                    run_id="post-fix",
                    hypothesis_id="H1_H2_H3_H4_H5",
                    location="train.py:wandb.config.update:skipped",
                    message="output_dir update skipped to avoid config mutation on resumed run",
                    data={
                        "existing_output_dir": str(existing_output_dir),
                        "new_output_dir": new_output_dir,
                    },
                )
                # #endregion
        except Exception as exc:
            # #region agent log
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="H1_H2_H3_H4_H5",
                location="train.py:wandb.config.update:exception",
                message="wandb.config.update failed",
                data={
                    "exc_type": type(exc).__name__,
                    "exc_message": str(exc),
                    "existing_output_dir": str(wandb.config.get("output_dir", None)),
                    "new_output_dir": str(self.output_dir),
                },
            )
            # #endregion
            raise
        # #region agent log
        try:
            wandb_summary_step = wandb_run.summary.get("_step", None)
        except Exception:
            wandb_summary_step = None
        _agent_debug_log(
            run_id="step-debug",
            hypothesis_id="S1_S2_S3_S4",
            location="train.py:wandb-step:after-init",
            message="wandb step state after init",
            data={
                "local_global_step": int(self.global_step),
                "wandb_run_step_attr": str(getattr(wandb_run, "step", None)),
                "wandb_summary_step": str(wandb_summary_step),
                "run_resumed": bool(getattr(wandb_run, "resumed", False)),
            },
        )
        # #endregion
        # #region agent log
        wandb_step_after_init = getattr(wandb_run, "step", None)
        if wandb_step_after_init is not None:
            try:
                wandb_step_after_init = int(wandb_step_after_init)
            except Exception:
                wandb_step_after_init = None
        if (
            bool(getattr(wandb_run, "resumed", False))
            and wandb_step_after_init is not None
            and int(self.global_step) < wandb_step_after_init
        ):
            prev_global_step = int(self.global_step)
            self.global_step = wandb_step_after_init
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="S1_S2_S3_S4",
                location="train.py:wandb-step:sync",
                message="synced local global_step to resumed wandb step",
                data={
                    "prev_global_step": prev_global_step,
                    "synced_global_step": int(self.global_step),
                    "wandb_step_after_init": wandb_step_after_init,
                },
            )
        else:
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="S1_S2_S3_S4",
                location="train.py:wandb-step:no-sync",
                message="no global_step sync needed",
                data={
                    "local_global_step": int(self.global_step),
                    "wandb_step_after_init": str(wandb_step_after_init),
                    "run_resumed": bool(getattr(wandb_run, "resumed", False)),
                },
            )
        # #endregion

        # When resuming a wandb run whose server-side step is already ahead
        # of our checkpointed `self.global_step` (e.g. the previous process
        # logged a few more steps before being killed / OOM / Colab timeout
        # / crashed after an intermediate save), every subsequent
        # `wandb_run.log(..., step=self.global_step)` fires a
        # "Tried to log to step X that is less than the current step Y"
        # warning and silently drops the data. Align our counter to wandb's
        # so logging resumes monotonically.
        try:
            wandb_step = int(getattr(wandb_run, "step", 0) or 0)
        except Exception:
            wandb_step = 0
        if getattr(wandb_run, "resumed", False) and wandb_step >= self.global_step:
            cprint(
                f"[WandB] resumed run at server step {wandb_step}; advancing "
                f"local global_step {self.global_step} -> {wandb_step + 1} "
                f"to keep log steps monotonically increasing.",
                "yellow")
            self.global_step = wandb_step + 1

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        # #region agent log
                        if batch_idx == 0 and local_epoch_idx == 0:
                            _agent_debug_log(
                                run_id="step-debug",
                                hypothesis_id="S1_S2_S3_S4",
                                location="train.py:wandb.log:first-batch",
                                message="first batch log step values",
                                data={
                                    "local_global_step_before_log": int(self.global_step),
                                    "wandb_run_step_attr_before_log": str(getattr(wandb_run, "step", None)),
                                },
                            )
                        # #endregion
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)

            
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)

        # Prioritas sumber checkpoint (fail-fast, no silent skip):
        #   1) cfg.eval.checkpoint_path (override eksplisit dari CLI Hydra)
        #   2) self.output_dir/checkpoints/latest.ckpt
        explicit_ckpt = None
        try:
            explicit_ckpt = cfg.get("eval", {}).get("checkpoint_path", None)
        except Exception:
            explicit_ckpt = None

        if explicit_ckpt:
            lastest_ckpt_path = pathlib.Path(str(explicit_ckpt)).expanduser()
        else:
            lastest_ckpt_path = self.get_checkpoint_path(tag="latest")

        if not lastest_ckpt_path.is_file():
            raise FileNotFoundError(
                "Tidak menemukan checkpoint untuk eval.\n"
                f"  Dicari di: {lastest_ckpt_path}\n"
                "Pastikan training sudah menyimpan checkpoint "
                "(`checkpoint.save_ckpt=True`) lalu salah satu:\n"
                "  - arahkan eval ke run_dir training yang benar "
                "(`hydra.run.dir=<train_run_dir>`), atau\n"
                "  - beri path ckpt langsung lewat "
                "`+eval.checkpoint_path=/path/ke/latest.ckpt`.")

        cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
        self.load_checkpoint(path=lastest_ckpt_path)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        dev = torch.device(cfg.training.device)
        policy.to(dev)

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'flow_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainFlowPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
