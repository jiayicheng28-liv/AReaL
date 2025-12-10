import os
import sys
from copy import deepcopy

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.core.dist_rollout import redistribute
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    concat_padded_tensors,
    get_batch_size,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow
#LORA data paralellism:
""""
On every rank:

A frozen base model (same weights, never updated)

A set of LoRA adapter weights (the only trainable parameters)

An FSDP wrapper around the trainable module(s) so we can shard params/grad/optimizer state

A different slice of data (different trajectories / samples for PPO/GRPO)

Each rank stores only 1/N of that parameter (plus its optimizer state).

When you run a forward pass on a rank:

FSDP all-gathers the shards so this rank temporarily has the full LoRA weights it needs.

During backward:

Gradients are computed on that rank,

Then reduced/sharded back across ranks (so each rank ends up with its own shard of the global gradient).
"""

def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])

#each branch get broadcast of part of the new rollout from rank0
#rank0 is data-parallel head
def bcast_and_split_from_rank0(batch: dict | None, granularity: int) -> dict:
    batch = broadcast_tensor_container(batch, src_rank=0)
    bs = get_batch_size(batch)
    assert bs % dist.get_world_size() == 0
    bs_per_rank = bs // dist.get_world_size()
    local_batch = []
    for i in range(dist.get_rank() * bs_per_rank, (dist.get_rank() + 1) * bs_per_rank):
        local_batch.append({k: v[i : i + 1] for k, v in batch.items()})
    local_batch = concat_padded_tensors(local_batch)
    # Make the sequences on each rank more balanced.
    return redistribute(local_batch, granularity=granularity).data


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    """
    FSDP strengths (specific to LoRA):

It shards trainable LoRA weights only
→ huge memory savings
→ no need to shard frozen weights

It keeps frozen base model on each GPU fully
→ fast autoregressive rollout
→ no TP/PP communication bottlenecks

It shards optimizer state and gradients for LoRA only
→ tiny communication cost

It can all-gather LoRA weights only at layer entry
→ LoRA modules are tiny, so all-gather is cheap

It works perfectly with data parallel RL

each rank does rollout independently

gradients sync only for LoRA

parameter update stays global
    """
    assert parallel_strategy is not None
    if parallel_strategy.data_parallel_size != parallel_strategy.world_size:
        raise ValueError("LoRA does not support parallelism other than FSDP.")

    # Initialize train engine
    """each rank create its own actor but fsdp link all actor into one conceptually"""
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    # NOTE: special design for lora, only rank 0 submits rollout
    train_dataloader = create_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    """
    Conceptually, train_data_parallel_size tells the remote rollout client:

“How many independent training data-parallel groups exist that will share this rollout server?”
    """
    rollout.initialize(train_data_parallel_size=1)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()
    """✅ Yes: both rollout and eval_rollout talk to the same rollout server job.

✅ Yes: with the launcher setup you described, you have one llm_server process that uses 4 GPUs (can be set during launching)"""

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.saver.experiment_name,
        config.saver.trial_name,
        config.saver.fileroot,
        use_lora=True,
    )

    actor.initialize(None, ft_spec)
    """actor.connect_engine(rollout, weight_update_meta) hooks up:

The training-side FSDP actor with the rollout engine so that when you call:

actor.update_weights(weight_update_meta)

the remote SGLang server knows which LoRA files to reload."""
    actor.connect_engine(rollout, weight_update_meta)
    #kl divergence control
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    #evaluate after a number of steps to log the progress of the training
    evaluator = Evaluator(config.evaluator, ft_spec)
    #the object responsible for resuming training from a checkpoint. not used yet
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = None
            # NOTE: Currently, if we use multiple ranks for LoRA rollout,
            # the algorithm performance will drop significantly. This may be
            # due to some concurrency issues. Use a single rank for rollout
            # as a temporary workaround.
            """
            Rank 0 → rollout.prepare_batch(...)

Pulls some prompts from train_dataloader.

Sends them plus RLVRWorkflow to the remote SGLang rollout server.

SGLang generates “thinking” + answers, logs, etc.

Server returns a TensorContainer of trajectories.

Rank 0 moves them to GPU.

Broadcast to all ranks

bcast_and_split_from_rank0(...) sends the batch from rank 0 to other ranks.

Each rank gets its slice according to config.actor.group_size."""
            if dist.get_rank() == 0:
                batch = rollout.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    should_accept_fn=lambda sample: True,
                )
                batch = tensor_container_to(batch, actor.device)
            batch = bcast_and_split_from_rank0(
                batch, granularity=config.actor.group_size
            )

        # Create barrier to synchronize all rollout processes.
        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                # Stats are logged in workflow
                # and will be exported later
                cnt = 0
                if dist.get_rank() == 0:
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Upload statistics to the logger (e.g., wandb)
        stats = actor.export_stats()
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
