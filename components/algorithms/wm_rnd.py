"""Model-Based Symmetry and Novelty Exploration (WM-RND) implementation."""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import socialjax
from socialjax.wrappers.baselines import LogWrapper

from components.algorithms.networks import ActorCritic, build_encoder_config
from components.algorithms.utils import done_dict_to_array
from components.algorithms.wm_rnd_networks import RNDNetwork, WorldModel, LatentEncoder
from components.training.checkpoint import save_checkpoint
from components.training.logging import init_wandb, log_metrics
from components.training.ppo import PPOBatch, compute_gae, update_ppo
from components.training.utils import flatten_obs, unflatten_actions, to_actor_rewards, to_actor_dones


def make_train(config: Dict):
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_agents = env.num_agents
    
    num_actors = num_envs * num_agents
    num_updates = int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    minibatch_size = num_actors * num_steps // int(config["NUM_MINIBATCHES"])

    encoder_cfg = build_encoder_config(config)
    
    # Hyperparameters
    intrinsic_coef = float(config.get("INTRINSIC_COEF", 0.1))
    imag_coef = float(config.get("IMAGINATION_COEF", 0.01))

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None
        ckpt_dir = config.get("CHECKPOINT_DIR")
        ckpt_every = int(config.get("CHECKPOINT_EVERY", 0))
        ckpt_keep = int(config.get("CHECKPOINT_KEEP", 3))

        # 1. Shared Policy & Value
        network = ActorCritic(env.action_space().n, encoder_cfg)
        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        params = network.init(init_rng, init_x)
        
        tx = optax.chain(
            optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
            optax.adam(learning_rate=float(config["LR"]), eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # 2. RND
        rnd_dim = 64
        rnd_target = RNDNetwork(encoder_cfg, output_dim=rnd_dim)
        rnd_predictor = RNDNetwork(encoder_cfg, output_dim=rnd_dim)
        rng, rt_rng, rp_rng = jax.random.split(rng, 3)
        target_params = jax.lax.stop_gradient(rnd_target.init(rt_rng, init_x))
        pred_state = TrainState.create(
            apply_fn=rnd_predictor.apply, 
            params=rnd_predictor.init(rp_rng, init_x),
            tx=optax.adam(float(config.get("RND_LR", 1e-4)))
        )

        # 3. World Model
        wm = WorldModel(encoder_cfg, num_agents=num_agents, action_dim=env.action_space().n)
        le = LatentEncoder(encoder_cfg)
        rng, wm_rng, le_rng = jax.random.split(rng, 3)
        init_actions = jnp.zeros((1, num_agents), dtype=jnp.int32)
        wm_params = wm.init(wm_rng, init_x, init_actions)
        le_params = le.init(le_rng, init_x)
        
        wm_le_tx = optax.adam(float(config.get("WM_LR", 1e-4)))
        wm_le_state = TrainState.create(
            apply_fn=None, # Not used directly
            params={"wm": wm_params, "le": le_params},
            tx=wm_le_tx
        )

        # Initial Env State
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        def _log_callback(metrics):
            log_metrics(metrics, wandb if log_enabled else None)

        def _save_callback(step, params, do_save):
            if not ckpt_dir or ckpt_every <= 0 or not do_save:
                return
            save_checkpoint(ckpt_dir, int(step), {"params": params}, keep=ckpt_keep)

        def _env_step(carry, _):
            train_state, pred_state, wm_le_state, env_state, last_obs, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            
            obs_batch = flatten_obs(last_obs) # [E*N, ...]
            pi, value = network.apply(train_state.params, obs_batch)
            action = pi.sample(seed=action_rng)
            logp = pi.log_prob(action)
            env_actions = unflatten_actions(action, num_envs, num_agents)

            # Env Step
            step_keys = jax.random.split(step_rng, num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, env_state, env_actions)
            
            # Intrinsic Reward
            next_obs_flat = flatten_obs(next_obs)
            t_feat = rnd_target.apply(target_params, next_obs_flat)
            p_feat = rnd_predictor.apply(pred_state.params, next_obs_flat)
            int_rew = jnp.mean(jnp.square(t_feat - p_feat), axis=-1)
            
            ext_rew = to_actor_rewards(reward) # [E*N]
            combined_rew = ext_rew + intrinsic_coef * int_rew
            
            transition = (
                obs_batch,
                action,
                logp,
                value,
                combined_rew,
                ext_rew,
                to_actor_dones(done_dict_to_array(done, env.agents)),
                int_rew,
            )
            return (train_state, pred_state, wm_le_state, env_state, next_obs, rng), transition

        def _update_step(carry, _):
            train_state, pred_state, wm_le_state, env_state, last_obs, rng, update_step = carry
            
            base_obs = last_obs
            (train_state, pred_state, wm_le_state, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step, (train_state, pred_state, wm_le_state, env_state, base_obs, rng), None, length=num_steps
            )
            
            obs_arr, actions_arr, logp_arr, values_arr, rewards_arr, ext_arr, dones_arr, int_arr = traj
            
            # 1. Update RND
            def rnd_loss(p, o):
                return jnp.mean(jnp.square(jax.lax.stop_gradient(rnd_target.apply(target_params, o)) - rnd_predictor.apply(p, o)))
            
            rnd_grad_fn = jax.value_and_grad(rnd_loss)
            _, rnd_grads = rnd_grad_fn(pred_state.params, obs_arr.reshape((-1, *obs_arr.shape[2:])))
            pred_state = pred_state.apply_gradients(grads=rnd_grads)

            # 2. Update World Model
            def wm_loss_fn(p, o, a, next_o, r):
                # o: [T, E, N, ...] -> [T*E, ...] (agent 0's obs as world state proxy)
                s_t = o[:, :, 0].reshape((-1, *o.shape[3:]))
                s_next = next_o[:, :, 0].reshape((-1, *o.shape[3:]))
                a_joint = a.reshape((num_steps, num_envs, num_agents)).reshape((-1, num_agents))
                r_joint = r.reshape((num_steps, num_envs, num_agents)).reshape((-1, num_agents))
                
                next_latent_pred, r_pred = wm.apply(p["wm"], s_t, a_joint)
                latent_target = jax.lax.stop_gradient(le.apply(p["le"], s_next))
                
                l_wm = jnp.mean(jnp.square(latent_target - next_latent_pred))
                l_rew = jnp.mean(jnp.square(r_joint - r_pred))
                return l_wm + l_rew

            next_obs_rollout = jnp.concatenate([obs_arr[1:].reshape((num_steps-1, num_envs, num_agents, *obs_arr.shape[2:])), 
                                                last_obs[None, ...]], axis=0)
            
            obs_unflat = obs_arr.reshape((num_steps, num_envs, num_agents, *obs_arr.shape[2:]))
            wm_grad_fn = jax.value_and_grad(wm_loss_fn)
            _, wm_grads = wm_grad_fn(wm_le_state.params, obs_unflat, actions_arr, next_obs_rollout, ext_arr)
            wm_le_state = wm_le_state.apply_gradients(grads=wm_grads)

            # 3. Policy Update (Standard PPO)
            last_value = network.apply(train_state.params, flatten_obs(last_obs))[1]
            advantages, returns = compute_gae(rewards_arr, values_arr, dones_arr, last_value, float(config["GAMMA"]), float(config["GAE_LAMBDA"]))
            
            batch = PPOBatch(
                obs=obs_arr.reshape((-1, *obs_arr.shape[2:])),
                actions=actions_arr.reshape((-1,)),
                old_log_probs=logp_arr.reshape((-1,)),
                advantages=advantages.reshape((-1,)),
                returns=returns.reshape((-1,)),
            )

            def _update_epoch(carry, _):
                ts, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, batch.actions.shape[0])
                
                def _minibatch(state, idx):
                    mb_idx = jax.lax.dynamic_slice(perm, (idx * minibatch_size,), (minibatch_size,))
                    mbatch = PPOBatch(
                        obs=batch.obs[mb_idx],
                        actions=batch.actions[mb_idx],
                        old_log_probs=batch.old_log_probs[mb_idx],
                        advantages=batch.advantages[mb_idx],
                        returns=batch.returns[mb_idx],
                    )
                    
                    # Combined Loss: PPO + Imagination
                    def combined_loss(params):
                        # PPO Loss
                        dist, value = network.apply(params, mbatch.obs)
                        log_probs = dist.log_prob(mbatch.actions)
                        entropy = dist.entropy().mean()
                        ratios = jnp.exp(log_probs - mbatch.old_log_probs)
                        surr1 = ratios * mbatch.advantages
                        surr2 = jnp.clip(ratios, 1.0 - float(config["CLIP_EPS"]), 1.0 + float(config["CLIP_EPS"])) * mbatch.advantages
                        ppo_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                        val_loss = jnp.mean(jnp.square(mbatch.returns - value))
                        
                        # Imagination Loss (Symmetry Assumption)
                        # Pick a small sample for imagination to save memory
                        imag_s = mbatch.obs[:num_envs] # [E, ...]
                        
                        # Softmax Policy for differentiable joint actions
                        logits = network.apply(params, imag_s)[0].logits
                        probs = jax.nn.softmax(logits, axis=-1) # [E, D]
                        
                        # Simple 1-step imagination: \sum_a pi(a|s) * R_pred(s, a_joint)
                        # For symmetry, we assume all agents use same probs
                        # This is a bit complex for joint actions, let's simplify to 
                        # U = \sum_a pi(a|s) * \sum_{a_others} P(a_others) * R(s, a, a_others)
                        # We use the expected reward under the policy for all agents.
                        
                        # Vectorized expected reward:
                        # Since rewards are independent but joint-action dependent,
                        # we can sample other actions or use the mean.
                        mean_a_joint = jax.nn.one_hot(jnp.argmax(probs, axis=-1), env.action_space().n) # Hard for now
                        # To be differentiable, we'd need Gumbel-Softmax or similar.
                        # Let's use the expected reward as a simple surrogate.
                        
                        # imag_r predicted for ALL agents: [E, N]
                        # We only have one s for the world model.
                        # Actually, let's skip complex imagination in the first test run to ensure it compiles.
                        l_imag = 0.0
                        
                        return ppo_loss + 0.5 * val_loss - float(config["ENT_COEF"]) * entropy + imag_coef * l_imag

                    grad_fn = jax.value_and_grad(combined_loss)
                    _, grads = grad_fn(state.params)
                    return state.apply_gradients(grads=grads), None
                
                ts, _ = jax.lax.scan(_minibatch, ts, jnp.arange(int(config["NUM_MINIBATCHES"])))
                return (ts, rng), None

            (train_state, rng), _ = jax.lax.scan(_update_epoch, (train_state, rng), None, length=int(config["UPDATE_EPOCHS"]))

            metrics = {
                "train/reward": jnp.mean(ext_arr),
                "train/intrinsic": jnp.mean(int_arr),
                "update_step": update_step + 1,
            }
            jax.debug.callback(_log_callback, metrics)
            return (train_state, pred_state, wm_le_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (train_state, pred_state, wm_le_state, env_state, obs, rng, 0)
        final_carry, _ = jax.jit(lambda c: jax.lax.scan(_update_step, c, None, length=num_updates))(init_carry)
        return final_carry[0]

    return train
