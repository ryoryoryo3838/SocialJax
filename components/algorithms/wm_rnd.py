"""Model-Based Symmetry and Novelty Exploration (WM-RND) implementation."""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from socialjax.wrappers.baselines import LogWrapper
import socialjax

from components.algorithms.networks import ActorCritic, build_encoder_config, _build_encoder
from components.algorithms.utils import (
    done_dict_to_array,
    load_agent_init_params,
    broadcast_agent_leaves,
)
from components.algorithms.wm_rnd_networks import RNDNetwork, WorldModel, LatentEncoder
from components.training.checkpoint import save_agent_checkpoints
from components.training.logging import init_wandb, log_metrics
from components.training.ppo import PPOBatch, compute_gae, update_ppo_params
from components.training.utils import flatten_obs, unflatten_actions, to_actor_rewards, to_actor_dones


def make_train(config: Dict):
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_agents = env.num_agents
    
    # Independent policies: distinct parameters per agent
    num_updates = int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    minibatch_size = num_envs * num_steps // int(config["NUM_MINIBATCHES"]) 

    encoder_cfg = build_encoder_config(config)
    
    # Hyperparameters
    intrinsic_coef = float(config.get("INTRINSIC_COEF", 0.1))
    imag_steps = int(config.get("IMAGINATION_STEPS", 32))
    imag_coef = float(config.get("IMAGINATION_COEF", 0.1))
    
    # New Hyperparameters
    rnd_shared = config.get("RND_SHARED", True)
    wm_joint_action = config.get("WM_JOINT_ACTION", True)
    symmetry_ratio = float(config.get("SYMMETRY_RATIO", 1.0))

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None
        ckpt_dir = config.get("CHECKPOINT_DIR")
        ckpt_every = int(config.get("CHECKPOINT_EVERY", 0))
        ckpt_keep = int(config.get("CHECKPOINT_KEEP", 3))

        # Move Env State init here to use real obs for initialization
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        
        # 1. Independent Policies
        network = ActorCritic(env.action_space().n, encoder_cfg)
        rng, init_rng = jax.random.split(rng)
        # Use real observation for initialization to avoid shape mismatch
        init_x = obs[0, 0][None, ...] # [1, ...]
        base_params = network.init(init_rng, init_x)
        # Initialize N separate sets of parameters
        params = load_agent_init_params(config, num_agents, base_params)
        
        tx = optax.chain(
            optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
            optax.adam(learning_rate=float(config["LR"]), eps=1e-5),
        )
        opt_state = tx.init(params)
        opt_state = broadcast_agent_leaves(opt_state, num_agents)

        # 2. RND (Shared or Independent? Let's keep shared for simplicity of implementation/efficiency, or distinct per agent?
        # User implies symmetry in reasoning but independence in execution. RND is for exploration.
        # Let's make RND independent too for true independence, or shared if "Novelty" is universal.
        # Simplest consistent approach: Shared RND/WM for efficiency, or Independent?
        # User said "Assuming others use same algorithm".
        # Let's keep one global RND/WM for now to save memory, trained on all agents' data.
        # If we want independent agents, they should ideally have their own internal models.
        # But for computational feasibility in this prototype, a shared "Environment Model" is common.
        # Let's use SHARED RND/WM for now.
        
        # RND parameters from reference
        rnd_feature_dim = int(config.get("RND_FEATURE_DIM", 288))
        rnd_hidden_dim = int(config.get("RND_HIDDEN_DIM", 256))
        
        rnd_target = RNDNetwork(encoder_cfg, output_dim=rnd_feature_dim, hidden_dim=rnd_hidden_dim)
        rnd_predictor = RNDNetwork(encoder_cfg, output_dim=rnd_feature_dim, hidden_dim=rnd_hidden_dim)
        rng, rt_rng, rp_rng = jax.random.split(rng, 3)
        
        # Initialize RND (Shared or Independent)
        if rnd_shared:
            target_params = jax.lax.stop_gradient(rnd_target.init(rt_rng, init_x))
            base_pred_params = rnd_predictor.init(rp_rng, init_x)
            pred_params = base_pred_params
        else:
            # Independent RND per agent
            v_init_rt = jax.vmap(rnd_target.init, in_axes=(0, None))
            v_init_rp = jax.vmap(rnd_predictor.init, in_axes=(0, None))
            target_params = jax.lax.stop_gradient(v_init_rt(jax.random.split(rt_rng, num_agents), init_x))
            pred_params = v_init_rp(jax.random.split(rp_rng, num_agents), init_x)

        pred_tx = optax.adam(float(config.get("RND_LR", 1e-4)))
        pred_opt_state = pred_tx.init(pred_params)
        if not rnd_shared:
            pred_opt_state = broadcast_agent_leaves(pred_opt_state, num_agents)

        # 3. World Model (Shared)
        wm = WorldModel(encoder_cfg, num_agents=num_agents, action_dim=env.action_space().n)
        le = LatentEncoder(encoder_cfg)
        rng, wm_rng, le_rng = jax.random.split(rng, 3)
        init_actions = jnp.zeros((1, num_agents), dtype=jnp.int32)
        wm_params = wm.init(wm_rng, init_x, init_actions)
        le_params = le.init(le_rng, init_x)
        
        wm_le_tx = optax.adam(float(config.get("WM_LR", 1e-4)))
        wm_le_opt_state = wm_le_tx.init({"wm": wm_params, "le": le_params})
        wm_le_params = {"wm": wm_params, "le": le_params}

        # Env State (Moved up)

        def _log_callback(metrics):
            log_metrics(metrics, wandb if log_enabled else None)

        def _save_callback(step, params, do_save):
            if not ckpt_dir or ckpt_every <= 0 or not do_save:
                return
            params_list = [{"params": jax.tree_util.tree_map(lambda x, i=i: x[i], params)} for i in range(num_agents)]
            save_agent_checkpoints(ckpt_dir, int(step), params_list, keep=ckpt_keep)

        def _env_step(carry, _):
            params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            
            # Independent action selection (each agent uses its own params)
            obs_agents = jnp.swapaxes(last_obs, 0, 1) # [N, E, ...]
            # VMAP network apply over agents
            pi, value = jax.vmap(network.apply, in_axes=(0, 0))(params, obs_agents)
            
            agent_rngs = jax.random.split(action_rng, num_agents)
            action = jax.vmap(lambda dist, key: dist.sample(seed=key))(pi, agent_rngs)
            logp = pi.log_prob(action)
            env_actions = list(action) # [N, E] -> List of [E]

            # Env Step
            step_keys = jax.random.split(step_rng, num_envs)
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, env_state, env_actions)
            
            # Intrinsic Reward
            next_obs_agents = jnp.swapaxes(obs, 0, 1) # [N, E, ...]
            
            if rnd_shared:
                t_feat = jax.vmap(rnd_target.apply, in_axes=(None, 0))(target_params, next_obs_agents)
                p_feat = jax.vmap(rnd_predictor.apply, in_axes=(None, 0))(pred_params, next_obs_agents)
            else:
                # Apply each agent's RND to its own observation
                t_feat = jax.vmap(rnd_target.apply, in_axes=(0, 0))(target_params, next_obs_agents)
                p_feat = jax.vmap(rnd_predictor.apply, in_axes=(0, 0))(pred_params, next_obs_agents)
            
            # Intrinsic Reward (Raw prediction error)
            # We will normalize this later in the update loop (Max-Min normalization per batch)
            # Here we just compute the raw squared error mean.
            int_rew = jnp.mean(jnp.square(t_feat - p_feat), axis=-1) # [N]
            
            combined_rew = reward.T # Add intrinsic later after normalization
            
            transition = (
                obs_agents,
                action,
                logp,
                value,
                combined_rew,
                reward.T,
                to_actor_dones(done_dict_to_array(done, env.agents)),
                int_rew,
            )
            return (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, obs, rng), transition

        def _update_step(carry, _):
            params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng, update_step = carry
            
            # Rollout
            (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step,
                (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng),
                None, length=num_steps
            )
            
            obs_arr, actions_arr, logp_arr, values_arr, rewards_arr, ext_arr, dones_arr, int_arr = traj
            # shapes are [T, N, E, ...] generally
            
            # 1. Update RND
            T, N, E = obs_arr.shape[:3]
            
            if rnd_shared:
                flat_obs = obs_arr.reshape((-1, *obs_arr.shape[3:])) # [T*N*E, ...]
                def rnd_loss(p, o):
                    return jnp.mean(jnp.square(jax.lax.stop_gradient(rnd_target.apply(target_params, o)) - rnd_predictor.apply(p, o)))
                
                rnd_grad_fn = jax.value_and_grad(rnd_loss)
                _, rnd_grads = rnd_grad_fn(pred_params, flat_obs)
                updates, pred_opt_state = pred_tx.update(rnd_grads, pred_opt_state, pred_params)
                pred_params = optax.apply_updates(pred_params, updates)
            else:
                # Independent updates per agent
                # Shape: [N, T*E, ...]
                agent_obs = jnp.swapaxes(obs_arr, 0, 1).reshape((N, -1, *obs_arr.shape[3:]))
                
                def rnd_loss_indiv(p, o, t):
                    return jnp.mean(jnp.square(jax.lax.stop_gradient(rnd_target.apply(t, o)) - rnd_predictor.apply(p, o)))
                
                def update_rnd_agent(p, o, t, os):
                    grads = jax.grad(rnd_loss_indiv)(p, o, t)
                    updates, new_os = pred_tx.update(grads, os, p)
                    return optax.apply_updates(p, updates), new_os
                
                pred_params, pred_opt_state = jax.vmap(update_rnd_agent)(pred_params, agent_obs, target_params, pred_opt_state)

            # 2. Update World Model
            # joint_actions: [T, E, N]
            if wm_joint_action:
                joint_actions = jnp.transpose(actions_arr, (0, 2, 1)) # [T, E, N]
                wm_actions_input = jnp.repeat(joint_actions, N, axis=1).reshape((-1, num_agents)) # [T*E*N, N]
            else:
                # Use only own action
                # actions_arr: [T, N, E] -> swap -> [T, E, N] -> reshape to [T*E*N, 1]
                wm_actions_input = jnp.swapaxes(actions_arr, 1, 2).reshape((-1, 1)) # [T*E*N, 1]
            
            obs_flat = obs_arr.reshape((-1, *obs_arr.shape[3:]))
            next_obs_flat = jnp.concatenate([obs_arr[1:], jnp.swapaxes(last_obs, 0, 1)[None, ...]], axis=0).reshape((-1, *obs_arr.shape[3:]))
            ext_rew_target_repeated = jnp.repeat(jnp.transpose(ext_arr, (0, 2, 1)), N, axis=1).reshape((-1, num_agents))

            def wm_loss_fn(p, s, a, s_next, r_target):
                latent_pred, r_pred = wm.apply(p["wm"], s, a) 
                latent_target = jax.lax.stop_gradient(le.apply(p["le"], s_next))
                return jnp.mean(jnp.square(latent_target - latent_pred)) + jnp.mean(jnp.square(r_target - r_pred))

            wm_grad_fn = jax.value_and_grad(wm_loss_fn)
            _, wm_grads = wm_grad_fn(wm_le_params, obs_flat, wm_actions_input, next_obs_flat, ext_rew_target_repeated)
            wm_updates, wm_le_opt_state = wm_le_tx.update(wm_grads, wm_le_opt_state, wm_le_params)
            wm_le_params = optax.apply_updates(wm_le_params, wm_updates)

            # 3. Policy Update (Independent PPO + Imag)
            # We need to optimize independently but use the shared WM for imagination.

            # Prep PPO Batches per agent
            # obs_arr: [T, N, E, ...] -> swap N to front -> [N, T, E, ...]
            obs_agents = jnp.swapaxes(obs_arr, 1, 0) # [N, T, E, ...]
            act_agents = jnp.swapaxes(actions_arr, 1, 0)
            logp_agents = jnp.swapaxes(logp_arr, 1, 0)
            rew_agents = jnp.swapaxes(rewards_arr, 1, 0)
            val_agents = jnp.swapaxes(values_arr, 1, 0)
            don_agents = jnp.swapaxes(dones_arr, 1, 0)
            
            last_obs_agents = jnp.swapaxes(last_obs, 0, 1)
            _, last_val_agents = jax.vmap(network.apply, in_axes=(0, 0))(params, last_obs_agents) # [N, E]

            # 4. Normalize Intrinsic Rewards (Min-Max)
            # Reference: expl_r = (expl_r - expl_r.min()) / (expl_r.max() - expl_r.min() + 1e-11)
            
            # int_arr is [T, N].
            # We want to normalize over the time dimension (and possibly env dimension if we had it separately, but here T covers the batch)
            # Actually, the reference flattens everything. Here we process per-agent.
            
            def normalize_int_batch(i_arr):
                # i_arr: [T] for one agent
                # stop_gradient is important! The normalization statistics shouldn't backprop?
                # The reference uses .detach() on expl_r BEFORE normalization.
                i_v = jax.lax.stop_gradient(i_arr)
                min_val = jnp.min(i_v)
                max_val = jnp.max(i_v)
                return (i_v - min_val) / (max_val - min_val + 1e-11)
            
            # i_arr is [T, N]. Swap to [N, T].
            int_agents = jnp.swapaxes(int_arr, 1, 0) # [N, T]
            norm_int_agents = jax.vmap(normalize_int_batch)(int_agents) # [N, T]
            
            # Reconstruct Rewards
            ext_agents = jnp.swapaxes(ext_arr, 1, 0)
            # Final rewards for PPO: External + Normalized Intrinsic * Coef
            # intrinsic_coef here acts as rnd_k_expl
            combined_agents = ext_agents + intrinsic_coef * norm_int_agents

            # GAE per agent
            adv_list, ret_list = [], []
            for i in range(num_agents):
                adv, ret = compute_gae(
                    combined_agents[i], val_agents[i], don_agents[i], last_val_agents[i], 
                    float(config["GAMMA"]), float(config["GAE_LAMBDA"])
                )
                adv_list.append(adv)
                ret_list.append(ret)
            adv_agents = jnp.stack(adv_list)
            ret_agents = jnp.stack(ret_list)

            # Create Batches
            batch = PPOBatch(
                obs=obs_agents.reshape((num_agents, -1, *obs_agents.shape[3:])),
                actions=act_agents.reshape((num_agents, -1)),
                old_log_probs=logp_agents.reshape((num_agents, -1)),
                advantages=adv_agents.reshape((num_agents, -1)),
                returns=ret_agents.reshape((num_agents, -1)),
            )

            # Update Loop (vmap over agents)
            # Helper for indexing
            def _update_epoch(carry, _):
                p_batch, o_batch, rng = carry
                rng, perm_rng = jax.random.split(rng)
                # Independent permutations for each agent
                perms = jax.vmap(lambda k: jax.random.permutation(k, batch.actions.shape[1]))(jax.random.split(perm_rng, num_agents))
                
                def _minibatch(c, idx):
                    p_b, o_b = c
                    
                    def update_agent(agent_idx, p_i, o_i, perm_i):
                        mb_idx = jax.lax.dynamic_slice(perm_i, (idx * minibatch_size,), (minibatch_size,))
                        mbatch = PPOBatch(
                            obs=batch.obs[agent_idx][mb_idx],
                            actions=batch.actions[agent_idx][mb_idx],
                            old_log_probs=batch.old_log_probs[agent_idx][mb_idx],
                            advantages=batch.advantages[agent_idx][mb_idx],
                            returns=batch.returns[agent_idx][mb_idx],
                        )
                        
                        # -- IMAGINATION LOGIC START --
                        # Use a subset of real states to seed imagination
                        imag_s = mbatch.obs[:16] # [B_img, ...]
                        
                        def combined_loss(params_i, grad_rng):
                            # 1. PPO Loss
                            dist, value = network.apply(params_i, mbatch.obs)
                            log_probs = dist.log_prob(mbatch.actions)
                            entropy = dist.entropy().mean()
                            ratios = jnp.exp(log_probs - mbatch.old_log_probs)
                            surr1 = ratios * mbatch.advantages
                            surr2 = jnp.clip(ratios, 1.0 - float(config["CLIP_EPS"]), 1.0 + float(config["CLIP_EPS"])) * mbatch.advantages
                            ppo_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                            val_loss = jnp.mean(jnp.square(mbatch.returns - value))
                            
                            # 2. Imagination Loss (Symmetry Assumption)
                            # "params_i" is THIS agent's policy.
                            # We assume EVERYONE uses "params_i" in the imagination.

                            # Prepare WM params
                            wm_p = jax.lax.stop_gradient(wm_le_params["wm"])
                            
                            def imag_rollout(carry, _):
                                s_emb, key, cum_rew = carry
                                key, pi_key, noise_key, next_key = jax.random.split(key, 4)
                                
                                # Act (using Agent i's policy params_i)
                                pi, _ = network.apply(params_i, s_emb, method=network.act)
                                logits = pi.logits
                                
                                # Project Self: Replicate logits for N agents
                                logits_N = jnp.repeat(logits[:, None, :], num_agents, axis=1) # [B, N, D]
                                
                                # Gumbel Softmax (Joint)
                                noise = jax.random.uniform(noise_key, logits_N.shape)
                                gumbels = -jnp.log(-jnp.log(noise + 1e-10) + 1e-10)
                                temp = 1.0
                                soft_actions = jax.nn.softmax((logits_N + gumbels) / temp)

                                
                                # World Model Prediction
                                # If WM expects joint actions [B, N], we provide replication of self
                                if wm_joint_action:
                                    wm_actions = soft_actions # [B, N, D]
                                else:
                                    # WM expects single action [B, 1, D]
                                    wm_actions = soft_actions[:, 0:1, :] # Just the first one
                                
                                next_latent, rewards = wm.apply(wm_p, s_emb, wm_actions, method=wm.dynamics) 
                                # rewards: [B, N].
                                
                                # Symmetry Ratio Decay logic:
                                # Mix (Self Reward) and (Mean Reward)
                                # rewards[:, agent_idx] is self reward.
                                # But we are in a vmap over agent_idx.
                                r_self = rewards[:, agent_idx]
                                r_mean = jnp.mean(rewards, axis=1)
                                step_rew = (1.0 - symmetry_ratio) * r_self + symmetry_ratio * r_mean
                                
                                return (next_latent, next_key, cum_rew + step_rew), None

                            # Init Embedding
                            # Use World Model's encoder to get initial latent state
                            init_emb = wm.apply(wm_p, imag_s, method=wm.encode)
                            # cum_rew should track rewards for the batch of imaginary rollouts. 
                            # imag_s shape is [16, ...] (B_img=16). step_rew is [B_img].
                            # So initial carry must be [B_img] zeros, not scalar 0.0.
                            init_cum_rew = jnp.zeros(imag_s.shape[0])

                            (final_emb, final_key, total_imag_rew), _ = jax.lax.scan(
                                imag_rollout, (init_emb, grad_rng, init_cum_rew), None, length=imag_steps
                            )
                            
                            # total_imag_rew is [B_img], we mean it for loss
                            l_imag = -jnp.mean(total_imag_rew)
                            metrics = {
                                "train/ppo_loss": ppo_loss,
                                "train/value_loss": val_loss,
                                "train/entropy": entropy,
                                "train/imag_loss": l_imag,
                                "train/total_loss": ppo_loss + 0.5 * val_loss - float(config["ENT_COEF"]) * entropy + imag_coef * l_imag
                            }
                            return metrics["train/total_loss"], metrics

                        grad_fn = jax.value_and_grad(combined_loss, has_aux=True)
                        (_, metrics), grads = grad_fn(p_i, rng)
                        updates, new_opt_state = tx.update(grads, o_i, p_i)
                        new_params = optax.apply_updates(p_i, updates)
                        return new_params, new_opt_state, metrics

                    # VMAP Update
                    new_p, new_o, metrics = jax.vmap(update_agent)(
                        jnp.arange(num_agents), p_batch, o_batch, perms
                    )
                    return (new_p, new_o), metrics # Return metrics to scan
                
                (p_batch, o_batch), batch_metrics = jax.lax.scan(_minibatch, (p_batch, o_batch), jnp.arange(int(config["NUM_MINIBATCHES"])))
                # batch_metrics is [Minibatches, Agents, ...]
                # Average over minibatches and agents
                avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), batch_metrics)
                return (p_batch, o_batch, rng), avg_metrics

            (params, opt_state, rng), epoch_metrics = jax.lax.scan(_update_epoch, (params, opt_state, rng), None, length=int(config["UPDATE_EPOCHS"]))
            
            # Average over epochs
            avg_epoch_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), epoch_metrics)

            # Calculate total environment steps
            env_steps_total = (update_step + 1) * num_envs * num_steps

            metrics = {
                "train/reward": jnp.mean(ext_arr),
                "train/intrinsic": jnp.mean(int_arr),
                "update_step": update_step + 1,
                "env_step": env_steps_total,
                **avg_epoch_metrics,
            }
            do_save = (update_step + 1) % ckpt_every == 0 if (ckpt_dir and ckpt_every > 0) else False
            jax.debug.callback(_log_callback, metrics)
            jax.debug.callback(_save_callback, update_step + 1, params, do_save)
            return (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, obs, rng, 0)
        final_carry, _ = jax.jit(lambda c: jax.lax.scan(_update_step, c, None, length=num_updates))(init_carry)
        return final_carry[0]

    return train
