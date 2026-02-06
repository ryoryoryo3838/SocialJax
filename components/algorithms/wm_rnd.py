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
        
        rnd_dim = 64
        rnd_target = RNDNetwork(encoder_cfg, output_dim=rnd_dim)
        rnd_predictor = RNDNetwork(encoder_cfg, output_dim=rnd_dim)
        rng, rt_rng, rp_rng = jax.random.split(rng, 3)
        target_params = jax.lax.stop_gradient(rnd_target.init(rt_rng, init_x))
        pred_params = rnd_predictor.init(rp_rng, init_x)
        pred_tx = optax.adam(float(config.get("RND_LR", 1e-4)))
        pred_opt_state = pred_tx.init(pred_params)

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
            
            # Intrinsic Reward (Computed centrally for now using shared RND)
            # We calculate novelty per agent view
            next_obs_agents = jnp.swapaxes(obs, 0, 1) # [N, E, ...]
            t_feat = jax.vmap(rnd_target.apply, in_axes=(None, 0))(target_params, next_obs_agents)
            p_feat = jax.vmap(rnd_predictor.apply, in_axes=(None, 0))(pred_params, next_obs_agents)
            int_rew = jnp.mean(jnp.square(t_feat - p_feat), axis=-1) # [N, E]
            
            combined_rew = reward.T + intrinsic_coef * int_rew
            
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
            
            # 1. Update RND (Shared) using all agents' experiences
            T, N, E = obs_arr.shape[:3]
            flat_obs = obs_arr.reshape((-1, *obs_arr.shape[3:])) # [T*N*E, ...]
            def rnd_loss(p, o):
                return jnp.mean(jnp.square(jax.lax.stop_gradient(rnd_target.apply(target_params, o)) - rnd_predictor.apply(p, o)))
            
            rnd_grad_fn = jax.value_and_grad(rnd_loss)
            _, rnd_grads = rnd_grad_fn(pred_params, flat_obs)
            updates, pred_opt_state = pred_tx.update(rnd_grads, pred_opt_state, pred_params)
            pred_params = optax.apply_updates(pred_params, updates)

            # 2. Update World Model (Shared)
            # Train on Agent 0's perspective + Other's actions? Or all agents?
            # To be robust, let's train on all agents (treating them as independent samples of "State -> Action -> Next State").
            # But WM input is [S, Actions].
            # For "CleanUp", the state is local view? 
            # If IPPO, we only see local view.
            # We assume S_local -> Actions -> S_local_next.
            # We need joint actions for every agent? 
            # In CleanUp, we can see others?
            # Let's simplify: We assume "Others" are part of the environment dynamics.
            # BUT the user wants "Project Self".
            # So the WM MUST take "Joint Actions" to predict outcome.
            # Local View + Joint Actions -> Next Local View.
            # For data: We have Obs [N, E], Actions [N, E].
            # We need to constructing inputs [S_i, A_all].
            # "Joint actions" from perspective of agent i: [a_i, a_others...].
            # Since strict ordering doesn't exist in local view without ids, we just assume fixed ordering [0..N].
            
            # Inputs:
            # S: [T*E*N, ...] (Everyone's observations)
            # A: [T*E, N] -> broadcast to [T*E*N, N]? 
            # We need [T*E*N, N] where for each agent i, we provide the full joint action vector of that step.
            
            joint_actions = jnp.transpose(actions_arr, (0, 2, 1)) # [T, E, N]
            joint_actions_repeated = jnp.repeat(joint_actions, N, axis=1) # [T, E*N, N]
            joint_actions_flat = joint_actions_repeated.reshape((-1, num_agents)) # [T*E*N, N]
            
            obs_flat = obs_arr.reshape((-1, *obs_arr.shape[3:]))
            next_obs_flat = jnp.concatenate([obs_arr[1:], jnp.swapaxes(last_obs, 0, 1)[None, ...]], axis=0).reshape((-1, *obs_arr.shape[3:]))
            ext_rew_flat = ext_arr.reshape((-1, 1)) # We predict own reward? Or joint?
            # Let's predict own reward [T*E*N, 1]? 
            # WM output for reward is [N] usually. 
            # Let's just predict [1] (Self Reward) for shared model simplicity, or [N] if we want full prediction.
            # The WM definition returns [B, N]. Let's say it predicts [Self, Others...] or just [N] fixed slots.
            # We'll train it to predict the rewards of the N agents given the state (which might be just i's view).
            
            # Target Reward: [T, E, N] -> [T*E*N, N] ?
            # Not easy to align "Agent i's view" with "Agent j's reward" without global knowledge.
            # Simplified WM: Predict ONLY my own reward and my own next latent. 
            # We rely on "Symmetry" in the *rollout policy*, not necessarily in the WM predicting everyone's state.
            # But we need everyone's reward to optimize "Cooperation".
            # "If I do A and they do A, we ALL get Reward R".
            # So WM should predict R_total or R_self? 
            # User: "prevent free riding". I need to see that cleaning helps ME.
            # But only if others also gather?
            # Let's predict [N] rewards from the perspective of the agent.
            
            # Align rewards: For obs i (agent i), the target rewards are the rewards of [0..N] at that step.
            ext_rew_target = jnp.transpose(ext_arr, (0, 2, 1)).reshape((-1, num_agents)) # [T*E, N] repeated?
            # No, for Obs of Agent i, the rewards are [r_0, r_1... r_N]
            # We repeat the joint reward vector N times.
            ext_rew_target_repeated = jnp.repeat(jnp.transpose(ext_arr, (0, 2, 1)), N, axis=1).reshape((-1, num_agents))

            def wm_loss_fn(p, s, a, s_next, r_target):

                # Use module logic.
                pass 
                # This is tricky with Flax functional style.
                # We need to call apply with p["wm"].
                latent_pred, r_pred = wm.apply(p["wm"], s, a) 
                latent_target = jax.lax.stop_gradient(le.apply(p["le"], s_next))
                
                return jnp.mean(jnp.square(latent_target - latent_pred)) + jnp.mean(jnp.square(r_target - r_pred))

            wm_grad_fn = jax.value_and_grad(wm_loss_fn)
            _, wm_grads = wm_grad_fn(wm_le_params, obs_flat, joint_actions_flat, next_obs_flat, ext_rew_target_repeated)
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

            # GAE per agent
            adv_list, ret_list = [], []
            for i in range(num_agents):
                adv, ret = compute_gae(
                    rew_agents[i], val_agents[i], don_agents[i], last_val_agents[i], 
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
                                # Use shared WM params (fixed, stop_gradient)
                                wm_p = jax.lax.stop_gradient(wm_le_params["wm"])
                                next_latent, rewards = wm.apply(wm_p, s_emb, soft_actions, method=wm.dynamics) 
                                # rewards: [B, N]. We care about SELF reward (agent i).
                                # But in symmetry, we are interchangeable? 
                                # Let's strictly maximize Agent i's reward (index 0 if we mapped self to 0? 
                                # No, WM output corresponds to fixed agent slots 0..N.
                                # If I assume I am Agent k, I should maximize index k?
                                # But I am simulating "Everyone is Me".
                                # A symmetric policy should yield high rewards for EVERYONE.
                                # So maximizing mean(rewards) is a safe cooperative proxy.
                                
                                step_rew = jnp.mean(rewards) 
                                return (next_latent, next_key, cum_rew + step_rew), None

                            # Init Embedding
                            # Use World Model's encoder to get initial latent state
                            init_emb = wm.apply(wm_p, imag_s, method=wm.encode)
                            (final_emb, final_key, total_imag_rew), _ = jax.lax.scan(
                                imag_rollout, (init_emb, grad_rng, 0.0), None, length=imag_steps
                            )
                            
                            l_imag = -total_imag_rew
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
            jax.debug.callback(_log_callback, metrics)
            return (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (params, opt_state, pred_params, pred_opt_state, wm_le_params, wm_le_opt_state, env_state, obs, rng, 0)
        final_carry, _ = jax.jit(lambda c: jax.lax.scan(_update_step, c, None, length=num_updates))(init_carry)
        return final_carry[0]

    return train
