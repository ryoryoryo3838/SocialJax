""" 
Model-Based IPPO (MB-IPPO) Algorithm Component.
Based on algorithms/MB_IPPO/mb_ippo_cnn_cleanup.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import functools
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper, GymnaxWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import wandb
import copy
import pickle
import os
from pathlib import Path

class AutoResetWrapper(GymnaxWrapper):
    """Auto-reset the environment when done."""
    def __init__(self, env):
        super().__init__(env)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        return self._env.reset(key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action)
        
        key, key_reset = jax.random.split(key)
        obs_reset, state_reset = self._env.reset(key_reset)
        
        reset_condition = done["__all__"]
        
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(reset_condition, x, y), obs_reset, obs
        )
        state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(reset_condition, x, y), state_reset, state
        )
        
        return obs, state, reward, done, info

from components.models.world_model import WorldModel
from components.models.encoder import CNNEncoder
from components.models.decoder import ImageDecoder, RewardDecoder
from components.models.dynamics import LatentDynamics

# --- ActorCritic using Latent ---

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, z):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # z is already embedding from WorldModel
        
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(z)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(z)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Minibatch size for PPO
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # Model Training Params
    # Defaults in case not in config
    if "MODEL_LR" not in config: config["MODEL_LR"] = 1e-3
    if "DREAM_HORIZON" not in config: config["DREAM_HORIZON"] = 5
    if "DREAM_RATIO" not in config: config["DREAM_RATIO"] = 0.5
    if "MODEL_COEF" not in config: config["MODEL_COEF"] = 1.0


    env = AutoResetWrapper(env)
    env = LogWrapper(env, replace_info=False)

    # Initialize WandB
    wandb_cfg = config["WANDB"].copy()
    if "enabled" in wandb_cfg:
        del wandb_cfg["enabled"]
        
    wandb.init(
        config=config,
        **wandb_cfg
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT COMPONENTS
        obs_shape = env.observation_space()[0].shape
        action_dim = env.action_space().n
        
        # World Model
        world_model = WorldModel(action_dim=action_dim, input_shape=obs_shape, activation=config["ACTIVATION"])

        # Actor Critic
        actor_critic = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *obs_shape))
        init_z = jnp.zeros((1, 64)) # Assuming latent dim 64

        # Init World Model Params
        wm_params = world_model.init(_rng, init_x, jnp.array([0])) # init with dummy action 0

        # Init AC Params
        ac_params = actor_critic.init(_rng, init_z)
        
        # Optimizers
        wm_tx = optax.adam(config["MODEL_LR"])
        if config["ANNEAL_LR"]:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        wm_train_state = TrainState.create(
            apply_fn=world_model.apply,
            params=wm_params,
            tx=wm_tx,
        )
        
        ac_train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=ac_params,
            tx=ac_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            wm_state, ac_state, env_state, last_obs, update_step, rng = runner_state
            
            # --- DATA COLLECTION ---
            def _env_step(runner_state, unused):
                wm_state, ac_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Encode observation to get latent for policy
                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *obs_shape)
                    z = world_model.apply(wm_state.params, obs_batch, method=world_model.get_latent)
                    
                    pi, value = actor_critic.apply(ac_state.params, z)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                     # Not implementing independent parameter version for now for simplicity
                     # raise NotImplementedError("Independent parameters not supported yet in MB-IPPO")
                     # Fallback to simple handling or error? 
                     # For now, MB-IPPO assumes Parameter Sharing = True as per design
                     pass

                env_act_list = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act_list)

                # Store Transition
                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch, # Store raw obs for model training
                    info,
                )
                
                runner_state = (wm_state, ac_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            wm_state, ac_state, env_state, last_obs, update_step, rng = runner_state

            # --- CALCULATE GAE (Policy) ---
            # Need next value. Encode last_obs first.
            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *obs_shape)
            last_z = world_model.apply(wm_state.params, last_obs_batch, method=world_model.get_latent)
            _, last_val = actor_critic.apply(ac_state.params, last_z)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # PREPARE BATCHES
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
            ), "batch size must be equal to number of steps * number of actors"

            obs = traj_batch.obs
            next_obs = jnp.concatenate([obs[1:], last_obs_batch[None, ...]], axis=0)
            
            dones = traj_batch.done
            
            # Flatten everything
            def flatten(x):
                return x.reshape((batch_size,) + x.shape[2:])
            
            batch_obs = flatten(obs)
            batch_next_obs = flatten(next_obs)
            batch_action = flatten(traj_batch.action)
            batch_reward = flatten(traj_batch.reward)
            batch_done = flatten(dones)
            batch_adv = flatten(advantages)
            batch_target = flatten(targets)
            batch_log_prob = flatten(traj_batch.log_prob)
            
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, batch_size)
            
            shuffled_batch = {
                "obs": batch_obs,
                "next_obs": batch_next_obs,
                "action": batch_action,
                "reward": batch_reward,
                "done": batch_done,
                "adv": batch_adv,
                "target": batch_target,
                "log_prob": batch_log_prob
            }
            
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), shuffled_batch
            )
            
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            # --- UPDATE LOOP ---
            def _update_minibatch(carry, batch):
                wm_state, ac_state, rng = carry
                
                # 1. Update World Model
                def wm_loss_fn(params, batch):
                    z, rec_obs, z_next_pred, r_pred = world_model.apply(params, batch["obs"], batch["action"])
                    
                    z_next_target = world_model.apply(params, batch["next_obs"], method=world_model.get_latent)
                    z_next_target = jax.lax.stop_gradient(z_next_target)
                    
                    rec_loss = jnp.mean(jnp.square(rec_obs - batch["obs"])) # MSE
                    dyn_loss = jnp.mean(jnp.square(z_next_pred - z_next_target) * (1 - batch["done"])[:, None])
                    rew_loss = jnp.mean(jnp.square(r_pred - batch["reward"]))
                    
                    total_loss = rec_loss + dyn_loss + rew_loss
                    return total_loss, (rec_loss, dyn_loss, rew_loss)

                wm_grad_fn = jax.value_and_grad(wm_loss_fn, has_aux=True)
                (wm_loss_val, _), wm_grads = wm_grad_fn(wm_state.params, batch)
                wm_state = wm_state.apply_gradients(grads=wm_grads)
                
                # 2. Update Policy (ActorCritic)
                
                # Dream Rollout
                z_start = world_model.apply(wm_state.params, batch["obs"], method=world_model.get_latent)
                z_start = jax.lax.stop_gradient(z_start)
                
                def dream_step_gae(carry, _):
                    z, rng = carry
                    rng, _rng = jax.random.split(rng)
                    pi, val = actor_critic.apply(ac_state.params, z)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    z_next, r_pred = world_model.apply(wm_state.params, z, action, method=world_model.predict_next)
                    return (z_next, rng), (z, action, r_pred, val, log_prob, z_next)
                    
                rng, _rng = jax.random.split(rng)
                _, (d_z, d_a, d_r, d_v, d_lp, d_zn) = jax.lax.scan(dream_step_gae, (z_start, _rng), None, config["DREAM_HORIZON"])
                
                d_done = jnp.zeros_like(d_r)
                
                # GAE loop for dreams
                def get_dream_adv(gae_and_next_val, item):
                     gae, next_val = gae_and_next_val
                     r, v, d = item
                     delta = r + config["GAMMA"] * next_val * (1-d) - v
                     gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1-d) * gae
                     return (gae, v), (gae, gae+v) # adv, target
                
                (_, _), (d_adv, d_target) = jax.lax.scan(
                    get_dream_adv,
                    (jnp.zeros_like(d_r[0]), jnp.zeros_like(d_r[0])), 
                    (d_r, d_v, d_done),
                    reverse=True
                )
                
                dream_data = {
                    "z": d_z,
                    "action": d_a,
                    "target": d_target,
                    "adv": d_adv,
                    "log_prob": d_lp
                }
                # Flatten dreams
                dream_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), dream_data)
                
                def ac_loss_fn(params, batch, dreamed_data):
                    # REAL DATA LOSS
                    z = world_model.apply(wm_state.params, batch["obs"], method=world_model.get_latent)
                    z = jax.lax.stop_gradient(z)
                    
                    pi, value = actor_critic.apply(params, z)
                    log_prob = pi.log_prob(batch["action"])
                    
                    value_pred_clipped = batch["target"] + (
                        value - batch["target"]
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - batch["target"])
                    value_losses_clipped = jnp.square(value_pred_clipped - batch["target"])
                    value_loss_real = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    
                    ratio = jnp.exp(log_prob - batch["log_prob"])
                    gae = batch["adv"]
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor_real = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    
                    entropy_real = pi.entropy().mean()
                    
                    total_loss_real = loss_actor_real + config["VF_COEF"] * value_loss_real - config["ENT_COEF"] * entropy_real
                    
                    # DREAM DATA LOSS
                    pi_d, value_d = actor_critic.apply(params, dreamed_data["z"])
                    log_prob_d = pi_d.log_prob(dreamed_data["action"])
                    
                    ratio_d = jnp.exp(log_prob_d - dreamed_data["log_prob"])
                    gae_d = dreamed_data["adv"]
                    
                    loss_actor1_d = ratio_d * gae_d
                    loss_actor2_d = jnp.clip(ratio_d, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_d
                    loss_actor_d = -jnp.minimum(loss_actor1_d, loss_actor2_d).mean()
                    
                    value_loss_d = jnp.mean(jnp.square(value_d - dreamed_data["target"]))
                    entropy_d = pi_d.entropy().mean()
                    
                    total_loss_dream = loss_actor_d + config["VF_COEF"] * value_loss_d - config["ENT_COEF"] * entropy_d
                    
                    return total_loss_real + config["DREAM_RATIO"] * total_loss_dream, (value_loss_real, loss_actor_real, entropy_real)

                ac_grad_fn = jax.value_and_grad(ac_loss_fn, has_aux=True)
                (ac_loss_val, _), ac_grads = ac_grad_fn(ac_train_state.params, batch, dream_data)
                ac_state = ac_state.apply_gradients(grads=ac_grads)
                
                return (wm_state, ac_state, rng), (wm_loss_val, ac_loss_val)

            (wm_state, ac_state, rng), (wm_loss, ac_loss) = jax.lax.scan(
                _update_minibatch, (wm_state, ac_train_state, rng), minibatches
            )
            
            update_state = (wm_state, ac_state, env_state, last_obs, update_step, rng)
            
            # Metrics
            metric = traj_batch.info
            metric["wm_loss"] = wm_loss.mean()
            metric["ac_loss"] = ac_loss.mean()
            
            def callback(metric):
                print(f"Metric keys: {list(metric.keys())}")
                if "returned_episode_returns" in metric:
                    print(f"Returned returns mean: {metric['returned_episode_returns']}")
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            
            jax.debug.callback(callback, metric)
            
            return update_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (wm_train_state, ac_train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
