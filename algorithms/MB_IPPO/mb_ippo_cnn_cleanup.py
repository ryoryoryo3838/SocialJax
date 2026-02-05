""" 
Based on algorithms/IPPO/ippo_cnn_cleanup.py
Modified for Model-Based RL (MB-IPPO)
"""
import sys
# sys.path.append('/home/shuqing/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper, SVOLogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# --- World Model Components ---

from components.models.encoder import CNNEncoder
from components.models.decoder import ImageDecoder, RewardDecoder
from components.models.dynamics import LatentDynamics

# --- World Model Components ---

class WorldModel(nn.Module):
    action_dim: int
    input_shape: Tuple[int, int, int] # Observation shape
    activation: str = "relu"

    def setup(self):
        self.encoder = CNNEncoder(activation=self.activation)
        self.decoder = ImageDecoder(output_shape=self.input_shape, activation=self.activation)
        self.dynamics = LatentDynamics(activation=self.activation)
        self.reward_model = RewardDecoder(activation=self.activation)

    def __call__(self, obs, action):
        z = self.encoder(obs)
        rec_obs = self.decoder(z)
        z_next = self.dynamics(z, action, self.action_dim)
        r_pred = self.reward_model(z, action, self.action_dim)
        return z, rec_obs, z_next, r_pred

    def get_latent(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        z_next = self.dynamics(z, action, self.action_dim)
        r_pred = self.reward_model(z, action, self.action_dim)
        return z_next, r_pred

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
    MODEL_BATCH_SIZE = config["MINIBATCH_SIZE"] # Reuse same batch size logic?

    env = LogWrapper(env, replace_info=False)

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
        wm_params = world_model.init(_rng, init_x, 0) # init with dummy action 0

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
                     raise NotImplementedError("Independent parameters not supported yet in MB-IPPO")

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

            # --- PREPARE BATCHES ---
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            
            # Use same batch for model (real data) and policy (real data part)
            # Need next observations for model training (z_next target)
            # Traj batch obs is [T, N, ...]
            # Next obs is twisted.
            # obs[t+1] corresponds to next state of obs[t] UNLESS done.
            # We can use traj_batch.obs[1:] and last_obs.
            
            obs = traj_batch.obs
            next_obs = jnp.concatenate([obs[1:], last_obs_batch[None, ...]], axis=0)
            
            # Filter dones? Model shouldn't predict across episodes.
            # But masked loss is easier.
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
                    
                    # Target z_next (need to encode next_obs with STOP GRADIENT to avoid collapsing)
                    z_next_target = world_model.apply(params, batch["next_obs"], method=world_model.get_latent)
                    z_next_target = jax.lax.stop_gradient(z_next_target)
                    
                    # Losses
                    rec_loss = jnp.mean(jnp.square(rec_obs - batch["obs"])) # MSE
                    dyn_loss = jnp.mean(jnp.square(z_next_pred - z_next_target) * (1 - batch["done"])[:, None])
                    rew_loss = jnp.mean(jnp.square(r_pred - batch["reward"]))
                    
                    total_loss = rec_loss + dyn_loss + rew_loss
                    return total_loss, (rec_loss, dyn_loss, rew_loss)

                wm_grad_fn = jax.value_and_grad(wm_loss_fn, has_aux=True)
                wm_loss, wm_grads = wm_grad_fn(wm_state.params, batch)
                wm_state = wm_state.apply_gradients(grads=wm_grads)
                
                # 2. Update Policy (ActorCritic) with Real + Dreamed Data
                
                # Dream!
                # Start from real latents in batch
                z_start = world_model.apply(wm_state.params, batch["obs"], method=world_model.get_latent)
                z_start = jax.lax.stop_gradient(z_start) # Don't backprop into model from policy here
                
                def dream_step(carry, _):
                    z, rng = carry
                    rng, _rng = jax.random.split(rng)
                    
                    # Policy Action
                    pi, value_pred = actor_critic.apply(ac_state.params, z)
                    action = pi.sample(seed=_rng)
                    # For stored log prob/value we use current policy outputs as 'old' for PPO clip?
                    # No, usually in Dyna-PPO we treat imagined data as fresh on-policy or use simple AC?
                    # We can use the imagined data in the PPO loss. 
                    # But we need 'old_log_prob' for PPO. If we generate it now, old=new, ratio=1.
                    # That is fine.
                    
                    log_prob = pi.log_prob(action)
                    
                    # Model Step
                    z_next, r_pred = world_model.apply(wm_state.params, z, action, method=world_model.predict_next)
                    
                    transition = {
                        "z": z,
                        "action": action,
                        "reward": r_pred,
                        "value": value_pred, # Estimate value of current z
                        "log_prob": log_prob,
                        "done": jnp.zeros(z.shape[0]), # Model assumes no terminal in horizon usually
                    }
                    
                    return (z_next, rng), transition

                rng, _rng = jax.random.split(rng)
                _, dreamed_traj = jax.lax.scan(dream_step, (z_start, _rng), None, config["DREAM_HORIZON"])
                
                # Collapse dreams
                # dreamed_traj: {key: [H, B, ...]}
                # Calculate Advantages for dreams?
                # Using GAE on dreams with Value estimate from critic.
                
                # Need value of last dreamed state
                last_z_dream = world_model.apply(wm_state.params, z_start, dreamed_traj["action"][-1], method=world_model.get_latent) # Wait, need dynamics not latent?
                # Actually scan returns last state
                # But let's act simply:
                # We have values for t=0..H-1. We need Value(H).
                # We can run critic on the final z_next from scan.
                # But for simplicity let's just use 1-step returns or GAE with 0 last val.
                # Let's compute values for all H+1 states?
                
                # Let's keep it simple: Treat dreams as extra batch for PPO.
                # Compute GAE for dreamed trajectory.
                # We need Value of next state.
                # In dream_step we got z_next. We can compute value there.
                
                pass # Already complex.
                
                # Let's simplify: Standard PPO update on REAL data + Imagine Loss?
                # Or Mix Real and Dreamed.
                
                # Implementation:
                # Calculate PPO Loss on REAL data (using batches) -> Standard
                # ADDITIONALLY: Calculate Policy Loss on DREAMED data (Dyna style)
                #   Loss = PPO_Loss(Real) + Dream_loss
                
                # Let's define the AC loss function
                def ac_loss_fn(params, batch, dreamed_data=None):
                    # REAL DATA LOSS
                    # Rerunning network on real obs
                    # Get latent (stop grad from model)
                    z = world_model.apply(wm_state.params, batch["obs"], method=world_model.get_latent)
                    z = jax.lax.stop_gradient(z)
                    
                    pi, value = actor_critic.apply(params, z)
                    log_prob = pi.log_prob(batch["action"])
                    
                    # Value Loss
                    value_pred_clipped = batch["target"] + (
                        value - batch["target"]
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - batch["target"])
                    value_losses_clipped = jnp.square(value_pred_clipped - batch["target"])
                    value_loss_real = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    
                    # Actor Loss
                    ratio = jnp.exp(log_prob - batch["log_prob"])
                    gae = batch["adv"]
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor_real = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    
                    entropy_real = pi.entropy().mean()
                    
                    total_loss_real = loss_actor_real + config["VF_COEF"] * value_loss_real - config["ENT_COEF"] * entropy_real
                    
                    # DREAM DATA LOSS (if ratio > 0)
                    if dreamed_data is not None:
                         # Calculate similar loss but for dreams
                         # dream data keys: z, action, reward, value(old), log_prob(old), adv, target
                         pi_d, value_d = actor_critic.apply(params, dreamed_data["z"])
                         log_prob_d = pi_d.log_prob(dreamed_data["action"])
                         
                         ratio_d = jnp.exp(log_prob_d - dreamed_data["log_prob"])
                         gae_d = dreamed_data["adv"]
                         # Normalize dream gae?
                         # gae_d = (gae_d - gae_d.mean()) / (gae_d.std() + 1e-8) 
                         
                         loss_actor1_d = ratio_d * gae_d
                         loss_actor2_d = jnp.clip(ratio_d, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_d
                         loss_actor_d = -jnp.minimum(loss_actor1_d, loss_actor2_d).mean()
                         
                         value_loss_d = jnp.mean(jnp.square(value_d - dreamed_data["target"]))
                         entropy_d = pi_d.entropy().mean()
                         
                         total_loss_dream = loss_actor_d + config["VF_COEF"] * value_loss_d - config["ENT_COEF"] * entropy_d
                         
                         return total_loss_real + config["DREAM_RATIO"] * total_loss_dream, (value_loss_real, loss_actor_real, entropy_real)

                    return total_loss_real, (value_loss_real, loss_actor_real, entropy_real)

                # Generate dreams and Compute GAE for them
                # Note: We do this OUTSIDE loss fn because we need to unroll model and compute targets first
                
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
                    # done is always 0 in dream
                    return (z_next, rng), (z, action, r_pred, val, log_prob, z_next)
                    
                _, (d_z, d_a, d_r, d_v, d_lp, d_zn) = jax.lax.scan(dream_step_gae, (z_start, rng), None, config["DREAM_HORIZON"])
                
                # Compute Dream GAE
                # Need Value of d_zn.
                # Reshape d_zn: [H, B, F]
                last_vals = []
                # Very inefficient loop, but specialized for small H
                # Ideally scan or vmap.
                _, d_next_v = actor_critic.apply(ac_state.params, d_zn) # Batch apply? [H, B, F] -> [H, B]
                # Flax apply handles extra dims? Usually yes if dense.
                
                # GAE Scan
                def dream_gae_scan(next_val, item):
                    reward, value, done = item
                    delta = reward + config["GAMMA"] * next_val * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * 0 # No carry gae?
                    # Wait, GAE recursive.
                    # This scan needs to go backwards.
                    return None # placeholder
                
                # Let's do simple calculation
                # Since H is small (5), use a loop or just 1-step?
                # Let's implement GAE properly backwards.
                
                d_done = jnp.zeros_like(d_r)
                
                # GAE loop
                def get_dream_adv(gae_and_next_val, item):
                     gae, next_val = gae_and_next_val
                     r, v, d = item
                     delta = r + config["GAMMA"] * next_val * (1-d) - v
                     gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1-d) * gae
                     return (gae, v), (gae, gae+v) # adv, target
                
                # Scan backwards
                (_, _), (d_adv, d_target) = jax.lax.scan(
                    get_dream_adv,
                    (jnp.zeros_like(d_r[0]), jnp.zeros_like(d_r[0])), # Zero terminal value approx
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
                # Flatten dreams [H, B, ...] -> [H*B, ...]
                dream_data = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), dream_data)
                
                ac_grad_fn = jax.value_and_grad(ac_loss_fn, has_aux=True)
                ac_loss, ac_grads = ac_grad_fn(ac_train_state.params, batch, dream_data)
                ac_state = ac_state.apply_gradients(grads=ac_grads)
                
                return (wm_state, ac_state, rng), (wm_loss, ac_loss)

            (wm_state, ac_state, rng), (wm_loss, ac_loss) = jax.lax.scan(
                _update_minibatch, (wm_state, ac_train_state, rng), minibatches
            )
            
            update_state = (wm_state, ac_state, env_state, last_obs, update_step, rng)
            
            # Metrics
            metric = traj_batch.info
            metric["wm_loss"] = wm_loss.mean()
            metric["ac_loss"] = ac_loss.mean()
            
            def callback(metric):
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

# --- Main & Config ---
@hydra.main(version_base=None, config_path="config", config_name="mb_ippo_cnn_cleanup")
def main(config):
    # Basic Main
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=list(config["WANDB_TAGS"]),
        config=OmegaConf.to_container(config),
        mode=config["WANDB_MODE"],
        name=f'mb_ippo_cnn_cleanup'
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    print("Training Finished")

if __name__ == "__main__":
    main()
