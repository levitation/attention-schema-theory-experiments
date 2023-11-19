from collections import namedtuple
'''
from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
'''
from aintelope.environments.savanna_gym import SavannaGymEnv
from aintelope.models.dqn import DQN
from aintelope.agents.instinct_agent import QAgent  # initialize agent registry
from aintelope.agents import get_agent_class
# from aintelope.agents.memory import ReplayBuffer, Experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("aintelope.experiment")
    
    episode_durations = []

    # Environment
    env = SavannaGymEnv(env_params=cfg.hparams.env_params) #TODO: get env from parameters
    n_actions = env.action_space.n # TODO what is this used for?
    #state, info = env.reset() #TODO: each agent has their own state, remove from here
    n_observations = len(state)  # get observation space from env
    action_space = self.env.action_space # removed dynamic actions per agent for now
    
    # Common trainer
    trainer = dqn_training()
    
    # Agents
    #agents = {}
    agent_id = 0
    #for agent_id in range(agents):
    #    agents["agent_"+agent_id] = 
    agent = get_agent_class(hparams.agent_id)(
        agent_id,
        trainer,
        hparams.warm_start_steps,
        **hparams.agent_params,
    )
    # buffer should be associated with agent
    #replay_buffer = ReplayBuffer(cfg.hparams.replay_size)
    # generalize to multi agent setup

    # networks should be part of the agent
    #policy_net = DQN(n_observations, n_actions).to(device)
    #target_net = DQN(n_observations, n_actions).to(device)
    #optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    #target_net.load_state_dict(policy_net.state_dict())
    
    # Warmup not supported atm, maybe not needed?
    #for _ in range(hparams.warm_start_steps):
    #     agents.play_step(self.net, epsilon=1.0) # TODO
    
    steps_done = 0

    for i_episode in range(num_episodes):
        # Reset
        state, info = env.reset() # remove state and info from here
        for agent in agents:
            agent.reset()
            agent.observe(env.observation(agent)) #TODO
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for step in range(1): #TODO
            '''
            # set epsilon value. should be in agent
            epsilon = max(
                cfg.hparams.eps_end,
                cfg.hparams.eps_start - step * 1 / cfg.hparams.eps_last_frame,
            )
            '''
            # for agent in agents:
            observation = self.env.observe(agent)
            action = agent.get_action(observation, step)#state, policy_net, epsilon, "cpu")
            
            # Env step
            if isinstance(env, GymEnv):
                observation, score, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            elif isinstance(env, PettingZooEnv):
                observation, score, terminateds, truncateds, _ = env.step(action)
                done = {
                    key: terminated or truncateds[key]
                    for (key, terminated) in terminateds.items()
                }
            else:
                logger.warning(f"{env} is not of type GymEnv or PettingZooEnv")
                observation, score, done, _ = env.step(action)
            ### TODO: move to support only pettingzoo?
            #observation, reward, terminated, truncated, _ = env.step(action)
            
            '''
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            exp = Experience(state, action, reward, done, next_state)
            replay_buffer.append(exp)

            # Move to the next state
            state = next_state
            '''
            # Agent is updated based on what the env shows. All commented above included ^
            done = terminated or truncated
            agent.update(observation, score, done) # note that score is used ONLY by baseline
            
            # Perform one step of the optimization (on the policy network)
            # TEST: if we call this every time, will it overlearn the initial steps? The buffer
            # is filled only with a batch worth of stuff, and it might overrepresent?
            trainer.optimize_models(step)

            if done:
                episode_durations.append(step + 1)
                break

# WIP: instantiate?
def reset() -> None:
    """Resets environment and agents."""
    self.done = False
    self.state = state
    if isinstance(self.state, tuple):
        self.state = self.state[0]
   
        
if __name__ == "__main__":
    main()
