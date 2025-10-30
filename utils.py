import random
import torch
import numpy as np
from tqdm import tqdm


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # If multiple GPUs are used
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_advantage(gamma, lmbda, td_error):
    """Implement GAE (Generalized Advantage Estimation).

    Args:
        gamma (float): discount factor
        lmbda (float): decay rate
        td_error (Tensor): TD error

    Formulation:
        \hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^{V}

    Reference:
        https://arxiv.org/abs/1506.02438
        https://www.yuque.com/chenxingjian/klyobx/gmeiz4ge1p9nquvx
    """
    td_error = td_error.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_error[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_tensor = torch.tensor(advantage_list, dtype=torch.float)
    return advantage_tensor

def train_on_policy_agent(env, agent, num_episodes, max_step=10):
    """
    NOTE: agent must implement the following interface:
    1. take_action(self, state) -> action
    2. update(self, transition_dict) -> None
    """
    return_list = []
    for i in range(max_step):
        with tqdm(total=int(num_episodes / max_step), desc=f"Iteration {i}") as pbar:
            for i_episode in range(int(num_episodes / max_step)):
                # set episode_return to zero and set environment to the initial state
                episode_return = 0
                transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
                state = env.reset()[0]

                # Sample trajectory based on behavior policy
                done = False
                while not done:
                    action = agent.take_action(state)
                    # NOTE: here we can get reward, but some in some situation, we may not
                    next_state, reward, done, _, _ = env.step(action)
                    # Record
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                # Once we get a sample trajectory, update the policy network and value network
                agent.update(transition_dict)
                if (i_episode + 1) % max_step == 0:
                    pbar.set_postfix({
                        "episode": f"{num_episodes / max_step * i + i_episode + 1}",
                        "return": f"{np.mean(return_list[-max_step:])}",
                    })
                pbar.update(1)
    return return_list
