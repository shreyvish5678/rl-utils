import gym
from policy_gradient_agent import PPOAgent
from tqdm import tqdm

def train(
        env_name: str,
        n_episodes: int = 1000,
        max_t: int = 1000,
        learning_rate: float = 3e-4
    ):
    env = gym.make(env_name)
    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        learning_rate,
        cnn_policy=True
    )
    max_rewards = 0

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_rewards = 0
        states, actions, log_probs, action_probs, rewards, next_states, dones = [], [], [], [], [], [], []

        for t in range(max_t):
            action, log_prob, action_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            action_probs.append(action_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            episode_rewards += reward
            state = next_state
            
            if done:
                break
        
        if episode_rewards > max_rewards:
            max_rewards = episode_rewards
            total_score = 0
            for _ in range(50):
                state, _ = env.reset()
                for _ in range(500):
                    action, _, _ = agent.select_action(state)
                    next_state, reward, done, score, _ = env.step(action)
                    state = next_state
                    if done:
                        break
                total_score += score
            print(f"Episode {episode}, Total Reward: {episode_rewards} Average Score: {total_score / 50}")
            agent.save('game.pth')
        
        agent.update(states, actions, log_probs, action_probs, rewards, next_states, dones)
    
    env.close()
import gym_game

if __name__ == "__main__":
    train(env_name='ALE/Adventure-v5', n_episodes=5000, max_t=500, learning_rate=3e-4)