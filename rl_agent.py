from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from turing_machine import TuringMachineEnv

class RLAgent:
    def __init__(self, env, model_params=None):
        """
        Initialize the RL agent with a PPO model.
        
        Args:
        env (gym.Env): The Turing machine environment
        model_params (dict): Parameters for the PPO model
        """
        if model_params is None:
            model_params = {
                "policy": "MlpPolicy",
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "verbose": 1
            }
        
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO(env=self.env, **model_params)

    def train(self, total_timesteps):
        """
        Train the agent.
        
        Args:
        total_timesteps (int): Total number of timesteps to train for
        """
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, n_eval_episodes=10):
        """
        Evaluate the agent's performance.
        
        Args:
        n_eval_episodes (int): Number of episodes to evaluate on
        
        Returns:
        tuple: Mean reward and standard deviation of reward
        """
        return evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes)

    def save(self, path):
        """
        Save the trained model.
        
        Args:
        path (str): Path to save the model to
        """
        self.model.save(path)

    @classmethod
    def load(cls, path, env):
        """
        Load a trained model.
        
        Args:
        path (str): Path to load the model from
        env (gym.Env): The Turing machine environment
        
        Returns:
        RLAgent: An RLAgent instance with the loaded model
        """
        agent = cls(env)
        agent.model = PPO.load(path, env=agent.env)
        return agent

    def predict(self, observation):
        """
        Make a prediction for a single observation.
        
        Args:
        observation (np.array): The current observation of the environment
        
        Returns:
        tuple: Predicted action and additional information
        """
        return self.model.predict(observation)

if __name__ == "__main__":
    # Test the RLAgent
    env = TuringMachineEnv()
    agent = RLAgent(env)
    print("Agent created successfully")
    
    # Train for a small number of steps
    agent.train(1000)
    print("Training completed")
    
    # Evaluate
    mean_reward, std_reward = agent.evaluate()
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save and load
    agent.save("test_model")
    loaded_agent = RLAgent.load("test_model", env)
    print("Model saved and loaded successfully")
