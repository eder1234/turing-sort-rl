import os
import numpy as np
from turing_machine import TuringMachineEnv
from rl_agent import RLAgent
from array_generator import generate_evaluation_array

def evaluate_agent(model_path, num_episodes=1000):
    """
    Evaluate the trained agent on arrays of length 10.
    
    Args:
    model_path (str): Path to the saved model
    num_episodes (int): Number of episodes to evaluate
    
    Returns:
    tuple: Mean reward, standard deviation of reward, mean steps, sorting accuracy
    """
    env = TuringMachineEnv(array_length=10)
    agent = RLAgent.load(model_path, env)
    
    rewards = []
    steps = []
    correct_sorts = 0
    
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action, _ = agent.predict(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
        
        # Check if the array is correctly sorted
        if np.array_equal(env.tape, sorted(env.tape)):
            correct_sorts += 1
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps)
    sorting_accuracy = correct_sorts / num_episodes
    
    return mean_reward, std_reward, mean_steps, sorting_accuracy

def print_evaluation_results(mean_reward, std_reward, mean_steps, sorting_accuracy):
    """Print the evaluation results in a formatted manner."""
    print(f"Evaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Steps: {mean_steps:.2f}")
    print(f"Sorting Accuracy: {sorting_accuracy:.2%}")

if __name__ == "__main__":
    MODEL_PATH = os.path.join("models", "sorting_agent")
    
    # Evaluate the agent
    mean_reward, std_reward, mean_steps, sorting_accuracy = evaluate_agent(MODEL_PATH)
    
    # Print results
    print_evaluation_results(mean_reward, std_reward, mean_steps, sorting_accuracy)
    
    # Additional analysis: distribution of steps for correct sorts
    env = TuringMachineEnv(array_length=10)
    agent = RLAgent.load(MODEL_PATH, env)
    
    correct_sort_steps = []
    for _ in range(100):  # Analyze 100 correct sorts
        observation = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = agent.predict(observation)
            observation, _, done, _ = env.step(action)
            steps += 1
        if np.array_equal(env.tape, sorted(env.tape)):
            correct_sort_steps.append(steps)
    
    if correct_sort_steps:
        print(f"\nAnalysis of steps for correct sorts:")
        print(f"Min steps: {min(correct_sort_steps)}")
        print(f"Max steps: {max(correct_sort_steps)}")
        print(f"Mean steps: {np.mean(correct_sort_steps):.2f}")
        print(f"Median steps: {np.median(correct_sort_steps):.2f}")
    else:
        print("\nNo correct sorts observed in the analysis.")
