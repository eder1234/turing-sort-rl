import os
import numpy as np
from turing_machine import TuringMachineEnv
from rl_agent import RLAgent
from array_generator import generate_curriculum

def train_agent(total_timesteps, curriculum_steps, save_path):
    """
    Train the RL agent using curriculum learning.
    
    Args:
    total_timesteps (int): Total number of timesteps to train for
    curriculum_steps (int): Number of steps in the curriculum
    save_path (str): Path to save the trained model
    
    Returns:
    RLAgent: The trained agent
    """
    # Generate curriculum
    curriculum = generate_curriculum(total_timesteps, curriculum_steps)
    
    # Initialize environment with the shortest array length
    env = TuringMachineEnv(array_length=len(curriculum[0]))
    agent = RLAgent(env)
    
    # Train through the curriculum
    timesteps_per_stage = total_timesteps // curriculum_steps
    for i in range(curriculum_steps):
        print(f"Curriculum stage {i+1}/{curriculum_steps}")
        
        # Update environment with new array length
        env.array_length = len(curriculum[i * timesteps_per_stage])
        agent.env.envs[0].array_length = len(curriculum[i * timesteps_per_stage])
        
        # Train for this curriculum stage
        agent.train(timesteps_per_stage)
        
        # Evaluate current performance
        mean_reward, std_reward = agent.evaluate(n_eval_episodes=100)
        print(f"Stage {i+1} performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save the trained model
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    return agent

if __name__ == "__main__":
    # Training parameters
    TOTAL_TIMESTEPS = 1_000_000  # Adjust based on your computational resources
    CURRICULUM_STEPS = 4
    SAVE_PATH = os.path.join("models", "sorting_agent")
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Train the agent
    trained_agent = train_agent(TOTAL_TIMESTEPS, CURRICULUM_STEPS, SAVE_PATH)
    
    # Final evaluation
    final_mean_reward, final_std_reward = trained_agent.evaluate(n_eval_episodes=1000)
    print(f"Final performance: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")
