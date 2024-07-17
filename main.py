import os
import argparse
import numpy as np
from turing_machine import TuringMachineEnv
from rl_agent import RLAgent
from train import train_agent
from evaluate import evaluate_agent, print_evaluation_results
from array_generator import generate_evaluation_array

def demo_sort(agent, env, max_steps=100):
    """
    Demonstrate the sorting process for a single array.
    
    Args:
    agent (RLAgent): The trained agent
    env (TuringMachineEnv): The Turing machine environment
    max_steps (int): Maximum number of steps to allow
    """
    observation = env.reset()
    print("Initial state:", env.tape)
    
    done = False
    steps = 0
    while not done and steps < max_steps:
        action, _ = agent.predict(observation)
        observation, reward, done, _ = env.step(action)
        steps += 1
        print(f"Step {steps}: Action {action}, New state: {env.tape}")
    
    if np.array_equal(env.tape, sorted(env.tape)):
        print("Sorting successful!")
    else:
        print("Sorting unsuccessful.")
    print(f"Total steps: {steps}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a sorting agent")
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agent')
    parser.add_argument('--demo', action='store_true', help='Run a demo of the sorting process')
    args = parser.parse_args()

    MODEL_PATH = os.path.join("models", "sorting_agent")
    
    if args.train:
        print("Training the agent...")
        TOTAL_TIMESTEPS = 1_000_000
        CURRICULUM_STEPS = 4
        train_agent(TOTAL_TIMESTEPS, CURRICULUM_STEPS, MODEL_PATH)
    
    if args.evaluate:
        print("Evaluating the agent...")
        mean_reward, std_reward, mean_steps, sorting_accuracy = evaluate_agent(MODEL_PATH)
        print_evaluation_results(mean_reward, std_reward, mean_steps, sorting_accuracy)
    
    if args.demo:
        print("Running a demo of the sorting process...")
        env = TuringMachineEnv(array_length=10)
        agent = RLAgent.load(MODEL_PATH, env)
        demo_sort(agent, env)

if __name__ == "__main__":
    main()
