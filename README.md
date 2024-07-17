# TuringSortRL: Reinforcement Learning for Array Sorting

TuringSortRL is a project that implements a reinforcement learning agent to sort arrays using a Turing machine-like interface. The agent learns to sort arrays of integers using basic operations like moving left, moving right, and swapping adjacent elements.

## Project Structure

The project consists of the following main components:

- `turing_machine.py`: Implements the Turing machine environment
- `array_generator.py`: Generates random arrays for training and evaluation
- `rl_agent.py`: Defines the reinforcement learning agent using PPO
- `train.py`: Implements the training process with curriculum learning
- `evaluate.py`: Evaluates the trained agent's performance
- `main.py`: Main script to run training, evaluation, and demos

## Requirements

- Python 3.7+
- NumPy
- Gym
- Stable-Baselines3

Install the required packages using:
```
pip install numpy gym stable-baselines3
```
## Usage

To use the TuringSortRL project, run the `main.py` script with the following options:

- Train the agent:
```
python main.py --train
```
- Evaluate the agent:
```
python main.py --evaluate
```
- Run a demo of the sorting process:
```
python main.py --demo
```
You can combine these options, for example:
```
python main.py --train --evaluate
```
## How It Works

1. The agent is trained on arrays of lengths 6 to 9 using curriculum learning.
2. The agent learns to sort arrays by moving a head left and right, and swapping adjacent elements.
3. The training process uses the PPO (Proximal Policy Optimization) algorithm.
4. The agent is evaluated on its ability to sort arrays of length 10, which it hasn't seen during training.

## Customization

You can adjust the following parameters in the scripts:

- `TOTAL_TIMESTEPS` in `main.py`: Total number of timesteps for training
- `CURRICULUM_STEPS` in `main.py`: Number of stages in the curriculum learning process
- Model hyperparameters in `rl_agent.py`

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions to TuringSortRL are welcome! Please feel free to submit a Pull Request.
