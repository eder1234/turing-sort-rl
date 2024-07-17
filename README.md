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

## Preiliminar result
| Metric                  | Value       |
|-------------------------|-------------|
| **rollout/**            |             |
|    ep_len_mean          | 27          |
|    ep_rew_mean          | 3.48        |
| **time/**               |             |
|    fps                  | 1804        |
|    iterations           | 123         |
|    time_elapsed         | 139         |
|    total_timesteps      | 251904      |
| **train/**              |             |
|    approx_kl            | 0.017784424 |
|    clip_fraction        | 0.137       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.377      |
|    explained_variance   | 0.7657558   |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00478     |
|    n_updates            | 1220        |
|    policy_gradient_loss | -0.0193     |
|    value_loss           | 0.185       |

- Stage 4 performance: 3.02 +/- 2.14
- Model saved to models/sorting_agent
- Evaluating the agent...
Using cpu device
### Evaluation Results:
- Mean Reward: 3.73 +/- 1.98
- Mean Steps: 26.98
- Sorting Accuracy: 0.00%

## Considerations

### Training Performance:
1. The agent is completing episodes with an average length of 27 steps.
2. The mean reward per episode (ep_rew_mean) is 3.48, which is positive but relatively low.
3. The explained variance of 0.7657558 suggests that the value function is doing a decent job of predicting returns.
4. The loss values are relatively low, indicating that the model is converging.

### Stage 4 Performance:
The performance in the final curriculum stage (3.02 +/- 2.14) is slightly lower than the overall training performance, which is not unusual as the task might be more challenging in the final stage.

### Evaluation Results:
1. Mean Reward: 3.73 +/- 1.98 - This is slightly better than the training performance, which is a good sign.
2. Mean Steps: 26.98 - This is consistent with the training episode length.
3. Sorting Accuracy: 0.00% - This is the most concerning aspect of the results.

### Analysis:
1. The agent is learning to accumulate some reward, which is positive. It's making some progress in reducing inversions in the array.
2. However, the 0% sorting accuracy during evaluation is a significant issue. This means that the agent is not successfully sorting any arrays completely.
3. The positive mean reward suggests that the agent is making partial progress (reducing some inversions) but is not completing the task.

### Possible reasons and suggestions:
1. Reward shaping: The current reward structure might not provide enough incentive for complete sorting. Consider increasing the bonus for a fully sorted array.

2. Curriculum difficulty: The jump in difficulty between curriculum stages might be too large. Try increasing the number of curriculum stages for a more gradual progression.

3. Training duration: The agent might need more training time. Try increasing the total_timesteps.

4. Model capacity: The neural network might not be complex enough to learn the task. Consider increasing the size of the policy network.

5. Exploration: The agent might be stuck in a local optimum. Try adjusting the entropy coefficient to encourage more exploration.

6. Task complexity: Sorting using only swap operations is a challenging task. You might want to consider adding more informative observations (like a count of inversions) or additional actions that could make the task easier to learn.

### Next steps:
1. Implement some of the suggestions above, particularly focusing on reward shaping and increasing training time.
2. Add more detailed logging during evaluation to see exactly what the agent is doing (e.g., print out the initial and final states of some episodes).
3. Visualize the agent's behavior on a few specific input arrays to get a better understanding of its strategy.

While the agent is learning something, there's clearly room for improvement to achieve successful sorting. Let me know if you'd like to focus on any specific area for improvement.

## Contributing

Contributions to TuringSortRL are welcome! Please feel free to submit a Pull Request.
