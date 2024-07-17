import numpy as np

def generate_array(length):
    """
    Generate a random array of unique integers.
    
    Args:
    length (int): Length of the array to generate (6 to 9 for training, 10 for evaluation)
    
    Returns:
    numpy.array: Array of unique integers between 0 and 9
    """
    if length < 6 or length > 10:
        raise ValueError("Array length must be between 6 and 10")
    
    return np.random.choice(10, length, replace=False)

def generate_training_array():
    """
    Generate a random array for training (length 6 to 9).
    
    Returns:
    numpy.array: Array of unique integers between 0 and 9
    """
    length = np.random.randint(6, 10)  # 6 to 9 inclusive
    return generate_array(length)

def generate_evaluation_array():
    """
    Generate a random array for evaluation (length 10).
    
    Returns:
    numpy.array: Array of 10 unique integers between 0 and 9
    """
    return generate_array(10)

def generate_curriculum(num_episodes, curriculum_steps):
    """
    Generate a curriculum of arrays with increasing length.
    
    Args:
    num_episodes (int): Total number of episodes
    curriculum_steps (int): Number of steps in the curriculum
    
    Returns:
    list: List of numpy arrays with increasing lengths
    """
    arrays = []
    episodes_per_step = num_episodes // curriculum_steps
    
    for i in range(curriculum_steps):
        length = 6 + i * (4 // curriculum_steps)  # Increase length from 6 to 9
        for _ in range(episodes_per_step):
            arrays.append(generate_array(length))
    
    return arrays

if __name__ == "__main__":
    # Test the functions
    print("Training array:", generate_training_array())
    print("Evaluation array:", generate_evaluation_array())
    print("Curriculum sample:")
    curriculum = generate_curriculum(100, 4)
    print(curriculum[:5])  # Print first 5 arrays in the curriculum
