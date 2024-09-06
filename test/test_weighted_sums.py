# weighted_sum.py

def calculate_weighted_sum(inputs, weights):
    """
    Calculate the weighted sum of inputs and weights.

    Args:
        inputs (list of float): List of input values (x₁, x₂, ..., xₙ).
        weights (list of float): Corresponding list of weights (w₁, w₂, ..., wₙ).

    Returns:
        float: The weighted sum of inputs and weights.
    """
    # Ensure the lists of inputs and weights are of the same length
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length.")
    
    # Calculate the weighted sum
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    return weighted_sum

# Test the function with sample inputs
if __name__ == "__main__":
    # Sample inputs and weights
    inputs = [1.0, 2.0, 3.0]  # Example inputs (x₁, x₂, x₃)
    weights = [0.5, 0.3, 0.2] # Corresponding weights (w₁, w₂, w₃)

    # Calculate and print the weighted sum
    result = calculate_weighted_sum(inputs, weights)
    print(f"The weighted sum is: {result}")
