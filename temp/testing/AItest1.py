# Function to calculate the sum of all numbers up to n
def sum_up_to(n):
    return sum(range(1, n + 1))

# Calculate the sum of numbers up to 1 billion
result = sum_up_to(1_000_000_000)

# Print the result
print(f"The sum of all numbers up to 10 million is: {result}")
