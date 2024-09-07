import matplotlib.pyplot as plt

# Example words mapped to their "importance" score
words = ["I", "Love", "Machine", "Learning", "in", "2024", "because", 
         "the", "world", "is", "changing", "at", "the", "speed", "of", "AI"]
importance = [1, 5, 4, 4, 1, 3, 2, 1, 3, 1, 5, 1, 1, 4, 1, 4]  # Arbitrary scores for demonstration

# Apply ReLU (only keep words with scores > 2)
filtered_words = [word if score > 2 else "" for word, score in zip(words, importance)]

# Plotting the words and their filtered states
plt.figure(figsize=(14, 4))
plt.plot(words, importance, 'ro-', label='Importance Score')
plt.axhline(y=2, color='blue', linestyle='--', label='ReLU Threshold')
plt.xticks(rotation=45)
plt.title('Neural Network ReLU Filter: Word Importance')
plt.ylabel('Importance Score')
plt.xlabel('Words')
plt.legend()
plt.grid(True)
plt.show()

# Display filtered words
print("Filtered Words:", [word for word in filtered_words if word])
