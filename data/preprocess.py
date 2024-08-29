import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# Ensure that NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load stopwords
stop_words = set(stopwords.words('english'))

def normalize_text(text: str) -> str:
    """
    Normalize the input text by removing punctuation, converting to lowercase, and stripping whitespace.
    Args:
    - text (str): The input text string to normalize.

    Returns:
    - str: The normalized text.
    """
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize the input text into individual words/tokens.
    Args:
    - text (str): The normalized text string to tokenize.

    Returns:
    - List[str]: A list of tokens extracted from the text.
    """
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from the list of tokens.
    Args:
    - tokens (List[str]): The list of tokens from which to remove stopwords.

    Returns:
    - List[str]: A list of tokens with stopwords removed.
    """
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize the list of tokens to reduce them to their base form.
    Args:
    - tokens (List[str]): The list of tokens to lemmatize.

    Returns:
    - List[str]: A list of lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess the input text by normalizing, tokenizing, removing stopwords, and lemmatizing.
    Args:
    - text (str): The input text to preprocess.

    Returns:
    - List[str]: A list of preprocessed tokens ready for graph representation.
    """
    # Step-by-step preprocessing
    normalized_text = normalize_text(text)
    tokens = tokenize_text(normalized_text)
    tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)
    return lemmatized_tokens

if __name__ == "__main__":
    # Example usage
    sample_text = "The quick brown fox jumps over the lazy dog!"
    processed_tokens = preprocess_text(sample_text)
    print(f"Processed Tokens: {processed_tokens}")
