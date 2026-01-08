# Hybrid Sentiment Analysis

A comprehensive sentiment analysis tool that combines the power of transformer models (RoBERTa) with domain-specific lexicons (Loughran McDonald Dictionary) for enhanced financial text analysis.

## Features

- **Hybrid Approach**: Combines RoBERTa transformer model with Loughran McDonald financial lexicon
- **Customizable Weights**: Adjustable weights for model vs dictionary contributions
- **Detailed Analysis**: Provides breakdown of sentiment scores, word counts, and detected sentiment words
- **Financial Focus**: Optimized for financial text analysis using specialized lexicon
- **Stemming Support**: Uses Porter Stemmer for better word matching

## Requirements

```bash
pip install transformers torch pandas nltk
```

## Setup

1. **Download the Loughran McDonald Dictionary**:
   - Download `LM-SA-2020.csv` (Loughran McDonald Sentiment Analysis Dictionary)
   - Ensure it contains columns: `word` and `sentiment` (Positive, Negative, Neutral)

2. **Download NLTK data** (if not already installed):
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

### Basic Usage

```python
# Initialize the analyzer
analyzer = HybridSentimentAnalyzer(
    lm_dict_path="path/to/LM-SA-2020.csv",
    model_weight=1.0,    # Weight for RoBERTa model
    dict_weight=0.5      # Weight for dictionary
)

# Analyze text
text = "The company reported strong earnings with positive outlook."
result = analyzer.analyze(text)

print(f"Sentiment: {result['final_sentiment']}")
print(f"Score: {result['final_score']:.4f}")
```

### Detailed Analysis

```python
# Get comprehensive analysis
result = analyzer.analyze(text)

# Access detailed results
print("=== RESULTS ===")
print(f"Final Sentiment: {result['final_sentiment']}")
print(f"Final Score: {result['final_score']:.4f}")
print(f"Model Score: {result['model_score']:.4f}")
print(f"Dictionary Score: {result['dict_score']:.4f}")

# Word counts from dictionary
dict_details = result['dict_details']
print(f"Positive words: {dict_details['positive_count']}")
print(f"Negative words: {dict_details['negative_count']}")
print(f"Neutral words: {dict_details['neutral_count']}")

# See detected words
detected_words = find_detected_words(text, analyzer.lm_dict)
for category, words in detected_words.items():
    if words:
        print(f"{category}: {', '.join(set(words))}")
```

## Classes and Methods

### `LoughranMcDonaldDict`

Handles the Loughran McDonald financial dictionary for sentiment analysis.

**Methods:**
- `__init__(csv_file_path)`: Initialize with dictionary file
- `analyze_text(text)`: Analyze text using dictionary approach

### `HybridSentimentAnalyzer`

Main class that combines RoBERTa model with Loughran McDonald dictionary.

**Parameters:**
- `lm_dict_path`: Path to the Loughran McDonald CSV file
- `model_weight`: Weight for transformer model contribution (default: 1.0)
- `dict_weight`: Weight for dictionary contribution (default: 0.5)

**Methods:**
- `analyze(text)`: Perform hybrid sentiment analysis
- `get_model_sentiment(text)`: Get sentiment from RoBERTa model only

### Helper Functions

- `find_detected_words(text, lm_dict)`: Extract and categorize detected sentiment words

## Output Format

The `analyze()` method returns a dictionary with:

```python
{
    'text': str,                    # Original text
    'final_sentiment': str,         # POSITIVE/NEGATIVE/NEUTRAL
    'final_score': float,           # Combined score (-1 to 1)
    'model_score': float,           # RoBERTa score (-1 to 1)
    'dict_score': float,            # Dictionary score
    'model_probs': tensor,          # Model probabilities [neg, neu, pos]
    'dict_details': {               # Dictionary analysis details
        'positive_count': int,
        'negative_count': int,
        'neutral_count': int,
        'total_words': int,
        'score': float
    },
    'breakdown': {                  # Contribution breakdown
        'model_contribution': float,
        'dict_contribution': float
    }
}
```

## Customization

### Adjusting Weights

```python
# More emphasis on dictionary
analyzer = HybridSentimentAnalyzer(
    lm_dict_path="LM-SA-2020.csv",
    model_weight=0.7,
    dict_weight=1.0
)

# Model-only analysis
analyzer = HybridSentimentAnalyzer(
    lm_dict_path="LM-SA-2020.csv",
    model_weight=1.0,
    dict_weight=0.0
)
```

### Sentiment Thresholds

The default thresholds for final sentiment classification:
- **POSITIVE**: score > 0.15
- **NEGATIVE**: score < -0.15
- **NEUTRAL**: -0.15 ≤ score ≤ 0.15

## Example Output

```
=== RESULTS ===
Final Sentiment: POSITIVE
Final Score: 0.2847

Details:
- Model Score: 0.4321
- Dictionary Score: 0.0873

Contributions:
- Model Contribution: 0.4321
- Dictionary Contribution: 0.0437

Dictionary Analysis:
- Positive words found: 3
- Negative words found: 1
- Neutral words found: 0
- Total Words Found: 45

Model Probabilities:
- NEGATIVE: 0.1234
- NEUTRAL: 0.2345
- POSITIVE: 0.6421

=== DETECTED WORDS ===
POSITIVE: growth, strong, optimistic
NEGATIVE: challenges
```

## Notes

- The model uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (update the model path in the code)
- Text is preprocessed with Porter Stemmer for better dictionary matching
- Neutral words are treated as slightly negative (weighted by 0.1)
- Maximum input length is 512 tokens for the transformer model

## File Structure

```
project/
├── Sentiment_analysis.ipynb                  # Main notebook
├── LM-SA-2020.csv                            # Loughran McDonald dictionary
├── roberta-base-sentiment-latest             # Model repository
└── README.md                                 # This file
```
