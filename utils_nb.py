import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources automatically
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def process_tweet(text):
    """
    Cleans and tokenizes raw text/lyrics:
    1. Converts to lowercase
    2. Removes URLs, punctuation, and digits
    3. Tokenizes into words
    4. Removes English stop words
    """
    if not isinstance(text, str):
        return []
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Clean URLs, punctuation, and digits
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove Stop Words
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return cleaned_tokens