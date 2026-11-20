import re
from typing import List

def split_into_claims(text: str) -> List[str]:
    """
    Splits an LLM response into individual sentence claims for granular audit.
    Handles abbreviations and standard punctuation.
    """
    text = text.strip()
    if not text:
        return []
        
    # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Split on sentence boundaries (. ! ?) while avoiding splitting on common abbreviations like Dr., e.g., i.e.
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_endings.split(text)
    
    claims = [s.strip() for s in sentences if s.strip()]
    return claims if claims else [text]
