import re
from typing import List, Set, Tuple

def extract_entities_and_facts(text: str) -> Set[str]:
    """
    Extracts high-risk factual tokens from text:
    - Numbers, years, percentages, dollar amounts
    - Capitalized proper nouns / entity names (excluding standard sentence starters)
    """
    facts = set()
    
    # 1. Numbers, percentages, years, currency
    numbers_and_dates = re.findall(r'\b(?:\$?\d+(?:[.,]\d+)?%?|\d{4}s?)\b', text)
    for item in numbers_and_dates:
        facts.add(item.strip().lower())
        
    # 2. Capitalized phrases (Proper Nouns / Named Entities)
    # Match sequences of capitalized words like "Arthur's Magazine", "Albert Einstein", "Paris"
    capitalized = re.findall(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b', text)
    for cap in capitalized:
        cap_clean = cap.strip()
        # Filter out common single-word English sentence starters
        if cap_clean.lower() not in {"the", "this", "that", "there", "these", "it", "they", "he", "she", "what", "where", "when", "how", "why", "in", "on", "at", "yes", "no"}:
            facts.add(cap_clean.lower())
            
    return facts

def check_entity_grounding(context: str, claim: str) -> Tuple[float, List[str]]:
    """
    Checks if factual entities in the claim exist in the context.
    Returns:
        grounding_score: float (0.0 to 1.0)
        unsupported_entities: List of entities in claim that aren't in context
    """
    context_lower = context.lower()
    claim_entities = extract_entities_and_facts(claim)
    
    if not claim_entities:
        return 1.0, []
        
    unsupported = []
    supported_count = 0
    
    for entity in claim_entities:
        if entity in context_lower:
            supported_count += 1
        else:
            unsupported.append(entity)
            
    grounding_score = supported_count / len(claim_entities)
    return grounding_score, unsupported
