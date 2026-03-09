"""
Dictionary-based Complexity Analyzer using WordNet

This module builds a graph from dictionary definitions where:
- Words point to other defined words in their definitions
- Complexity is computed recursively:
  - Words with no defined words in definition → complexity 0
  - Otherwise → complexity = max(children's complexity) + 1

Uses lazy loading for memory efficiency.
"""

import nltk
from nltk.corpus import wordnet
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
import re


class DictionaryAnalyzer:
    """Analyzes word complexity based on dictionary definition dependencies."""
    
    def __init__(self):
        self.word_definitions: Dict[str, str] = {}  # Keep only first definition per word
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.complexity_cache: Dict[str, int] = {}
        self.all_defined_words: Set[str] = set()
        self._built = False
        
    def _ensure_built(self, max_words: int = 500):
        """Lazy build the dictionary on first use."""
        if not self._built:
            self.build_dictionary(max_words)
            self._built = True
        
    def build_dictionary(self, max_words: int = 500) -> None:
        """Build dictionary from WordNet with up to max_words entries."""
        print(f"Building dictionary from WordNet (up to {max_words} words)...")
        
        collected = 0
        seen_words = set()
        
        # Use iterator to avoid loading all synsets at once
        try:
            all_synsets = wordnet.all_synsets()
        except:
            # Fallback: create minimal test dictionary
            print("WordNet not available, using fallback dictionary...")
            self._create_fallback_dictionary()
            return
        
        for synset in all_synsets:
            if collected >= max_words:
                break
                
            # Get lemma names (words)
            lemmas = synset.lemma_names()
            
            # Get definition
            definition = synset.definition()
            
            # Process each lemma
            for lemma in lemmas:
                word = lemma.lower().replace('_', ' ')
                
                # Skip very short or very long words, or already seen
                if len(word) < 2 or len(word) > 20:
                    continue
                if word in seen_words:
                    continue
                    
                # Store only first definition per word (memory efficient)
                self.word_definitions[word] = definition
                self.all_defined_words.add(word)
                seen_words.add(word)
                collected += 1
                    
                if collected >= max_words:
                    break
        
        print(f"Collected {len(self.word_definitions)} unique words")
        self._build_dependency_graph()
    
    def _create_fallback_dictionary(self):
        """Create a minimal fallback dictionary for testing."""
        fallback = {
            'cat': 'a small domesticated carnivorous mammal',
            'dog': 'a domesticated carnivorous mammal',
            'mat': 'a piece of coarse material placed on a floor',
            'sat': 'past tense of sit',
            'data': 'facts and statistics collected together for reference or analysis',
            'fact': 'a thing that is known or proved to be true',
            'true': 'in accordance with fact or reality',
            'analysis': 'detailed examination of the elements or structure of something',
            'analyze': 'examine methodically and in detail',
            'information': 'facts provided or learned about something or someone',
            'complex': 'consisting of many different and connected parts',
            'system': 'a set of things working together as parts of a mechanism',
            'simple': 'easily understood or done; presenting no difficulty',
            'basic': 'forming an essential foundation or starting point',
            'word': 'a single distinct meaningful element of speech or writing',
            'definition': 'a statement of the exact meaning of a word',
            'meaning': 'what is meant by a word, text, concept, or action',
            'reality': 'the world or the state of things as they actually exist',
            'existence': 'the fact or state of living or having objective reality',
            'knowledge': 'facts, information, and skills acquired through experience',
        }
        
        for word, definition in fallback.items():
            self.word_definitions[word] = definition
            self.all_defined_words.add(word)
        
        print(f"Created fallback dictionary with {len(self.word_definitions)} words")
        self._build_dependency_graph()
        
    def _normalize_word(self, text: str) -> str:
        """Normalize text to extract base words."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except spaces and hyphens
        text = re.sub(r'[^\w\s\-]', '', text)
        return text
    
    def _extract_defined_words(self, definition: str) -> Set[str]:
        """Extract words from definition that exist in our dictionary."""
        normalized = self._normalize_word(definition)
        words = set(normalized.split())
        
        # Filter to only words we have definitions for
        defined_in_dict = words.intersection(self.all_defined_words)
        return defined_in_dict
    
    def _build_dependency_graph(self) -> None:
        """Build graph where edges point from word to words in its definition."""
        print("Building dependency graph...")
        
        for word, definition in self.word_definitions.items():
            defined_words = self._extract_defined_words(definition)
            # Don't include self-references
            defined_words.discard(word)
            self.dependency_graph[word].update(defined_words)
        
        print(f"Built graph with {len(self.dependency_graph)} nodes")
    
    def compute_complexity(self, word: str, memo: Optional[Dict[str, int]] = None) -> int:
        """
        Compute complexity of a word recursively.
        
        - If word has no dependencies (no defined words in definition) → complexity 0
        - Otherwise → complexity = max(dependencies' complexity) + 1
        
        Uses memoization to avoid recomputation.
        """
        if memo is None:
            memo = {}
        
        if word in memo:
            return memo[word]
        
        # Base case: word not in dictionary or no dependencies
        if word not in self.dependency_graph or len(self.dependency_graph[word]) == 0:
            memo[word] = 0
            return 0
        
        # Prevent infinite recursion for circular dependencies
        memo[word] = -1  # Mark as being computed
        
        max_child_complexity = 0
        for child in self.dependency_graph[word]:
            child_complexity = self.compute_complexity(child, memo)
            if child_complexity >= 0:  # Skip cycles
                max_child_complexity = max(max_child_complexity, child_complexity)
        
        # If we detected a cycle, use 0 for that path
        if max_child_complexity == -1:
            max_child_complexity = 0
        
        complexity = max_child_complexity + 1
        memo[word] = complexity
        return complexity
    
    def get_word_complexity(self, word: str) -> int:
        """Get cached complexity for a word, computing if necessary."""
        word = word.lower().strip()
        
        if word in self.complexity_cache:
            return self.complexity_cache[word]
        
        complexity = self.compute_complexity(word)
        self.complexity_cache[word] = complexity
        return complexity
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze complexity of all words in a text.
        
        Returns dict with:
        - word_complexities: {word: complexity}
        - avg_complexity: average complexity
        - max_complexity: maximum complexity found
        - complexity_distribution: {complexity_level: count}
        """
        normalized = self._normalize_word(text)
        words = [w for w in normalized.split() if len(w) > 1]
        
        word_complexities = {}
        complexity_distribution = defaultdict(int)
        
        for word in words:
            if word in self.all_defined_words:
                complexity = self.get_word_complexity(word)
                word_complexities[word] = complexity
                complexity_distribution[complexity] += 1
        
        if not word_complexities:
            return {
                'word_complexities': {},
                'avg_complexity': 0,
                'max_complexity': 0,
                'complexity_distribution': {},
                'words_analyzed': 0
            }
        
        complexities = list(word_complexities.values())
        return {
            'word_complexities': word_complexities,
            'avg_complexity': sum(complexities) / len(complexities),
            'max_complexity': max(complexities),
            'complexity_distribution': dict(complexity_distribution),
            'words_analyzed': len(word_complexities)
        }
    
    def get_complexity_examples(self, n_per_level: int = 5) -> Dict[int, List[Tuple[str, int]]]:
        """Get example words for each complexity level."""
        examples = defaultdict(list)
        
        # Sample words to check
        sample_words = list(self.all_defined_words)[:1000]
        
        for word in sample_words:
            complexity = self.get_word_complexity(word)
            if len(examples[complexity]) < n_per_level:
                examples[complexity].append((word, complexity))
        
        return dict(examples)
    
    def visualize_dependencies(self, word: str, max_depth: int = 3) -> str:
        """Create a text visualization of word dependencies up to max_depth."""
        lines = []
        visited = set()
        
        def traverse(w: str, depth: int, indent: str = ""):
            if depth > max_depth or w in visited:
                return
            
            visited.add(w)
            complexity = self.get_word_complexity(w)
            lines.append(f"{indent}{w} (complexity={complexity})")
            
            deps = sorted(self.dependency_graph.get(w, set()))[:5]  # Limit children
            for i, dep in enumerate(deps):
                is_last = (i == len(deps) - 1)
                prefix = "└─ " if is_last else "├─ "
                traverse(dep, depth + 1, indent + ("   " if is_last else "│  "))
        
        traverse(word, 0)
        return "\n".join(lines)


def main():
    """Demonstrate the DictionaryAnalyzer."""
    print("=" * 70)
    print("DICTIONARY-BASED COMPLEXITY ANALYZER")
    print("=" * 70)
    
    analyzer = DictionaryAnalyzer()
    # Use fallback dictionary (WordNet causes OOM on this system)
    analyzer._create_fallback_dictionary()
    
    print("\n" + "=" * 70)
    print("COMPLEXITY EXAMPLES BY LEVEL")
    print("=" * 70)
    
    examples = analyzer.get_complexity_examples(n_per_level=3)
    for complexity in sorted(examples.keys()):
        words = examples[complexity]
        word_list = ", ".join([f"'{w}'" for w, _ in words])
        print(f"Complexity {complexity}: {word_list}")
    
    print("\n" + "=" * 70)
    print("TEXT ANALYSIS EXAMPLES")
    print("=" * 70)
    
    test_texts = [
        "The cat sat on the mat",
        "Data analysis requires information",
        "True facts about reality",
        "Complex systems need knowledge"
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        result = analyzer.analyze_text(text)
        print(f"  Words analyzed: {result['words_analyzed']}")
        print(f"  Average complexity: {result['avg_complexity']:.2f}")
        print(f"  Max complexity: {result['max_complexity']}")
        if result['word_complexities']:
            details = ", ".join([f"{w}={c}" for w, c in result['word_complexities'].items()])
            print(f"  Word complexities: {details}")
    
    print("\n" + "=" * 70)
    print("DEPENDENCY GRAPH VISUALIZATION")
    print("=" * 70)
    
    demo_words = ['information', 'data', 'analyze', 'true', 'fact', 'knowledge']
    for word in demo_words:
        if word in analyzer.all_defined_words:
            print(f"\n'{word}' dependencies:")
            viz = analyzer.visualize_dependencies(word, max_depth=2)
            print(viz)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
