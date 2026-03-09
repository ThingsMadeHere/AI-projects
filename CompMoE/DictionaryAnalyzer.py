from collections import defaultdict, deque
from typing import Dict, List, Set, Optional
import re

class DictionaryGraph:
    """
    Builds a knowledge graph from dictionary definitions with:
    - Parent-Child relationships (definition dependencies)
    - Sibling/Analogy relationships (shared parents/hypernyms)
    - Similarity relationships (semantic overlap)
    - Complexity scoring based on definition depth
    """
    
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.complexity_cache: Dict[str, int] = {}
        
    def build_graph(self, custom_dict: Dict[str, str] = None):
        """Build graph from custom dictionary or use built-in"""
        if custom_dict is None:
            custom_dict = self._get_builtin_dict()
        
        print(f"Building Dictionary Graph with {len(custom_dict)} words...")
        
        # Initialize nodes
        for word, definition in custom_dict.items():
            word = word.lower()
            self.nodes[word] = {
                'definition': definition,
                'complexity': None,
                'children': set(),
                'parents': set(),
                'siblings': set(),
                'similar': set(),
                'hypernyms': set(),
                'hyponyms': set()
            }
        
        # Build dependency edges from definitions
        for word, node in self.nodes.items():
            def_words = set(re.findall(r'\b[a-z]+\b', node['definition'].lower()))
            for def_word in def_words:
                if def_word in self.nodes and def_word != word:
                    self.nodes[word]['children'].add(def_word)
                    self.nodes[def_word]['parents'].add(word)
        
        # Add explicit hypernym/hyponym relationships
        self._add_taxonomy_relationships()
        
        # Compute complexity
        for word in self.nodes:
            self._compute_complexity(word)
        
        # Find siblings and similar words
        self._find_siblings()
        self._find_similar()
        
        print(f"Graph built successfully with {len(self.nodes)} nodes.")
        self._print_stats()
    
    def _get_builtin_dict(self) -> Dict[str, str]:
        """Built-in dictionary with common terms"""
        return {
            # Base concepts (C0)
            'reality': 'the state of things as they actually exist',
            'existence': 'the fact or state of living or having objective reality',
            'entity': 'a thing with distinct and independent existence',
            'object': 'a material thing that can be seen and touched',
            'being': 'the nature or essence of a thing',
            
            # Simple concepts (C1)
            'fact': 'a thing that is known to be consistent with objective reality',
            'truth': 'that which is true or in accordance with fact or reality',
            'life': 'the existence of an individual human being or animal',
            'death': 'the end of life',
            'time': 'the indefinite continued progress of existence',
            'space': 'a continuous area or expanse that is free or unoccupied',
            
            # Medium concepts (C2)
            'knowledge': 'facts, information, and skills acquired through experience or education',
            'information': 'facts provided or learned about something or someone',
            'data': 'facts and statistics collected together for reference or analysis',
            'analysis': 'detailed examination of the elements or structure of something',
            
            # Complex concepts (C3+)
            'science': 'the intellectual and practical activity encompassing the systematic study of the structure and behavior of the physical and natural world through observation and experiment',
            'philosophy': 'the study of the fundamental nature of knowledge, reality, and existence',
            'consciousness': 'the state of being aware of and responsive to one\'s surroundings',
            'intelligence': 'the ability to acquire and apply knowledge and skills',
            
            # Animals and categories
            'animal': 'a living organism that feeds on organic matter and typically has specialized sense organs',
            'mammal': 'an animal that nourishes its young with milk and has hair',
            'dog': 'a domesticated carnivorous mammal',
            'cat': 'a small domesticated carnivorous mammal',
            'human': 'a human being; a person',
            'person': 'a human being regarded as an individual',
            
            # More relationships
            'cause': 'a person or thing that gives rise to an action or phenomenon',
            'effect': 'a change which is a result or consequence of an action',
            'good': 'to be desired or approved of',
            'bad': 'of poor quality or low standard',
        }
    
    def _add_taxonomy_relationships(self):
        """Add explicit taxonomy (hypernym/hyponym) relationships"""
        taxonomy = {
            'dog': ['mammal', 'animal'],
            'cat': ['mammal', 'animal'],
            'human': ['mammal', 'animal'],
            'mammal': ['animal'],
            'person': ['human'],
        }
        
        for child, parents in taxonomy.items():
            if child in self.nodes:
                for parent in parents:
                    if parent in self.nodes:
                        self.nodes[child]['hypernyms'].add(parent)
                        self.nodes[child]['children'].add(parent)
                        self.nodes[parent]['hyponyms'].add(child)
                        self.nodes[parent]['parents'].add(child)
    
    def _compute_complexity(self, word: str, visited: Set[str] = None) -> int:
        """Compute complexity recursively: max(children's complexity) + 1"""
        if visited is None:
            visited = set()
        
        if word in self.complexity_cache:
            return self.complexity_cache[word]
        
        if word in visited:
            return 0
        
        visited.add(word)
        
        node = self.nodes.get(word)
        if not node:
            return 0
        
        children = node['children']
        if not children:
            complexity = 0
        else:
            valid_children = [c for c in children if c in self.nodes]
            if not valid_children:
                complexity = 0
            else:
                max_child = max(self._compute_complexity(c, visited.copy()) for c in valid_children)
                complexity = max_child + 1
        
        self.complexity_cache[word] = complexity
        node['complexity'] = complexity
        return complexity
    
    def _find_siblings(self):
        """Find siblings: words that share parents or hypernyms"""
        # Group by shared parents
        parent_to_children = defaultdict(set)
        for word, node in self.nodes.items():
            for parent in node['parents']:
                if parent in self.nodes:
                    parent_to_children[parent].add(word)
        
        for parent, children in parent_to_children.items():
            children_list = list(children)
            for i, c1 in enumerate(children_list):
                for c2 in children_list[i+1:]:
                    self.nodes[c1]['siblings'].add(c2)
                    self.nodes[c2]['siblings'].add(c1)
        
        # Group by shared hypernyms
        hyper_to_hypos = defaultdict(set)
        for word, node in self.nodes.items():
            for hyper in node['hypernyms']:
                if hyper in self.nodes:
                    hyper_to_hypos[hyper].add(word)
        
        for hyper, hypos in hyper_to_hypos.items():
            hypo_list = list(hypos)
            for i, h1 in enumerate(hypo_list):
                for h2 in hypo_list[i+1:]:
                    self.nodes[h1]['siblings'].add(h2)
                    self.nodes[h2]['siblings'].add(h1)
    
    def _find_similar(self):
        """Find similar words based on overlapping definitions"""
        words_list = list(self.nodes.keys())
        for i, w1 in enumerate(words_list):
            for w2 in words_list[i+1:]:
                if w2 in self.nodes[w1]['siblings']:
                    continue
                
                def1 = set(re.findall(r'\b[a-z]+\b', self.nodes[w1]['definition'].lower()))
                def2 = set(re.findall(r'\b[a-z]+\b', self.nodes[w2]['definition'].lower()))
                
                overlap = len(def1 & def2)
                total = len(def1 | def2)
                
                if total > 0 and overlap / total > 0.2:
                    self.nodes[w1]['similar'].add(w2)
                    self.nodes[w2]['similar'].add(w1)
    
    def _print_stats(self):
        """Print graph statistics"""
        complexities = [n['complexity'] for n in self.nodes.values() if n['complexity'] is not None]
        if not complexities:
            return
        
        print(f"Complexity range: {min(complexities)} to {max(complexities)}")
        most_complex = max(self.nodes.items(), key=lambda x: x[1]['complexity'])
        print(f"Most complex word: '{most_complex[0]}' (C{most_complex[1]['complexity']})")
        
        total_sib = sum(len(n['siblings']) for n in self.nodes.values()) // 2
        total_sim = sum(len(n['similar']) for n in self.nodes.values()) // 2
        print(f"Sibling relationships: {total_sib}, Similarity relationships: {total_sim}")
    
    def get_word_info(self, word: str) -> Optional[dict]:
        return self.nodes.get(word)
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two words using BFS"""
        if start not in self.nodes or end not in self.nodes:
            return None
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            
            neighbors = (self.nodes[current]['children'] | 
                        self.nodes[current]['parents'] | 
                        self.nodes[current]['siblings'])
            
            for neighbor in neighbors:
                if neighbor in self.nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def analyze_text(self, text: str) -> dict:
        """Analyze text complexity"""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        complexities = []
        details = {}
        
        for word in words:
            if word in self.nodes and self.nodes[word]['complexity'] is not None:
                c = self.nodes[word]['complexity']
                complexities.append(c)
                details[word] = c
        
        if not complexities:
            return {'avg_complexity': 0, 'max_complexity': 0, 'word_count': 0, 'word_details': {}}
        
        return {
            'avg_complexity': sum(complexities) / len(complexities),
            'max_complexity': max(complexities),
            'word_count': len(complexities),
            'word_details': details
        }


class QuestionParser:
    def __init__(self, graph: DictionaryGraph):
        self.graph = graph
    
    def parse(self, question: str) -> str:
        q = question.lower().strip()
        
        # Most complex - check FIRST before other patterns
        if 'most complex' in q or ('complex' in q and 'most' in q):
            best = max(self.graph.nodes.items(), key=lambda x: x[1]['complexity'] or 0)
            return f"The most complex word is '{best[0]}' (C{best[1]['complexity']})."
        
        # Definition: "What is X?" or "What is the definition of X?" (but not "most complex")
        m = re.search(r'(?:what is the definition of|what is)\s+([a-z_]+)', q)
        if m:
            word = m.group(1).replace('_', ' ')
            info = self.graph.get_word_info(word)
            if info:
                return f"{word.capitalize()}: {info['definition']} (Complexity: C{info['complexity']})"
            return f"I don't have '{word}' in my dictionary."
        
        # Complexity: "How complex is X?"
        m = re.search(r'(?:how complex is|complexity of)\s+([a-z_]+)', q)
        if m:
            word = m.group(1).replace('_', ' ')
            info = self.graph.get_word_info(word)
            if info and info['complexity'] is not None:
                return f"'{word}' has complexity C{info['complexity']}."
            return f"I don't know the complexity of '{word}'."
        
        # Relationships: "How is X related to Y?"
        m = re.search(r'how is\s+([a-z_]+)\s+related to\s+([a-z_]+)', q)
        if m:
            w1, w2 = m.group(1).replace('_', ' '), m.group(2).replace('_', ' ')
            info1, info2 = self.graph.get_word_info(w1), self.graph.get_word_info(w2)
            if info1 and info2:
                rels = []
                if w2 in info1['children']: rels.append(f"'{w1}' uses '{w2}' in its definition")
                if w2 in info1['parents']: rels.append(f"'{w1}' helps define '{w2}'")
                if w2 in info1['siblings']: rels.append(f"They share categories")
                if w2 in info1['similar']: rels.append(f"They are semantically similar")
                if w2 in info1['hypernyms']: rels.append(f"'{w2}' is a broader category")
                if w2 in info1['hyponyms']: rels.append(f"'{w2}' is more specific")
                
                if rels:
                    return "; ".join(rels) + "."
                
                path = self.graph.find_path(w1, w2)
                if path and len(path) <= 4:
                    return f"Connected via: {' → '.join(path)}"
            return f"No clear relationship found."
        
        # Is-a: "Is X a Y?" (but not "is to") - handle "an" as well as "a"
        if ' is to ' not in q:
            m = re.search(r'\bis\s+([a-z_]+)\s+a(?:n)?\s+([a-z_]+)', q)
            if m:
                w1, w2 = m.group(1).replace('_', ' '), m.group(2).replace('_', ' ')
                info1, info2 = self.graph.get_word_info(w1), self.graph.get_word_info(w2)
                if info1 and info2:
                    if w2 in info1['hypernyms'] or w2 in info1['children']:
                        return f"Yes, '{w1}' is a type of '{w2}'."
                    if w1 in info2['hyponyms']:
                        return f"Yes, '{w1}' is a type of '{w2}'."
                    if w2 in info1['siblings']:
                        return f"They are in the same category."
                return f"I cannot confirm that relationship."
        
        # Examples: "Give me an example of X"
        m = re.search(r'(?:give me an example of|examples? of)\s+([a-z_]+)', q)
        if m:
            word = m.group(1).replace('_', ' ')
            info = self.graph.get_word_info(word)
            if info and info['hyponyms']:
                examples = list(info['hyponyms'])[:5]
                return f"Examples of '{word}': {', '.join(examples)}."
            return f"I don't have specific examples for '{word}'."
        
        # Path: "Explain the path from X to Y"
        m = re.search(r'(?:explain the path from|path from)\s+([a-z_]+)\s+to\s+([a-z_]+)', q)
        if m:
            w1, w2 = m.group(1).replace('_', ' '), m.group(2).replace('_', ' ')
            path = self.graph.find_path(w1, w2)
            if path:
                return f"Path: {' → '.join(path)}"
            return f"No path found."
        
        # Analogy: "X is to Y as A is to B"
        m = re.search(r'([a-z_]+)\s+is to\s+([a-z_]+)\s+as\s+([a-z_]+)\s+is to\s+([a-z_]+)', q)
        if m:
            w1, w2, w3, w4 = [x.replace('_', ' ') for x in m.groups()]
            infos = [self.graph.get_word_info(w) for w in [w1, w2, w3, w4]]
            if all(infos):
                # Check relationship types
                def get_rel(a, b, info_a):
                    if b in info_a['hypernyms'] or b in info_a['children']: return 'child'
                    if b in info_a['hyponyms'] or b in info_a['parents']: return 'parent'
                    if b in info_a['siblings']: return 'sibling'
                    return None
                
                r1 = get_rel(w1, w2, infos[0])
                r2 = get_rel(w3, w4, infos[2])
                
                if r1 and r2 and r1 == r2:
                    return f"Valid analogy! Both are '{r1}' relationships."
                elif r1 and r2:
                    return f"Not quite: '{w1}→{w2}' is {r1}, but '{w3}→{w4}' is {r2}."
            return "Cannot verify analogy."
        
        # Most complex - multiple patterns
        if 'most complex' in q or ('complex' in q and 'most' in q):
            best = max(self.graph.nodes.items(), key=lambda x: x[1]['complexity'] or 0)
            return f"The most complex word is '{best[0]}' (C{best[1]['complexity']})."
        
        # Siblings/Similar
        m = re.search(r'(?:siblings|similar) of\s+([a-z_]+)', q)
        if m:
            word = m.group(1).replace('_', ' ')
            info = self.graph.get_word_info(word)
            if info:
                sibs = list(info['siblings'])[:5]
                sims = list(info['similar'])[:5]
                result = []
                if sibs: result.append(f"Siblings: {', '.join(sibs)}")
                if sims: result.append(f"Similar: {', '.join(sims)}")
                return "; ".join(result) if result else f"No relationships found for '{word}'."
        
        return "Try asking about definitions, complexity, relationships, examples, paths, analogies, or 'is a' questions!"


if __name__ == "__main__":
    # Build graph
    graph = DictionaryGraph()
    graph.build_graph()
    
    parser = QuestionParser(graph)
    
    # Demo questions
    print("\n--- Q&A Demo ---\n")
    demos = [
        "What is the definition of life?",
        "How complex is knowledge?",
        "How is data related to information?",
        "Explain the path from dog to reality",
        "Is a dog an animal?",
        "Give me an example of mammal",
        "What is the most complex word?",
        "Dog is to mammal as cat is to mammal"
    ]
    
    for q in demos:
        print(f"Q: {q}")
        print(f"A: {parser.parse(q)}\n")
    
    # Show sample nodes
    print("--- Sample Nodes ---\n")
    for word in ['dog', 'mammal', 'knowledge', 'reality'][:4]:
        node = graph.nodes[word]
        print(f"{word} (C{node['complexity']}): {node['definition'][:60]}...")
        print(f"  Hypernyms: {list(node['hypernyms'])[:3]}")
        print(f"  Hyponyms: {list(node['hyponyms'])[:3]}")
        print(f"  Siblings: {list(node['siblings'])[:3]}\n")
    
    # Interactive
    print("Enter questions (type 'exit' to quit):")
    while True:
        try:
            inp = input("\nQ: ").strip()
            if inp.lower() == 'exit':
                break
            if inp:
                print(f"A: {parser.parse(inp)}")
        except EOFError:
            break
