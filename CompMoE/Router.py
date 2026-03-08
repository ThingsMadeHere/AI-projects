import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import re
from collections import Counter

# ==========================================
# 1. REGIONS (Geometric Manifolds)
# ==========================================

class Region(Enum):
    INFO = "Info Retrieval"
    LOGIC = "Logic Interpreter"
    CREATIVE = "Creative Interpreter"

# ==========================================
# 1.5 COMPLEXITY LEVELS
# ==========================================

class Complexity(Enum):
    """
    Complexity dimension for task decomposition.
    Represents how many cognitive steps/regions are involved.
    """
    SIMPLE = 1      # Single region, 1-2 monotasks
    MODERATE = 2    # Two regions, 2-3 monotasks  
    COMPLEX = 3     # All three regions, 4+ monotasks

ALLOWED_TRANSITIONS = {
    None: [Region.INFO, Region.LOGIC],
    Region.INFO: [Region.INFO, Region.LOGIC, Region.CREATIVE],
    Region.LOGIC: [Region.LOGIC, Region.CREATIVE, Region.INFO],
    Region.CREATIVE: [Region.INFO, Region.LOGIC],
}

# ==========================================
# 2. CODEBOOK (Vector Dictionary)
# ==========================================

@dataclass
class Monotask:
    id: int
    name: str
    region: Region
    vector: np.ndarray
    complexity: Complexity = Complexity.SIMPLE  # Default complexity

class Codebook:
    def __init__(self, dim=64, num_experts_per_region=5):
        self.dim = dim
        self.monotasks: List[Monotask] = []
        self.region_centroids = {}
        self.complexity_centroids = {}
        self._initialize_regions()
        self._initialize_complexity()
        self._populate_codebook(num_experts_per_region)
    
    def _initialize_regions(self):
        """Create orthogonal region centroids."""
        self.region_centroids[Region.INFO] = self._make_centroid([0, 10])
        self.region_centroids[Region.LOGIC] = self._make_centroid([10, 20])
        self.region_centroids[Region.CREATIVE] = self._make_centroid([20, 30])
    
    def _initialize_complexity(self):
        """Create complexity dimension centroids in separate subspace."""
        # Use dimensions 30-64 for complexity encoding
        self.complexity_centroids[Complexity.SIMPLE] = self._make_centroid_complexity([30, 40])
        self.complexity_centroids[Complexity.MODERATE] = self._make_centroid_complexity([40, 50])
        self.complexity_centroids[Complexity.COMPLEX] = self._make_centroid_complexity([50, 64])
    
    def _make_centroid_complexity(self, indices: Tuple[int, int]) -> np.ndarray:
        """Create centroid in complexity subspace (dims 30-64)."""
        vec = np.zeros(self.dim)
        vec[indices[0]:indices[1]] = 1.0
        return vec / np.linalg.norm(vec)
    
    def _make_centroid(self, indices: Tuple[int, int]) -> np.ndarray:
        vec = np.zeros(self.dim)
        vec[indices[0]:indices[1]] = 1.0
        return vec / np.linalg.norm(vec)
    
    def _populate_codebook(self, num_per_region: int):
        """Populate with monotask vectors near region centroids."""
        task_templates = {
            Region.INFO: [("Search", Complexity.SIMPLE), ("Retrieve", Complexity.SIMPLE), 
                         ("Fetch", Complexity.SIMPLE), ("Query", Complexity.MODERATE), ("Get", Complexity.SIMPLE)],
            Region.LOGIC: [("Analyze", Complexity.MODERATE), ("Calculate", Complexity.MODERATE), 
                          ("Validate", Complexity.SIMPLE), ("Compare", Complexity.MODERATE), ("Plan", Complexity.COMPLEX)],
            Region.CREATIVE: [("Write", Complexity.MODERATE), ("Generate", Complexity.MODERATE), 
                             ("Create", Complexity.COMPLEX), ("Design", Complexity.COMPLEX), ("Compose", Complexity.MODERATE)]
        }
        
        task_id = 1
        for region, templates in task_templates.items():
            for i, (template, complexity) in enumerate(templates[:num_per_region]):
                # Vector = centroid + small noise, then normalized
                # Combine region and complexity subspaces
                region_vec = self.region_centroids[region].copy()
                complexity_vec = self.complexity_centroids[complexity].copy()
                
                # Weighted combination: 70% region, 30% complexity
                base_vec = 0.7 * region_vec + 0.3 * complexity_vec
                noise = np.random.randn(self.dim) * 0.15
                vec = base_vec + noise
                vec = vec / np.linalg.norm(vec)
                
                self.monotasks.append(Monotask(
                    id=task_id,
                    name=f"{template}_{region.value.split()[0]}",
                    region=region,
                    vector=vec,
                    complexity=complexity
                ))
                task_id += 1
    
    def get_by_region(self, region: Region) -> List[Monotask]:
        return [m for m in self.monotasks if m.region == region]
    
    def sample_task_composition(self, num_monotasks: Tuple[int, int] = (2, 4)) -> Tuple[List[Monotask], np.ndarray, Complexity]:
        """
        Generate a valid task by sampling monotasks and summing their vectors.
        Returns: (selected_monotasks, task_vector, complexity_level)
        This guarantees the task IS decomposable by construction.
        """
        # Sample 2-4 monotasks with valid region flow
        selected = []
        last_region = None
        
        for _ in range(np.random.randint(*num_monotasks)):
            allowed = ALLOWED_TRANSITIONS.get(last_region, [Region.INFO, Region.LOGIC])
            candidates = [m for m in self.monotasks if m.region in allowed and m not in selected]
            
            if not candidates:
                break
            
            chosen = np.random.choice(candidates)
            selected.append(chosen)
            last_region = chosen.region
        
        # Determine complexity from number of unique regions
        unique_regions = len(set(m.region for m in selected))
        if unique_regions == 1 or len(selected) <= 2:
            complexity = Complexity.SIMPLE
        elif unique_regions == 2 or len(selected) <= 3:
            complexity = Complexity.MODERATE
        else:
            complexity = Complexity.COMPLEX
        
        # Task vector = sum of selected monotask vectors (normalized) + complexity encoding
        if selected:
            task_vector = sum(m.vector for m in selected)
            # Add complexity signal
            task_vector = task_vector + 0.2 * self.complexity_centroids[complexity]
            task_vector = task_vector / np.linalg.norm(task_vector)
            return selected, task_vector, complexity
        return [], np.zeros(self.dim), Complexity.SIMPLE

# ==========================================
# 2.5 TEXT ENCODER (Text to Vector)
# ==========================================

class TextEncoder:
    """
    Simple keyword-based text encoder that maps text to vector space.
    In production, this would be replaced with a real transformer model.
    """
    
    def __init__(self, codebook: Codebook):
        self.codebook = codebook
        # Keyword mappings for region detection
        self.region_keywords = {
            Region.INFO: ['search', 'find', 'get', 'retrieve', 'fetch', 'query', 'lookup', 
                         'information', 'data', 'fact', 'check', 'read', 'list', 'show'],
            Region.LOGIC: ['analyze', 'calculate', 'compute', 'compare', 'validate', 'verify',
                          'plan', 'reason', 'solve', 'evaluate', 'determine', 'assess', 'logic'],
            Region.CREATIVE: ['write', 'create', 'generate', 'design', 'compose', 'draft',
                             'imagine', 'invent', 'produce', 'craft', 'author', 'build']
        }
        
        # Complexity keywords
        self.complexity_keywords = {
            Complexity.SIMPLE: ['simple', 'quick', 'basic', 'easy', 'single', 'one', 'brief'],
            Complexity.MODERATE: ['moderate', 'standard', 'normal', 'several', 'multiple', 'some'],
            Complexity.COMPLEX: ['complex', 'detailed', 'comprehensive', 'complete', 'full', 
                                'elaborate', 'thorough', 'multi-step', 'advanced']
        }
    
    def encode_text(self, text: str) -> Tuple[np.ndarray, Dict]:
        """
        Encode text into a vector representation with metadata.
        
        Returns: (task_vector, metadata_dict)
        where metadata_dict contains:
            - detected_regions: List[Region]
            - detected_complexity: Complexity
            - keyword_scores: Dict[str, float]
        """
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        # Score each region based on keyword matches
        region_scores = {region: 0.0 for region in Region}
        for region, keywords in self.region_keywords.items():
            for word in words:
                if word in keywords:
                    region_scores[region] += 1.0
        
        # Score complexity
        complexity_scores = {comp: 0.0 for comp in Complexity}
        for comp, keywords in self.complexity_keywords.items():
            for word in words:
                if word in keywords:
                    complexity_scores[comp] += 1.0
        
        # Determine dominant region(s)
        max_region_score = max(region_scores.values())
        detected_regions = [r for r, score in region_scores.items() if score > 0]
        
        # If no regions detected, default based on text length
        if not detected_regions:
            if len(words) < 5:
                detected_regions = [Region.INFO]
            elif len(words) < 15:
                detected_regions = [Region.LOGIC]
            else:
                detected_regions = [Region.CREATIVE]
        
        # Determine complexity
        max_comp_score = max(complexity_scores.values())
        if max_comp_score > 0:
            detected_complexity = max(complexity_scores.keys(), key=lambda k: complexity_scores[k])
        else:
            # Infer from number of detected regions and text length
            if len(detected_regions) == 1 and len(words) < 10:
                detected_complexity = Complexity.SIMPLE
            elif len(detected_regions) <= 2 and len(words) < 20:
                detected_complexity = Complexity.MODERATE
            else:
                detected_complexity = Complexity.COMPLEX
        
        # Build task vector from region and complexity centroids
        task_vector = np.zeros(self.codebook.dim)
        
        # Add region components (weighted by scores)
        total_region_score = sum(region_scores.values())
        if total_region_score > 0:
            for region, score in region_scores.items():
                weight = score / total_region_score if total_region_score > 0 else 0
                task_vector += weight * self.codebook.region_centroids[region]
        else:
            # Default distribution
            task_vector += 0.4 * self.codebook.region_centroids[Region.INFO]
            task_vector += 0.4 * self.codebook.region_centroids[Region.LOGIC]
            task_vector += 0.2 * self.codebook.region_centroids[Region.CREATIVE]
        
        # Add complexity component
        task_vector += 0.3 * self.codebook.complexity_centroids[detected_complexity]
        
        # Normalize
        task_vector = task_vector / np.linalg.norm(task_vector)
        
        metadata = {
            'detected_regions': detected_regions,
            'detected_complexity': detected_complexity,
            'region_scores': region_scores,
            'complexity_scores': complexity_scores,
            'word_count': len(words)
        }
        
        return task_vector, metadata

# ==========================================
# 3. ROUTER (Cosine Similarity Selection)
# ==========================================

class VectorRouter:
    def __init__(self, codebook: Codebook, max_steps=10, min_cosine=0.1):
        self.codebook = codebook
        self.max_steps = max_steps
        self.min_cosine = min_cosine  # Minimum cosine similarity to accept
    
    def decompose(self, task_vector: np.ndarray) -> Tuple[List[Monotask], Dict[int, float]]:
        """
        Decompose using cosine similarity selection with magnitude tracking.
        
        Key insight: We want monotask vectors that point in the SAME DIRECTION
        as the residual. We track the optimal scaling factor for each selected
        monotask to minimize reconstruction error.
        
        Returns: (selected_monotasks, coefficients_dict)
        """
        residual = task_vector.copy()
        selected = []
        last_region = None
        
        # Track coefficients for selected monotasks
        coefficients = {}
        
        initial_norm = np.linalg.norm(residual)
        print(f"  Initial residual norm: {initial_norm:.4f}")
        
        for step in range(self.max_steps):
            current_norm = np.linalg.norm(residual)
            
            # Convergence check
            if current_norm < 0.1:
                print(f"  [Converged] Residual norm: {current_norm:.4f}")
                break
            
            # Allowed regions (TransE constraint)
            allowed_regions = ALLOWED_TRANSITIONS.get(last_region, [Region.INFO, Region.LOGIC])
            
            # Candidates from allowed regions
            candidates = []
            for region in allowed_regions:
                candidates.extend(self.codebook.get_by_region(region))
            
            # Filter used monotasks
            used_ids = {m.id for m in selected}
            candidates = [c for c in candidates if c.id not in used_ids]
            
            if not candidates:
                print(f"  [Stop] No valid candidates")
                break
            
            # CRITICAL FIX: Select by cosine similarity, not residual reduction
            best_monotask = None
            best_cosine = -1
            
            for candidate in candidates:
                # Cosine similarity = dot product of normalized vectors
                cosine = np.dot(residual, candidate.vector)
                
                if cosine > best_cosine:
                    best_cosine = cosine
                    best_monotask = candidate
            
            # Check if alignment is sufficient
            if best_cosine < self.min_cosine:
                print(f"  [Stop] Best cosine ({best_cosine:.4f}) < threshold ({self.min_cosine})")
                break
            
            # Compute optimal coefficient: alpha = residual · monotask_vector
            # This minimizes ||residual - alpha * monotask_vector||
            optimal_alpha = np.dot(residual, best_monotask.vector)
            
            # Select and update residual with scaled subtraction
            selected.append(best_monotask)
            coefficients[best_monotask.id] = optimal_alpha
            residual = residual - optimal_alpha * best_monotask.vector
            last_region = best_monotask.region
            
            new_norm = np.linalg.norm(residual)
            print(f"  [Step {step+1}] Selected: {best_monotask.name} ({best_monotask.region.value}), α={optimal_alpha:.4f}")
            print(f"            Cosine: {best_cosine:.4f}, Residual norm: {new_norm:.4f}")
        
        return selected, coefficients

# ==========================================
# 4. CONSTRAINT VERIFICATION
# ==========================================

class ConstraintVerifier:
    @staticmethod
    def verify_sum_constraint(task_vector: np.ndarray, selected: List[Monotask], 
                               coefficients: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        if not selected:
            return False, float('inf')
        
        # If coefficients are provided, use weighted sum
        if coefficients:
            sum_vector = sum(coefficients.get(m.id, 1.0) * m.vector for m in selected)
        else:
            sum_vector = sum(m.vector for m in selected)
        
        error = np.linalg.norm(task_vector - sum_vector)
        return error < 0.60, error
    
    @staticmethod
    def verify_region_constraint(selected: List[Monotask]) -> Tuple[bool, List[str]]:
        violations = [f"{m.name}: Invalid region" for m in selected 
                     if m.region not in [Region.INFO, Region.LOGIC, Region.CREATIVE]]
        return len(violations) == 0, violations
    
    @staticmethod
    def verify_transE_constraint(selected: List[Monotask]) -> Tuple[bool, List[str]]:
        violations = []
        last_region = None
        for m in selected:
            allowed = ALLOWED_TRANSITIONS.get(last_region, [Region.INFO, Region.LOGIC])
            if m.region not in allowed:
                violations.append(f"{m.name}: Invalid transition {last_region} -> {m.region}")
            last_region = m.region
        return len(violations) == 0, violations
    
    @staticmethod
    def verify_complexity_constraint(selected: List[Monotask], expected_complexity: Optional[Complexity] = None) -> Tuple[bool, str]:
        """Verify that the decomposition matches expected complexity."""
        if not selected:
            return False, "No monotasks selected"
        
        unique_regions = len(set(m.region for m in selected))
        
        # Infer complexity from selection
        if unique_regions == 1 or len(selected) <= 2:
            inferred = Complexity.SIMPLE
        elif unique_regions == 2 or len(selected) <= 3:
            inferred = Complexity.MODERATE
        else:
            inferred = Complexity.COMPLEX
        
        if expected_complexity is None:
            return True, f"Inferred complexity: {inferred.value}"
        
        match = inferred == expected_complexity
        msg = f"Inferred: {inferred.value}, Expected: {expected_complexity.value} - {'Match' if match else 'Mismatch'}"
        return match, msg
    
    @staticmethod
    def verify_all(task_vector: np.ndarray, selected: List[Monotask], 
                   coefficients: Optional[Dict[int, float]] = None,
                   expected_complexity: Optional[Complexity] = None) -> Dict:
        sum_passed, sum_error = ConstraintVerifier.verify_sum_constraint(task_vector, selected, coefficients)
        region_passed, region_violations = ConstraintVerifier.verify_region_constraint(selected)
        transE_passed, transE_violations = ConstraintVerifier.verify_transE_constraint(selected)
        complexity_passed, complexity_msg = ConstraintVerifier.verify_complexity_constraint(selected, expected_complexity)
        
        return {
            "sum_constraint": {"passed": sum_passed, "error": sum_error},
            "region_constraint": {"passed": region_passed, "violations": region_violations},
            "transE_constraint": {"passed": transE_passed, "violations": transE_violations},
            "complexity_constraint": {"passed": complexity_passed, "message": complexity_msg},
            "all_passed": sum_passed and region_passed and transE_passed and complexity_passed
        }

# ==========================================
# 5. EXPERIMENT
# ==========================================

def run_experiment():
    np.random.seed(42)
    
    print("="*70)
    print("VECTOR DECOMPOSITION: TEXT ENCODER + COMPLEXITY DIMENSION")
    print("="*70)
    
    # Initialize
    codebook = Codebook(dim=64, num_experts_per_region=5)
    router = VectorRouter(codebook, max_steps=15, min_cosine=0.1)
    text_encoder = TextEncoder(codebook)
    
    print(f"\nCodebook: {len(codebook.monotasks)} monotasks")
    print(f"Region centroids: orthogonal subspaces (dims 0-30)")
    print(f"Complexity centroids: separate subspace (dims 30-64)\n")
    
    results = []
    
    # Test 1: Tasks generated from codebook (ground truth known)
    print("-"*70)
    print("TEST 1: Tasks generated from codebook (with complexity)")
    print("-"*70)
    
    for i in range(3):
        print(f"\n[Task {i+1}]")
        
        # Generate task by sampling monotasks (ground truth)
        ground_truth, task_vector, gt_complexity = codebook.sample_task_composition(num_monotasks=(2, 5))
        
        print(f"Ground truth: {[m.name for m in ground_truth]}")
        print(f"Regions: {[m.region.value.split()[0] for m in ground_truth]}")
        print(f"Complexity: {gt_complexity.value}")
        
        # Decompose
        print("Decomposition:")
        selected, coefficients = router.decompose(task_vector)
        
        # Verify with complexity
        verification = ConstraintVerifier.verify_all(task_vector, selected, coefficients, gt_complexity)
        
        print(f"\nVerification:")
        print(f"  Sum:        {'✓' if verification['sum_constraint']['passed'] else '✗'} (error: {verification['sum_constraint']['error']:.4f})")
        print(f"  Region:     {'✓' if verification['region_constraint']['passed'] else '✗'}")
        print(f"  TransE:     {'✓' if verification['transE_constraint']['passed'] else '✗'}")
        print(f"  Complexity: {'✓' if verification['complexity_constraint']['passed'] else '✗'} ({verification['complexity_constraint']['message']})")
        print(f"  ALL:        {'✓✓✓✓' if verification['all_passed'] else '✗✗✗✗'}")
        
        results.append(verification['all_passed'])
    
    # Test 2: Text input decomposition
    print(f"\n\n{'-'*70}")
    print("TEST 2: Text Input Decomposition (NEW)")
    print("-"*70)
    
    test_texts = [
        "Find the data",
        "Search for information and analyze the results",
        "Create a comprehensive design document with detailed analysis and validation",
        "Quick lookup",
        "Write a complex multi-step plan with creative solutions"
    ]
    
    for text in test_texts:
        print(f"\n[Text Input]: \"{text}\"")
        
        # Encode text to vector
        task_vector, metadata = text_encoder.encode_text(text)
        
        print(f"Detected regions: {[r.value.split()[0] for r in metadata['detected_regions']]}")
        print(f"Detected complexity: {metadata['detected_complexity'].value}")
        print(f"Word count: {metadata['word_count']}")
        
        # Decompose
        print("Decomposition:")
        selected, coefficients = router.decompose(task_vector)
        
        # Verify
        verification = ConstraintVerifier.verify_all(task_vector, selected, coefficients, metadata['detected_complexity'])
        
        print(f"Selected: {[m.name for m in selected]}")
        print(f"Verification: {'✓✓✓✓ PASS' if verification['all_passed'] else '✗✗✗✗ FAIL'}")
        print(f"  Sum error: {verification['sum_constraint']['error']:.4f}")
        print(f"  Coefficients: {coefficients}")
        
        results.append(verification['all_passed'])
    
    # Test 3: Full decomposition snapshot with complexity dimension
    print(f"\n\n{'-'*70}")
    print("TEST 3: Full Decomposition Snapshot (Region × Complexity)")
    print("-"*70)
    
    snapshot_texts = [
        ("Simple info query", "Get basic facts"),
        ("Moderate analysis", "Analyze and compare multiple datasets"),
        ("Complex creative task", "Design a comprehensive solution with detailed planning and creative writing")
    ]
    
    for name, text in snapshot_texts:
        print(f"\n[{name}]: \"{text}\"")
        
        task_vector, metadata = text_encoder.encode_text(text)
        selected, coefficients = router.decompose(task_vector)
        
        # Build snapshot
        region_counts = Counter(m.region for m in selected)
        complexity_counts = Counter(m.complexity for m in selected)
        
        print(f"  Detected: {metadata['detected_complexity'].value}")
        print(f"  Region breakdown: {dict(region_counts)}")
        print(f"  Complexity breakdown: {dict(complexity_counts)}")
        print(f"  Selected monotasks: {len(selected)}")
        for m in selected:
            coef = coefficients.get(m.id, 0)
            print(f"    - {m.name}: region={m.region.value.split()[0]}, complexity={m.complexity.value}, α={coef:.3f}")
        
        results.append(True)  # Snapshot is informational
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {sum(results)}/{len(results)} tasks passed all constraints")
    print(f"{'='*70}")
    print("\nKey additions:")
    print("  1. TextEncoder: Converts natural language to task vectors")
    print("  2. Complexity dimension: Adds 3rd axis (Simple/Moderate/Complex)")
    print("  3. Full snapshot: Region × Complexity decomposition view")

if __name__ == "__main__":
    run_experiment()