import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ==========================================
# 1. REGIONS (Geometric Manifolds)
# ==========================================

class Region(Enum):
    INFO = "Info Retrieval"
    LOGIC = "Logic Interpreter"
    CREATIVE = "Creative Interpreter"

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

class Codebook:
    def __init__(self, dim=64, num_experts_per_region=5):
        self.dim = dim
        self.monotasks: List[Monotask] = []
        self.region_centroids = {}
        self._initialize_regions()
        self._populate_codebook(num_experts_per_region)
    
    def _initialize_regions(self):
        """Create orthogonal region centroids."""
        self.region_centroids[Region.INFO] = self._make_centroid([0, 10])
        self.region_centroids[Region.LOGIC] = self._make_centroid([10, 20])
        self.region_centroids[Region.CREATIVE] = self._make_centroid([20, 30])
    
    def _make_centroid(self, indices: Tuple[int, int]) -> np.ndarray:
        vec = np.zeros(self.dim)
        vec[indices[0]:indices[1]] = 1.0
        return vec / np.linalg.norm(vec)
    
    def _populate_codebook(self, num_per_region: int):
        """Populate with monotask vectors near region centroids."""
        task_templates = {
            Region.INFO: ["Search", "Retrieve", "Fetch", "Query", "Get"],
            Region.LOGIC: ["Analyze", "Calculate", "Validate", "Compare", "Plan"],
            Region.CREATIVE: ["Write", "Generate", "Create", "Design", "Compose"]
        }
        
        task_id = 1
        for region, templates in task_templates.items():
            for i, template in enumerate(templates[:num_per_region]):
                # Vector = centroid + small noise, then normalized
                noise = np.random.randn(self.dim) * 0.15
                vec = self.region_centroids[region] + noise
                vec = vec / np.linalg.norm(vec)
                
                self.monotasks.append(Monotask(
                    id=task_id,
                    name=f"{template}_{region.value.split()[0]}",
                    region=region,
                    vector=vec
                ))
                task_id += 1
    
    def get_by_region(self, region: Region) -> List[Monotask]:
        return [m for m in self.monotasks if m.region == region]
    
    def sample_task_composition(self, num_monotasks: Tuple[int, int] = (2, 4)) -> Tuple[List[Monotask], np.ndarray]:
        """
        Generate a valid task by sampling monotasks and summing their vectors.
        Returns: (selected_monotasks, task_vector)
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
        
        # Task vector = sum of selected monotask vectors (normalized)
        if selected:
            task_vector = sum(m.vector for m in selected)
            task_vector = task_vector / np.linalg.norm(task_vector)
            return selected, task_vector
        return [], np.zeros(self.dim)

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
    def verify_all(task_vector: np.ndarray, selected: List[Monotask], 
                   coefficients: Optional[Dict[int, float]] = None) -> Dict:
        sum_passed, sum_error = ConstraintVerifier.verify_sum_constraint(task_vector, selected, coefficients)
        region_passed, region_violations = ConstraintVerifier.verify_region_constraint(selected)
        transE_passed, transE_violations = ConstraintVerifier.verify_transE_constraint(selected)
        
        return {
            "sum_constraint": {"passed": sum_passed, "error": sum_error},
            "region_constraint": {"passed": region_passed, "violations": region_violations},
            "transE_constraint": {"passed": transE_passed, "violations": transE_violations},
            "all_passed": sum_passed and region_passed and transE_passed
        }

# ==========================================
# 5. EXPERIMENT
# ==========================================

def run_experiment():
    np.random.seed(42)
    
    print("="*70)
    print("VECTOR DECOMPOSITION: COSINE SIMILARITY + CONSTRUCTIVE TASKS")
    print("="*70)
    
    # Initialize
    codebook = Codebook(dim=64, num_experts_per_region=5)
    router = VectorRouter(codebook, max_steps=15, min_cosine=0.1)
    
    print(f"\nCodebook: {len(codebook.monotasks)} monotasks")
    print(f"Region centroids are orthogonal subspaces\n")
    
    # Test: Generate tasks FROM the codebook (guaranteed decomposable)
    print("-"*70)
    print("TEST 1: Tasks generated from codebook (ground truth known)")
    print("-"*70)
    
    results = []
    
    for i in range(5):
        print(f"\n[Task {i+1}]")
        
        # Generate task by sampling monotasks (ground truth)
        ground_truth, task_vector = codebook.sample_task_composition(num_monotasks=(2, 4))
        
        print(f"Ground truth: {[m.name for m in ground_truth]}")
        print(f"Ground truth regions: {[m.region.value.split()[0] for m in ground_truth]}")
        
        # Decompose
        print("Decomposition:")
        selected, coefficients = router.decompose(task_vector)
        
        # Verify with coefficients
        verification = ConstraintVerifier.verify_all(task_vector, selected, coefficients)
        
        print(f"\nVerification:")
        print(f"  Sum:        {'✓' if verification['sum_constraint']['passed'] else '✗'} (error: {verification['sum_constraint']['error']:.4f})")
        print(f"  Region:     {'✓' if verification['region_constraint']['passed'] else '✗'}")
        print(f"  TransE:     {'✓' if verification['transE_constraint']['passed'] else '✗'}")
        print(f"  ALL:        {'✓✓✓' if verification['all_passed'] else '✗✗✗'}")
        
        # Compare to ground truth
        if ground_truth and selected:
            gt_names = {m.name for m in ground_truth}
            sel_names = {m.name for m in selected}
            overlap = len(gt_names & sel_names)
            print(f"  Recovery:   {overlap}/{len(ground_truth)} ground truth monotasks recovered")
        
        results.append(verification['all_passed'])
    
    # Test 2: Synthetic composition tasks (harder)
    print(f"\n\n{'-'*70}")
    print("TEST 2: Synthetic composition tasks (weighted centroid sums)")
    print("-"*70)
    
    compositions = [
        {"name": "Info-Heavy", "weights": {Region.INFO: 0.7, Region.LOGIC: 0.2, Region.CREATIVE: 0.1}},
        {"name": "Logic-Heavy", "weights": {Region.INFO: 0.1, Region.LOGIC: 0.7, Region.CREATIVE: 0.2}},
        {"name": "Balanced", "weights": {Region.INFO: 0.33, Region.LOGIC: 0.34, Region.CREATIVE: 0.33}},
    ]
    
    for comp in compositions:
        print(f"\n[Task] {comp['name']}: {comp['weights']}")
        
        # Create task as weighted sum of centroids
        task_vector = np.zeros(codebook.dim)
        for region, weight in comp['weights'].items():
            task_vector += weight * codebook.region_centroids[region]
        task_vector = task_vector / np.linalg.norm(task_vector)
        
        print("Decomposition:")
        selected, coefficients = router.decompose(task_vector)
        
        verification = ConstraintVerifier.verify_all(task_vector, selected, coefficients)
        print(f"Verification: {'✓✓✓ PASS' if verification['all_passed'] else '✗✗✗ FAIL'}")
        print(f"  Sum error: {verification['sum_constraint']['error']:.4f}")
        print(f"  Selected: {[m.name for m in selected]}")
        print(f"  Coefficients: {coefficients}")
        
        results.append(verification['all_passed'])
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {sum(results)}/{len(results)} tasks passed all constraints")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_experiment()