from san_engine import SensoryAssociativeNetwork
import os
import json

def run_large_scale_experiment():
    san_path = os.path.join("data", "large_san_5000.json")

    print("==================================================================")
    print("        SENTIENT TOPOLOGY LARGE-SCALE SIMULATION RUNNER")
    print("==================================================================")

    # 1. Load the complete 5,000-Node SAN Database
    if not os.path.exists(san_path):
        raise FileNotFoundError(f"SAN database '{san_path}' not found. Run build_large_san_5000.py first.")

    # [B.3] Type-aware bias: amplify excitation onto sensation/association receivers,
    # dampen concept->concept hub flooding so affective semantics surface in the active set.
    TYPE_BIAS = {
        'sensation': 1.8,
        'association': 1.5,
        'context': 1.2,
        'concept': 0.7,
    }

    # Load SAN from JSON database (E.5 Integration)
    # Using GCN normalization, we lower the threshold to allow normalized activation to propagate!
    san = SensoryAssociativeNetwork.load_from_json(
        san_path,
        damping_factor=0.35, # Higher propagation rate
        threshold=0.005,    # Lower threshold corresponding to GCN-normalized activations
        max_steps=30,
        type_bias=TYPE_BIAS,
    )

    # [B.2] Both contexts are genuine V_context nodes in the 5,000-node vocabulary,
    # resolving the prior category violation where context_b='love' was a sensation node.
    seed_concept = "sunset"
    context_a = "war"     # destructive / loss-bearing context (battlefield analogue)
    context_b = "garden"  # cultivated / domestic context (peaceful analogue)
    
    print("==================================================================")
    
    # 2. Execute Scenario A: Sunset under War Context
    print(f"\n[SCENARIO A] Spreading Activation: concept='{seed_concept}', context='{context_a}'")
    act_a = san.propagate(seed_concept, context_a)
    vec_a = san.compute_topological_vector(act_a, seed_concept)
    report_a = san.describe_state(seed_concept, context_a, act_a, vec_a)
    print(report_a)
    
    # 3. Execute Scenario B: Sunset under Love Context
    print(f"\n[SCENARIO B] Spreading Activation: concept='{seed_concept}', context='{context_b}'")
    act_b = san.propagate(seed_concept, context_b)
    vec_b = san.compute_topological_vector(act_b, seed_concept)
    report_b = san.describe_state(seed_concept, context_b, act_b, vec_b)
    print(report_b)
    
    # 4. Compare Topological Vectors
    print("==================================================================")
    print("             SCALED 5,000-NODE TOPOLOGICAL COMPARISON TABLE")
    print("==================================================================")
    print(f"{'Dimension':<20} | {'War':<14} | {'Garden':<14} | {'Delta':<10}")
    print("-" * 64)
    for dim in vec_a.keys():
        val_a = vec_a[dim]
        val_b = vec_b[dim]
        delta = val_a - val_b
        print(f"{dim:<20} | {val_a:<14.4f} | {val_b:<14.4f} | {delta:<+10.4f}")
    print("==================================================================")

    return {
        'context_a': context_a,
        'context_b': context_b,
        'activation_a': act_a,
        'activation_b': act_b,
        'vector_a': vec_a,
        'vector_b': vec_b,
        'san': san,
        'seed_concept': seed_concept,
    }


if __name__ == "__main__":
    run_large_scale_experiment()
