import networkx as nx
import numpy as np
import json
import hashlib
import math

class SensoryAssociativeNetwork:
    def __init__(self, damping_factor=0.15, threshold=0.10, max_steps=50, tolerance=1e-5,
                 type_bias=None):
        """
        Sensory Associative Network (SAN) Engine
        Implements context-conditional damped spreading activation and pure-Python TDA persistent homology.

        type_bias: dict mapping target node type -> multiplicative bias on incoming edge contribution.
                   Used to surface affective semantics by amplifying excitation onto sensation/association
                   nodes and dampening pure concept->concept hub-flooding. Defaults to no bias.
        """
        self.graph = nx.Graph()
        self.gamma = damping_factor
        self.theta = threshold
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.type_bias = type_bias if type_bias is not None else {
            'sensation': 1.0,
            'association': 1.0,
            'context': 1.0,
            'concept': 1.0,
        }
        
    def add_node(self, node_id, node_type, description=""):
        """
        Add a node to the SAN.
        Types: 'concept', 'context', 'sensation', 'association'
        """
        self.graph.add_node(node_id, type=node_type, description=description)
        
    def add_association(self, u, v, weight, compatibility_context=None):
        """
        Add an associative weighted edge. Positive weight is excitatory, negative is inhibitory.
        """
        if compatibility_context is not None:
            if isinstance(compatibility_context, str):
                contexts = [compatibility_context]
            else:
                contexts = list(compatibility_context)
        else:
            contexts = []
            
        self.graph.add_edge(u, v, weight=weight, compatible_contexts=contexts)
        
    @classmethod
    def load_from_json(cls, path, damping_factor=0.15, threshold=0.10, max_steps=50, tolerance=1e-5,
                       type_bias=None):
        """
        [E.5 Integration] Load the complete SAN database (nodes, categories, weights) from JSON.
        """
        san = cls(damping_factor, threshold, max_steps, tolerance, type_bias=type_bias)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for node in data["nodes"]:
            san.add_node(node["id"], node["type"], node.get("description", ""))
            
        for edge in data["edges"]:
            san.add_association(
                edge["source"], 
                edge["target"], 
                edge["weight"], 
                edge.get("compatible_contexts", None)
            )
            
        print(f" -> SAN loaded successfully from '{path}' ({len(san.graph.nodes)} nodes, {len(san.graph.edges)} edges).")
        return san

    def propagate(self, seed_concept, context):
        """
        Damped Context-Conditional Spreading Activation
        GCN-style symmetric degree normalization to prevent hub-flooding.
        Clamped linear activation with seed node continuous clamping and hard thresholding.
        """
        if seed_concept not in self.graph:
            raise ValueError(f"Seed concept '{seed_concept}' not in network.")
            
        nodes = list(self.graph.nodes())
        num_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Initialize activation vector
        alpha = np.zeros(num_nodes)
        
        # Seed concept and context start fully active
        idx_seed = node_to_idx[seed_concept]
        alpha[idx_seed] = 1.0
        
        idx_context = None
        if context in self.graph:
            idx_context = node_to_idx[context]
            alpha[idx_context] = 1.0
            
        for step in range(self.max_steps):
            alpha_next = np.zeros(num_nodes)
            
            for u in self.graph.nodes():
                idx_u = node_to_idx[u]
                
                # Keep seeds clamped to 1.0
                if idx_u == idx_seed or idx_u == idx_context:
                    alpha_next[idx_u] = 1.0
                    continue
                
                # [B.3] Type-aware bias on the *receiver* u: amplify when u is sensation/association,
                # dampen pure concept hub-flooding. Applied multiplicatively to the aggregated input.
                u_type = self.graph.nodes[u].get('type', 'concept')
                u_bias = self.type_bias.get(u_type, 1.0)

                # Dynamic inputs sum
                input_sum = 0.0
                for v in self.graph.neighbors(u):
                    idx_v = node_to_idx[v]
                    if alpha[idx_v] <= 0.0:
                        continue

                    edge_data = self.graph.get_edge_data(u, v)
                    weight = edge_data['weight']
                    compat_contexts = edge_data.get('compatible_contexts', [])

                    is_compatible = (len(compat_contexts) == 0 or context in compat_contexts)

                    if is_compatible:
                        # [GCN Normalization] Normalize weight by square root of degrees to prevent hub flooding
                        deg_u = self.graph.degree(u)
                        deg_v = self.graph.degree(v)
                        norm = math.sqrt(deg_u * deg_v) if deg_u * deg_v > 0 else 1.0

                        input_sum += (weight / norm) * alpha[idx_v]

                # Apply receiver-type bias after aggregation (preserves sign of inhibitory inputs)
                input_sum *= u_bias
                        
                # Propagation & hard thresholding to keep activation localized (preventing flooding)
                val = (1 - self.gamma) * alpha[idx_u] + self.gamma * input_sum
                val = max(0.0, min(1.0, val))
                alpha_next[idx_u] = val if val >= self.theta else 0.0
                
            # Re-clamp seeds
            alpha_next[idx_seed] = 1.0
            if idx_context is not None:
                alpha_next[idx_context] = 1.0
                
            diff = np.linalg.norm(alpha_next - alpha, ord=1)
            alpha = alpha_next
            
            if diff < self.tolerance:
                break
                
        return {nodes[i]: alpha[i] for i in range(num_nodes)}
        
    def extract_active_subgraph(self, activation_map):
        active_nodes = [node for node, act in activation_map.items() if act > self.theta]
        return self.graph.subgraph(active_nodes).copy(), active_nodes
        
    def _calculate_gini(self, x):
        if len(x) == 0 or np.sum(x) == 0:
            return 0.0
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))
        
    def _calculate_continuous_symmetry(self, subgraph):
        """
        [B.1 Symmetry Redesign] Structural symmetry as the normalized algebraic connectivity
        (Fiedler value of the normalized Laplacian) of the largest connected component of the
        active subgraph.

            S = lambda_2( L_norm(G_active^{LCC}) ) in [0, 1]

        Intuition: a uniform / regular structure (k-regular, complete) approaches 1 (perfect
        balance / equanimity); a star-like or path-like structure collapses toward 0
        (asymmetric tension). This is a well-defined graph invariant, agnostic to node labels,
        and directly comparable across activations of different sizes.
        """
        n = len(subgraph)
        if n <= 1:
            return 1.0

        components = sorted(nx.connected_components(subgraph), key=len, reverse=True)
        largest = subgraph.subgraph(components[0])
        if len(largest) <= 1:
            return 0.0
        try:
            fiedler = nx.algebraic_connectivity(largest, normalized=True)
        except Exception:
            return 0.0
        return float(max(0.0, min(1.0, fiedler)))

    def compute_persistent_homology(self, subgraph, activation_map):
        r"""
        Pure Python $\mathbb{Z}_2$ Persistent Homology Solver for Vietoris-Rips Graph Complexes.
        Computes persistence diagrams for $H_0$ and $H_1$.
        Filtration values represent "birth times" based on node activation: f(v) = 1.0 - act(v)
        """
        vertices = list(subgraph.nodes())
        edges = [tuple(sorted(e)) for e in subgraph.edges()]
        
        triangles = []
        for clique in nx.enumerate_all_cliques(subgraph):
            if len(clique) == 3:
                triangles.append(tuple(sorted(clique)))
            elif len(clique) > 3:
                break
                
        filt_vals = {}
        for v in vertices:
            filt_vals[(v,)] = float(1.0 - activation_map[v])
            
        for e in edges:
            filt_vals[e] = float(max(filt_vals[(e[0],)], filt_vals[(e[1],)]))
            
        for t in triangles:
            e0 = (t[0], t[1])
            e1 = (t[1], t[2])
            e2 = (t[0], t[2])
            filt_vals[t] = float(max(filt_vals[e0], filt_vals[e1], filt_vals[e2]))
            
        simplices = []
        for v in vertices:
            simplices.append(((v,), 0))
        for e in edges:
            simplices.append((e, 1))
        for t in triangles:
            simplices.append((t, 2))
            
        simplices.sort(key=lambda s: (filt_vals[s[0]], s[1]))
        
        simplex_to_idx = {s[0]: i for i, s in enumerate(simplices)}
        n_simplices = len(simplices)
        
        columns = {}
        for j in range(n_simplices):
            simplex, dim = simplices[j]
            faces = []
            if dim == 1:
                faces = [(simplex[0],), (simplex[1],)]
            elif dim == 2:
                faces = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])]
                
            non_zero_rows = sorted([simplex_to_idx[f] for f in faces if f in simplex_to_idx])
            columns[j] = set(non_zero_rows)
            
        pivot_to_col = {}
        pairs = []
        
        for j in range(n_simplices):
            while len(columns[j]) > 0:
                pivot = max(columns[j])
                if pivot in pivot_to_col:
                    k = pivot_to_col[pivot]
                    columns[j] = columns[j] ^ columns[k]
                else:
                    pivot_to_col[pivot] = j
                    pairs.append((pivot, j))
                    break
                    
        all_births = set(range(n_simplices))
        for birth, death in pairs:
            if birth in all_births:
                all_births.remove(birth)
                
        h0_diagram = []
        h1_diagram = []
        
        for birth, death in pairs:
            b_simplex, b_dim = simplices[birth]
            d_simplex, d_dim = simplices[death]
            
            b_time = filt_vals[b_simplex]
            d_time = filt_vals[d_simplex]
            lifetime = d_time - b_time
            
            if b_dim == 0:
                h0_diagram.append((b_time, d_time, lifetime))
            elif b_dim == 1:
                h1_diagram.append((b_time, d_time, lifetime))
                
        for birth in all_births:
            b_simplex, b_dim = simplices[birth]
            b_time = filt_vals[b_simplex]
            lifetime = 1.0 - b_time
            if b_dim == 0:
                h0_diagram.append((b_time, 1.0, lifetime))
            elif b_dim == 1:
                h1_diagram.append((b_time, 1.0, lifetime))
                
        return h0_diagram, h1_diagram

    def compute_topological_vector(self, activation_map, seed_concept):
        """
        Compute the 5-dimensional Topological Signature T(alpha*) = (D, S, C, H, B)
        """
        subgraph, active_nodes = self.extract_active_subgraph(activation_map)
        num_active = len(active_nodes)
        
        if num_active == 0:
            return {
                'Density (D)': 0.0,
                'Symmetry (S)': 1.0,
                'Centrality (C)': 0.0,
                'Depth (H)': 0.0,
                'Boundary (B)': 0.0
            }
            
        # 1. Density (D)
        max_possible_edges = num_active * (num_active - 1) / 2.0
        density = subgraph.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # 2. Symmetry (S)
        symmetry = self._calculate_continuous_symmetry(subgraph)
        
        # 3. Centrality (C)
        if num_active > 1:
            try:
                cent_dict = nx.eigenvector_centrality_numpy(subgraph)
            except Exception:
                cent_dict = nx.degree_centrality(subgraph)
            centralities = np.array(list(cent_dict.values()))
            gini_centrality = self._calculate_gini(centralities)
        else:
            gini_centrality = 0.0
            
        # 4. Depth (H)
        depth = 0.0
        if seed_concept in active_nodes:
            lengths = nx.single_source_shortest_path_length(subgraph, seed_concept)
            if lengths:
                depth = float(max(lengths.values()))
        else:
            if len(subgraph) > 1 and nx.is_connected(subgraph):
                depth = float(nx.diameter(subgraph))
            elif len(subgraph) > 1:
                components = sorted(nx.connected_components(subgraph), key=len, reverse=True)
                largest_comp = subgraph.subgraph(components[0])
                if len(largest_comp) > 1:
                    depth = float(nx.diameter(largest_comp))
                    
        # 5. Boundary (B)
        h0_dia, h1_dia = self.compute_persistent_homology(subgraph, activation_map)
        if h1_dia:
            boundary = sum(val[2] for val in h1_dia if val[2] > 0.01)
        else:
            try:
                components = sorted(nx.connected_components(subgraph), key=len, reverse=True)
                largest_comp = subgraph.subgraph(components[0])
                fiedler = nx.algebraic_connectivity(largest_comp, normalized=True)
                boundary = float(fiedler)
            except Exception:
                boundary = 0.0
        
        return {
            'Density (D)': float(density),
            'Symmetry (S)': float(symmetry),
            'Centrality (C)': float(gini_centrality),
            'Depth (H)': float(depth),
            'Boundary (B)': float(boundary)
        }
        
    def describe_state(self, seed_concept, context, activation_map, top_vector):
        subgraph, active_nodes = self.extract_active_subgraph(activation_map)
        sorted_nodes = sorted(
            [(node, activation_map[node]) for node in active_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        description = f"=== SENTIENT TOPOLOGY ANALYTICAL REPORT ===\n"
        description += f"Stimulus: concept='{seed_concept}', context='{context}'\n"
        description += f"Active Nodes Count: {len(active_nodes)} (out of {len(self.graph)} nodes)\n\n"
        
        description += f"Top 10 Active Nodes:\n"
        for node, act in sorted_nodes[:10]:
            node_desc = self.graph.nodes[node].get('description', '')
            node_type = self.graph.nodes[node].get('type', '')
            description += f" - {node} ({act:.3f}) [Category: {node_type}]: {node_desc}\n"
            
        description += f"\nTopological Signature T(alpha*):\n"
        for dim, val in top_vector.items():
            description += f" - {dim}: {val:.4f}\n"
            
        description += f"\nQualitative Interpretation:\n"
        
        d = top_vector['Density (D)']
        if d > 0.6:
            description += " - Density (High): Sensation is intense, tightly integrated, and overwhelming.\n"
        elif d > 0.3:
            description += " - Density (Moderate): Sensation is balanced and cohesive.\n"
        else:
            description += " - Density (Low): Sensation is sparse, subtle, with quiet resonance.\n"
            
        s = top_vector['Symmetry (S)']
        if s > 0.7:
            description += " - Symmetry (High): Exceptional emotional equanimity, balance, and poise (Zen-like state).\n"
        elif s > 0.4:
            description += " - Symmetry (Moderate): Relational stability with dynamic adjustments.\n"
        else:
            description += " - Symmetry (Low): Severe asymmetric tension, dynamic conflict, and high emotional volatility.\n"
            
        c = top_vector['Centrality (C)']
        if c > 0.6:
            description += " - Centrality (High): Sensation is single-point focused, dominated by a core affective hub.\n"
        else:
            description += " - Centrality (Low): Sensation is distributed, complex, and multifaceted.\n"
            
        return description
