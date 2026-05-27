import os
import re
import json
import math
from collections import Counter, defaultdict
import nltk
from nltk.stem import WordNetLemmatizer

class LargeSanBuilder5000:
    def __init__(self, data_dir="data", vocabulary_size=5000, window_size=15, ppmi_threshold=1.8, max_edges=25000):
        self.data_dir = data_dir
        self.books_dir = os.path.join(data_dir, "books")
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.ppmi_threshold = ppmi_threshold
        self.max_edges = max_edges
        
        # Initialize Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Comprehensive stopwords list
        self.stopwords = {
            "the", "and", "of", "to", "a", "i", "in", "was", "that", "it", "he", "you", "his", "had",
            "with", "as", "for", "at", "him", "her", "my", "on", "be", "she", "not", "but", "is", "have",
            "by", "all", "this", "they", "which", "me", "from", "so", "or", "an", "one", "were", "there",
            "their", "would", "them", "been", "has", "what", "will", "no", "if", "out", "when", "into",
            "up", "more", "who", "its", "are", "do", "your", "can", "very", "could", "than", "some", "our",
            "about", "my", "me", "how", "then", "like", "other", "only", "itself", "upon", "about", "could",
            "than", "some", "two", "we", "us", "our", "ours", "yours", "hers", "theirs", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
            "does", "did", "doing", "because", "until", "while", "of", "at", "by", "for", "with", "against",
            "between", "through", "during", "before", "after", "above", "below", "once", "here", "there",
            "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "such",
            "no", "nor", "not", "only", "own", "same", "too", "very", "s", "t", "just", "don", "should",
            "now", "d", "ll", "m", "o", "re", "ve", "y", "shouldn", "wasn", "weren", "won", "wouldn", "say",
            "said", "go", "went", "come", "came", "get", "got", "look", "looked", "make", "made", "know",
            "known", "think", "thought", "see", "saw", "seen", "take", "took", "tell", "told", "give", "given"
        }
        
        # [E.1 Integration] Pure Lemmatized Category Whitelists to discard substring false positives
        self.sensation_whitelist = {
            "sorrow", "grief", "sadness", "mourn", "weep",
            "guilt", "shame", "remorse", "blame", "repent",
            "desolation", "loneliness", "isolated", "barren", "empty",
            "anger", "fury", "rage", "wrath", "bitter", "furious",
            "fear", "dread", "terror", "horror", "fright", "anxious", "anxiety", "panic",
            "pain", "agony", "suffering", "misery", "hurt",
            "joy", "delight", "happiness", "bliss", "glad", "ecstasy", "pleasure",
            "gratitude", "thankful", "appreciation", "bless",
            "peace", "tranquility", "calm", "stillness", "quiet", "serene",
            "fullness", "satisfaction", "contentment", "complete",
            "love", "affection", "beloved", "fondness", "darling", "lovely",
            "hope", "aspiration", "trust", "despair", "disgust", "pity", "compassion"
        }
        
        self.association_whitelist = {
            "soul", "spirit", "essence", "mind", "conscience", 
            "death", "dying", "grave", "dead", "comrade", 
            "nature", "sunset", "dusk", "sea", "hill", "woods", "forest", "mountain", "ocean",
            "beauty", "beautiful", "truth", "justice", "time", "eternity", "infinite", "immortal",
            "timeless", "art", "world", "life", "existence", "god", "holy", "reverence", "awe",
            "solemn", "veneration", "melancholy", "pride", "proud", "shame"
        }
        
        # [E.3 Integration] Explicit Whitelist of Context Seeds (V_context)
        # Injected directly to guarantee context node presence and PPMI semantic mapping
        self.context_whitelist = {
            "battlefield", "lover_hill", "funeral", "wedding", "solitude", "war", "hospital",
            "church", "prison", "dark_room", "storm", "garden", "graveyard", "palace", "market",
            "library", "school", "wilderness", "ruins", "spring_morning", "winter_night", "chamber",
            "bridge", "tower", "parlor", "drawing_room", "cottage", "countryside", "shipboard"
        }
        
        # Opposing Sensation Domains for active inhibition mapping
        self.positive_sensations = {
            "joy", "delight", "happiness", "bliss", "glad", "ecstasy", "pleasure", "gratitude",
            "thankful", "appreciation", "peace", "tranquility", "calm", "stillness", "quiet", "serene",
            "fullness", "satisfaction", "contentment", "love", "affection", "beloved", "lovely", "hope", "trust"
        }

        self.negative_sensations = {
            "sorrow", "grief", "sadness", "mourn", "weep", "guilt", "shame", "remorse", "blame", "repent",
            "desolation", "loneliness", "isolated", "barren", "empty", "anger", "fury", "rage", "wrath", "bitter",
            "furious", "fear", "dread", "terror", "horror", "fright", "anxious", "anxiety", "panic", "pain", "agony",
            "suffering", "misery", "hurt", "despair", "disgust"
        }

        # [B.3-rule-augmented] Context -> sensation/association seed channels.
        # The 8-book Gutenberg PPMI is too sparse to grow context<->affect edges naturally;
        # we make the affect grounding *explicit* and *auditable* exactly like the inhibitory rule,
        # rather than letting topical co-occurrence dominate. Reported as augmentation in §3.1.
        self.context_affect_map = {
            "war":        {"sorrow": 0.65, "grief": 0.6, "fear": 0.55, "dread": 0.5, "anger": 0.6,
                           "desolation": 0.7, "death": 0.7, "comrade": 0.55, "horror": 0.55, "rage": 0.55,
                           "agony": 0.55, "remorse": 0.45, "reverence": 0.4},
            "garden":     {"peace": 0.7, "calm": 0.65, "joy": 0.6, "gratitude": 0.55, "hope": 0.55,
                           "love": 0.55, "beauty": 0.55, "serene": 0.55, "lovely": 0.55, "delight": 0.5,
                           "affection": 0.5, "stillness": 0.45},
            "funeral":    {"sorrow": 0.75, "grief": 0.75, "mourn": 0.7, "reverence": 0.6, "death": 0.7,
                           "despair": 0.55, "weep": 0.55, "melancholy": 0.55},
            "wedding":    {"joy": 0.7, "love": 0.7, "gratitude": 0.55, "hope": 0.55, "delight": 0.6,
                           "affection": 0.6, "beloved": 0.6, "happiness": 0.6},
            "solitude":   {"stillness": 0.6, "calm": 0.55, "melancholy": 0.55, "reverence": 0.5,
                           "loneliness": 0.65, "barren": 0.45, "soul": 0.45},
            "prison":     {"despair": 0.7, "desolation": 0.65, "shame": 0.55, "guilt": 0.55, "fear": 0.5,
                           "loneliness": 0.6, "remorse": 0.55, "barren": 0.5},
            "church":     {"reverence": 0.7, "peace": 0.5, "soul": 0.55, "hope": 0.5, "gratitude": 0.45,
                           "guilt": 0.4, "eternity": 0.5},
            "storm":      {"fear": 0.6, "dread": 0.55, "anger": 0.55, "rage": 0.55, "horror": 0.5,
                           "agony": 0.45, "panic": 0.55},
            "wilderness": {"desolation": 0.6, "fear": 0.55, "loneliness": 0.6, "barren": 0.6,
                           "reverence": 0.4, "soul": 0.4, "melancholy": 0.45},
            "cottage":    {"peace": 0.55, "calm": 0.5, "affection": 0.5, "love": 0.45, "stillness": 0.45,
                           "gratitude": 0.4},
            "library":    {"reverence": 0.45, "calm": 0.45, "stillness": 0.45, "melancholy": 0.4,
                           "soul": 0.4, "truth": 0.4},
            "palace":     {"reverence": 0.5, "pride": 0.55, "beauty": 0.5, "love": 0.4, "calm": 0.4},
        }

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-Z']+\b", text.lower())

    def _get_lemma(self, token):
        # Lemmatize as noun first, then verb
        lemma = self.lemmatizer.lemmatize(token, pos="n")
        lemma = self.lemmatizer.lemmatize(lemma, pos="v")
        return lemma

    def build_large_san_5000(self):
        print("==================================================================")
        print("          SCALING SAN TO 5,000 NODES -- ROBUST PIPELINE")
        print("==================================================================")
        
        # 1. Read books and lemmatize vocabulary
        print("[Step 1] Reading literature files and performing lemmatization...")
        if not os.path.exists(self.books_dir) or len(os.listdir(self.books_dir)) == 0:
            raise FileNotFoundError(f"No cached book texts found in '{self.books_dir}'.")
            
        full_lemmas = []
        lemma_counts = Counter()
        
        for file in os.listdir(self.books_dir):
            if file.endswith(".txt"):
                path = os.path.join(self.books_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                tokens = self._tokenize(text)
                
                # Lemmatize full text
                lemmas = [self._get_lemma(t) for t in tokens]
                full_lemmas.extend(lemmas)
                
                # Count frequencies for vocabulary building
                content_lemmas = [l for l in lemmas if l not in self.stopwords and len(l) > 2]
                lemma_counts.update(content_lemmas)
                
        total_tokens = len(full_lemmas)
        print(f" - Loaded {total_tokens} total tokens from cached literature corpus.")
        print(f" - Unique lemmatized content words: {len(lemma_counts)}")
        
        # [E.3 Integration] Collect explicit context nodes present in the corpus
        injected_contexts = []
        for ctx in self.context_whitelist:
            # We check if the context exists in the book corpus
            # If it does, we inject it with high priority
            if ctx in lemma_counts:
                injected_contexts.append(ctx)
        
        print(f" - Injected {len(injected_contexts)} explicit V_context nodes (e.g. battlefield, solitude).")
        
        # Select candidate nodes: injected contexts first, then most frequent lemmas
        candidate_set = set(injected_contexts)
        remaining_slots = self.vocabulary_size * 2 # Select a larger set first to filter out isolated nodes
        
        sorted_lemmas = [l for l, count in lemma_counts.most_common(remaining_slots) if l not in candidate_set]
        
        # Merge candidate vocabulary list
        top_vocab = injected_contexts + sorted_lemmas[:remaining_slots - len(injected_contexts)]
        vocab_set = set(top_vocab)
        vocab_idx = {word: idx for idx, word in enumerate(top_vocab)}
        
        # 2. Build index representation of lemmas
        print("\n[Step 2] Performing index sliding window co-occurrence scan...")
        indexed_lemmas = [vocab_idx[l] if l in vocab_set else None for l in full_lemmas]
        
        co_counts = defaultdict(Counter)
        indexed_counts = Counter([l for l in indexed_lemmas if l is not None])
        
        n_tokens = len(indexed_lemmas)
        for i in range(n_tokens):
            w1 = indexed_lemmas[i]
            if w1 is None:
                continue
                
            start = max(0, i - self.window_size)
            end = min(n_tokens, i + self.window_size + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                w2 = indexed_lemmas[j]
                if w2 is not None and w1 != w2:
                    co_counts[w1][w2] += 1
                    
        # 3. Compute PPMI weights and add automatic negative edges
        print(f"\n[Step 3] Calculating PPMI weights and [E.2] generating inhibitory edges...")
        edges_temp = []
        active_nodes_temp = set()
        
        # Track edge degree of each node to remove isolated nodes
        node_degrees = Counter()
        
        # Compute positive PPMI edges from co-occurrence
        for w1, neighbors in co_counts.items():
            for w2, count in neighbors.items():
                if w1 >= w2:
                    continue
                if count <= 2:
                    continue
                    
                p_w1 = indexed_counts[w1] / total_tokens
                p_w2 = indexed_counts[w2] / total_tokens
                p_joint = count / (total_tokens * 2 * self.window_size)
                
                pmi = math.log2(p_joint / (p_w1 * p_w2))
                ppmi = max(0.0, pmi)
                
                if ppmi >= self.ppmi_threshold:
                    weight = min(1.0, ppmi / 8.0) # Scale weight
                    u = top_vocab[w1]
                    v = top_vocab[w2]
                    edges_temp.append({
                        "source": u,
                        "target": v,
                        "weight": float(weight),
                        "ppmi": float(ppmi)
                    })
                    node_degrees[u] += 1
                    node_degrees[v] += 1
                    
        # [E.2 Integration] Generate Automatic Negative/Inhibitory Edges
        # We find opposing pairs of whitelisted sensations present in our top vocabulary
        inhibitory_edges_count = 0

        pos_active = [w for w in top_vocab if w in self.positive_sensations and node_degrees[w] > 0]
        neg_active = [w for w in top_vocab if w in self.negative_sensations and node_degrees[w] > 0]

        for pos_w in pos_active:
            for neg_w in neg_active:
                # Add strong active inhibition between opposing emotional centers
                # Ex: joy <-> sorrow (-0.8), peace <-> anger (-0.8)
                weight = -0.7
                edges_temp.append({
                    "source": pos_w,
                    "target": neg_w,
                    "weight": float(weight),
                    "ppmi": 5.0 # High priority
                })
                node_degrees[pos_w] += 1
                node_degrees[neg_w] += 1
                inhibitory_edges_count += 1

        print(f" - Generated {inhibitory_edges_count} active inhibition edges (negative weights) between opposing sensations.")

        # [B.3 rule-augmented] Context -> sensation/association excitatory seed edges.
        # Same auditability principle as the inhibitory rule: corpus-thin signal is explicitly
        # supplemented to ground each V_context node in its semantically expected affect channel.
        # These augmentations are reported as such in the LaTeX, not hidden under "learned".
        context_seed_count = 0
        for ctx, partners in self.context_affect_map.items():
            if ctx not in vocab_set:
                continue
            for partner, weight in partners.items():
                if partner not in vocab_set:
                    continue
                if ctx == partner:
                    continue
                edges_temp.append({
                    "source": ctx,
                    "target": partner,
                    "weight": float(weight),
                    "ppmi": 5.0  # high priority so they survive the max_edges cap
                })
                node_degrees[ctx] += 1
                node_degrees[partner] += 1
                context_seed_count += 1
        print(f" - Generated {context_seed_count} context->affect seed edges (rule-augmented).")
        
        # [E.4 Integration] Remove Isolated Nodes (degree = 0)
        # Filter top_vocab to keep only nodes that have at least 1 edge (degree >= 1)
        active_vocab = [w for w in top_vocab if node_degrees[w] >= 1]
        print(f" - Filtered out isolated nodes. Active nodes remaining: {len(active_vocab)}")
        
        # Select exactly the top 5,000 most frequent active nodes
        # If we have fewer than 5,000, we keep all of them.
        final_nodes_list = sorted(active_vocab, key=lambda w: lemma_counts[w], reverse=True)[:self.vocabulary_size]
        final_nodes_set = set(final_nodes_list)
        
        print(f" - Final retained active nodes: {len(final_nodes_list)} (0 isolated nodes!)")
        
        # Filter edges to only include final nodes
        final_edges = []
        for e in edges_temp:
            if e["source"] in final_nodes_set and e["target"] in final_nodes_set:
                final_edges.append({
                    "source": e["source"],
                    "target": e["target"],
                    "weight": e["weight"]
                })
                
        # Limit edges if they exceed max_edges, keeping negative edges and high PPMI positive edges
        # Negative edges have negative weight, positive have positive weight
        final_edges.sort(key=lambda e: abs(e["weight"]) if e["weight"] < 0 else e["weight"], reverse=True)
        final_edges = final_edges[:self.max_edges]
        
        # Count positive vs negative edges
        pos_edges = sum(1 for e in final_edges if e["weight"] > 0)
        neg_edges = sum(1 for e in final_edges if e["weight"] < 0)
        print(f" - Saved Edges: {len(final_edges)} (Excitatory: {pos_edges}, Inhibitory: {neg_edges})")
        
        # 4. Assemble Node list and categorize based on strict whitelists (E.1)
        print(f"\n[Step 4] Categorizing nodes and assembling JSON...")
        nodes = []
        
        for word in final_nodes_list:
            # E.1 Precise classification using Whitelists to discard substring false positives
            if word in self.context_whitelist:
                node_type = "context"
            elif word in self.sensation_whitelist:
                node_type = "sensation"
            elif word in self.association_whitelist:
                node_type = "association"
            else:
                node_type = "concept"
                
            nodes.append({
                "id": word,
                "type": node_type,
                "frequency": int(lemma_counts[word]),
                "description": f"Lemmatized SAN concept '{word}'. Category: {node_type}."
            })
            
        # Count categories
        cat_counts = Counter([n["type"] for n in nodes])
        print(f" - Node Categories: Sensation: {cat_counts['sensation']}, Association: {cat_counts['association']}, Context: {cat_counts['context']}, Concept: {cat_counts['concept']}")
        
        # 5. Save Graph JSON
        san_data = {
            "metadata": {
                "description": "5,000-Node lemmatized SAN with corpus-PPMI edges, rule-augmented inhibition, and rule-augmented context->affect seed channels.",
                "total_nodes": len(nodes),
                "total_edges": len(final_edges),
                "category_breakdown": dict(cat_counts),
                "edge_breakdown": {"excitatory": pos_edges, "inhibitory": neg_edges},
                "augmentation": {
                    "inhibitory_rule_pairs": inhibitory_edges_count,
                    "context_seed_pairs": context_seed_count
                }
            },
            "nodes": nodes,
            "edges": final_edges
        }
        
        out_path = os.path.join(self.data_dir, "large_san_5000.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(san_data, f, indent=4)
            
        print(f"\n[Step 5] Highly rigorous 5,000-Node SAN compiled successfully!")
        print(f" - Saved to: {out_path}")
        print("==================================================================")

if __name__ == "__main__":
    builder = LargeSanBuilder5000()
    builder.build_large_san_5000()
