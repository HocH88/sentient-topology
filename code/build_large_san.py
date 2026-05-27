import os
import re
import json
import urllib.request
from collections import Counter, defaultdict
import math

class GutenbergSanBuilder:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.books_dir = os.path.join(data_dir, "books")
        
        # Ensure directories exist
        os.makedirs(self.books_dir, exist_ok=True)
        
        # Curated affect-rich works from Project Gutenberg's Top 100 (extended 2026-05-24).
        # Selection criteria: human emotional depth, narrative interiority, philosophical reflection.
        # Excluded: factbooks, manuals, dictionaries, foreign-language-only works without English edition.
        self.curated_books = {
            # === Original 8 (Phase 1.2 baseline) ===
            "174":  {"title": "The Picture of Dorian Gray",       "author": "Oscar Wilde"},
            "84":   {"title": "Frankenstein",                     "author": "Mary Shelley"},
            "2701": {"title": "Moby Dick",                        "author": "Herman Melville"},
            "1342": {"title": "Pride and Prejudice",              "author": "Jane Austen"},
            "219":  {"title": "Heart of Darkness",                "author": "Joseph Conrad"},
            "5200": {"title": "Metamorphosis",                    "author": "Franz Kafka"},
            "11":   {"title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
            "1661": {"title": "The Adventures of Sherlock Holmes","author": "Arthur Conan Doyle"},

            # === Russian Affective Masters ===
            "2554": {"title": "Crime and Punishment",             "author": "Fyodor Dostoyevsky"},
            "28054":{"title": "The Brothers Karamazov",           "author": "Fyodor Dostoyevsky"},
            "36034":{"title": "White Nights, and Other Stories",  "author": "Fyodor Dostoyevsky"},

            # === British Interiority ===
            "1260": {"title": "Jane Eyre",                        "author": "Charlotte Bronte"},
            "768":  {"title": "Wuthering Heights",                "author": "Emily Bronte"},
            "145":  {"title": "Middlemarch",                      "author": "George Eliot"},
            "394":  {"title": "Cranford",                         "author": "Elizabeth Gaskell"},
            "98":   {"title": "A Tale of Two Cities",             "author": "Charles Dickens"},
            "1400": {"title": "Great Expectations",               "author": "Charles Dickens"},
            "2641": {"title": "A Room with a View",               "author": "E. M. Forster"},
            "16389":{"title": "The Enchanted April",              "author": "Elizabeth von Arnim"},

            # === American Interiority ===
            "64317":{"title": "The Great Gatsby",                 "author": "F. Scott Fitzgerald"},
            "37106":{"title": "Little Women",                     "author": "Louisa May Alcott"},
            "76":   {"title": "Adventures of Huckleberry Finn",   "author": "Mark Twain"},
            "74":   {"title": "The Adventures of Tom Sawyer",     "author": "Mark Twain"},
            "75201":{"title": "A Farewell to Arms",               "author": "Ernest Hemingway"},
            "23":   {"title": "Narrative of the Life of Frederick Douglass","author": "Frederick Douglass"},
            "205":  {"title": "Walden",                           "author": "Henry David Thoreau"},

            # === Tragic / Existential ===
            "43":   {"title": "Dr. Jekyll and Mr. Hyde",          "author": "Robert Louis Stevenson"},
            "345":  {"title": "Dracula",                          "author": "Bram Stoker"},
            "1695": {"title": "The Man Who Was Thursday",         "author": "G. K. Chesterton"},
            "8492": {"title": "The King in Yellow",               "author": "Robert W. Chambers"},

            # === Poetry / Drama ===
            "1513": {"title": "Romeo and Juliet",                 "author": "William Shakespeare"},
            "1727": {"title": "The Odyssey",                      "author": "Homer"},
            # "26471": Spoon River Anthology — pg26471 cache 404 as of 2026-05-24, find alternative ID later
            "2542": {"title": "A Doll's House",                   "author": "Henrik Ibsen"},
            "844":  {"title": "The Importance of Being Earnest",  "author": "Oscar Wilde"},
            "16328":{"title": "Beowulf",                          "author": "Anonymous"},

            # === Philosophy / Wisdom ===
            "1998": {"title": "Thus Spake Zarathustra",           "author": "Friedrich Nietzsche"},
            "2680": {"title": "Meditations",                      "author": "Marcus Aurelius"},
            "3296": {"title": "The Confessions of St. Augustine", "author": "Saint Augustine"},
            "1080": {"title": "A Modest Proposal",                "author": "Jonathan Swift"},

            # === Additional Bronte / Austen / Twain coverage ===
            "161":  {"title": "Sense and Sensibility",            "author": "Jane Austen"},
            "86":   {"title": "A Connecticut Yankee in King Arthur's Court","author": "Mark Twain"},
        }
        
        # Core concept dictionary to extract (representing core human/affective nodes)
        self.core_concepts = {
            # Heavy / Negative Sensation
            "sorrow": ["sorrow", "grief", "sadness", "mourn", "weep"],
            "guilt": ["guilt", "shame", "remorse", "blame", "repent"],
            "desolation": ["desolation", "loneliness", "isolated", "barren", "empty"],
            "anger": ["anger", "fury", "rage", "wrath", "bitter"],
            "fear": ["fear", "dread", "terror", "horror", "fright"],
            "pain": ["pain", "agony", "suffering", "misery", "hurt"],
            
            # Bright / Stable Sensation
            "joy": ["joy", "delight", "happiness", "bliss", "glad"],
            "gratitude": ["gratitude", "thankful", "appreciation", "bless"],
            "peace": ["peace", "tranquility", "calm", "stillness", "quiet"],
            "fullness": ["fullness", "satisfaction", "contentment", "complete"],
            "love": ["love", "affection", "beloved", "darling", "fondness"],
            "hope": ["hope", "aspiration", "expectation", "trust"],
            
            # Aesthetic / Existential
            "reverence": ["reverence", "awe", "solemn", "veneration", "holy"],
            "melancholy": ["melancholy", "wistful", "pensive", "sadly"],
            "eternity": ["eternity", "infinite", "immortal", "forever", "timeless"],
            "death": ["death", "dying", "grave", "dead", "comrade"],
            "soul": ["soul", "spirit", "essence", "mind", "conscience"],
            "nature": ["nature", "sunset", "dusk", "sea", "hill", "woods"]
        }
        
        # Flatten dictionary to map any synonym to its core concept
        self.synonym_to_concept = {}
        for concept, synonyms in self.core_concepts.items():
            for syn in synonyms:
                self.synonym_to_concept[syn.lower()] = concept

    def download_book(self, book_id):
        """
        Download book from Project Gutenberg cache.
        """
        local_path = os.path.join(self.books_dir, f"pg{book_id}.txt")
        if os.path.exists(local_path):
            print(f" - [Cached] pg{book_id}.txt already exists.")
            return local_path
            
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f" - Downloading pg{book_id} from Gutenberg...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                
            # Clean Project Gutenberg headers/footers roughly
            content_clean = self._clean_gutenberg_text(content)
            
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content_clean)
            print(f"   -> Saved and cleaned: pg{book_id}.txt")
            return local_path
        except Exception as e:
            print(f"   -> Error downloading pg{book_id}: {e}")
            return None

    def _clean_gutenberg_text(self, text):
        """
        Strip Project Gutenberg headers and footers.
        """
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        
        start_match = re.search(start_pattern, text)
        end_match = re.search(end_pattern, text)
        
        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(text)
        
        return text[start_idx:end_idx]

    def _tokenize(self, text):
        """
        Basic regex tokenizer. Returns a list of lowercase alphanumeric tokens.
        """
        return re.findall(r"\b[a-zA-Z']+\b", text.lower())

    def extract_associations(self, window_size=40, ppmi_threshold=1.5):
        """
        Parses all downloaded books, calculates co-occurrences in a sliding window,
        and computes Pointwise Mutual Information (PMI) to build weighted edges.
        """
        print(f"\n[Step 2] Processing texts and calculating co-occurrences (Window: {window_size})...")
        
        word_counts = Counter()
        co_counts = defaultdict(Counter)
        total_tokens = 0
        
        # Read and tokenize all cached texts
        for file in os.listdir(self.books_dir):
            if file.endswith(".txt"):
                path = os.path.join(self.books_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                tokens = self._tokenize(text)
                total_tokens += len(tokens)
                print(f" - Parsed {file}: {len(tokens)} tokens")
                
                # Sliding window co-occurrence counting
                # To speed up, we only track occurrences of our core concepts (mapped synonyms)
                mapped_tokens = [self.synonym_to_concept.get(t, None) for t in tokens]
                
                for i in range(len(mapped_tokens)):
                    w1 = mapped_tokens[i]
                    if w1 is None:
                        continue
                        
                    word_counts[w1] += 1
                    
                    # Check window
                    start = max(0, i - window_size)
                    end = min(len(mapped_tokens), i + window_size + 1)
                    
                    for j in range(start, end):
                        if i == j:
                            continue
                        w2 = mapped_tokens[j]
                        if w2 is not None and w1 != w2:
                            co_counts[w1][w2] += 1
                            
        # Compute PPMI as edge weight
        # PMI(w1, w2) = log2( P(w1, w2) / (P(w1) * P(w2)) )
        # PPMI(w1, w2) = max(0, PMI)
        edges = []
        concepts = list(self.core_concepts.keys())
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                c1 = concepts[i]
                c2 = concepts[j]
                
                c_joint = co_counts[c1][c2]
                if c_joint <= 2: # Ignore rare co-occurrences
                    continue
                    
                p_c1 = word_counts[c1] / total_tokens
                p_c2 = word_counts[c2] / total_tokens
                p_joint = c_joint / (total_tokens * 2 * window_size) # Approximate joint probability in window
                
                pmi = math.log2(p_joint / (p_c1 * p_c2))
                ppmi = max(0.0, pmi)
                
                if ppmi >= ppmi_threshold:
                    # Normalize weight to [0, 1] range roughly
                    weight = min(1.0, ppmi / 10.0)
                    edges.append({
                        "source": c1,
                        "target": c2,
                        "weight": float(weight)
                    })
                    
        print(f" - Extracted {len(edges)} weighted edges above PPMI threshold {ppmi_threshold}.")
        return edges, word_counts

    def build_and_save_san(self):
        """
        Executes downloading, parsing, and saves the large-scale SAN graph.
        """
        print("==================================================================")
        print("          AUTOMATED PROJECT GUTENBERG SAN BUILDER")
        print("==================================================================")
        
        # 1. Download curated books
        print("[Step 1] Downloading curated literature corpus...")
        for bid, info in self.curated_books.items():
            print(f" - Book: '{info['title']}' by {info['author']}")
            self.download_book(bid)
            
        # 2. Extract weighted edges using PMI
        edges, counts = self.extract_associations()
        
        # 3. Assemble JSON file
        nodes = []
        for concept, synonyms in self.core_concepts.items():
            # Estimate node base priority/size from corpus occurrence counts
            occurrence = counts.get(concept, 0)
            nodes.append({
                "id": concept,
                "type": "sensation" if concept in ["sorrow", "guilt", "desolation", "anger", "fear", "pain", "joy", "gratitude", "peace", "fullness", "love", "hope"] else "association",
                "description": f"Core concept representing {concept}. Synonyms: {', '.join(synonyms)}. Frequency: {occurrence}",
                "frequency": occurrence
            })
            
        san_data = {
            "metadata": {
                "description": "Large-Scale SAN extracted from Gutenberg literature corpus.",
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            },
            "nodes": nodes,
            "edges": edges
        }
        
        out_path = os.path.join(self.data_dir, "large_san.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(san_data, f, indent=4)
            
        print(f"\n[Step 3] SAN compiled successfully!")
        print(f" - Nodes: {len(nodes)}")
        print(f" - Edges: {len(edges)}")
        print(f" - Saved to: {out_path}")
        print("==================================================================")

if __name__ == "__main__":
    builder = GutenbergSanBuilder()
    builder.build_and_save_san()
