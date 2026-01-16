import os
import pickle
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# COLLABORATIVE FILTERING MODELS
# ============================================================================

class KNNRecommenderCF:
    """K-Nearest Neighbors Collaborative Filtering (User-based and Item-based)"""
    
    def __init__(self, k_vals=[5, 10, 20, 50]):
        self.k_vals = k_vals
        self.best_k = {'user': 50, 'item': 5}
        self.tune_results = {'user': {}, 'item': {}}
        
        self.pivot = None
        self.sparse_user = None
        self.sparse_item = None
        self.user_knn = None
        self.item_knn = None

    def _recommend_core(self, pivot, sparse_data, knn, user_id, n_recs, mode, sim_graph=None, revisit=True):
        """Core recommendation logic for both user-based and item-based."""
        
        # Ensure user_id is string and exists
        user_id_str = str(user_id)
        if user_id_str not in pivot.index:
            print(f"Warning: User {user_id_str} not found in training data")
            return []
        
        u_idx = pivot.index.get_loc(user_id_str)

        # ---------- USER-BASED ----------
        if mode == "user":
            # Find nearest neighbor users
            _, idx = knn.kneighbors(sparse_data[u_idx])
            neighbors = idx.flatten()[1:]  # Exclude self (first neighbor)

            # Aggregate items from neighbor users
            scores = pivot.iloc[neighbors].sum(axis=0)
            if not revisit:
                scores[pivot.iloc[u_idx] > 0] = -1  # Mask already seen items

            return [str(i) for i in scores.sort_values(ascending=False).head(n_recs).index]

        # ---------- ITEM-BASED ----------
        seen = np.where(pivot.iloc[u_idx] > 0)[0]
        if len(seen) == 0:
            return []
        
        # Use pre-computed similarity graph
        if hasattr(sim_graph[seen].sum(axis=0), 'A1'):
            item_scores = sim_graph[seen].sum(axis=0).A1  # Sparse matrix
        else:
            item_scores = np.asarray(sim_graph[seen].sum(axis=0)).flatten()  # Dense matrix
            
        if not revisit:
            item_scores[seen] = -1  # Mask already seen items

        top = np.argsort(item_scores)[::-1][:n_recs]
        return [str(pivot.columns[i]) for i in top]
    
    def get_k_recommend(self, user_id, k=5, mode="user"):
        """Generate top-k recommendations for a user."""
        
        if self.pivot is None:
            raise ValueError("Call fit() before get_k_recommend().")

        if mode == "user":
            return self._recommend_core(
                self.pivot, self.sparse_user, self.user_knn, user_id, k, "user"
            )

        # Pre-compute similarity graph for item-based
        sim_graph = self.item_knn.kneighbors_graph(self.sparse_item, mode='connectivity')
        return self._recommend_core(
            self.pivot, self.sparse_item, self.item_knn, user_id, k, "item", sim_graph
        )


class ALSRecommender:
    """Alternating Least Squares Matrix Factorization Recommender"""
    
    def __init__(self, factors_list=[32, 64, 128], reg_list=[0.01, 0.1, 1.0]):
        self.factors_list = factors_list
        self.reg_list = reg_list
        
        self.best_params = {"factors": 64, "regularization": 0.1}
        self.tune_results = {}
        self.final_model = None
        
        # Stored after fit()
        self.pivot = None
        self.sparse = None
        self.user_list = None
        self.item_list = None
    
    def get_k_recommend(self, user_id, k=5):
        """Generate top-k recommendations for a user."""
        
        if self.final_model is None:
            raise ValueError("Call fit() before get_k_recommend().")

        user_id_str = str(user_id)
        if user_id_str not in self.pivot.index:
            return []
            
        u_idx = self.pivot.index.get_loc(user_id_str)

        ids, _ = self.final_model.recommend(
            u_idx,
            self.sparse[u_idx],
            N=k,
            filter_already_liked_items=False
        )

        return self.item_list[ids].tolist()


class KMeansRecommender:
    """Cluster-based Collaborative Filtering using K-Means"""
    
    def __init__(self):
        self.best_params = {"n_clusters": 10}
        self.tune_results = {}
        self.final_model = None
        
        # Stored after fit()
        self.features = None
        self.interactions = None
        self.user_clusters = None
        self.cluster_top_items = {}
    
    def get_k_recommend(self, user_id, k=5):
        """Generate top-k recommendations for a user."""
        
        if self.final_model is None:
            raise ValueError("Call fit() before get_k_recommend().")

        user_str = str(user_id)
        
        if user_str not in self.features.index:
            return []
        
        cluster = self.features.loc[user_str, 'cluster']
        return self.cluster_top_items[cluster][:k]


# ============================================================================
# CONTENT-BASED MODELS
# ============================================================================

class ContentModel:
    """Base class for content-based recommenders"""
    
    def __init__(self, train_df: pd.DataFrame, attr_df: pd.DataFrame,
                 id_map: Dict, idx_map: Dict):
        self.train_df = train_df
        self.attr_df = attr_df
        
        # Pre-compute user histories for fast lookup
        self.user_histories = train_df.groupby('user_id')['vroot_id'].apply(set).to_dict()
        self.id_to_idx = id_map
        self.idx_to_id = idx_map

    def get_user_history(self, user_id) -> Set:
        """Get the set of items a user has interacted with."""
        return self.user_histories.get(user_id, set())

    def _filter_recs(self, candidates: np.ndarray, history: Set, k: int) -> List:
        """Filters out items already seen by the user."""
        recs = []
        for idx in candidates:
            v_id = self.idx_to_id[idx]
            if v_id not in history:
                recs.append(v_id)
                if len(recs) >= k:
                    break
        return recs


class CosineRecommender(ContentModel):
    """Content-based recommender using Cosine Similarity"""
    
    def __init__(self, tfidf_matrix, train_data: pd.DataFrame,
                 attr_data: pd.DataFrame, id_map: Dict, idx_map: Dict):
        super().__init__(train_data, attr_data, id_map, idx_map)
        
        # Pre-compute similarity matrix
        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    def recommend(self, user_id, k: int = 5) -> List:
        """Generate top-k recommendations for a user."""
        
        user_history = self.get_user_history(user_id)
        if not user_history:
            return []

        # Aggregate similarity scores
        total_scores = np.zeros(self.sim_matrix.shape[0])

        for item_id in user_history:
            if item_id in self.id_to_idx:
                idx = self.id_to_idx[item_id]
                total_scores += self.sim_matrix[idx]

        # Sort indices by score (descending)
        top_indices = total_scores.argsort()[::-1]

        # Filter out items already visited
        return self._filter_recs(top_indices, user_history, k)


class KNNRecommender(ContentModel):
    """Content-based recommender using K-Nearest Neighbors"""
    
    def __init__(self, tfidf_matrix, train_df: pd.DataFrame,
                 attr_df: pd.DataFrame, id_map: Dict, idx_map: Dict,
                 k_neighbors: int = 10):
        super().__init__(train_df, attr_df, id_map, idx_map)
        self.k_neighbors = k_neighbors
        self.tfidf_matrix = tfidf_matrix
        
        # Model: Find items similar to the user's profile
        self.model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=k_neighbors
        )
        self.model.fit(tfidf_matrix)

    def recommend(self, user_id, k: int = 5) -> List:
        """Generate top-k recommendations for a user."""
        
        history = self.get_user_history(user_id)
        if not history:
            return []

        # 1. Build User Profile (Average of TF-IDF vectors)
        indices = [self.id_to_idx[i] for i in history if i in self.id_to_idx]
        if not indices:
            return []

        user_profile = np.asarray(self.tfidf_matrix[indices].mean(axis=0))

        # 2. Find Neighbors
        n_query = min(self.k_neighbors + len(history), self.tfidf_matrix.shape[0])
        _, neighbor_indices = self.model.kneighbors(user_profile, n_neighbors=n_query)

        return self._filter_recs(neighbor_indices.flatten(), history, k)


class SVDRecommender(ContentModel):
    """Content-based recommender using SVD/LSA for dimensionality reduction"""
    
    def __init__(self, tfidf_matrix, train_df: pd.DataFrame,
                 attr_df: pd.DataFrame, id_map: Dict, idx_map: Dict,
                 n_components: int = 50):
        super().__init__(train_df, attr_df, id_map, idx_map)
        
        # Dimensionality Reduction (LSA)
        n_comps = min(n_components, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comps, random_state=42)
        matrix_reduced = self.svd.fit_transform(tfidf_matrix)

        # Compute Similarity on Reduced Matrix
        self.sim_matrix = cosine_similarity(matrix_reduced)

    def recommend(self, user_id, k: int = 5) -> List:
        """Generate top-k recommendations for a user."""
        
        history = self.get_user_history(user_id)
        if not history:
            return []

        # Aggregate similarity scores
        total_scores = np.zeros(self.sim_matrix.shape[0])
        for item_id in history:
            if item_id in self.id_to_idx:
                idx = self.id_to_idx[item_id]
                total_scores += self.sim_matrix[idx]

        # Sort by score
        top_indices = total_scores.argsort()[::-1]

        return self._filter_recs(top_indices, history, k)


# ============================================================================
# HYBRID RECOMMENDATION FUNCTIONS
# ============================================================================

def get_candidates(user_str, train_binary, item_sim_df, item_popularity, 
                   X_test_tfidf, n_candidates=100):
    """Generate candidate items using multiple strategies."""
    
    scored_candidates = {}

    # Strategy 1: Collaborative (Weight 1.0)
    if user_str in train_binary.index:
        user_hist = train_binary.columns[train_binary.loc[user_str] > 0]
        if len(user_hist) > 0:
            sim_scores = item_sim_df.loc[:, user_hist].mean(axis=1).sort_values(
                ascending=False).head(40)
            for i, (idx, score) in enumerate(sim_scores.items()):
                scored_candidates[idx] = scored_candidates.get(idx, 0) + (1.0 - (i/40))

    # Strategy 2: Content-Based (Weight 1.0)
    if user_str in X_test_tfidf.index:
        content_scores = X_test_tfidf.loc[user_str].sort_values(ascending=False).head(40)
        for i, (idx, score) in enumerate(content_scores.items()):
            if score > 0:
                scored_candidates[idx] = scored_candidates.get(idx, 0) + (1.0 - (i/40))

    # Strategy 3: Popularity Fallback
    pop_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:40]
    for i, (idx, count) in enumerate(pop_items):
        scored_candidates[idx] = scored_candidates.get(idx, 0) + (0.5 - (i/80))

    # Sort by combined score and take top N
    final_candidates = sorted(scored_candidates.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in final_candidates[:n_candidates]]


def generate_hybrid_recommendations(stacking_model, user_str, candidates, 
                                   train_binary, item_popularity, item_sim_df, 
                                   top_n=5):
    """Generate top-N recommendations using stacking model."""
    
    # Extract features for all candidates
    cand_features = []
    valid_candidates = []
    max_pop = max(item_popularity.values()) if item_popularity else 1
    max_items = train_binary.shape[1]

    for i_id in candidates:
        # Feature 1: User activity level
        f1 = train_binary.loc[user_str].sum() / max_items if user_str in train_binary.index else 0
        
        # Feature 2: Item popularity
        f2 = item_popularity.get(i_id, 0) / max_pop

        # Feature 3: Average similarity to user's history
        if user_str in train_binary.index:
            user_history = train_binary.columns[train_binary.loc[user_str] > 0]
            other_items = [item for item in user_history if item != i_id]
            f3 = (item_sim_df.loc[other_items, i_id].mean() 
                  if len(other_items) > 0 and i_id in item_sim_df.columns else 0)
        else:
            f3 = 0
        
        cand_features.append([f1, f2, f3])
        valid_candidates.append(i_id)

    if len(cand_features) == 0:
        return [item[0] for item in sorted(item_popularity.items(), 
                                          key=lambda x: x[1], reverse=True)[:top_n]]

    # Get predictions
    X_cand = np.array(cand_features)
    proba = stacking_model.predict_proba(X_cand)[:, 1]

    # Get top-N
    top_indices = np.argsort(proba)[-top_n:][::-1]
    recommendations = [valid_candidates[i] for i in top_indices]

    return recommendations


# ============================================================================
# DISPLAY UTILITIES
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_models(models_dir='saved_models'):
    """Load all saved models and data from pickle files."""
    
    print(f"\n{Colors.BLUE}Loading models from '{models_dir}/'...{Colors.END}")
    
    models = {}
    
    try:
        # ================================================================
        # COLLABORATIVE FILTERING MODELS
        # ================================================================
        print(f"\n{Colors.CYAN}[Collaborative Filtering]{Colors.END}")
        
        try:
            with open(os.path.join(models_dir, 'knn_cf_recommender.pkl'), 'rb') as f:
                models['knn_cf'] = pickle.load(f)
        except FileNotFoundError:
            with open(os.path.join(models_dir, 'knn_recommender.pkl'), 'rb') as f:
                models['knn_cf'] = pickle.load(f)
        print(f"  ✓ KNN CF (User-K={models['knn_cf'].best_k['user']}, "
              f"Item-K={models['knn_cf'].best_k['item']})")
        
        with open(os.path.join(models_dir, 'kmeans_recommender.pkl'), 'rb') as f:
            models['kmeans'] = pickle.load(f)
        print(f"  ✓ KMeans (n_clusters={models['kmeans'].best_params['n_clusters']})")
        
        with open(os.path.join(models_dir, 'full_train_matrix.pkl'), 'rb') as f:
            models['train_matrix'] = pickle.load(f)
        print(f"  ✓ Training matrix loaded")
        
        with open(os.path.join(models_dir, 'ground_truth.pkl'), 'rb') as f:
            models['ground_truth'] = pickle.load(f)
        print(f"  ✓ Ground truth ({len(models['ground_truth'])} users)")
        
        with open(os.path.join(models_dir, 'attributes.pkl'), 'rb') as f:
            models['attributes'] = pickle.load(f)
        print(f"  ✓ Attributes loaded")
        
        # ================================================================
        # CONTENT-BASED MODELS
        # ================================================================
        print(f"\n{Colors.YELLOW}[Content-Based]{Colors.END}")
        
        try:
            with open(os.path.join(models_dir, 'cosine_model.pkl'), 'rb') as f:
                models['cosine'] = pickle.load(f)
            print(f"  ✓ Cosine Model loaded")
        except FileNotFoundError:
            models['cosine'] = None
            print(f"  ⚠ Cosine Model not found")
        
        try:
            with open(os.path.join(models_dir, 'knn_content_model.pkl'), 'rb') as f:
                models['knn_content'] = pickle.load(f)
            print(f"  ✓ KNN Content Model loaded")
        except FileNotFoundError:
            models['knn_content'] = None
            print(f"  ⚠ KNN Content Model not found")
        
        try:
            with open(os.path.join(models_dir, 'svd_model.pkl'), 'rb') as f:
                models['svd'] = pickle.load(f)
            print(f"  ✓ SVD Model loaded")
        except FileNotFoundError:
            models['svd'] = None
            print(f"  ⚠ SVD Model not found")
        
        # ================================================================
        # HYBRID MODELS
        # ================================================================
        print(f"\n{Colors.WHITE}[Hybrid]{Colors.END}")
        
        try:
            with open(os.path.join(models_dir, 'stacking1.pkl'), 'rb') as f:
                models['stacking1'] = pickle.load(f)
            print(f"  ✓ Stacking1 (RF+GB/LR) loaded")
        except FileNotFoundError:
            models['stacking1'] = None
            print(f"  ⚠ Stacking1 not found")
        
        try:
            with open(os.path.join(models_dir, 'stacking2.pkl'), 'rb') as f:
                models['stacking2'] = pickle.load(f)
            print(f"  ✓ Stacking2 (KNN+XGB+GB/LR) loaded")
        except FileNotFoundError:
            models['stacking2'] = None
            print(f"  ⚠ Stacking2 not found")
        
        try:
            with open(os.path.join(models_dir, 'stacking3.pkl'), 'rb') as f:
                models['stacking3'] = pickle.load(f)
            print(f"  ✓ Stacking3 (LR+RF+XGB/XGB) loaded")
        except FileNotFoundError:
            models['stacking3'] = None
            print(f"  ⚠ Stacking3 not found")
        
        # Load hybrid data dependencies
        try:
            with open(os.path.join(models_dir, 'train_binary.pkl'), 'rb') as f:
                models['train_binary'] = pickle.load(f)
            with open(os.path.join(models_dir, 'item_popularity.pkl'), 'rb') as f:
                models['item_popularity'] = pickle.load(f)
            with open(os.path.join(models_dir, 'item_sim_df.pkl'), 'rb') as f:
                models['item_sim_df'] = pickle.load(f)
            with open(os.path.join(models_dir, 'X_test_tfidf.pkl'), 'rb') as f:
                models['X_test_tfidf'] = pickle.load(f)
            print(f"  ✓ Hybrid data loaded")
        except FileNotFoundError:
            models['train_binary'] = None
            models['item_popularity'] = None
            models['item_sim_df'] = None
            models['X_test_tfidf'] = None
            print(f"  ⚠ Hybrid data not found")
        
        return models
        
    except FileNotFoundError as e:
        print(f"\n{Colors.RED}Error: Required model files not found!{Colors.END}")
        print(f"Please run the notebook first to train and save models.")
        print(f"Missing file: {e}")
        return None


def get_page_title(vroot_id, attributes):
    """Get the title of a page given its vroot_id."""
    
    try:
        vroot_id_int = int(vroot_id) if isinstance(vroot_id, str) else vroot_id
        match = attributes[attributes['vroot_id'] == vroot_id_int]
        if len(match) > 0:
            return match.iloc[0]['title']
    except (ValueError, KeyError, IndexError):
        pass
    return f"Page {vroot_id}"


def display_pages(page_ids, attributes, title, ground_truth_items=None):
    """Display a list of pages with their titles."""
    
    if title:
        print(f"\n{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
        print("-" * 60)
    
    if not page_ids:
        print(f"  {Colors.YELLOW}(No pages){Colors.END}")
        return 0
    
    correct_count = 0
    for i, pid in enumerate(page_ids[:7], 1):
        page_title = get_page_title(pid, attributes)
        
        if ground_truth_items is not None:
            if str(pid) in ground_truth_items or pid in ground_truth_items:
                color = Colors.GREEN
                marker = "✓"
                correct_count += 1
            else:
                color = Colors.RED
                marker = "✗"
            print(f"  {color}{marker} {i}. [{pid}] {page_title}{Colors.END}")
        else:
            print(f"  {i}. [{pid}] {page_title}")
    
    return correct_count


def display_model_section(title, color):
    """Print section header."""
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{color}  {title}{Colors.END}")
    print(f"  {Colors.GREEN}Green = Correct{Colors.END} | "
          f"{Colors.RED}Red = Not in Test{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")


def display_single_model(model_name, recs, attributes, test_items_set, color):
    """Display recommendations from a single model with score."""
    
    print(f"\n{color}{Colors.BOLD}🔹 {model_name}{Colors.END}")
    if not recs:
        print(f"  {Colors.YELLOW}(No recommendations){Colors.END}")
        return
    
    correct = display_pages(recs, attributes, "", ground_truth_items=test_items_set)
    print(f"  {Colors.CYAN}Score: {correct}/5 correct ({correct/5*100:.1f}%){Colors.END}")


def try_content_recommend(model, user_id):
    """Try recommending with different user ID formats for content-based models."""
    
    if hasattr(model, 'user_histories') and model.user_histories:
        sample_key = next(iter(model.user_histories.keys()))
        key_type = type(sample_key)
        
        try:
            if key_type == int:
                lookup_id = int(user_id)
            elif key_type == str:
                lookup_id = str(user_id)
            else:
                lookup_id = user_id
            
            recs = model.recommend(lookup_id, k=5)
            return recs if recs else []
        except (ValueError, KeyError, IndexError):
            pass
    
    # Fallback: try both formats
    for uid in [int(user_id) if str(user_id).isdigit() else user_id, str(user_id)]:
        try:
            recs = model.recommend(uid, k=5)
            if recs:
                return recs
        except (ValueError, KeyError, IndexError):
            pass
    
    return []


def display_user_info(user_id, models):
    """Display user's training history and test pages."""
    
    train_matrix = models['train_matrix']
    ground_truth = models['ground_truth']
    attributes = models['attributes']
    user_str = str(user_id)
    
    print("\n" + "=" * 70)
    print(f"{Colors.CYAN}{Colors.BOLD}  🎯 RECOMMENDATION VIEWER - "
          f"User ID: {user_id}{Colors.END}")
    print("=" * 70)
    
    # Get user history
    if user_str not in train_matrix.index:
        print(f"\n{Colors.YELLOW}Warning: User {user_id} not found in "
              f"training data{Colors.END}")
        user_history = []
    else:
        user_row = train_matrix.loc[user_str]
        user_history = [str(col) for col in user_row.index[user_row > 0]]
    
    # Get test items
    test_items = ground_truth.get(user_str, [])
    test_items_set = set(str(x) for x in test_items)
    
    display_pages(user_history, attributes, 
                 f"📚 Pages Visited in Training ({len(user_history)} pages)")
    display_pages(test_items, attributes, 
                 f"🎯 Pages Visited in Test File ({len(test_items)} pages)")
    
    return test_items_set


def display_cf_recommendations(user_id, models, test_items_set):
    """Display Collaborative Filtering model recommendations."""
    
    attributes = models['attributes']
    
    display_model_section("📊 COLLABORATIVE FILTERING MODELS", Colors.MAGENTA)
    
    # KNN User-Based
    knn_cf = models['knn_cf']
    try:
        recs = knn_cf.get_k_recommend(user_id, k=5, mode='user')
        display_single_model(f"KNN User-Based (k={knn_cf.best_k['user']})", 
                           recs, attributes, test_items_set, Colors.MAGENTA)
    except Exception as e:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}🔹 KNN User-Based{Colors.END}")
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    # KNN Item-Based
    try:
        recs = knn_cf.get_k_recommend(user_id, k=5, mode='item')
        display_single_model(f"KNN Item-Based (k={knn_cf.best_k['item']})", 
                           recs, attributes, test_items_set, Colors.MAGENTA)
    except Exception as e:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}🔹 KNN Item-Based{Colors.END}")
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    # KMeans
    kmeans = models['kmeans']
    try:
        recs = kmeans.get_k_recommend(user_id, k=5)
        display_single_model(
            f"KMeans (n_clusters={kmeans.best_params['n_clusters']})", 
            recs, attributes, test_items_set, Colors.MAGENTA)
    except Exception as e:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}🔹 KMeans{Colors.END}")
        print(f"  {Colors.RED}Error: {e}{Colors.END}")


def display_content_recommendations(user_id, models, test_items_set):
    """Display Content-Based model recommendations."""
    
    if not any([models.get('cosine'), models.get('knn_content'), models.get('svd')]):
        return
    
    attributes = models['attributes']
    display_model_section("📝 CONTENT-BASED MODELS", Colors.YELLOW)
    
    # Cosine
    if models.get('cosine'):
        try:
            recs = try_content_recommend(models['cosine'], user_id)
            display_single_model("Cosine Similarity", recs, attributes, 
                               test_items_set, Colors.YELLOW)
        except Exception as e:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}🔹 Cosine Similarity{Colors.END}")
            print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    # KNN Content
    if models.get('knn_content'):
        try:
            recs = try_content_recommend(models['knn_content'], user_id)
            display_single_model("KNN Content-Based", recs, attributes, 
                               test_items_set, Colors.YELLOW)
        except Exception as e:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}🔹 KNN Content-Based{Colors.END}")
            print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    # SVD
    if models.get('svd'):
        try:
            recs = try_content_recommend(models['svd'], user_id)
            display_single_model("SVD/LSA", recs, attributes, 
                               test_items_set, Colors.YELLOW)
        except Exception as e:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}🔹 SVD/LSA{Colors.END}")
            print(f"  {Colors.RED}Error: {e}{Colors.END}")


def display_hybrid_recommendations(user_id, models, test_items_set):
    """Display Hybrid model recommendations."""
    
    if not any([models.get('stacking1'), models.get('stacking2'), models.get('stacking3')]):
        return
    if models.get('train_binary') is None:
        return
    
    attributes = models['attributes']
    user_str = str(user_id)
    
    display_model_section("🔀 HYBRID MODELS (Stacking)", Colors.WHITE)
    
    try:
        candidates = get_candidates(
            user_str, 
            models['train_binary'], 
            models['item_sim_df'], 
            models['item_popularity'], 
            models['X_test_tfidf']
        )
    except Exception as e:
        print(f"  {Colors.RED}Error getting candidates: {e}{Colors.END}")
        return
    
    if not candidates:
        print(f"  {Colors.YELLOW}(No candidates generated){Colors.END}")
        return
    
    # Stacking models
    stacking_models = [
        ('stacking1', "Hybrid Combo1 (RF+GB / LR)"),
        ('stacking2', "Hybrid Combo2 (KNN+XGB+GB / LR)"),
        ('stacking3', "Hybrid Combo3 (LR+RF+XGB / XGB)")
    ]
    
    for model_key, model_name in stacking_models:
        if models.get(model_key):
            try:
                recs = generate_hybrid_recommendations(
                    models[model_key], user_str, candidates,
                    models['train_binary'], models['item_popularity'], 
                    models['item_sim_df']
                )
                display_single_model(model_name, recs, attributes, 
                                   test_items_set, Colors.WHITE)
            except Exception as e:
                print(f"\n{Colors.WHITE}{Colors.BOLD}🔹 {model_name}{Colors.END}")
                print(f"  {Colors.RED}Error: {e}{Colors.END}")


def display_recommendations(user_id, models):
    """Display recommendations from all models for a given user."""
    
    # Display user info and get test items
    test_items_set = display_user_info(user_id, models)
    
    # Display recommendations from each model type
    display_cf_recommendations(user_id, models, test_items_set)
    display_content_recommendations(user_id, models, test_items_set)
    display_hybrid_recommendations(user_id, models, test_items_set)
    
    print("\n" + "=" * 70)


def list_sample_users(models, n=10):
    """List sample users that have both training and test data."""
    
    ground_truth = models['ground_truth']
    train_matrix = models['train_matrix']
    
    common_users = set(train_matrix.index) & set(ground_truth.keys())
    sample = list(common_users)[:n]
    
    print(f"\n{Colors.CYAN}Sample users with both training and test data:{Colors.END}")
    for uid in sample:
        train_count = (train_matrix.loc[uid] > 0).sum()
        test_count = len(ground_truth[uid])
        print(f"  • User {uid}: {train_count} training visits, {test_count} test visits")
    
    return sample


def main():
    """Main interactive loop."""
    
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.BLUE}  🎯 RECOMMENDATION SYSTEM VIEWER{Colors.END}")
    print(f"  CF | Content-Based | Hybrid")
    print("=" * 70)
    
    models = load_models()
    if models is None:
        return
    
    sample_users = list_sample_users(models)
    
    print(f"\n{Colors.YELLOW}Commands:{Colors.END}")
    print(f"  • Enter user ID to see recommendations")
    print(f"  • 'sample' - show more sample users")
    print(f"  • 'random' - pick a random user")
    print(f"  • 'q' - quit")
    
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}Enter User ID: {Colors.END}").strip()
            
            if user_input.lower() == 'q':
                print(f"\n{Colors.CYAN}Goodbye!{Colors.END}\n")
                break
            
            if user_input.lower() == 'sample':
                list_sample_users(models, n=20)
                continue
            
            if user_input.lower() == 'random':
                import random
                user_input = random.choice(list(models['ground_truth'].keys()))
                print(f"  → Random user selected: {user_input}")
            
            display_recommendations(user_input, models)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Interrupted. Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()