import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from schedule_base import Schedule, parse_ckpt_spec
from budget_utils import budget_allocation
import numpy as np
import time
import glob
import re
from pdb import set_trace as breakpoint

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def attn_svd_stability(metrics):
    """"""
    stability_vals = []
    for i in range(len(metrics)):
        traj = metrics[i].cpu().numpy()
        stability_vals.append(float(np.sum(np.abs(np.diff(traj)))))
    return np.array(stability_vals)


def parse_source_config(source_spec):
    """"""
    if not source_spec:
        return None
    
    source_config = {}
    for part in source_spec.split(','):
        part = part.strip()
        if ':' not in part or '-' not in part:
            raise ValueError(f"Invalid source spec format: {part}. Expected format: source:n_clusters-n_samples")
        
        source, config = part.split(':')
        n_clusters, n_samples = config.split('-')
        source_config[source.strip()] = (int(n_clusters), int(n_samples))
    
    return source_config


def compute_softmax_allocation(difficulty_dict, size_dict, total_budget, temperature, name_prefix):
    """"""
    names = list(difficulty_dict.keys())
    
    difficulty_values = np.array([difficulty_dict[name] for name in names])
    softmax_probs = np.exp(difficulty_values / temperature) / np.sum(np.exp(difficulty_values / temperature))
    
    weights = {name: float(softmax_probs[i]) for i, name in enumerate(names)}
    
    print(f"\n*** {name_prefix} softmax weights:")
    for name in sorted(names):
        print(f"  {name}: prob={weights[name]:.4f}, size={size_dict[name]}")
    
    final_allocation = budget_allocation(weights, size_dict, total_budget)
    
    print(f"\n*** Final {name_prefix} allocation:")
    for name in sorted(names):
        initial = int(weights[name] * total_budget)
        final = final_allocation[name]
        if initial != final:
            diff = final - initial
            print(f"  {Colors.RED}{name}: {final} samples (changed by {diff:+d} from initial {initial}){Colors.RESET}")
        else:
            print(f"  {Colors.GREEN}{name}: {final} samples{Colors.RESET}")
    
    total_allocated = sum(final_allocation.values())
    print(f"\n*** Total {name_prefix} allocated: {total_allocated}/{total_budget}")
    
    return final_allocation, weights


def compute_softmax_source_allocation(losses, sources, domains, temperature, total_samples, n_sample_per_cluster=3, score_type='geo_mean', no_domain=False, same_learnability=None, original_losses=None):
    """"""
    if score_type == 'geo_mean':
        if losses.shape[1] < 2:
            raise ValueError(f"Need at least 2 checkpoints for geo_mean softmax allocation, but got {losses.shape[1]}")
    elif score_type in ['loss', 'neg_loss']:
        if losses.shape[1] < 1:
            raise ValueError(f"Need at least 1 checkpoint for {score_type} softmax allocation, but got {losses.shape[1]}")
    else:
        raise ValueError(f"Unknown score_type for softmax allocation: {score_type}")
    
    first_loss = losses[:, 0]
    last_loss = losses[:, -1]
    
    print(f"\n*** Hierarchical softmax-based allocation (T={temperature})")
    if no_domain:
        print(f"*** Skipping domain-level allocation (--no_domain=True)")
    if score_type == 'geo_mean':
        if same_learnability == 1000:
            print(f"*** Using difficulty metric: sqrt(last_loss * learnability1000) where learnability1000 uses original loss")
        else:
            print(f"*** Using difficulty metric: sqrt((last_loss - first_loss) * last_loss)")
    elif score_type == 'loss':
        print(f"*** Using difficulty metric: last_loss")
    elif score_type == 'neg_loss':
        print(f"*** Using difficulty metric: -last_loss (lower loss -> more budget)")
    
    if same_learnability == 1000 and score_type == 'geo_mean':
        if original_losses is None:
            raise ValueError("original_losses must be provided when same_learnability=1000")
        learnability_first_loss = original_losses[:, 0]
        learnability_last_loss = original_losses[:, -1]
        learnability1000 = learnability_last_loss - learnability_first_loss  
    
    unique_domains = np.unique(domains)
    unique_sources = np.unique(sources)
    
    domain_to_sources = {}
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_sources = np.unique(sources[domain_mask])
        domain_to_sources[domain] = domain_sources
        print(f"*** Domain '{domain}' contains sources: {list(domain_sources)}")
    
    if no_domain:
        print(f"\n*** Direct source-level allocation (no domain hierarchy)")
        
        source_mean_difficulty = {}
        source_sizes = {}
        for source in unique_sources:
            source_mask = sources == source
            source_first_loss = first_loss[source_mask].mean().item()
            source_last_loss = last_loss[source_mask].mean().item()
            if score_type == 'geo_mean':
                if same_learnability == 1000:
                    source_learnability = learnability1000[source_mask].mean().item()
                    source_mean_difficulty[source] = np.sqrt(source_last_loss * source_learnability)
                else:
                    source_mean_difficulty[source] = np.sqrt((source_last_loss - source_first_loss) * source_last_loss)
            elif score_type == 'loss':
                source_mean_difficulty[source] = source_last_loss
            elif score_type == 'neg_loss':
                source_mean_difficulty[source] = -source_last_loss
            source_sizes[source] = source_mask.sum()
        
        print("*** Mean difficulty by source:")
        for source in sorted(unique_sources):
            print(f"  {source}: {source_mean_difficulty[source]:.6f} (size: {source_sizes[source]})")
        
        final_source_allocation, source_weights = compute_softmax_allocation(
            source_mean_difficulty, source_sizes, total_samples, temperature, "source"
        )
    else:
        print(f"\n*** STEP 1: Domain-level allocation")
        
        domain_mean_difficulty = {}
        domain_sizes = {}
        for domain in unique_domains:
            domain_mask = domains == domain
            domain_first_loss = first_loss[domain_mask].mean().item()
            domain_last_loss = last_loss[domain_mask].mean().item()
            if score_type == 'geo_mean':
                if same_learnability == 1000:
                    domain_learnability = learnability1000[domain_mask].mean().item()
                    domain_mean_difficulty[domain] = np.sqrt(domain_last_loss * domain_learnability)
                else:
                    domain_mean_difficulty[domain] = np.sqrt((domain_last_loss - domain_first_loss) * domain_last_loss)
            elif score_type == 'loss':
                domain_mean_difficulty[domain] = domain_last_loss
            elif score_type == 'neg_loss':
                domain_mean_difficulty[domain] = -domain_last_loss
            domain_sizes[domain] = domain_mask.sum()
        
        print("*** Mean difficulty by domain:")
        for domain in sorted(unique_domains):
            print(f"  {domain}: {domain_mean_difficulty[domain]:.6f} (size: {domain_sizes[domain]})")
        
        final_domain_allocation, _ = compute_softmax_allocation(
            domain_mean_difficulty, domain_sizes, total_samples, temperature, "domain"
        )
        
        print(f"\n*** STEP 2: Source-level allocation within each domain")
        
        final_source_allocation = {}
        source_weights = {}
        
        for domain in unique_domains:
            domain_budget = final_domain_allocation[domain]
            domain_sources_list = domain_to_sources[domain]
            
            print(f"\n*** Processing domain '{domain}' with budget {domain_budget}")
            
            source_mean_difficulty = {}
            source_sizes = {}
            for source in domain_sources_list:
                source_mask = (sources == source) & (domains == domain)
                source_first_loss = first_loss[source_mask].mean().item()
                source_last_loss = last_loss[source_mask].mean().item()
                if score_type == 'geo_mean':
                    if same_learnability == 1000:
                        source_learnability = learnability1000[source_mask].mean().item()
                        source_mean_difficulty[source] = np.sqrt(source_last_loss * source_learnability)
                    else:
                        source_mean_difficulty[source] = np.sqrt((source_last_loss - source_first_loss) * source_last_loss)
                elif score_type == 'loss':
                    source_mean_difficulty[source] = source_last_loss
                elif score_type == 'neg_loss':
                    source_mean_difficulty[source] = -source_last_loss
                source_sizes[source] = source_mask.sum()
            
            print("*** Mean difficulty by source:")
            for source in sorted(domain_sources_list):
                print(f"  {source}: {source_mean_difficulty[source]:.6f} (size: {source_sizes[source]})")
            
            domain_source_allocation, domain_source_weights = compute_softmax_allocation(
                source_mean_difficulty, source_sizes, domain_budget, temperature, f"source in domain '{domain}'"
            )
            
            final_source_allocation.update(domain_source_allocation)
            source_weights.update(domain_source_weights)
    
    print(f"\n*** Final allocation across all sources:")
    for source in sorted(unique_sources):
        print(f"  {source}: {final_source_allocation[source]} samples")
    
    total_allocated = sum(final_source_allocation.values())
    print(f"\n*** Total allocated: {total_allocated}/{total_samples}")
    
    source_config = {}
    for src, n_samples in final_source_allocation.items():
        n_clusters = max(1, n_samples // n_sample_per_cluster)
        source_config[src] = (n_clusters, n_samples)
    
    return source_config, source_weights


class S2L(Schedule):
    def __init__(self,
        args,
    ):
        super(S2L, self).__init__(
            args,
        )
        
        raw_sources = np.array([data['source'] for data in self.train_data], dtype=object)
        if 'domain' in self.train_data[0]:
            self.domains = np.array([data['domain'] for data in self.train_data], dtype=object)
            self.sources = np.array([
                f"{str(domain)}_{str(source)}" for source, domain in zip(raw_sources, self.domains)
            ], dtype=object)
            print("*** Domain field found. Namespacing sources as domain_source to prevent cross-domain collisions.")
        else:
            print("*** 'domain' field not found in dataset. Treating all data as a single 'default' domain.")
            self.domains = np.array(['default'] * len(self.train_data), dtype=object)
            self.sources = raw_sources
        if "is_reasoning_answer" not in self.train_data[0]:
            self.is_reasoning_answer=None
        else:
            self.is_reasoning_answer=np.array([data['is_reasoning_answer'] for data in self.train_data])
            np.save("is_reasoning_answer.npy",self.is_reasoning_answer)
        if "reasoning_dynamic_chunk" not in self.train_data[0]:
            self.effective_reasoning_length=None
            self.effective_reasoning_percentage=None
        else:
            self.effective_reasoning_length=np.array([len(data['reasoning_dynamic_chunk']) for data in self.train_data])
            self.effective_reasoning_percentage=np.array([len(data['reasoning_dynamic_chunk'])/len(data['reasoning']) for data in self.train_data])

        self.n_sources = len(set(self.sources))
        self.distance = args["distance"]
        self.n_components = args["n_components"]
        self.code_n_components = args["code_n_components"]
        
        self.source_config = parse_source_config(args.get("source"))
        if self.source_config:
            print(f"*** Using custom source configuration: {self.source_config}")
        
        if args.get("softmax_t") is not None:
            if self.source_config is not None:
                raise ValueError("softmax_t should not be used when source_config is already set")
            
            assert self.difficulty_losses is not None, "Softmax source allocation requires difficulty_losses to be provided"
            
            self.source_config, self._softmax_source_weights = compute_softmax_source_allocation(
                self.difficulty_losses, 
                self.sources,
                self.domains,
                args["softmax_t"],
                args["init_label_num"],
                args["n_sample_per_cluster"],
                args["softmax_score"],
                args["no_domain"],
            )
            print(f"*** Generated source configuration from softmax: {self.source_config}")
            print(f"*** Source weights: {self._softmax_source_weights}")
        
        if self.losses is not None and self.losses.shape[0] != self.n_pool:
            raise AssertionError(
                f"Loss rows ({self.losses.shape[0]}) != dataset size n_pool ({self.n_pool}). "
                f"losses.shape={tuple(self.losses.shape)}, full_data_path={self.full_data_path!r}, "
                f"ref_model_path={self.args.get('ref_model_path')!r}. "
                "Usually the loss .pt was computed on a different corpus or is incomplete."
            )
        
        self.cluster_idx = np.zeros(len(self.sources), dtype=int)
        self.source_cluster_assignments = {}


    def initialize_labeled_data(self):
        """"""
        if self.args.get("sampling_method") == "embedding":
            return self._initialize_labeled_data_embedding_baseline()
        
        if self.args.get("sampling_method") == "middle_perplexity":
            return self._initialize_labeled_data_middle_perplexity_baseline()
        
        if self.args.get("sampling_method") == "longest_reasoning":
            return self._initialize_labeled_data_longest_reasoning_baseline()
        
        if self.args.get("sampling_method") == "learnability_baseline":
            return self._initialize_labeled_data_learnability_baseline()
        
        if self.args.get("ignore_sources", False):
            return self._initialize_labeled_data_ignore_sources()
        
        if self.source_config:
            return self._initialize_labeled_data_custom_source()
        
        num = self.init_label_num
        sources, counts = np.unique(self.sources, return_counts=True)
        sorted_idx = np.argsort(counts)

        sampled_indices = []
        used_sources = []
        for i in range(len(sorted_idx)):
            src = sources[sorted_idx[i]]
            indices = np.where(self.sources == src)[0]
            remaining_sources = len(sorted_idx) - len(used_sources)
            n_per_source = num // remaining_sources
            if len(indices) <= n_per_source:
                if self.args.get("sample_small_clusters", "select") == "select":
                    sampled_indices.append(indices)
                    num -= len(indices)
                    used_sources.append(src)
                elif self.args.get("sample_small_clusters") == "ignore":
                    used_sources.append(src)
                    continue

        for i in range(len(sorted_idx)):
            src = sources[sorted_idx[i]]
            if src in used_sources:
                continue

            indices = np.where(self.sources == src)[0]
            
            min_filter = self.args.get("min_filter")
            if min_filter is not None and len(indices) < min_filter:
                print(f"*** Source {src} has {len(indices)} samples (below min_filter={min_filter}), taking ALL samples")
                sampled_indices.append(indices)
                num -= len(indices)
                used_sources.append(src)
                continue
            
            remaining_sources = len(sorted_idx) - len(used_sources)
            n_per_source = num // remaining_sources
            k = min(n_per_source, len(indices))

            is_cluster_all_samples = ("clusters" in self.args["sampling_method"] and 
                                     self.args.get("all_samples", False))
            n_param = None if is_cluster_all_samples else k
            
            code_sources = {"taco", "code_contests", "apps", "codeforces"}
            n_comp = self.code_n_components if src in code_sources else self.n_components
            if src in code_sources:
                print(f"Source {src} uses code_n_components={n_comp}")
            
            np.random.seed(self.args["seed"])
            
            new_indices, cluster_assignments = self.faiss_kmeans_selection(
                self.losses[indices], n_param, indices, n_comp, 
                seed=self.args["seed"]
            )
            if len(new_indices) == 0:
                return 0
            sampled_indices.append(indices[new_indices])
            actual_samples = len(indices[new_indices])
            num -= actual_samples
            used_sources.append(src)
            print(f"Sampled {actual_samples} samples from source {src} with max loss trajectory coverage")
            self.source_cluster_assignments[str(src)] = {
                int(indices[i]): int(cluster_assignments[i]) for i in range(len(indices))}

        sampled_indices = np.concatenate(sampled_indices)
        print("Total sampled num:",len(sampled_indices))
        
        if not self.args.get("all_samples", False):
            sampled_indices = sampled_indices[:self.init_label_num]
            print(f"Capped to init_label_num: {self.init_label_num}")
        else:
            print(f"Using all samples (no cap applied): {len(sampled_indices)}")

        self.labeled_idx[sampled_indices] = True

    def _initialize_labeled_data_ignore_sources(self):
        """"""
        print("*** Ignoring sources - treating all data as one source")
        
        all_indices = np.arange(self.n_pool)
        
        is_cluster_all_samples = ("clusters" in self.args["sampling_method"] and 
                                 self.args.get("all_samples", False))
        n_param = None if is_cluster_all_samples else self.init_label_num
        
        difficulty_feats = None
        new_indices, cluster_assignments = self.faiss_kmeans_selection(
            self.losses, n_param, all_indices, self.n_components, 
            seed=self.args["seed"]
        )
        
        if len(new_indices) == 0:
            return 0
        
        sampled_indices = all_indices[new_indices]
        actual_samples = len(sampled_indices)
        
        print(f"Sampled {actual_samples} samples from entire dataset (ignoring sources)")
        
        self.source_cluster_assignments["all"] = {
            int(all_indices[i]): int(cluster_assignments[i]) for i in range(len(all_indices))
        }
        
        if not self.args.get("all_samples", False):
            sampled_indices = sampled_indices[:self.init_label_num]
            print(f"Capped to init_label_num: {self.init_label_num}")
        else:
            print(f"Using all samples (no cap applied): {len(sampled_indices)}")

        self.labeled_idx[sampled_indices] = True

    def _initialize_labeled_data_middle_perplexity_baseline(self):
        """"""
        print("*** Using middle_perplexity baseline (no clustering, no sources)")
        
        avg_losses = self.losses.mean(dim=1)
        perplexities = torch.exp(avg_losses)
        
        print(f"*** Calculated perplexities for {len(perplexities)} samples")
        print(f"*** Perplexity range: min={perplexities.min():.4f}, max={perplexities.max():.4f}, median={perplexities.median():.4f}")
        
        sorted_indices = np.argsort(perplexities.numpy())
        
        total_samples = len(sorted_indices)
        n_select = self.init_label_num
        
        if n_select > total_samples:
            print(f"*** Warning: Requested {n_select} samples but only {total_samples} available")
            n_select = total_samples
        
        start_idx = (total_samples - n_select) // 2
        end_idx = start_idx + n_select
        
        selected_indices = sorted_indices[start_idx:end_idx]
        
        print(f"*** Selected {len(selected_indices)} middle-perplexity samples")
        print(f"*** Selection range: indices {start_idx} to {end_idx} (out of {total_samples})")
        print(f"*** Selected perplexity range: min={perplexities[selected_indices].min():.4f}, max={perplexities[selected_indices].max():.4f}")
        
        self.labeled_idx[selected_indices] = True

    def _initialize_labeled_data_embedding_baseline(self):
        """"""
        print("*** Using embedding baseline (k-medoids on all data)")
        
        embeddings = self._load_embeddings()
        
        n_clusters = self.init_label_num
        
        if n_clusters > len(embeddings):
            print(f"*** Warning: Requested {n_clusters} clusters but only {len(embeddings)} samples available")
            n_clusters = len(embeddings)
        
        print(f"*** Running k-means with {n_clusters} clusters on {len(embeddings)} samples")
        
        import faiss
        
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.float().numpy()
        else:
            embeddings_np = embeddings.astype(np.float32)

        kmeans = faiss.Kmeans(
            d=embeddings_np.shape[1], 
            k=n_clusters, 
            niter=20, 
            verbose=True, 
            seed=self.args["seed"]
        )
        kmeans.train(embeddings_np)
        
        distances, assignments = kmeans.index.search(embeddings_np, 1)
        assignments = assignments.flatten()
        
        print(f"*** Finding medoids (samples closest to cluster centers)")
        
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                print(f"*** Warning: Cluster {cluster_id} is empty, skipping")
                continue
            
            cluster_center = kmeans.centroids[cluster_id]
            
            cluster_embeddings = embeddings_np[cluster_indices]
            distances_to_center = np.linalg.norm(
                cluster_embeddings - cluster_center, axis=1
            )
            
            medoid_idx_in_cluster = np.argmin(distances_to_center)
            medoid_idx = cluster_indices[medoid_idx_in_cluster]
            
            selected_indices.append(medoid_idx)
        
        selected_indices = np.array(selected_indices)
        
        print(f"*** Selected {len(selected_indices)} medoids from {n_clusters} clusters")
        
        unique_clusters, cluster_counts = np.unique(assignments, return_counts=True)
        print(f"*** Cluster size statistics:")
        print(f"    Mean: {cluster_counts.mean():.2f}")
        print(f"    Min: {cluster_counts.min()}")
        print(f"    Max: {cluster_counts.max()}")
        print(f"    Non-empty clusters: {len(unique_clusters)}/{n_clusters}")
        
        self.labeled_idx[selected_indices] = True

    def _initialize_labeled_data_custom_source(self):
        """"""
        resample_mode = self.args.get("resample", False)
        if resample_mode:
            print("*** Using custom source configuration (RESAMPLE mode - saving full orderings)")
        else:
            print("*** Using custom source configuration")
        
        sources = np.unique(self.sources)
        
        if "other" in sources and "other" in self.source_config:
            raise ValueError(
                "Conflict: Found a real data source named 'other', but 'other' is reserved as a special "
                "catch-all category in custom source configuration. Please rename the data source or use "
                "a different name in your --source specification."
            )
        
        missing_sources = []
        for src in self.source_config.keys():
            if src != "other" and src not in sources:
                missing_sources.append(src)
        
        if missing_sources:
            raise ValueError(
                f"Error: The following sources in --source configuration do not exist in the dataset: {missing_sources}\n"
                f"Available sources in dataset: {list(sources)}"
            )
        
        sampled_indices = []
        if resample_mode:
            self._cluster_orderings = {}
            self._source_budgets = {}
            self._source_weights = getattr(self, '_softmax_source_weights', None)
        processed_sources = set()
        
        for src in sources:
            indices = np.where(self.sources == src)[0]
            if src in self.source_config:
                n_clusters, n_samples = self.source_config[src]
            else:
                n_clusters, n_samples = self.source_config.get("other", (1, 0))
                
            print(f"\n*** Processing source '{src}' with custom config: {n_clusters} clusters, {n_samples} samples")
            
            if len(indices) == 0:
                print(f"  Warning: No samples found for source '{src}'")
                continue
            
            print(f"  Source '{src}': {len(indices)} total samples available")
            
            if len(indices) < n_samples:
                print(f"\033[91mWARNING: Source '{src}' requested {n_samples} samples but only {len(indices)} available. Taking all {len(indices)} samples.\033[0m")
                n_samples = len(indices)
            
            np.random.seed(self.args["seed"])
            
            if resample_mode:
                self._source_budgets[str(src)] = n_samples

                def to_original_idx(subset_local_idx):
                    local_idx = int(indices[subset_local_idx])
                    if self.subset_to_original_idx is not None:
                        return int(self.subset_to_original_idx[local_idx])
                    return local_idx
                
                if n_samples == 0:
                    self._cluster_orderings[str(src)] = {}
                    cluster_assignments = np.zeros(len(indices), dtype=int)
                else:
                    cluster_orderings_local, cluster_assignments = self.faiss_kmeans_cluster_orderings(
                        self.losses[indices],
                        indices,
                        n_clusters,
                        seed=self.args["seed"]
                    )

                    cluster_orderings_original = {}
                    for cid, local_idxs in cluster_orderings_local.items():
                        cluster_orderings_original[cid] = [
                            to_original_idx(i) for i in local_idxs
                        ]
                    self._cluster_orderings[str(src)] = cluster_orderings_original
                
                self.source_cluster_assignments[str(src)] = {
                    to_original_idx(i): int(cluster_assignments[i]) for i in range(len(indices))
                }
                processed_sources.add(src)
                print(f"  Stored cluster orderings for source '{src}' ({len(self._cluster_orderings[str(src)])} clusters)")
                continue
            
            if n_samples == 0 and (not self.args.get("all_samples", False)):
                new_indices = []
                cluster_assignments = np.zeros(len(indices), dtype=int)
            elif n_samples == len(indices) and (not self.args.get("all_samples", False)):
                new_indices = np.arange(len(indices))
                cluster_assignments = np.zeros(len(indices), dtype=int)
            else:   
                new_indices, cluster_assignments = self.faiss_kmeans_selection(
                    self.losses[indices], 
                    n_samples,
                    indices, 
                    n_clusters,
                    seed=self.args["seed"]
                )
            
            
            sampled_indices.append(indices[new_indices])
            actual_samples = len(indices[new_indices])
            print(f"  Sampled {actual_samples} samples from source '{src}'")
            
            self.source_cluster_assignments[str(src)] = {
                int(indices[i]): int(cluster_assignments[i]) for i in range(len(indices))
            }
            
            processed_sources.add(src)
        
        if resample_mode:
            print(f"\n*** Resample mode: stored orderings for {len(self._cluster_orderings)} sources")
            total_candidates = sum(
                sum(len(idxs) for idxs in src_orderings.values())
                for src_orderings in self._cluster_orderings.values()
            )
            total_budget = sum(self._source_budgets.values())
            print(f"*** Total candidates: {total_candidates}, Total budget: {total_budget}")
            return
        
        if len(sampled_indices) == 0:
            print("ERROR: No samples were selected!")
            return 0
        
        sampled_indices = np.concatenate(sampled_indices)
        print(f"\n*** Total sampled: {len(sampled_indices)} samples across all configured sources")
        
        self.labeled_idx[sampled_indices] = True

    
    def faiss_kmeans_selection(self, features, n, original_indices, n_components=None, seed=None):
        """"""
        if self.args.get("all_samples", False) and "clusters" not in self.args["sampling_method"]:
            raise ValueError("--all_samples flag can only be used with cluster-based sampling methods "
                           f"(hard_clusters, high_loss_clusters, low_loss_clusters), "
                           f"but got sampling_method='{self.args['sampling_method']}'")
        import faiss
        if n_components is None:
            n_components = self.n_components
        effective_k = min(n_components, features.shape[0])
        
        standardized_features = features.clone()
        if self.args["sampling_method"] == "standardized-learnability" or self.args.get("cluster_features", "raw") == "standardized":
            start_mean = features[:, 0].mean()
            start_std = features[:, 0].std()
            if start_std > 0:
                standardized_features[:, 0] = (features[:, 0] - start_mean) / start_std
            
            end_mean = features[:, -1].mean()
            end_std = features[:, -1].std()
            if end_std > 0:
                standardized_features[:, -1] = (features[:, -1] - end_mean) / end_std
            
            print(f"*** Standardized features for clustering/ranking (start: μ={start_mean:.4f}, σ={start_std:.4f}; end: μ={end_mean:.4f}, σ={end_std:.4f})")
        
        clustering_features = standardized_features if self.args.get("cluster_features", "raw") == "standardized" else features
        
        if self.args.get("standardize_kmeans", False):
            clustering_features = clustering_features.clone()
            for i in range(clustering_features.shape[1]):
                c_mean = clustering_features[:, i].mean()
                c_std = clustering_features[:, i].std()
                if c_std > 0:
                    clustering_features[:, i] = (clustering_features[:, i] - c_mean) / c_std
            print(f"*** Applied --standardize-kmeans to all {clustering_features.shape[1]} dimensions")
            
        kmeans = faiss.Kmeans(clustering_features.shape[1], effective_k, niter=20, verbose=False, seed=seed if seed is not None else self.args["seed"])
        print(clustering_features.numpy().shape)
        kmeans.train(clustering_features.numpy())
        
        D, I = kmeans.index.search(clustering_features.numpy(), 1)
        if self.args.get("save_cluster_assignments", False):
            cluster_assignments = {}
            dataset_indices = self.train_idx[features._index] if hasattr(features, "_index") else None

            for local_idx, cluster_id in enumerate(I.flatten()):
                global_idx = features._original_indices[local_idx] if hasattr(features, "_original_indices") else local_idx
                cluster_assignments[int(global_idx)] = int(cluster_id)

            save_path = self.args.get("cluster_assignment_save_path", "cluster_assignments.json")
            import json
            with open(save_path, "w") as f:
                json.dump(cluster_assignments, f, indent=2)
            print(f"Saved cluster assignments to {save_path}")
            return np.array([])
        
        
        clusters, counts = np.unique(I, return_counts=True)
        sorted_idx = np.argsort(counts)
        
        print(f"*** Sample from clusters with size > 2")
        sorted_idx = sorted_idx[counts[sorted_idx] >= 2]
        
        sampled_indices = []
        sampling_method=self.args["sampling_method"]
        source_sort_scores = (
            self.sort_scores[original_indices]
            if self.sort_scores is not None
            else None
        )
        if "clusters" not in sampling_method:
            for i in range(len(sorted_idx)):
                n_per_cluster = n // (len(sorted_idx) - i)
                indices = np.where(I == clusters[sorted_idx[i]])[0]
                if source_sort_scores is not None:
                    if self.sort_order == "descending":
                        sorted_idx_cluster = indices[np.argsort(-source_sort_scores[indices])]
                    else:
                        sorted_idx_cluster = indices[np.argsort(source_sort_scores[indices])]
                    
                    if len(indices) > n_per_cluster:
                        sampled_indices.append(sorted_idx_cluster[:n_per_cluster])
                        n -= n_per_cluster
                    else:
                        sampled_indices.append(sorted_idx_cluster)
                        n -= len(indices)
                elif sampling_method == "random":
                    if len(indices) > n_per_cluster:
                        sampled_indices.append(np.random.choice(indices, n_per_cluster, replace=False))
                        n -= n_per_cluster
                    else:
                        sampled_indices.append(indices)
                        n -= len(indices)
                else:
                    raise ValueError(f"No sort_score_path provided and sampling_method '{sampling_method}' is not 'random' or a cluster-level method")
            if n > 0:
                clusters_to_sample = clusters[np.where(counts < 2)[0]]
                indices = np.where(np.isin(I, clusters_to_sample))[0]
                if len(indices) >= n:
                    sampled_indices.append(np.random.choice(indices, n, replace=False))
                else:
                    sampled_indices.append(indices)
                    print(f"\033[91mWARNING: Requested {n} samples but only {len(indices)} available. Taking all {len(indices)} samples.\033[0m")
        else:
            cluster_metric = np.zeros(len(clusters))
            for i, cluster_id in enumerate(clusters):
                indices = np.where(I == cluster_id)[0]
                features_cluster = features[indices]
                
                if sampling_method == "hard_clusters":
                    cluster_metric[i] = (features_cluster[:, 0] - features_cluster[:, -1]).mean()
                elif sampling_method == "high_loss_clusters":
                    cluster_metric[i] = features_cluster.mean()
                elif sampling_method == "low_loss_clusters":
                    cluster_metric[i] = features_cluster.mean()

            print(f"Cluster metric: {cluster_metric}")

            if sampling_method == "low_loss_clusters":
                sorted_by_metric = np.argsort(cluster_metric)
                metric_name = "lowest average loss"
            elif sampling_method == "high_loss_clusters":
                sorted_by_metric = np.argsort(-cluster_metric)
                metric_name = "highest average loss"
            else:
                sorted_by_metric = np.argsort(-cluster_metric)
                metric_name = "highest learnability"

            top_k_clusters = int(0.5 * len(clusters))
            top_k_clusters = max(1, top_k_clusters)
            selected_clusters = clusters[sorted_by_metric[:top_k_clusters]]

            print(f"Selected top {top_k_clusters}/{len(clusters)} clusters with {metric_name}")
            print(f"Top cluster metrics: {cluster_metric[sorted_by_metric[:top_k_clusters]]}")

            if (
                self.args.get("all_samples", False)
                and sampling_method == "high_loss_clusters"
                and len(clusters) == 2
            ):
                initial_selected_count = 0
                for cluster_id in selected_clusters:
                    initial_selected_count += int(np.sum(I.flatten() == cluster_id))
                selected_ratio = initial_selected_count / len(I) if len(I) > 0 else 0.0

                fallback_keep = None
                if selected_ratio > 0.90:
                    fallback_keep = 1
                elif selected_ratio < 0.10:
                    fallback_keep = 2

                if fallback_keep is not None:
                    fallback_k = min(3, features.shape[0])
                    if fallback_k == 3:
                        print(
                            f"*** 2-cluster selection ratio={selected_ratio:.2%} triggered fallback: "
                            f"reclustering to 3 and keeping top-{fallback_keep} highest-loss cluster(s)"
                        )
                        kmeans_fallback = faiss.Kmeans(
                            clustering_features.shape[1],
                            3,
                            niter=20,
                            verbose=False,
                            seed=seed if seed is not None else self.args["seed"],
                        )
                        kmeans_fallback.train(clustering_features.numpy())
                        _, I_fallback = kmeans_fallback.index.search(clustering_features.numpy(), 1)
                        I = I_fallback

                        clusters_fallback, _ = np.unique(I, return_counts=True)
                        cluster_metric_fallback = np.zeros(len(clusters_fallback))
                        for i, cluster_id in enumerate(clusters_fallback):
                            indices = np.where(I == cluster_id)[0]
                            features_cluster = features[indices]
                            cluster_metric_fallback[i] = features_cluster.mean()

                        sorted_by_metric_fallback = np.argsort(-cluster_metric_fallback)
                        fallback_keep = min(fallback_keep, len(clusters_fallback))
                        selected_clusters = clusters_fallback[sorted_by_metric_fallback[:fallback_keep]]
                        print(
                            f"*** Fallback selected {fallback_keep}/{len(clusters_fallback)} clusters "
                            f"with highest average loss"
                        )
                    else:
                        print(
                            f"*** Skipping 3-cluster fallback (only {features.shape[0]} samples available)"
                        )

            if self.args.get("all_samples", False):
                print(f"*** Taking ALL samples from selected clusters (all_samples=True)")
                for i in range(len(selected_clusters)):
                    cluster_id = selected_clusters[i]
                    indices = np.where(I == cluster_id)[0]
                    sampled_indices.append(indices)
                print(f"*** Total samples collected: {sum(len(idx) for idx in sampled_indices)}")
            else:
                for i in range(len(selected_clusters)):
                    cluster_id = selected_clusters[i]
                    n_per_cluster = n // (len(selected_clusters) - i)
                    indices = np.where(I == cluster_id)[0]

                    if len(indices) > n_per_cluster:
                        sampled = np.random.choice(indices, n_per_cluster, replace=False)
                        sampled_indices.append(sampled)
                        n -= n_per_cluster
                    else:
                        sampled_indices.append(indices)
                        n -= len(indices)

                if n > 0:
                    print(f"*** Warning: {n} samples left to sample after processing all selected clusters")
                    print(f"*** Stopping sampling - not enough samples in top 50% {metric_name} clusters")

        return np.concatenate(sampled_indices), I.flatten()

    def _initialize_labeled_data_longest_reasoning_baseline(self):
        print("*** Initializing labeled data using longest_reasoning baseline")
        reasoning_lengths = []
        for idx, item in enumerate(self.train_data):
            if "content" not in item["conversations"][1]:
                content=item["conversations"][1]["value"]
            else:
                content = item["conversations"][1]["content"]
            
            think_start = content.find("<|im_start|>think")
            if think_start != -1:
                answer_start = content.find("<|im_start|>answer", think_start)
                if answer_start != -1:
                    think_text = content[think_start + len("<|im_start|>think"):answer_start]
                    answer_end = content.find("<|im_end|>", answer_start)
                    if answer_end == -1:
                        answer_text = content[answer_start + len("<|im_start|>answer"):]
                    else:
                        answer_text = content[answer_start + len("<|im_start|>answer"):answer_end]
                    length = len(think_text) + len(answer_text)
                else:
                    raise ValueError(f"No <|im_start|>answer tag found after <|im_start|>think in content at index {idx}")
            else:
                thought_start = content.find("<|begin_of_thought|>")
                if thought_start != -1:
                    thought_end = content.find("<|end_of_thought|>", thought_start)
                    if thought_end != -1:
                        thought_text = content[thought_start + len("<|begin_of_thought|>"):thought_end]
                        length = len(thought_text)
                    else:
                        raise ValueError(f"No <|end_of_thought|> tag found after <|begin_of_thought|> in content at index {idx}")
                else:
                    raise ValueError(f"No reasoning tags found in content at index {idx}. Expected either <|im_start|>think or <|begin_of_thought|>")
            
            reasoning_lengths.append((length, item, idx))
        
        reasoning_lengths.sort(key=lambda x: x[0], reverse=True)
        
        selected_items = reasoning_lengths[:self.init_label_num]
        self.labeled_data = [item for _, item, _ in selected_items]
        selected_indices = [idx for _, _, idx in selected_items]
        
        self.labeled_idx[selected_indices] = True
        
        print(f"*** Selected {len(self.labeled_data)} samples with longest reasoning")

    def _initialize_labeled_data_learnability_baseline(self):
        print("*** Initializing labeled data using learnability_baseline")
        first_ckpt = self.args["first_ckpt"]
        last_ckpt = self.args["last_ckpt"]
        first_idx = self.ckpt_nums.index(first_ckpt)
        last_idx = self.ckpt_nums.index(last_ckpt)
        
        first_loss = self.losses[:, first_idx]
        last_loss = self.losses[:, last_idx]
        
        mean_first = first_loss.mean()
        mean_last = last_loss.mean()
        assert mean_first > mean_last, f"Mean first loss {mean_first} should be > mean last loss {mean_last}"
        
        diffs = first_loss - last_loss
        
        sorted_indices = torch.argsort(diffs, descending=True)
        
        selected_indices = sorted_indices[:self.init_label_num]
        
        self.labeled_data = [self.train_data[i] for i in selected_indices.tolist()]
        self.labeled_idx[selected_indices] = True
        print(f"*** Selected {len(self.labeled_data)} samples with highest learnability (biggest loss improvement from ckpt {first_ckpt} to {last_ckpt})")

    def faiss_kmeans_cluster_orderings(self, features, original_indices, n_components=None, seed=None):
        """"""
        import faiss
        if n_components is None:
            n_components = self.n_components
        effective_k = min(n_components, features.shape[0])
        
        standardized_features = features.clone()
        if self.args.get("cluster_features", "raw") == "standardized":
            start_mean = features[:, 0].mean()
            start_std = features[:, 0].std()
            if start_std > 0:
                standardized_features[:, 0] = (features[:, 0] - start_mean) / start_std
            end_mean = features[:, -1].mean()
            end_std = features[:, -1].std()
            if end_std > 0:
                standardized_features[:, -1] = (features[:, -1] - end_mean) / end_std
        
        clustering_features = standardized_features if self.args.get("cluster_features", "raw") == "standardized" else features
        
        if self.args.get("standardize_kmeans", False):
            clustering_features = clustering_features.clone()
            for i in range(clustering_features.shape[1]):
                c_mean = clustering_features[:, i].mean()
                c_std = clustering_features[:, i].std()
                if c_std > 0:
                    clustering_features[:, i] = (clustering_features[:, i] - c_mean) / c_std
            print(f"*** Applied --standardize-kmeans to all {clustering_features.shape[1]} dimensions")
            
        kmeans = faiss.Kmeans(clustering_features.shape[1], effective_k, niter=20, verbose=False, 
                             seed=seed if seed is not None else self.args["seed"])
        kmeans.train(clustering_features.numpy())
        D, I = kmeans.index.search(clustering_features.numpy(), 1)
        
        clusters, counts = np.unique(I, return_counts=True)
        sampling_method = self.args["sampling_method"]
        source_sort_scores = (
            self.sort_scores[original_indices]
            if self.sort_scores is not None
            else None
        )
        
        cluster_orderings = {}
        
        for cluster_id in clusters:
            indices = np.where(I.flatten() == cluster_id)[0]
            if source_sort_scores is not None:
                if self.sort_order == "descending":
                    ordered = indices[np.argsort(-source_sort_scores[indices])]
                else:
                    ordered = indices[np.argsort(source_sort_scores[indices])]
            elif sampling_method == "random":
                rng = np.random.RandomState(seed if seed is not None else self.args["seed"])
                ordered = rng.permutation(indices)
            else:
                raise ValueError(f"No sort_score_path provided and sampling_method '{sampling_method}' is not 'random'")
            
            cluster_orderings[int(cluster_id)] = np.atleast_1d(ordered).tolist()
        
        print(f"*** Cluster orderings: {len(cluster_orderings)} clusters, "
              f"total {sum(len(v) for v in cluster_orderings.values())} samples")
        
        return cluster_orderings, I.flatten()

    def save_cluster_orderings(self):
        """"""
        import json
        
        if not hasattr(self, '_cluster_orderings') or not hasattr(self, '_source_budgets'):
            raise ValueError("No cluster orderings available. Did you call initialize_labeled_data() with resample=True?")
        
        def to_py(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_py(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_py(v) for v in obj]
            return obj

        orderings_json = {}
        for src, clusters in self._cluster_orderings.items():
            orderings_json[src] = {str(k): to_py(v) for k, v in clusters.items()}
        
        orderings_path = os.path.join(self.data_path_root, "cluster_orderings.json")
        with open(orderings_path, "w") as f:
            json.dump(orderings_json, f, indent=2)
        print(f"*** Saved cluster orderings to {orderings_path}")
        
        budgets_path = os.path.join(self.data_path_root, "source_budgets.json")
        with open(budgets_path, "w") as f:
            json.dump(to_py(self._source_budgets), f, indent=2)
        print(f"*** Saved source budgets to {budgets_path}")
        
        if self._source_weights is not None:
            weights_path = os.path.join(self.data_path_root, "source_weights.json")
            with open(weights_path, "w") as f:
                json.dump(to_py(self._source_weights), f, indent=2)
            print(f"*** Saved source weights to {weights_path}")
        else:
            print("*** No source weights to save (not using softmax allocation)")
        
        cluster_dir = os.path.join(self.data_path_root, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        for src, assignment_dict in self.source_cluster_assignments.items():
            safe_src = str(src).replace("/", "__")
            save_path = os.path.join(cluster_dir, f"{safe_src}.json")
            with open(save_path, "w") as f:
                json.dump(assignment_dict, f, indent=2)
            print(f"*** Saved cluster assignments for source '{src}' to {save_path}")
        
        if self.losses is not None:
            save_path = os.path.join(self.data_path_root, "loss.npy")
            np.save(save_path, self.losses.numpy())
            print(f"*** Saved losses to {save_path}")

    def save_labeled_unlabeled_data(self):
        """"""
        super().save_labeled_unlabeled_data()
        
        if self.losses is not None:
            save_path = os.path.join(self.data_path_root, "loss.npy")
            np.save(save_path, self.losses.numpy())
            print(f"*** Saved losses to {save_path}")
