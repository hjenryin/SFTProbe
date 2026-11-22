import numpy as np
import json
import torch
import os
import glob
import re
from datasets import load_dataset
from utils import jload, jdump
from pdb import set_trace as breakpoint

def parse_ckpt_spec(spec):
    """"""
    if any(c.isalpha() and c != 'e' for c in spec):
        return [p.strip() for p in spec.split(',')]
        
    ckpt_numbers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            for i in range(int(start), int(end) + 1):
                ckpt_numbers.add(i)
        else:
            ckpt_numbers.add(int(part))
    return ckpt_numbers

class Schedule:
    def __init__(self, 
        args,
    ):
        self.full_data_path = args["full_data_path"]
        self.init_label_num = args["init_label_num"] if "init_label_num" in args else 0
        self.args = args
        
        if self.full_data_path.endswith(".jsonl"):
            with open(self.full_data_path, "r") as f:
                self.train_data = [json.loads(line) for line in f]
            
        elif self.full_data_path.endswith(".json"):
            with open(self.full_data_path, "r") as f:
                self.train_data = json.load(f)
                
        else:
            data_df = load_dataset(self.full_data_path)["train"]
            self.train_data = [dict(data_df[i]) for i in range(len(data_df))]
        
        self.train_idx = torch.arange(len(self.train_data))
        
        if len(self.train_data) > 0:
            if 'source' not in self.train_data[0]:
                raise ValueError(
                    "Dataset must contain 'source' field for stratified sampling. "
                    f"First data entry has fields: {list(self.train_data[0].keys())}"
                )
            
            if 'reasoning_dynamic_chunk' in self.train_data[0] and 'reasoning' not in self.train_data[0]:
                raise ValueError(
                    "Dataset has 'reasoning_dynamic_chunk' but missing 'reasoning' field. "
                    "Both are required for effective reasoning percentage calculation."
                )
        
        if args.get("subset_indices_path") is not None:
            subset_path = args["subset_indices_path"]
            print(f"*** Loading subset indices from {subset_path}")
            subset_indices = np.load(subset_path)
            print(f"*** Original dataset size: {len(self.train_data)}")
            print(f"*** Subset size: {len(subset_indices)}")
            
            self.subset_to_original_idx = subset_indices
            
            self.train_data = [self.train_data[i] for i in subset_indices]
            
            self.train_idx = torch.arange(len(self.train_data))
            
            print(f"*** Using subset with {len(self.train_data)} samples")
        else:
            self.subset_to_original_idx = None
        
        self.n_pool = len(self.train_data)
        self.labeled_idx = torch.zeros(self.n_pool, dtype=bool)  
        self.data_path_root = args["data_path_root"]
        
        if args.get("ref_model_path") is not None and args.get("use_ckpts") != "None":
            self._load_losses()
            if args.get("softmax_t") is not None and args.get("difficulty_loss_file_name") is not None:
                self._load_difficulty_losses()
            else:
                self.difficulty_losses = None
        else:
            self.losses = None
            self.difficulty_losses = None
        
        if args.get("sort_score_path") is not None:
            sort_path = args["sort_score_path"]
            print(f"*** Loading pre-computed sort scores from {sort_path}")
            self.sort_scores = torch.load(sort_path).float()
            self.sort_order = "ascending" if args.get("sampling_method") == "ascending_loss" else "descending"
            print(f"*** Sort scores shape: {self.sort_scores.shape}, order: {self.sort_order}")
        else:
            self.sort_scores = None
            self.sort_order = "descending"

    def _load_losses_from_file(self, loss_file_name, description="losses", use_ckpts_arg=None):
        """"""
        assert description in ["losses", "difficulty_losses"]
        
        if use_ckpts_arg is None:
            use_ckpts_arg = self.args.get("use_ckpts")

        def get_checkpoint_number(path):
            """"""
            match = re.search(r'checkpoint-(\d+)', path)
            return int(match.group(1)) if match else 0
        
        ckpt_paths = [
            p for p in glob.glob(f'{self.args["ref_model_path"]}/*')
            if os.path.isdir(p)
        ]
        ckpt_paths = sorted(ckpt_paths, key=get_checkpoint_number)
        
        print(f"*** Found {len(ckpt_paths)} total checkpoints")
        for i, ckpt in enumerate(ckpt_paths):
            print(f"  [{i+1}] {os.path.basename(ckpt)}")
        
        if use_ckpts_arg is not None:
            if use_ckpts_arg == "None":
                ckpt_paths = []
                print(f"\n*** Selected 0 checkpoints (use_ckpts='None')")
            else:
                total_ckpts = len(ckpt_paths)
                selected_ckpt_numbers = parse_ckpt_spec(use_ckpts_arg)
                
                if isinstance(selected_ckpt_numbers, list):
                    filtered_paths = []
                    for name in selected_ckpt_numbers:
                        p = os.path.join(self.args["ref_model_path"], name)
                        if os.path.exists(p):
                            filtered_paths.append(p)
                    ckpt_paths = filtered_paths
                else:
                    filtered_paths = []
                    for ckpt_path in ckpt_paths:
                        ckpt_num = get_checkpoint_number(ckpt_path)
                        if ckpt_num in selected_ckpt_numbers:
                            filtered_paths.append(ckpt_path)
                    
                    ckpt_paths = filtered_paths

                print(f"\n*** Selected {len(ckpt_paths)} out of {total_ckpts} checkpoints using spec '{use_ckpts_arg}'")
                for i, ckpt in enumerate(ckpt_paths):
                    print(f"  [{i+1}] {os.path.basename(ckpt)}")
        
        if isinstance(parse_ckpt_spec(use_ckpts_arg) if use_ckpts_arg and use_ckpts_arg != "None" else None, list):
            self.ckpt_nums = [os.path.basename(p) for p in ckpt_paths]
        else:
            self.ckpt_nums = [get_checkpoint_number(p) for p in ckpt_paths]
        
        losses = []
        for ckpt in ckpt_paths:
            print(f"*** {ckpt} ** Loading {description}...")
            loaded = torch.load(f"{ckpt}/{loss_file_name}", map_location="cpu", weights_only=False)
            if torch.is_tensor(loaded):
                losses.append(loaded.detach().float().reshape(-1).cpu())
            else:
                losses.append(torch.as_tensor(loaded, dtype=torch.float32).reshape(-1).cpu())
        
        if len(losses) == 0:
            raise ValueError(f"Could not load any {description} from {loss_file_name}")
        
        if description == "losses" and (self.args.get("num_loss_ckpts", -1) > -1) and (len(losses) > self.args["num_loss_ckpts"]):
            losses = np.stack(losses)
            keep_every = len(losses) // self.args["num_loss_ckpts"]
            print(f"*** Keeping every {keep_every} losses from {len(losses)} losses")
            losses = losses[np.arange(0, len(losses), keep_every)]
            losses_tensor = torch.from_numpy(losses).t()
        else:
            print(f"*** Using all {len(losses)} {description}")
            losses_tensor = torch.stack(losses).t()
            print(f"*** Full {description} shape: {losses_tensor.shape}")
        
        losses_tensor[torch.isnan(losses_tensor)] = 0
        
        if self.subset_to_original_idx is not None:
            print(f"*** Filtering {description} from {losses_tensor.shape[0]} to {len(self.subset_to_original_idx)} samples (subset)")
            losses_tensor = losses_tensor[self.subset_to_original_idx]
            print(f"*** Filtered {description} shape: {losses_tensor.shape}")
            
            if (losses_tensor < 0).any():
                neg_count = (losses_tensor < 0).sum().item()
                raise ValueError(
                    f"FATAL: Found {neg_count} negative values in {description} after filtering! "
                    "This means the provided subset indices point to samples that were not computed "
                    "(marked with -1.0). Please check that your subset indices match the computed losses."
                )
        
        print(f"*** {description.capitalize()} loaded and ready: {losses_tensor.shape}")
        return losses_tensor

    def _load_losses(self):
        """"""
        loss_file_name = self.args.get("loss_file_name", "losses-fullanswer.pt")
        self.losses = self._load_losses_from_file(loss_file_name, "losses")

    def _load_difficulty_losses(self):
        """"""
        difficulty_loss_file_name = self.args.get("difficulty_loss_file_name")
        print(f"\n*** Loading difficulty losses for softmax allocation from: {difficulty_loss_file_name}")
        self.difficulty_losses = self._load_losses_from_file(difficulty_loss_file_name, "difficulty_losses", use_ckpts_arg=self.args.get("difficulty_use_ckpts"))

    def _load_embeddings(self):
        """"""
        embedding_file_name = self.args.get("embedding_file_name", "embedding.pt")
        
        ref_model_path = self.args.get("ref_model_path")
        if ref_model_path is None:
            raise ValueError("ref_model_path is required for embedding-based selection")
        
        ckpt_paths = sorted(glob.glob(os.path.join(ref_model_path, "checkpoint*")))
        
        if len(ckpt_paths) == 0:
            raise ValueError(f"No checkpoints found in {ref_model_path}")
        
        use_ckpts = self.args.get("use_ckpts")
        if use_ckpts is None:
            raise ValueError("--use_ckpts is required when using embedding baseline")
        
        if ',' in use_ckpts or '-' in use_ckpts:
            raise ValueError(f"embedding baseline requires exactly one checkpoint, got: {use_ckpts}")
        
        ckpt_num = int(use_ckpts)
        
        selected_ckpt = None
        for ckpt_path in ckpt_paths:
            match = re.search(r'checkpoint-(\d+)', os.path.basename(ckpt_path))
            if match and int(match.group(1)) == ckpt_num:
                selected_ckpt = ckpt_path
                break
        
        if selected_ckpt is None:
            raise ValueError(f"Checkpoint {ckpt_num} not found in {ref_model_path}")
        
        embedding_path = os.path.join(selected_ckpt, embedding_file_name)
        print(f"\n*** Loading embeddings from: {embedding_path}")
        
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding file not found: {embedding_path}")
        
        embeddings = torch.load(embedding_path)
        print(f"*** Loaded embeddings with shape: {embeddings.shape}")
        
        if self.subset_to_original_idx is not None:
            print(f"*** Filtering embeddings from {embeddings.shape[0]} to {len(self.subset_to_original_idx)} samples (subset)")
            embeddings = embeddings[self.subset_to_original_idx]
            print(f"*** Filtered embeddings shape: {embeddings.shape}")
        
        return embeddings.float()


    def initialize_labeled_data(self):
        """"""
        tmp_idxs = torch.randperm(self.n_pool)
        self.labeled_idx[tmp_idxs[:self.init_label_num]] = True

    def save_labeled_unlabeled_data(self):
        """"""
        labeled_idx = np.where(self.labeled_idx)[0]
        unlabeled_idx = np.where(~self.labeled_idx)[0]

        if self.subset_to_original_idx is not None:
            original_labeled_idx = self.subset_to_original_idx[labeled_idx]
            original_unlabeled_idx = self.subset_to_original_idx[unlabeled_idx]
            print(f"*** Mapped {len(labeled_idx)} subset indices to original indices")
        else:
            original_labeled_idx = labeled_idx
            original_unlabeled_idx = unlabeled_idx

        labeled_data_json_format = [self.train_data[_] for _ in labeled_idx]

        labeled_data_path = f"{self.data_path_root}/labeled.json"
        labeled_idx_path = f"{self.data_path_root}/labeled_idx.npy"

        retry = 0
        while True:
            jdump(labeled_data_json_format, labeled_data_path)
            try:
                temp_labeled = jload(labeled_data_path)
                print(f"*** jdump(labeled_data_json_format, labeled_data_path) SUCCESSFUL to --> {labeled_data_path}")
                break
            except:
                retry += 1
                print(f"*** jdump(labeled_data_json_format, labeled_data_path) FAILED to --> {labeled_data_path}")
                if retry > 5:
                    raise
                continue


        np.save(labeled_idx_path, original_labeled_idx)
        print(f"*** Saved subset indices to {labeled_idx_path}")

        import os, json
        cluster_dir = os.path.join(self.data_path_root, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)

        for src, assignment_dict in self.source_cluster_assignments.items():
            safe_src = str(src).replace("/", "__")
            save_path = os.path.join(cluster_dir, f"{safe_src}.json")
            with open(save_path, "w") as f:
                json.dump(assignment_dict, f, indent=2)
            print(f"*** Saved cluster assignments for source '{src}' to {save_path}")
