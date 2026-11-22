import argparse
import os
import yaml
import random
import numpy as np
import torch

from schedules import Full, S2L
from schedule_base import parse_ckpt_spec

DOMAIN_TO_SOURCES = {
    'biology': ['camelai_biology'],
    'chemistry': ['camelai_chemistry'],
    'code': ['apps', 'code_contests', 'codeforces', 'taco'],
    'math': ['amc_aime', 'aops_forum', 'math', 'olympiads', 'synthetic_math'],
    'physics': ['camelai_physics'],
    'puzzle': ['riddle_sense']
}

def expand_domain_to_source(domain_spec):
    """"""
    if not domain_spec:
        return None
    
    domain_configs = {}
    for part in domain_spec.split(','):
        part = part.strip()
        if ':' not in part or '-' not in part:
            raise ValueError(f"Invalid domain spec format: {part}. Expected format: domain:n_clusters-n_samples")
        
        domain, config = part.split(':')
        domain = domain.strip()
        domain_configs[domain] = config
    
    all_domains = set(DOMAIN_TO_SOURCES.keys())
    specified_domains = set(domain_configs.keys()) - {'other'}
    
    invalid_domains = specified_domains - all_domains
    if invalid_domains:
        raise ValueError(f"Unknown domains: {list(invalid_domains)}. Valid domains are: {list(all_domains)}")
    
    source_parts = []
    
    for domain, config in domain_configs.items():
        if domain == 'other':
            continue
            
        sources = DOMAIN_TO_SOURCES[domain]
        for source in sources:
            source_parts.append(f"{source}:{config}")
    
    if 'other' in domain_configs:
        other_config = domain_configs['other']
        other_domains = all_domains - specified_domains
        
        for domain in other_domains:
            sources = DOMAIN_TO_SOURCES[domain]
            for source in sources:
                source_parts.append(f"{source}:{other_config}")
    
    return ','.join(source_parts)

def get_schedule(schedule_name):
    if schedule_name == "Full":
        return Full
    elif schedule_name == "S2L":
        return S2L
    
def set_default_values(args):
    if "ref_model_path" not in args:
        args["ref_model_path"] = None
    if "n_components" not in args:
        args["n_components"] = -1
    if "code_n_components" not in args:
        args["code_n_components"] = args["n_components"]
    if "num_loss_ckpts" not in args:
        args["num_loss_ckpts"] = -1
    if "distance" not in args:
        args["distance"] = 'euclidean'
    if "seed" not in args:
        args["seed"] = 42
    if "cache_dir" not in args:
        args["cache_dir"] = None
    if "use_ckpts" not in args:
        args["use_ckpts"] = None
    if "subset_indices_path" not in args:
        args["subset_indices_path"] = None
    if "all_samples" not in args:
        args["all_samples"] = False
    if "ignore_sources" not in args:
        args["ignore_sources"] = False
    if "source" not in args:
        args["source"] = None
    if "min_filter" not in args:
        args["min_filter"] = None
    if "difficulty_use_ckpts" not in args:
        args["difficulty_use_ckpts"] = None
    if "softmax_t" not in args:
        args["softmax_t"] = None
    if "softmax_score" not in args:
        args["softmax_score"] = None
    if "no_domain" not in args:
        args["no_domain"] = False
    if "n_sample_per_cluster" not in args:
        args["n_sample_per_cluster"] = None
    if "sort_score_path" not in args:
        args["sort_score_path"] = None
    if "sort_order" not in args:
        sm = args.get("sampling_method", "")
        if sm == "ascending_loss":
            args["sort_order"] = "ascending"
        else:
            args["sort_order"] = "descending"
    return args


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script with flexible configuration')
    
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML config file (optional)')
    
    parser.add_argument('--full_data_path', type=str,
                        help='Path to full dataset')
    parser.add_argument('--cache_dir', type=str,
                        help='Cache directory for HuggingFace models')
    parser.add_argument('--schedule_name', type=str, choices=['Full', 'S2L'],
                        help='Schedule name (Full or S2L)')
    parser.add_argument('--result_dir_name', type=str,
                        help='Result directory name')
    parser.add_argument('--ref_model_path', type=str,
                        help='Reference model path')
    parser.add_argument('--init_label_num', type=int,
                        help='Initial number of labeled samples')
    parser.add_argument('--n_round', type=int,
                        help='Number of active learning rounds')
    parser.add_argument('--n_components', type=int,
                        help='Number of PCA components')
    parser.add_argument('--code_n_components', type=int,
                        help='Number of PCA components for code sources (taco, code_contests, apps, codeforces). If not set, uses n_components.')
    parser.add_argument('--source', type=str,
                        help='Custom source configuration. Format: source1:n_clusters-n_samples,source2:n_clusters-n_samples,... '
                             'Example: code:10-100,math:20-400,other:15-300. When set, --n_components and --init_label_num are ignored.')
    parser.add_argument('--domain', type=str,
                        help='Custom domain configuration. Format: domain1:n_clusters-n_samples,domain2:n_clusters-n_samples,... '
                             'Example: code:10-100,math:20-400,other:5-50. This broadcasts settings to all sources within each domain. '
                             'Special case: "other" broadcasts to all sources from domains not explicitly specified. '
                             'Cannot be used together with --source. When set, --n_components and --init_label_num are ignored.')
    parser.add_argument('--sample_small_clusters', type=str, choices=['select', 'ignore'],
                        help='How to handle small clusters')
    parser.add_argument('--sampling_method', type=str,
                        choices=['random', 'ascending_loss', 'descending_loss',
                                 'high_loss_clusters', 'low_loss_clusters',
                                 'embedding', 'middle_perplexity',
                                 'longest_reasoning', 'learnability_baseline'],
                        default=None,
                        help='Sampling strategy. '
                             '"ascending_loss"/"descending_loss" require --sort_score_path. '
                             '"random" uses random intra-cluster sort. '
                             '"high_loss_clusters"/"low_loss_clusters" are for step-3 cluster-level selection. '
                             '"embedding"/"middle_perplexity"/"longest_reasoning"/"learnability_baseline" are standalone baselines.')
    parser.add_argument('--sort_score_path', type=str, default=None,
                        help='Path to a pre-computed (N,) .pt tensor of per-sample sort scores. '
                             'Required when --sampling_method is ascending_loss or descending_loss.')
    parser.add_argument('--num_loss_ckpts', type=int,
                        help='Number of loss checkpoints')
    parser.add_argument('--distance', type=str,
                        help='Distance metric for clustering')
    parser.add_argument('--seed', type=int,
                        help='Random seed')
    parser.add_argument('--loss_file_name', type=str, default='losses-fullanswer.pt',
                        help='Name of the loss file to read')
    parser.add_argument('--use_ckpts', type=str, default=None,
                        help='Checkpoint numbers to use (e.g., "8-15" for checkpoint8 through checkpoint15). Examples: "1-10", "1,3,5,7", "1-5,8,10-12"')
    parser.add_argument('--subset_indices_path', type=str, default=None,
                        help='Path to .npy file containing subset indices from original dataset')
    parser.add_argument('--all_samples', action='store_true',
                        help='For cluster-based sampling methods, return all samples from selected clusters instead of sampling')
    parser.add_argument('--ignore_sources', action='store_true',
                        help='Ignore source field and treat all data as a single source for clustering')
    parser.add_argument('--softmax_t', type=float, default=None,
                        help='Temperature for softmax-based source selection. When set, automatically determines sample allocation '
                             'across sources using softmax on mean losses from the last two checkpoints. '
                             'Exclusive with --source, --domain, and --n_components.')
    parser.add_argument('--softmax_score', type=str, choices=['geo_mean', 'loss', 'neg_loss'], default=None,
                        help='Difficulty score type for softmax allocation. "geo_mean" uses sqrt((last_loss - first_loss) * last_loss), '
                             '"loss" uses last_loss, "neg_loss" uses -last_loss (lower loss → more budget). '
                             'Must be set when --softmax_t is specified.')
    parser.add_argument('--no_domain', action='store_true',
                        help='Skip domain-level allocation and directly allocate across sources. '
                             'Only used when --softmax_t is specified.')
    parser.add_argument('--n_sample_per_cluster', type=int, default=None,
                        help='Number of samples per cluster for softmax-based allocation. Only used when --softmax_t is specified. '
                             'Defaults to 3 if --softmax_t is set but this is not specified.')
    parser.add_argument('--difficulty_loss_file_name', type=str, default='loss-openthoughts-100token-r-losses.pt',
                        help='Loss file name for difficulty calculation in softmax allocation. Only used when --softmax_t is specified. '
                             'Defaults to loss-openthoughts-100token-r-losses.pt')
    parser.add_argument('--difficulty_use_ckpts', type=str, default=None,
                        help='Checkpoint numbers or names specifically for difficulty losses. If not provided, fallback to --use_ckpts.')
    parser.add_argument('--no-standardize-kmeans', action='store_false', dest='standardize_kmeans',
                        help='Do not standardize all loss dimensions before k-means clustering')
    parser.add_argument('--resample', action='store_true',
                        help='Resample mode: output full cluster orderings instead of fixed selection. '
                             'Outputs cluster_orderings.json and source_budgets.json for iterative generation.')

    return parser.parse_args()


def load_config(config_file, cli_args):
    """"""
    config = {}
    
    if config_file is not None:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.full_load(f)
        print(f'Configuration loaded from {config_file}')
    else:
        print('No config file provided, using CLI arguments only')
    
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key in ['config_file', 'wandb_key']:
            continue
        if value is not None:
            config[key] = value
            print(f'Overriding config: {key} = {value}')
    
    if config.get('domain') is not None:
        if config.get('source') is not None:
            raise ValueError("Cannot use both --domain and --source. Please use only one.")
        print(f"*** Expanding domain configuration: {config['domain']}")
        config['source'] = expand_domain_to_source(config['domain'])
        print(f"*** Expanded to source configuration: {config['source']}")
    
    if config.get('source') is not None or config.get('domain') is not None:
        required_params = [
            'full_data_path', 'schedule_name', 'result_dir_name', 'n_round'
        ]
    else:
        required_params = [
            'full_data_path', 'schedule_name', 'result_dir_name', 
            'init_label_num', 'n_round'
        ]
    
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}\n"
                        f"Please provide them either in config file or via command line arguments")
    
    return config


def main():
    cli_args = parse_arguments()
    
    args = load_config(cli_args.config_file, cli_args)
    
    if args.get("source") is not None or args.get("domain") is not None:
        if args.get("n_components") is not None and args.get("n_components") != -1:
            print("Warning: --source or --domain is set, ignoring --n_components")
            args["n_components"] = -1
        if args.get("init_label_num") is not None:
            print("Warning: --source or --domain is set, ignoring --init_label_num")
            args["init_label_num"] = 0
    
    if args.get("softmax_t") is not None:
        if args.get("source") is not None:
            raise ValueError("--softmax_t cannot be used with --source")
        if args.get("domain") is not None:
            raise ValueError("--softmax_t cannot be used with --domain")
        if args.get("n_components") is not None and args.get("n_components") != -1:
            print("Warning: --softmax_t is set, ignoring --n_components")
            args["n_components"] = -1
        
        if args.get("softmax_score") is None:
            raise ValueError("--softmax_score must be set when --softmax_t is specified")
        
        
        if args.get("n_sample_per_cluster") is None:
            args["n_sample_per_cluster"] = 3
    else:
        if args.get("softmax_score") is not None:
            raise ValueError("--softmax_score can only be used when --softmax_t is specified")
        
        if args.get("no_domain"):
            raise ValueError("--no_domain can only be used when --softmax_t is specified")
        
        if args.get("n_sample_per_cluster") is not None:
            raise ValueError("--n_sample_per_cluster can only be used when --softmax_t is specified")
    
    if args.get("ignore_sources") and (args.get("source") is not None or args.get("domain") is not None):
        raise ValueError("--ignore-sources cannot be set when --source or --domain is provided")
    
    if args.get("ignore_sources") and args.get("softmax_t") is not None:
        raise ValueError("--ignore-sources cannot be set when --softmax_t is provided")
    
    if args.get("min_filter") is not None and not args.get("all_samples"):
        raise ValueError("--min-filter can only be used with --all_samples")
    
    sm = args.get("sampling_method")
    has_sort_score = args.get("sort_score_path") is not None
    if has_sort_score:
        if sm not in ("ascending_loss", "descending_loss"):
            args["sampling_method"] = "descending_loss"
    elif sm in ("ascending_loss", "descending_loss"):
        raise ValueError(f"--sampling_method {sm} requires --sort_score_path")
    
    loss_file_name = args.get("loss_file_name", "")
    if "/" in loss_file_name or "\\" in loss_file_name:
        raise ValueError("loss_file_name must be a filename, not a path. Paths are not allowed.")
    
    args = set_default_values(args)
    
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    
    print('\n=== Final Configuration ===')
    print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    result_dir = args['result_dir_name']
    if result_dir.startswith('/'):
        base_path = result_dir
    else:
        base_path = f"res/{result_dir}"

    args["data_path_root"] = f"{base_path}/data"
    
    resample_mode = args.get("resample", False)
    
    if resample_mode:
        output_file = os.path.join(args["data_path_root"], "cluster_orderings.json")
    else:
        output_file = os.path.join(args["data_path_root"], "labeled_idx.npy")
    if os.path.exists(output_file):
        raise FileExistsError(f"Output file already exists: {output_file}")
    

    schedule = get_schedule(schedule_name=args["schedule_name"])(
        args=args
    )
    print('*** Schedule built!')


    schedule.initialize_labeled_data()
    
    os.makedirs(args["data_path_root"], exist_ok=True)
    
    if resample_mode:
        schedule.save_cluster_orderings()
        print("*** Resample mode: cluster orderings saved!")
    else:
        schedule.save_labeled_unlabeled_data()
        print(f"*** Selected-Data-Size = {len(schedule.labeled_idx[schedule.labeled_idx==True])}")
    print("*** Data selection complete!")


if __name__ == '__main__':
    
    main()
