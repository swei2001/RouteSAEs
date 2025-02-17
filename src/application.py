from utils import parse_args, load_config, Applier


if __name__ == '__main__':
    cfg = parse_args()
    # cfg = load_config('../config.yaml')
    applier = Applier(cfg)

    applier.get_context(
        threshold=15, 
        max_length=64, 
        max_per_token=2, 
        lines=4
    )
    
