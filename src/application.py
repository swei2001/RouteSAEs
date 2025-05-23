from utils import parse_args, load_config, Applier


if __name__ == '__main__':
    cfg = parse_args()
    applier = Applier(cfg)
    applier.get_context()
    
    applier.get_context(
        threshold=15, 
        max_length=64, 
        max_per_token=2, 
        lines=4
    )

    applier.clamp(
        max_length=128, 
        set_high=[
            [13523, 15, 0]
        ],
        output_path='../clamp/clamped_output.json'
    )
