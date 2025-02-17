from utils import parse_args, SAE_pipeline


if __name__ == '__main__':
    cfg = parse_args()
    pipeline = SAE_pipeline(cfg)
    pipeline.run()