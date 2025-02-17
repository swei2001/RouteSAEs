from utils import parse_args, load_config, set_seed, Trainer


if __name__ == '__main__':
    cfg = parse_args()
    # cfg = load_config('../config.yaml')
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.run()