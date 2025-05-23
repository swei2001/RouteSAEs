from utils import parse_args, set_seed, Trainer


if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.run()