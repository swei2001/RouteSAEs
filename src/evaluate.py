from utils import parse_args, load_config, Evaluater


if __name__ == '__main__':
    cfg = parse_args()
    # cfg = load_config('../config.yaml')
    evaluater = Evaluater(cfg)
    evaluater.run()