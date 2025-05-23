from utils import parse_args, Evaluater


if __name__ == '__main__':
    cfg = parse_args()
    evaluater = Evaluater(cfg)
    evaluater.run()