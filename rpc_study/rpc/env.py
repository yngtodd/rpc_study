import os


class Env:
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    num_workers = world_size - 1
