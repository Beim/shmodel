import os
import json

curr_dir = os.path.split(os.path.abspath(__file__))[0]


class ConfigLoader:

    def __init__(self):
        with open('%s/config.json' % curr_dir, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        env_activate = cfg['env']
        with open('%s/config-%s.json' % (curr_dir, env_activate), encoding='utf-8') as f:
            env_cfg = json.load(f)
        self.cfg = env_cfg

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(ConfigLoader, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_config(self):
        return self.cfg


config_loader = ConfigLoader()

if __name__ == '__main__':
    print(config_loader.get_config())