from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_config(config_name: str, config_dir: Path = Path("./config")) -> DictConfig:
    OmegaConf.register_new_resolver("path", lambda x: Path(x), replace=True)
    global_config = OmegaConf.load(config_dir / "global.yaml")
    exp_config = OmegaConf.load(config_dir / f"{config_name}.yaml")
    config = OmegaConf.merge(global_config, exp_config)
    config.output_path = config.output_path / config.exp
    config.output_path.mkdir(exist_ok=True, parents=True)
    return config
