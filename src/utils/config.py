from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


def get_config(exp: str, config_dir: PosixPath = Path("./config")) -> DictConfig:
    OmegaConf.register_new_resolver("path", lambda x: Path(x), replace=True)
    global_config = OmegaConf.load(config_dir / "global.yaml")
    exp_config = OmegaConf.load(config_dir / f"exp_{exp}.yaml")
    config = OmegaConf.merge(global_config, exp_config)
    config.output_path = config.output_path / exp
    config.output_path.mkdir(exist_ok=True, parents=True)
    return config
