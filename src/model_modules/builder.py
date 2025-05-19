from src.model_modules.multimodal_model import MultiModalModel


def build_model_from_config(config):
    return MultiModalModel(config)
