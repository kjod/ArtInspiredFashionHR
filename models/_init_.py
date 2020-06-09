import importlib

def find_model_using_name(model_name, dir):
    model = getattr(importlib.import_module(dir), model_name)
    return model

def create_model(args, logger, dir):
    model = find_model_using_name(args.model, dir)
    instance = model(args, logger)
    return instance
