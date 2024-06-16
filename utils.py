import json

def load_json_config(app_path='data/Application', infra_path='data/Graph_Infra'):
    with open(app_path, 'r') as app_file, open(infra_path, 'r') as infra_file:
        app_config = json.load(app_file)
        infra_config = json.load(infra_file)
    return app_config, infra_config

