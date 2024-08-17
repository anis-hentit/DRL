import json

def load_json_config(app_path='data/Application', infra_path='data/Graph_Infra'):
    with open(app_path, 'r') as app_file, open(infra_path, 'r') as infra_file:
        app_config = json.load(app_file)
        infra_config = json.load(infra_file)
    return app_config, infra_config



'''
IoT devices are indexed from 0 to 29.
Edge servers are indexed from 30 to 44.
Cloud servers are indexed from 45 to 49.
'''