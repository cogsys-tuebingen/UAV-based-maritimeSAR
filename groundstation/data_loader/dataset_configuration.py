import json
import os


def load_dataset_configuration(network, dataset_name, type, config_variant=None, ignore_keys=[], server_name=None, config_file=None):
    project_base_path = os.getcwd()
    project_name = 'crow'
    project_base_path = project_base_path[:project_base_path.find(project_name)+len(project_name)]

    if config_file is not None:
        j = json.load(open(config_file, 'r'))
    else:
        config_file_name = '%s_%s.json' % (dataset_name, type) if config_variant is None else\
            '%s_%s_%s.json' % (dataset_name, type, config_variant)

        if server_name is not None:
            assert server_name in ['avalon1']
            j = json.load(open(os.path.join(project_base_path, 'configs', network, server_name, config_file_name)))
        else:
            j = json.load(open(os.path.join(project_base_path, 'configs', network, config_file_name)))

    if ignore_keys is not None or len(ignore_keys) > 0:
        for k in ignore_keys:
            j.pop(k)

    return j