import wandb
import re
import os
import glob
import shutil
import argparse
import tqdm
import json
from data_loader.flight_detection_dataset import PrepareBatch
from torchvision import transforms
from torch.utils.data import DataLoader
from evaluation.cocoeval import COCOeval
from tidecv import TIDE
import random
import string
import numpy as np

from utils.tide_util import coco_dt_to_tide_data, coco_gt_to_tide_data
from data_loader.utils import ResizerSquare, ResizerRectangle

# urls = [
#     # "https://wandb.ai/heuchelmoerder/crow-centernet/runs/new_id_0001?workspace=user-heuchelmoerder",
#     # "https://wandb.ai/heuchelmoerder/crow-centernet/runs/39i9lkq6?workspace=user-heuchelmoerder",
#     "https://wandb.ai/martinmessmer/centernet/runs/2q48o1xy?workspace=user-martinmessmer"
# ]

def generate_result(type, network_name, backbone, img_size, coco_stats_labels, coco_stats, tide_stats, random_cropping,
                    url=None):
    if url is None:
        url = input("Url of the run:")

    obj = {
        "network": network_name,
        "backbone": backbone,
        "type": type,
        "img_size": img_size,
        "random_cropping": random_cropping,
        "reference": url,
        "val": {
            "mAP": {
                l: v for (k, v) in zip(coco_stats_labels, coco_stats)
            },
            "TIDE": {
                "Cls": tide_stats['main']['default']['Cls'],
                "Loc": tide_stats['main']['default']['Loc'],
                "Both": tide_stats['main']['default']['Both'],
                "Dupe": tide_stats['main']['default']['Dupe'],
                "Bkg": tide_stats['main']['default']['Bkg'],
                "Miss": tide_stats['main']['default']['Miss'],
                "FalsePos": tide_stats['special']['default']['FalsePos'],
                "FalseNeg": tide_stats['special']['default']['FalseNeg']
            }
        }
    }

    return obj


def get_run_id(url):
    match = re.match(
        "https:\/\/wandb.ai\/([^?]+\/runs\/[a-z0-9_]+)(\/model|\/overview|\?workspace=user\-martinmessmer)?.*",
        url)
    return match.group(1)


def get_wandb_runs_with_tag(username, project, tags: dict) -> wandb.apis.public.Runs:
    runs = get_wandb_runs(username, project)
    for k, v in tags.items():
        def perform_check(run):
            config = json.loads(run.json_config)
            for key, value in config.items():
                if key == k and value['value'] == v:
                    return True
            return False

        runs = list(filter(perform_check, runs))

    return runs


def get_wandb_runs(username, project) -> wandb.apis.public.Runs:
    api = wandb.Api()
    run = api.runs(username + '/' + project)
    return run


def get_wandb_run(run_id) -> wandb.apis.public.Run:
    api = wandb.Api()
    run = api.run(run_id)
    return run


def get_wandb_run_url(user, project, id):
    return "https://wandb.ai/%s/%s/runs/%s" % (user, project, id)


def get_slurm_job_checkpoint_folder(dataset, slurm_job_id, project_name, wandb_id, is_batch_run=False, on_msstore=False):
    if is_batch_run:
        path = os.path.join(get_batch_slurm_job_folder(dataset, slurm_job_id, on_msstore=on_msstore), project_name, wandb_id, "checkpoints")
    else:
        path = os.path.join(get_slurm_job_folder(dataset, slurm_job_id, on_msstore=on_msstore), project_name, wandb_id, "checkpoints")
    return path


def get_slurm_job_folder(dataset, slurm_job_id, on_msstore):
    if on_msstore:
        path = os.path.join("/home/lvarga/mnt/msstore/tcml/logs/##Finished", dataset, slurm_job_id)
    else:
        path = os.path.join("/home/lvarga/mnt/tcml/logs/##Finished", dataset, slurm_job_id)
    return path


def get_batch_slurm_job_folder(dataset, slurm_job_id, on_msstore):
    if on_msstore:
        path = os.path.join("/home/lvarga/mnt/msstore/tcml/logs/##Finished/batch_run", dataset, slurm_job_id)
    else:
        path = os.path.join("/home/lvarga/mnt/tcml/logs/##Finished/batch_run", dataset, slurm_job_id)
    return path


def get_best_checkpoint_path(dataset, slurm_job_id, project_name, wandb_id):
    if slurm_job_id.startswith('CS-WS'):
        # is a local run
        return get_best_checkpoint_path_for_local(slurm_job_id, project_name, wandb_id)
    elif slurm_job_id.startswith('cuda'):
        return get_best_checkpoint_path_for_cuda(slurm_job_id, project_name, wandb_id)
    elif slurm_job_id.startswith('avalon1'):
        return get_best_checkpoint_path_for_avalon1(slurm_job_id, project_name, wandb_id)
    # else cluster run
    try:
        checkpoint_path = get_best_checkpoint_path_from_msstore(dataset, slurm_job_id, project_name, wandb_id)
        return checkpoint_path
    except Exception as e:
        print(f"! Could not find checkpoint on msstore: {e}")

    return get_best_checkpoint_path_from_msstore(dataset, slurm_job_id, project_name, wandb_id)


def get_best_checkpoint_path_from_tcml(dataset, slurm_job_id, project_name, wandb_id):
    path = get_slurm_job_checkpoint_folder(dataset, slurm_job_id, project_name, wandb_id)
    matches = glob.glob(os.path.join(path, "epoch*"))

    if len(matches) == 0:
        # maybe it is a batch run
        path = get_slurm_job_checkpoint_folder(dataset, slurm_job_id, project_name, wandb_id, is_batch_run=True)
        matches = glob.glob(os.path.join(path, "epoch*"))

    assert len(matches) == 1
    return matches[0]


def get_best_checkpoint_path_from_msstore(dataset, slurm_job_id, project_name, wandb_id):
    path = get_slurm_job_checkpoint_folder(dataset, slurm_job_id, project_name, wandb_id, on_msstore=False)
    matches = glob.glob(os.path.join(path, "epoch*"))

    if len(matches) == 0:
        # maybe it is a batch run
        path = get_slurm_job_checkpoint_folder(dataset, slurm_job_id, project_name, wandb_id, is_batch_run=True, on_msstore=False)
        matches = glob.glob(os.path.join(path, "epoch*"))

    if len(matches) != 1:
        raise Exception("No checkpoint")

    return matches[0]


def get_best_checkpoint_path_for_cuda(slurm_job_id, project_name, wandb_id):
    match = re.match("cuda.:(.*)", slurm_job_id)
    path = match.group(1).replace("/home", "/cshome").replace("/rahome", "/cshome").replace("/crow", "/output")
    path = os.path.join(path, project_name, wandb_id, "checkpoints")

    matches = glob.glob(os.path.join(path, "epoch*"))

    if len(matches) == 0:
        import paramiko
        from scp import SCPClient

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("cuda1")

        scp_client = SCPClient(client.get_transport())
        os.makedirs(os.path.join("/tmp", "remote_checkpoints", wandb_id), exist_ok=True)

        path = os.path.join("/data/lvarga/logs/", project_name, wandb_id, "checkpoints")
        scp_client.get(path, os.path.join("/tmp", "remote_checkpoints", wandb_id), recursive=True)

        matches = glob.glob(os.path.join("/tmp", "remote_checkpoints", wandb_id, "checkpoints", "epoch*"))

    assert len(matches) == 1
    return matches[0]


def get_best_checkpoint_path_for_avalon1(slurm_job_id, project_name, wandb_id):
    # path = os.path.join("/home/messmer/mnt/avalon1/data/log_outsource/wandb", project_name, wandb_id, "checkpoints")
    path = os.path.join("/home/messmer/mnt/avalon1/home/messmer/PycharmProjects/crow/centernet_better/*/*",
                        project_name, wandb_id, "checkpoints")
    matches = glob.glob(os.path.join(path, "epoch*"))

    assert len(matches) == 1
    return matches[0]


def get_best_checkpoint_path_for_local(slurm_job_id, project_name, wandb_id):
    match = re.match("CS-.*:(.*)", slurm_job_id)
    path = match.group(1)
    path = os.path.join(path, project_name, wandb_id, "checkpoints")

    matches = glob.glob(os.path.join(path, "epoch*"))

    assert len(matches) == 1
    return matches[0]


def get_filename_of_path(path):
    return os.path.basename(path)


def copy_to_local(path):
    file_name = random_filename("ckpt")
    local_path = os.path.join("/tmp", "checking")

    if not os.path.exists(local_path):
        os.mkdir(local_path)

    dst = os.path.join(local_path, file_name)
    shutil.copyfile(path, dst)

    return dst


def rm_local_checkpoint(path):
    os.remove(path)


def get_val_generator_and_gt(network, image_size, img_folder, ann_file, meta_file):
    from data_loader.flight_detection_dataset import FlightDetectionDataset

    if network == 'centernet_better':
        from centernet_better.train import ConvertAnnotation as CenterNetConvertAnnotation, convert_to_coco_api, \
            collater
        preprocessing = [PrepareBatch(), ResizerSquare(image_size), CenterNetConvertAnnotation()]
        val_set = FlightDetectionDataset(img_folder, ann_file, meta_file, image_size,
                                         transform=transforms.Compose(preprocessing))
        coco_gt = convert_to_coco_api(val_set)

        val_params = {"batch_size": 1,
                      "shuffle": False,
                      "drop_last": False,
                      "collate_fn": collater,
                      "num_workers": 0,
                      "pin_memory": True}
        val_generator = DataLoader(val_set, **val_params)

        return val_generator, coco_gt

    if network == 'efficientdet':
        from efficientdet.train import Normalizer, FitIntoGrid, convert_to_coco_api, collater
        preprocessing = [PrepareBatch(), Normalizer(), ResizerRectangle(image_size), FitIntoGrid(128)]
        val_set = FlightDetectionDataset(img_folder, ann_file, image_size,
                                         transform=transforms.Compose(preprocessing))
        coco_gt = convert_to_coco_api(val_set)

        val_params = {"batch_size": 1,
                      "shuffle": False,
                      "drop_last": False,
                      "collate_fn": collater,
                      "num_workers": 0,
                      "pin_memory": True}
        val_generator = DataLoader(val_set, **val_params)

        return val_generator, coco_gt

    if network == 'yolov4':
        from yolov4.train import ConvertAnnotation as Yolov4ConvertAnnotation, convert_to_coco_api, collater
        preprocessing = [PrepareBatch(), ResizerSquare(image_size), Yolov4ConvertAnnotation()]
        val_set = FlightDetectionDataset(img_folder, ann_file, image_size,
                                         transform=transforms.Compose(preprocessing))
        coco_gt = convert_to_coco_api(val_set)

        val_params = {"batch_size": 1,
                      "shuffle": False,
                      "drop_last": False,
                      "collate_fn": collater,
                      "num_workers": 0,
                      "pin_memory": True}
        val_generator = DataLoader(val_set, **val_params)

        return val_generator, coco_gt


def load_checkpoint(network, path, checkpoint_path):
    if network == 'centernet_better':
        from centernet_better.train import CenterNetBetterModule
        model = CenterNetBetterModule.load_from_checkpoint(path, checkpoints_path=checkpoint_path)
        return model

    if network == 'efficientdet':
        from efficientdet.train import EfficientDetModule
        model = EfficientDetModule.load_from_checkpoint(path)
        return model

    if network == 'yolov4':
        from yolov4.train import YoloV4Module
        model = YoloV4Module.load_from_checkpoint(path, checkpoints_path=checkpoint_path)
        return model


def predict(network, model, val_generator, device, test_time_augmentation=False):
    if network == 'centernet_better':
        if test_time_augmentation:
            from centernet_better.test_checkpoint import predict_whole_valset_augmented
            coco_pred_img_ids, coco_pred = predict_whole_valset_augmented(model, val_generator, device)
        else:
            from centernet_better.test_checkpoint import predict_whole_valset
            coco_pred_img_ids, coco_pred = predict_whole_valset(model, val_generator, device)
        return coco_pred_img_ids, coco_pred

    if network == 'efficientdet':
        if test_time_augmentation:
            raise Exception("Not implemented")

        from efficientdet.test_checkpoint import predict_whole_valset
        coco_pred_img_ids, coco_pred = predict_whole_valset(model, val_generator, device)
        return coco_pred_img_ids, coco_pred

    if network == 'yolov4':
        if test_time_augmentation:
            raise Exception("Not implemented")

        from yolov4.test_checkpoint import predict_whole_valset
        coco_pred_img_ids, coco_pred = predict_whole_valset(model, val_generator, device)
        return coco_pred_img_ids, coco_pred

    raise Exception("Not yet implemented")


def get_backbone(model):
    network = model.hparams['network']
    if network == 'centernet_better':
        return model.hparams['backbone']

    if network == 'efficientdet':
        return model.hparams['efficientnet_model']

    if network == 'yolov4':
        return None


def random_filename(file_ext):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(8))

    return "%s.%s" % (result_str, file_ext)


def dump_json(j):
    filename = os.path.join("/tmp", "%s" % (random_filename('json')))
    print("# Dumped json into %s" % filename)
    json.dump(j, open(filename, 'w'))


def calc_stats(model, coco_gt, coco_pred, coco_pred_img_ids, ap70=False):
    coco_dt = coco_gt.loadRes(coco_pred)

    print("Calculate mAP")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox', dataset=model.hparams['dataset_name'])
    coco_eval.params.imgIds = coco_pred_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_stats = coco_eval.stats
    coco_class_wise_stats = coco_eval.class_wise_stats

    if ap70:
        coco_eval.params.iouThrs = np.array([0.7])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_stats = np.array([z for z in coco_stats] + [coco_eval.stats[0]], dtype=np.float64)
    else:
        print('Evaluation op AP70 is OFF!')

    print("Calculate TIDE")
    tide = TIDE()
    tide.evaluate(coco_gt_to_tide_data(coco_gt),
                  coco_dt_to_tide_data(coco_dt),
                  mode=TIDE.BOX)
    tide.summarize()
    # tide.plot()
    tide_stats = tide.get_all_errors()

    return coco_stats, tide_stats, coco_class_wise_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--set", type=str, required=False, help="train/val/test", default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--test_time_augmentation", action="store_true", default=False)
    parser.add_argument("--dataset_name", default=None, required=True, choices=['uav_dt', 'vis_drone'])
    parser.add_argument("--ap70", type=lambda x: x == 'True', default=True)
    opt = parser.parse_args()

    list_of_checkpoints = []
    resuls_list = []

    device = 'cuda'

    username = 'martinmessmer'
    project = 'centernet'
    dataset_name = opt.dataset_name
    dataset_saved_name = 'UAVDT' if dataset_name == 'uav_dt' else 'visdrone_json_birdview'


    # urls = get_wandb_runs_with_tag(username=username, project=project, tags={'dataset_name': dataset_name})

    PATHS = {
        "train": (os.path.join(opt.data_path, dataset_saved_name, "images", "train"),
                  os.path.join(opt.data_path, dataset_saved_name, "annotations", 'instances_train.json'),
                  os.path.join(opt.data_path, dataset_saved_name, "annotations", 'meta.csv')),
        "val": (os.path.join(opt.data_path, dataset_saved_name, "images", "val"),
                os.path.join(opt.data_path, dataset_saved_name, "annotations", 'instances_val.json'),
                os.path.join(opt.data_path, dataset_saved_name, "annotations", 'meta.csv')),
        "test": (os.path.join(opt.data_path, dataset_saved_name, "images", "test"),
                 os.path.join(opt.data_path, dataset_saved_name, "annotations", 'instances_test.json'),
                 os.path.join(opt.data_path, dataset_saved_name, "annotations", 'meta.csv')),
    }

    # For vis drone we use the validation set, because there are no labels for test
    dataset_type = 'val' if dataset_name in ['vis_drone', 'dota_20', 'uav_dt', 'UAVDT', 'visdrone_json_birdview'] else 'test'

    if opt.set is not None:
        dataset_type = opt.set
        print(f"! Overwrite default data set type to: {dataset_type}")

    img_folder, ann_file, meta_file = PATHS[dataset_type]

    # assert dataset_name in opt.data_path, "Different dataset for train and validation!"
    # Iterators von Leon:
        # _iter = tqdm.tqdm([get_wandb_run(get_run_id(u)) for u in urls], desc="Checkpoint..")
        # _iter = tqdm.tqdm(get_wandb_run_with_tag(username, project, tag, dataset_name=dataset_name), desc="Checkpoint..")
    _iter = tqdm.tqdm(get_wandb_runs_with_tag(username=username, project=project, tags={'dataset_name': dataset_name}))
    for run in _iter:
        u = get_wandb_run_url(username, run.project, run.id)
        _iter.set_description("Checkpoint %s" % u)
        run_name = run.name
        run_config = run.config
        dataset = run_config['dataset_name']
        slurm_job_id = run_config['slurm_job_id']
        project_name = run.project
        wandb_id = run.id

        if not opt.force and "Checked" in run.summary.keys():
            print("Skip %s, because already checked" % u)
            continue

        best_checkpoint_path = get_best_checkpoint_path(dataset, slurm_job_id, project_name, wandb_id)
        best_checkpoint_path = copy_to_local(best_checkpoint_path)
        c = {
            'checkpoint_path': best_checkpoint_path,
            'run': run,
            'run_url': u,
            'network': run_config['network']
        }

        print("Collected the checkpoint, now validate..")

        checkpoint_path = c['checkpoint_path']
        model = load_checkpoint(c['network'], checkpoint_path, opt.checkpoint_path).to(device)
        model.eval()

        max_input_image_size = None
        # if model.hparams['type'] == "splitter":
        #     max_input_image_size = model.hparams['val_image_size']
        # elif model.hparams['type'] == "whole":
        #     max_input_image_size = model.hparams['image_size']
        if model.hparams['type'] == 'birdview':
            max_input_image_size = model.hparams['image_size']

        assert max_input_image_size is not None

        val_generator, coco_gt = get_val_generator_and_gt(c['network'], max_input_image_size, img_folder, ann_file, meta_file)
        coco_pred_img_ids, coco_pred = predict(c['network'], model, val_generator, device)
        coco_stats, tide_stats, coco_class_wise_stats = calc_stats(model, coco_gt, coco_pred, coco_pred_img_ids, opt.ap70)
        coco_stats_labels = COCOeval(None, None, 'bbox', dataset=model.hparams['dataset_name']).get_labels()
        coco_stats_labels += ['Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=500 ]']
        for (l, v) in zip(coco_stats_labels, coco_stats):
            c['run'].summary["best/" + l] = v

        used_cat_ids = [c['id'] for c in coco_gt.cats.values()]
        used_cats = [val_generator.dataset.cats[c] for c in used_cat_ids] + ["overall"]

        for (l, v) in zip(coco_stats_labels, coco_class_wise_stats):
            c['run'].summary["best/class_wise_" + l] = None
            c['run'].summary["best/class_wise_" + l] = {a: b for a, b in zip(used_cats, v)}
        if opt.test_time_augmentation:
            try:
                coco_pred_img_ids, coco_pred = predict(c['network'], model, val_generator, device,
                                                       test_time_augmentation=True)
                coco_stats, tide_stats, coco_class_wise_stats = calc_stats(model, coco_gt, coco_pred, coco_pred_img_ids)
                coco_stats_labels = COCOeval(None, None, 'bbox', dataset=model.hparams['dataset_name']).get_labels()
                for (l, v) in zip(coco_stats_labels, coco_stats):
                    c['run'].summary["best_tta/" + l] = v
            except Exception as e:
                print(f"Doesn't support test time augmentation: {e}")

        c['run'].summary['Checked'] = 1
        c['run'].summary.update()

        backbone = get_backbone(model)
        random_cropping = model.hparams['random_cropping'] if 'random_cropping' in model.hparams.keys() else False

        result = generate_result(model.hparams['type'], model.hparams['network'], backbone, model.hparams['image_size'],
                                 coco_stats_labels, coco_stats,
                                 tide_stats, random_cropping, url=c['run_url'])

        resuls_list.append(result)
        rm_local_checkpoint(c['checkpoint_path'])

    print(resuls_list)
    dump_json(resuls_list)
