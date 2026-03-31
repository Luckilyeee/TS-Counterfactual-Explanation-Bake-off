import torch
import numpy as np
import os
import json
import csv
import wandb
import ssl
import time
from nte.experiment.utils import get_model, dataset_mapper, backgroud_data_configuration, get_run_configuration
import shortuuid
from nte.models.saliency_model.counterfactual_cels import CFExplainer
import random
from nte.experiment.default_args import parse_arguments
import seaborn as sns
from nte.experiment.utils import number_to_dataset, set_global_seed
from nte.utils import CustomJsonEncoder

sns.set_style("darkgrid")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = True
WANDB_DRY_RUN = False


BASE_SAVE_DIR = 'results_v1/2312/'

if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    args = parse_arguments()
    print("Config: \n", json.dumps(args.__dict__, indent=2))

    dataset_name = args.dataset

    if args.dataset in number_to_dataset.keys():
        args.dataset = number_to_dataset[args.dataset]

    if args.enable_seed:
        set_global_seed(args.seed_value)

    ENABLE_SAVE_PERTURBATIONS = args.save_perturbations
    PROJECT_NAME = args.pname

    # Ensure data and meta are loaded
    dataset = dataset_mapper(DATASET=args.dataset)
    dataset.load_data()
    num_classes = dataset.meta["n_classes"]


    TAG = f'{args.algo}-{args.dataset}-{args.background_data}-{args.background_data_perc}-run-{args.run_id}'
    BASE_SAVE_DIR = BASE_SAVE_DIR + "/" + TAG

    model = get_model(dataset=args.dataset, input_size=1, num_classes=num_classes)

    softmax_fn = torch.nn.Softmax(dim=-1)

    bg_data, bg_label, bg_len = backgroud_data_configuration(BACKGROUND_DATA=args.background_data,
                                                   BACKGROUND_DATA_PERC=args.background_data_perc,
                                                   dataset=dataset)

    print(f"Using {args.background_data_perc}% of background data. Samples: {bg_len}")

    config = args.__dict__

    explainer = None

    if args.algo == 'cf':
        explainer = CFExplainer(background_data=bg_data[:bg_len], background_label=bg_label[:bg_len],
                                  predict_fn=model, enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)

    config = {**config, **{
        "tag": TAG,
        "algo": args.algo,
    }}

    dataset_len = len(dataset.test_data)

    ds = get_run_configuration(args=args, dataset=dataset, TASK_ID=args.task_id)

    res_path = f"CELS/{args.dataset}/"
    os.system(f'mkdir -p "{res_path}"')

    cf_res = []
    cf_res0 = []
    cf_probs = []
    cf_maps = []
    cf_flips = []
    cf_times = []  # Track time for each instance

    dataset_start_time = time.time()  # Track total time for the dataset

    for ind, (original_signal, original_label) in ds: #original_signal is from the test set
        try:
            instance_start_time = time.time()  # Start timing for this instance

            if args.enable_seed_per_instance:
                set_global_seed(random.randint())
            metrics = {'epoch': {}}
            cur_ind = args.single_sample_id if args.run_mode == 'single' else (
                ind + (int(args.task_id) * args.samples_per_task))
            UUID = dataset.valid_name[cur_ind] if args.dataset_type == 'valid' else shortuuid.uuid()
            EXPERIMENT_NAME = f'{args.algo}-{cur_ind}-R{args.run_id}-{UUID}-C{ind}-T{args.task_id}-S{args.samples_per_task}-TS{(int(args.task_id) * args.samples_per_task)}-TT{(ind+int(args.task_id) * args.samples_per_task)}'
            print(
                f" {args.algo}: Working on dataset: {args.dataset} index: {cur_ind} [{((cur_ind + 1) / dataset_len * 100):.2f}% Done]")
            SAVE_DIR = f'{BASE_SAVE_DIR}/{EXPERIMENT_NAME}'
            os.system(f'mkdir -p "{SAVE_DIR}"')
            os.system(f'mkdir -p "./wandb/{TAG}/"')
            config['save_dir'] = SAVE_DIR

            # if args.run_mode == 'single' and args.dynamic_replacement == False:
            #     config = {**config}

            if args.run_mode == 'single':
                config = {**config}

            json.dump(config, open(SAVE_DIR + "/config.json", 'w'), indent=2, cls=CustomJsonEncoder)
            if ENABLE_WANDB:
                wandb.init(project=PROJECT_NAME, name=EXPERIMENT_NAME, tags=TAG,
                           config=config, reinit=True, force=True, dir=f"./wandb/{TAG}/")

            original_signal = torch.tensor(original_signal, dtype=torch.float32) # original_signal ->ntest instance

            with torch.no_grad():
                if args.bbm == 'fcn':
                    # print(original_signal.shape)
                    target = softmax_fn(model(original_signal.reshape(1, 1, original_signal.shape[0])))

                    # target = softmax_fn(model(original_signal)) # multivariate

                    # data_train = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
                      #✅
                elif args.bbm == "cnn":
                    target = softmax_fn(model(original_signal))[0]
                else:
                    raise Exception(f"Black Box model not supported: {args.bbm}")
            # print("target", target)

            category = np.argmax(target.cpu().data.numpy()) # prediction label
            args.dataset = dataset
            if ENABLE_WANDB:
                wandb.run.summary[f"ori_prediction_class"] = category
                wandb.run.summary[f"ori_prediction_prob"] = np.max(target.cpu().data.numpy())
                wandb.run.summary[f"ori_label"] = original_label


            if args.background_data == "none":
                explainer.background_data = original_signal
                explainer.background_label = original_label


            mask, perturbation_res, target_prob, flip = explainer.generate_saliency(
                data=original_signal.cpu().detach().numpy(), label=original_label,
                save_dir=SAVE_DIR, pred=target, dataset=dataset)
            cf_flips.append(flip)

            cf = perturbation_res.cpu().detach().numpy().flatten()# Convert tensor to NumPy array
            cf_res0.append(cf.copy())  # Save the original cf for reference
            # used to evaluation part, since we want to only calculate the valid cfs
            if not flip:
                cf = np.full_like(cf, -1)
            cf_res.append(cf)

            perturbation_res = torch.tensor(perturbation_res, dtype=torch.float32)

            print(perturbation_res.shape)
            pert_res = softmax_fn(model(perturbation_res.reshape(1, 1, perturbation_res.shape[0]))) #infocels

            pert_label = np.argmax(pert_res.cpu().data.numpy())  # prediction label

            cf_probs.append(target_prob)

            cf_maps.append(mask)

            if ENABLE_WANDB:
                wandb.run.summary[f"pert_prediction_class"] = pert_label
                wandb.run.summary[f"target_prob"] = target_prob
                wandb.run.summary[f"mask"] = mask

            instance_end_time = time.time()  # End timing for this instance
            instance_time = instance_end_time - instance_start_time
            cf_times.append(instance_time)
            print(f"Instance {ind} completed in {instance_time:.2f} seconds")

        except Exception as e:
            with open(f'/tmp/{TAG}_error.log', 'a+') as f:
                f.write(e)
                f.write(e.__str__())
                f.write("\n\n")


    np.save(res_path + 'valid_cfs.npy', np.array(cf_res)) # valid cfs with -1 unvalid
    np.save(res_path + 'all_cfs.npy', np.array(cf_res0)) # all cfs without check the flip
    np.save(res_path + 'saliency_cf_prob.npy', np.array(cf_probs))
    np.save(res_path + 'map_cf.npy', np.array(cf_maps))
    np.save(res_path + 'instance_times.npy', np.array(cf_times))  # Save timing for each instance

    dataset_end_time = time.time()
    total_dataset_time = dataset_end_time - dataset_start_time
    avg_instance_time = np.mean(cf_times) if cf_times else 0

    flip_rate = np.mean(cf_flips)
    print(f"Flip rate: {flip_rate:.4f}")
    print(f"Total dataset time: {total_dataset_time:.2f} seconds")
    print(f"Average instance time: {avg_instance_time:.2f} seconds")
    print(f"Processed {len(cf_times)} instances")

    csv_path = 'CELS/flip_rates.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['dataset', 'flip_rate', 'total_time_seconds', 'avg_time_per_instance', 'num_instances'])
        writer.writerow([str(dataset_name), float(flip_rate), float(total_dataset_time), float(avg_instance_time), len(cf_times)])

