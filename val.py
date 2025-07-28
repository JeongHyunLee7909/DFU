# CONFIG
import argparse

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='val_LRRU_with_Three_DFU.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
# if len(config.gpus) == 1:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)

# BASIC PACKAGES
import emoji
import time
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# MODULES
from dataloaders.kitti_loader import KittiDepth
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import *

# VARIANCES
sample_, output_ = None, None
metric_txt_dir = None

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def test(args):

    # DATASET
    print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"), end=' ')
    global sample_, output_, metric_txt_dir
    data_test = KittiDepth(args.test_option, args)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print('Done!')

    # NETWORK
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model = get_model(args)
    net = model(args)
    net.cuda()
    print('Done!')
    # total_params = count_parameters(net)

    # METRIC
    print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"), end=' ')
    metric = get_metric(args)
    metric = metric(args)
    print('Done!')

    # SUMMARY
    print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"), end=' ')
    summary = get_summary(args)
    try:
        if not os.path.isdir(args.test_dir):
            os.makedirs(args.test_dir)
        os.makedirs(args.test_dir, exist_ok=True)
        os.makedirs(args.test_dir + '/test', exist_ok=True)
        metric_txt_dir = os.path.join(args.test_dir + '/test/result_metric.txt')
        with open(metric_txt_dir, 'w') as f:
            f.write('test_model: {} \ntest_option: {} \nval:{} \ntest_name: {} \n'
                    'test_not_random_crop: {} \n'
                    'tta: {}\n \n'.format(args.test_model, args.test_option, 'val', args.test_name,
                                       args.test_not_random_crop,
                                       args.tta))
    except OSError:
        pass
    writer_test = summary(args.test_dir, 'test', args, None, metric.metric_name)
    print('Done!')

    # LOAD MODEL
    print(emoji.emojize('Load model... :writing_hand:', variant="emoji_type"), end=' ')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), \
            "file not found: {}".format(args.test_model)

        checkpoint_ = torch.load(args.test_model, map_location='cpu')
        model = remove_moudle(checkpoint_)
        key_m, key_u = net.load_state_dict(model, strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)

    net = nn.DataParallel(net)
    net.eval()
    print('Done!')

    num_sample = len(loader_test) * loader_test.batch_size
    pbar_ = tqdm(total=num_sample)
    t_total = 0
    with torch.no_grad():
        for batch_, sample_ in enumerate(loader_test):

            torch.cuda.synchronize()
            t0 = time.time()
            if args.tta:
                print("TTA")
                samplep = {key: val.cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                samplep['d_path'] = sample_['d_path']
                outputp = net(samplep)
                predp = outputp['results'][-1]

                samplef = {'dep': torch.flip(sample_['dep'], [-1]),
                           'rgb': torch.flip(sample_['rgb'], [-1]),
                           'ip': torch.flip(sample_['ip'], [-1]),
                           'dep_clear': torch.flip(sample_['dep_clear'], [-1])
                           }
                samplef = {key: val.cuda() for key, val in samplef.items()
                           if val is not None}
                outputf = net(samplef)
                predf = torch.flip(outputf['results'][-1], [-1])

                output_ = {'results': [(predp + predf) / 2.]}

            else:
                print("Not TTA")
                samplep = {key: val.float().cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                print("Input keys:", samplep.keys())
                samplep['d_path'] = sample_['d_path']

                output_ = net(samplep)

            torch.cuda.synchronize()
            t1 = time.time()
            t_total += (t1 - t0)
            if 'test' not in args.test_option:
                rgb_np = samplep['rgb'].squeeze().cpu().detach().numpy()
                rgb_np = np.transpose(rgb_np, (1, 2, 0))  # Reorder to (height, width, channels)
                depth_gt_np = samplep['gt'].squeeze().cpu().detach().numpy()
                dep_np = samplep['dep'].squeeze().cpu().detach().numpy()
                dep_clear_np = samplep['dep_clear'].squeeze().cpu().detach().numpy()
                ip_np = samplep['ip'].squeeze().cpu().detach().numpy()
                depth_pred_np = output_['results'][-1].squeeze().cpu().detach().numpy()
                print(f"Max for ip_np is {np.max(ip_np)}")
                print(f"Min for ip_np is {np.min(ip_np)}")

                # Create a figure with two subplots (one for gt, one for pred)
                fig, ax = plt.subplots(2, 3, figsize=(14, 12))

                # Plot Ground Truth (gt) on the top
                ax[0][0].imshow(rgb_np)  # Use gray colormap for grayscale images
                ax[0][0].set_title("RGB")
                ax[0][0].axis('off')  # Hide the axis for clarity

                # Plot Ground Truth (gt) on the top
                ax[0][1].imshow(dep_np, cmap='gray')  # Use gray colormap for grayscale images
                ax[0][1].set_title("Raw Lidar points")
                ax[0][1].axis('off')  # Hide the axis for clarity

                ax[0][2].imshow(dep_clear_np, cmap='gray')  # Use gray colormap for grayscale images
                ax[0][2].set_title("Raw Lidar points")
                ax[0][2].axis('off')  # Hide the axis for clarity

                # Plot Ground Truth (gt) on the top
                ax[1][0].imshow(ip_np, cmap='gray')  # Use gray colormap for grayscale images
                ax[1][0].set_title("Reconstructed Lidar points")
                ax[1][0].axis('off')  # Hide the axis for clarity

                # Plot Prediction (pred) on the bottom
                ax[1][1].imshow(depth_gt_np, cmap='gray')  # Use gray colormap for grayscale images
                ax[1][1].set_title("Ground truth")
                ax[1][1].axis('off')  # Hide the axis for clarity

                # Plot Prediction (pred) on the bottom
                ax[1][2].imshow(depth_pred_np, cmap='gray')  # Use gray colormap for grayscale images
                ax[1][2].set_title("Prediction")
                ax[1][2].axis('off')  # Hide the axis for clarity

                # Show the plot
                plt.tight_layout()
                plt.show()
                metric_test = metric.evaluate(output_['results'][-1], samplep['gt'], 'test')
            else:
                metric_test = metric.evaluate(output_['results'][-1], samplep['dep'], 'test')

            depth_validpoint_number = count_validpoint(samplep['dep'])
            depth_validpoint_number_clear = count_validpoint(samplep['dep'])
            with open(metric_txt_dir, 'a') as f:
                f.write('{}; RMSE:{}; MAE:{}; vp_pre:{}; vp_post:{}\n'.format(samplep['d_path'][0].split('/')[-1],
                                                                         metric_test.data.cpu().numpy()[0, 0]*1000,
                                                                         metric_test.data.cpu().numpy()[0, 1]*1000,
                                                                         depth_validpoint_number,
                                                                         depth_validpoint_number_clear))

            writer_test.add(None, metric_test)
            if args.save_test_image:
                writer_test.save(args.epochs, batch_, samplep, output_)

            current_time = time.strftime('%y%m%d@%H:%M:%S')
            error_str = '{} | Test'.format(current_time)
            pbar_.set_description(error_str)
            pbar_.update(loader_test.batch_size)

    pbar_.close()
    _ = writer_test.update(args.epochs, samplep, output_,
                           online_loss=False, online_metric=False, online_rmse_only=False, online_img=False)
    t_avg = t_total / num_sample
    with open(metric_txt_dir, 'a') as f:
        f.write('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


if __name__ == '__main__':
    test(config)
