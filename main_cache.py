"""
CuBID: Cubic Discrete Diffusion for High-Dimensional Representation Tokens
"""

import argparse
import datetime
import numpy as np
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import util.misc as misc
from util.loader import ImageFolderWithFilename
from util.crop import center_crop_arr

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rae import create_rae


def cache_latents(rae_model,
                  data_loader,
                  device: torch.device,
                  args=None):
    """Cache RAE latents to disk."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            encoder_features = rae_model.encode(samples)
            encoder_features_flip = rae_model.encode(samples.flip(dims=[3]))

            moments = encoder_features
            moments_flip = encoder_features_flip

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path,
                    moments=moments[i].cpu().numpy(),
                    moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return


def get_args_parser():
    parser = argparse.ArgumentParser('Cache RAE latents', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # RAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size (RAE internally resizes to 224)')
    parser.add_argument('--encoder_size', default=224, type=int,
                        help='RAE encoder input size')
    parser.add_argument('--encoder_name', default='facebook/dinov2-with-registers-base', type=str,
                        help='DINOv2 encoder name')
    parser.add_argument('--decoder_path',
                        default='/mnt/bn/dq-storage-ckpt/wangyuqing/huggingface/hub/RAE-collections/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
                        type=str, help='Path to RAE decoder weights')
    parser.add_argument('--stats_path',
                        default='/mnt/bn/dq-storage-ckpt/wangyuqing/huggingface/hub/RAE-collections/stats/dinov2/wReg_base/imagenet1k/stat.pt',
                        type=str, help='Path to RAE normalization stats')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
    ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Loading RAE model...")
    print(f"  Encoder: {args.encoder_name}")
    print(f"  Encoder size: {args.encoder_size} (internal resize from {args.img_size})")
    print(f"  Decoder path: {args.decoder_path}")
    print(f"  Stats path: {args.stats_path}")

    rae_model = create_rae(
        encoder_name=args.encoder_name,
        encoder_size=args.encoder_size,
        decoder_path=args.decoder_path,
        stats_path=args.stats_path,
        device=device
    )

    rae_model.eval()
    print("RAE model loaded and set to eval mode")

    print(f"Start caching RAE latents")
    print(f"  Input images: {args.img_size}x{args.img_size} (center cropped)")
    print(f"  RAE internally resizes to: {args.encoder_size}x{args.encoder_size}")
    print(f"  Output cache path: {args.cached_path}")

    start_time = time.time()
    cache_latents(
        rae_model,
        data_loader_train,
        device,
        args=args
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if not args.cached_path:
        raise ValueError("--cached_path must be specified for saving latents")

    main(args)
