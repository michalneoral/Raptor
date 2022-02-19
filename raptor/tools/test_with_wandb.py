import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from raptor.tools.wandb_tools import *
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--eval_without_images', action='store_true')
    parser.add_argument('--save_output_images', action='store_true')
    parser.add_argument('--save_towards_path', type=str, default=None)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)


    # INIT WANDB
    print("start init wandb")
    _, init_kwargs = prepare_cfg_for_wandb(cfg, args.debug)
    import wandb
    #_ = init_kwargs.pop('config')
    if args.wandb_id is not None:
        init_kwargs.update({'resume': 'allow', 'id': args.wandb_id})
    if args.wandb_project is not None:
        init_kwargs.update({'project': args.wandb_project})
    if args.wandb_entity is not None:
        init_kwargs.update({'entity': args.wandb_entity})

    print(init_kwargs)
    wandb.init(**init_kwargs)
    # wandb.init(project='debug', name='debug')

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is None:
        print('WITHOUT CHECKPOINT')
        checkpoint = dict()
        checkpoint['meta'] = dict()
        checkpoint['meta']['iter'] = 16000
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if args.save_output_images:
        from raptor.demo.inference_tools import custom_image, save_masks, save_labels, save_single_mask, create_mask, mask_separation

        for i in range(len(outputs)):
            result = outputs[i]
            file_name_path = dataset[i]['img_metas'][0].data['filename']

            loaded_img, label_mask = create_mask(file_name_path, result, threshold=0.5)
            sep_masks = mask_separation(file_name_path, result, threshold=0.5)

            # if args.save_outputs:
            #     save_path = os.path.join(save_root, sequence, file_name)[:-4] + '.jpg'
            #     save_path_masks = os.path.join(save_root_masks, sequence, file_name)
            #     save_path_single_mask = os.path.join(save_root_single_mask, sequence, file_name)
            #
            #     save_labels(label_mask, loaded_img, deepcopy(save_path))
            #     save_masks(sep_masks, deepcopy(save_path_masks))
            #     save_single_mask(sep_masks, loaded_img, deepcopy(save_path_single_mask))

            # if args.save_custom_outputs:
            sequence = args.config.split('/')[-1].split('.')[0]
            save_root_custom_mask = '/datagrid/personal/neoral/raptor_outputs/mask/'
            save_root_custom_mask_w = '/datagrid/personal/neoral/raptor_outputs/mask_w/'
            save_root_custom_with_gt_mask = '/datagrid/personal/neoral/raptor_outputs/gt/'
            file_name = file_name_path.replace('/datagrid/public_datasets/', '')

            save_path_custom = os.path.join(save_root_custom_mask, sequence, file_name)
            save_path_custom_w = os.path.join(save_root_custom_mask_w, sequence, file_name)
            im_w = loaded_img.shape[1] / loaded_img.shape[0] * 3
            custom_image(loaded_img, sep_masks, contour=True, contour_color='w', active=True, figsize=(im_w, 3), interpolation='nearest', save_path=save_path_custom,
                         show_image=False)

            im_h = loaded_img.shape[0] / loaded_img.shape[1] * 10
            custom_image(loaded_img, sep_masks, contour=True, contour_color='w', active=True, figsize=(10, im_h), interpolation='nearest', save_path=save_path_custom_w,
                         show_image=False)
            # gt_file_path = os.path.join(gt_path, sequence, file_name)
            # if os.path.isfile(gt_file_path):
            #     gt_input = imageio.imread(gt_file_path)
            #     gt_input[gt_input == 255] = 1
            #     gt_input[gt_input == 0] = 0
            #
            #     save_path_custom_with_gt = os.path.join(save_root_custom_with_gt_mask, file_name)
            #     custom_image(loaded_img, sep_masks, labels_gt=gt_input, contour=True, contour_color='w', active=True, figsize=(19.8, 3), interpolation='nearest',
            #                  save_path=save_path_custom_with_gt,
            #                  show_image=False)
    if args.save_towards_path:
        from raptor.demo.inference_tools import mkdir_from_full_file_path_if_not_exist, custom_image, save_masks, save_labels, save_single_mask, create_mask, mask_separation
        import pickle
        for i in range(len(outputs)):
            # outputs = self.retransform_masks_bboxes(segm, bbox, ori_shape)
            result = outputs[i]
            file_name_path = dataset[i]['img_metas'][0].data['filename']

            bbox = result[0]
            segm = result[1]

            # path_image = path_image.replace('image_2/', 'towards_kitti_padding64/')
            if 'KITTI' in file_name_path:
                filename = os.path.basename(file_name_path).split('.')[0]
                seq = filename.split('_')[0]
            elif 'DAVIS' in file_name_path:
                filename = os.path.basename(file_name_path).split('.')[0]
                seq = file_name_path.split('/')[-2]
            else:
                filename = os.path.basename(file_name_path).split('.')[0]
                seq = file_name_path.split('/')[-2]
            #
            filename = filename.replace('.jpg', '').replace('.png', '')
            pickle_file_path = '%s/%s/joint/detections/%s.pickle' % (args.save_towards_path, seq, filename)
            mkdir_from_full_file_path_if_not_exist(pickle_file_path)
            # pickle_file = np.load(pickle_file_path, allow_pickle=True)
            # segm = pickle_file['segmentations']
            # bbox = pickle_file['boxes']

            with open(pickle_file_path, "wb") as f:
                pickle.dump({'segmentations': segm, 'boxes': bbox}, f, protocol=2)

    rank, _ = get_dist_info()
    if rank == 0:
        step = checkpoint['meta']['iter']
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, step=step, with_images=not(args.eval_without_images), **kwargs))
            results = dataset.evaluate(outputs, **eval_kwargs)
            # print(results)

            results = {'val/'+k:v for k,v in results.items()}
            wandb.log(results, step=step)
            if args.checkpoint is None:
                wandb.log(results, step=100000)

        wandb.finish()

if __name__ == '__main__':
    main()
