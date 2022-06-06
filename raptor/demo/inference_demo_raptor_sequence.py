from raptor.demo.inference_tools import *
from tqdm import tqdm


def get_image_size(root, sequence, specific_subdir=None):
    if specific_subdir is None:
        specific_subdir = ''
    images_seq_dir = os.path.join(root, sequence, args.specific_subdir)
    file_list = [f for f in os.listdir(os.path.join(images_seq_dir)) if os.path.isfile(os.path.join(images_seq_dir, f))]
    file_list.sort()

    img_tmp_path = os.path.join(root, sequence, args.specific_subdir, file_list[0])
    img_tmp = imageio.imread(img_tmp_path)
    return img_tmp.shape[1], img_tmp.shape[0]

def set_calib(image_size, calib_focal, calib_dx, calib_dy):
    # setting parameters according to first image in the sequence
    fl = min(image_size[0], image_size[1]) * 2 if calib_focal is None else calib_focal
    cx = image_size[0] / 2. if calib_dx is None else calib_dx
    cy = image_size[1] / 2. if calib_dy is None else calib_dy

    K0 = np.eye(3)
    K0[0, 0] = fl
    K0[1, 1] = fl
    K0[0, 2] = cx
    K0[1, 2] = cy
    calib_K = K0
    calib_K_2 = K0
    calib_K_0 = K0
    return calib_K, calib_K_2, calib_K_0

def build_model(checkpoint_file, image_size):
    if args.config_file is None:
        checkpoint_file_split = checkpoint_file.split('/')
        config_file = os.path.join('/', *checkpoint_file_split[:-2], checkpoint_file_split[-2], checkpoint_file_split[-2] + '.py')
    else:
        config_file = args.config_file

    config = load_and_change_config(config_file, image_size)
    if 'CACHE_TORCH' in os.environ:
        model = init_detector(config, checkpoint_file, device='cuda:' + args.gpuid)
    else:
        checkpoint_file_filtered = filter_checkpoint_file(checkpoint_file)
        # build the model from a config file and a checkpoint file
        model = init_detector(config, checkpoint_file_filtered, device='cuda:' + args.gpuid)
    
    try:
        a = model.cfg.data.test.pipeline
    except:
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0]['pipeline']
    return model

def main(args):

    calib_baseline = args.baseline
    calib_focal = args.focal
    calib_dx = args.camera_dx
    calib_dy = args.camera_dy

    root = args.input_dict
    checkpoint_file = args.checkpoint_file


    if args.search_subdirs:
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        subdirs.sort()
    else:
        subdirs = ['']

    image_size = get_image_size(root, subdirs[0], specific_subdir=args.specific_subdir)
    calib_K, calib_K_2, calib_K_0 = set_calib(image_size, calib_focal, calib_dx, calib_dy)

    model = build_model(checkpoint_file, image_size)

    save_root = '{:s}/{:s}/images/'.format(args.output_dict, checkpoint_file.split('/')[-2]+'_'+os.path.basename(checkpoint_file)[:-4])
    save_root_masks = '{:s}/{:s}/masks/'.format(args.output_dict, checkpoint_file.split('/')[-2]+'_'+os.path.basename(checkpoint_file)[:-4])
    save_root_single_mask = '{:s}/{:s}/single_mask/'.format(args.output_dict, checkpoint_file.split('/')[-2] + '_' + os.path.basename(checkpoint_file)[:-4])
    save_root_custom_mask = '{:s}/{:s}/custom/'.format(args.output_dict, checkpoint_file.split('/')[-2]+'_'+os.path.basename(checkpoint_file)[:-4])



    T = Timer()
    for sequence in subdirs:
        print(sequence)

        image_size_current = get_image_size(root, sequence, specific_subdir=args.specific_subdir)
        if image_size != image_size_current:
            print('RELOADING MODEL!!! DIFFERENT IMAGE SIZE')
            image_size = image_size_current
            calib_K, calib_K_2, calib_K_0 = set_calib(image_size, calib_focal, calib_dx, calib_dy)
            model = build_model(checkpoint_file, image_size)

        current_sequence_paths = {}
        label_mask_prev = None
        sep_masks_prev = None

        images_seq_dir = os.path.join(root, sequence, args.specific_subdir)

        file_list = [f for f in os.listdir(os.path.join(images_seq_dir)) if os.path.isfile(os.path.join(images_seq_dir, f))]
        if args.file_prefix is not None:
            file_list = [f for f in file_list if f.startswith(args.file_prefix)]
        if args.extension is not None:
            file_list = [f for f in file_list if f.endswith(args.extension)]
        file_list.sort()

        sequence_length_counter = 0
        skip_length_counter = 0

        pbar = tqdm(range(len(file_list)))
        for i in pbar:

            if args.split_sequence:
                if sequence_length_counter > args.max_sequence_length:
                    if skip_length_counter > args.skip_steps:
                        sequence_length_counter = 0
                        skip_length_counter = 0
                    else:
                        skip_length_counter += 1
                    continue
                else:
                    sequence_length_counter += 1

            file_name = file_list[i]

            tqdm.write(f"Working on: {file_name} ({sequence})")

            file_name_path = os.path.join(images_seq_dir, file_list[i])

            # in case of first or last image in the sequence - mirroring images
            if i == 0:
                file_name_0_path = os.path.join(images_seq_dir, file_list[1])
            else:
                file_name_0_path = os.path.join(images_seq_dir, file_list[i-1])

            if i == len(file_list)-1:
                file_name_2_path = os.path.join(images_seq_dir, file_list[-2])
            else:
                file_name_2_path = os.path.join(images_seq_dir, file_list[i+1])

            if not os.path.isfile(file_name_2_path) or not os.path.isfile(file_name_0_path):
                continue
            #print(file_name_path)

            T.iter()
            result = inference_moseg_detector(model, file_name_path, file_name_2_path, calib_K=calib_K, calib_K_2=calib_K_2, calib_baseline=calib_baseline, calib_K_0=calib_K_0, img0=file_name_0_path, additional_outputs_setting=['flow_t_t-1'])
            t = T.iter()
            print(t)
            if isinstance(result[1], dict):
                result, flow_t_tm1_torch = result[0], result[1]['flow_t_t-1']
            else:
                result, flow_t_tm1_torch = result[0], None

            if args.debug:
                show_result_pyplot(model, file_name_path, result)

            loaded_img, label_mask = create_mask(file_name_path, result, threshold=args.confidence_threshold)
            sep_masks = mask_separation(file_name_path, result, threshold=args.confidence_threshold)
            sep_masks, sorted_masks_list = sort_sep_masks(sep_masks)

            flow_t_tm1_numpy = None
            if flow_t_tm1_torch is not None:
                flow_t_tm1_numpy = flow_torch2numpy_resize_rescale(flow_t_tm1_torch, loaded_img.shape)
                if sep_masks_prev is not None or label_mask_prev is not None:
                    sep_masks, label_mask = simple_tracker(flow_t_tm1_numpy, sep_masks, label_mask, sep_masks_prev, label_mask_prev)

            if args.save_outputs:
                save_path = os.path.join(save_root, sequence, file_name)[:-4] + '.jpg'
                save_path_masks = os.path.join(save_root_masks, sequence, file_name)
                save_path_single_mask = os.path.join(save_root_single_mask, sequence, file_name)

                save_labels(label_mask, loaded_img, deepcopy(save_path))
                save_masks(sep_masks, deepcopy(save_path_masks))
                save_single_mask(sep_masks, loaded_img, deepcopy(save_path_single_mask), sort_masks=False)

            if args.save_custom_outputs:
                save_path_custom = os.path.join(save_root_custom_mask, sequence, file_name)
                fig_size = (10.0, 10.0 * loaded_img.shape[0]/loaded_img.shape[1])
                custom_image(loaded_img, sep_masks, contour=True, contour_color='w', active=True, figsize=fig_size, interpolation='nearest', save_path=save_path_custom, show_image=args.debug, sort_masks=False)

            label_mask_prev = label_mask
            sep_masks_prev = sep_masks

def get_input_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='input arguments for evaluation')
    parser.add_argument('--gpuid', default='1')
    parser.add_argument('--config_file', default=None, help="Configuration file.py path. If not set, it is parsed from checkpoint_file")
    parser.add_argument('--checkpoint_file', required=True, help="Model weights path")
    parser.add_argument('--debug', action='store_true', help="Debug mode with showing images during estimation process")

    parser.add_argument('--confidence_threshold', default=0.3, help="Confidence threshold for output images")

    parser.add_argument('--input_dict', required=True, help="Full path to your dictionary with image sequence (png/jpeg/jpg)")
    parser.add_argument('--output_dict', required=True, help="Dictionary for saving outputs")
    parser.add_argument('--search_subdirs', action='store_true', help="Search subdirectories (i.e. False for KITTI, True for DAVIS)")

    parser.add_argument('--save_outputs', action='store_true', )
    parser.add_argument('--save_custom_outputs', action='store_true')

    parser.add_argument('--baseline', default=1, help="Original camera baseline (default=1)")
    parser.add_argument('--focal', default=None, help="Camera calib focal lenght")
    parser.add_argument('--camera_dx', default=None, help="Camera calib dx")
    parser.add_argument('--camera_dy', default=None, help="Camera calib dy")

    parser.add_argument('--specific_subdir', default='', help='Use this if path of images dir is contained in another subdirs. Look to code for usage.')
    parser.add_argument('--skip_if_exist', action='store_true')
    parser.add_argument('--extension', default=None, help='Specific file extension (png|jpg|...), default -> All files')
    parser.add_argument('--file_prefix', default=None, help='File Prefix i.e. 000046_')

    parser.add_argument('--split_sequence', action='store_true')
    parser.add_argument('--max_sequence_length', default=-1, type=int)
    parser.add_argument('--skip_steps', default=0, type=int)

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = get_input_args()
    main(args)