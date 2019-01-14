config = {}

config['batch_size'] = 4
config['thread_num'] = 10
config['img_inp_shape'] = [None, None, 1]
config['out_shape'] = [None, None, 1]
config['min_size'] = 512
config['max_size'] = 512
config['img_pad_val'] = 0
config['label_pad_val'] = 0
config['aug_pix_enlarge'] = 32
config['pad_divisable'] = 32
config['debug'] = False

config['data_root'] = '/Users/aviavidan/data/viz/takehome'
config['log_root'] = '/Users/aviavidan/data/viz/logs'
config['samples_root'] = config['data_root'] + '/jpg'
config['labels_root'] = config['data_root'] + '/bmp'

config['train_val_lists'] = config['data_root'] + '/train_val_lists.pickle'
