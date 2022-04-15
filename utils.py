import numpy as np

def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)

def get_class_weightage(multihot_labels):
   counts = np.sum(multihot_labels, axis=0)
   max_counts = np.repeat(np.max(counts), len(multihot_labels[0]))
   return max_counts/counts

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)

def write_to_file(dname, dmap, cmap, itr, mode='goal'):
    fid = open('./run-logs/' + dname + '.log', 'a+')
    string_to_write = mode + " " + str(itr)
    average_score = np.mean(np.array(dmap))
    for item in dmap:
        string_to_write += ' ' + '%.2f' %item
    string_to_write += ' ' + '%.2f' %average_score
    string_to_write += ' ' + '%.2f' %cmap
    fid.write(string_to_write + '\n')
    fid.close()

def create_logger(cfg, phase='train'):
    """
    create a logger for experiment record
    To use a logger to publish message m, just run logger.info(m)
    :param cfg: global config
    :param phase: train or val
    :return: a logger
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(args.model_name, time_str, phase)
    final_log_file = Path(cfg.OUTPUT_DIR) / log_file
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
   
    
   




