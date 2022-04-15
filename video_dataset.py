import numpy as np
import utils
import torch

class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.path_to_annotations = self.dataset_name + '-Annotations/'
        self.vidnames = np.load(self.path_to_annotations + 'videoname.npy', allow_pickle=True)
        self.goal_labels = np.load(self.path_to_annotations + 'goal_labels.npy', allow_pickle=True)     # Specific to Thumos14
        self.wentwrong_labels = np.load(self.path_to_annotations + 'wentwrong_labels.npy', allow_pickle=True)
        self.goal_segments = np.load(self.path_to_annotations + 'goal_segments.npy', allow_pickle=True)
        self.wentwrong_segments = np.load(self.path_to_annotations + 'wentwrong_segments.npy', allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + 'subset.npy', allow_pickle=True)
                    
        if args.feature_type == 'i3d':
            self.feature_size = 1024
            self.path_to_features = './video_features/i3d_25fps_feats.npy'
            self.dict_features = np.load(self.path_to_features, encoding='bytes', allow_pickle=True)
            self.features = np.array([self.dict_features.item()[name] for name in self.vidnames])
        elif args.feature_type == 'r3d':
            self.feature_size = 512
            self.path_to_features = './video_features/r2plus1d_25fps_feats.pth'
            self.dict_features = torch.load(self.path_to_features)
            self.features = np.array([self.dict_features[name].detach().cpu().numpy() for name in self.vidnames])
        self.pose_dict_features = np.load('./video_features/pose_feats.npy', encoding='bytes', allow_pickle=True)
        self.pose_features = np.array([self.pose_dict_features.item()[name] for name in self.vidnames])


        self.goal_classlist = np.load(self.path_to_annotations + 'goal_list.npy', allow_pickle=True)
        self.wentwrong_classlist = np.load(self.path_to_annotations + 'wentwrong_list.npy', allow_pickle=True)
        self.num_goal_class = len(self.goal_classlist)
        self.num_wentwrong_class = len(self.wentwrong_classlist)
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.goal_classwiseidx = []
        self.wentwrong_classwiseidx = []
        self.remove_ixs = []
        self.currenttestidx = 0
        self.goal_labels_multihot = [utils.strlist2multihot(labs,self.goal_classlist) for labs in self.goal_labels]
        self.wentwrong_labels_multihot = [utils.strlist2multihot(labs,self.wentwrong_classlist) for labs in self.wentwrong_labels]
        self.goal_class_weightage = utils.get_class_weightage(self.goal_labels_multihot)
        self.wentwrong_class_weightage = utils.get_class_weightage(self.wentwrong_labels_multihot)
       
        self.train_test_idx()
        print(f"There are {len(self.trainidx)} train samples and {len(self.testidx)} test samples")
        self.classwise_feature_mapping()
        
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s == 'train':   # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.goal_classlist:
            idx = []
            for i in self.trainidx:
                for label in self.goal_labels[i]:
                    if label == category:
                        idx.append(i); break;
            self.goal_classwiseidx.append(idx)
        
        for category in self.wentwrong_classlist:
            idx = []
            for i in self.trainidx:
                for label in self.wentwrong_labels[i]:
                    if label == category:
                        idx.append(i); break;
            self.wentwrong_classwiseidx.append(idx)


    def load_data(self, n_similar=3, is_training=True):
        if is_training==True:
            idx = []
        
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feats = np.array([utils.process_feat(self.features[i], self.t_max) for i in idx])
            pose_feats = np.array([utils.process_feat(self.pose_features[i], self.t_max) for i in idx])
            # feats = np.array([utils.process_feat(np.concatenate([self.features[i], self.pose_features[i]],axis=-1), self.t_max) for i in idx])
            goal_multihot = np.array([self.goal_labels_multihot[i] for i in idx])
            wentwrong_multihot = np.array([self.wentwrong_labels_multihot[i] for i in idx])
            return feats,pose_feats, goal_multihot, wentwrong_multihot

        else:
            goal_labs = self.goal_labels_multihot[self.testidx[self.currenttestidx]]
            wentwrong_labs = self.wentwrong_labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            pose_feat = self.pose_features[self.testidx[self.currenttestidx]]
            # feat = np.concatenate([feat,pose_feat],axis=-1)
            vid_name = self.vidnames[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array(feat), np.array(pose_feat), np.array(goal_labs), np.array(wentwrong_labs), done, vid_name

