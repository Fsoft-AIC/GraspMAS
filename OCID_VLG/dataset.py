import os
import json
import cv2
import numpy as np
import torch.utils.data as data
import torch
from torchvision import transforms
from torchvision.transforms import functional as tfn
import functools
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.filters import gaussian


class GraspTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=150, width=640, height=480):
        self.width_factor = width_factor
        self.width = width 
        self.height = height

    def __call__(self, grasp_rectangles, target):
        # grasp_rectangles: (M, 4, 2)
        M = grasp_rectangles.shape[0]
        p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)
        
        center_x = (p1[..., 0] + p3[..., 0]) / 2
        center_y = (p1[..., 1] + p3[..., 1]) / 2
        
        width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
        height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))
        
        theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
        theta = np.where(theta > 0, theta - 90, theta + 90)

        target = np.tile(np.array([[target]]), (M,1))

        return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta, _ = rect
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta, _ = rect
            
            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < self.width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < self.height
            cc = cc[mask_cc]
            rr = rr[mask_cc]


            pos_out[cc, rr] = 1.0
            ang_out[cc, rr] = theta * np.pi / 180
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor
        
        qua_out = gaussian(pos_out, 3, preserve_range=True)
        #ang_out = gaussian(ang_out, 2, preserve_range=True)
        wid_out = gaussian(wid_out, 3, preserve_range=True)
        
        return {'pos': pos_out, 
                'qua': qua_out, 
                'ang': ang_out, 
                'wid': wid_out}



class OCIDVLGDataset(data.Dataset):
    
    """ OCID-Vision-Language-Grasping dataset with referring expressions and grasps """

    def __init__(self, 
                 root_dir,
                 split, 
                 transform_img = None,
                 transform_grasp = GraspTransforms(),
                 with_depth = True, 
                 with_segm_mask = True,
                 with_grasp_masks = True,
                 version = "multiple"
    ):
        super(OCIDVLGDataset, self).__init__()
        self.root_dir = root_dir
        self.version = version
        self.split_map = {'train': 'train_expressions.json', 
                          'val': 'val_expressions.json',
                          'test': 'test_expressions.json'
                         }
        self.split = split
        self.refer_dir = os.path.join(root_dir, "refer", version)
        
        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        self.with_depth = with_depth
        self.with_segm_mask = with_segm_mask
        self.with_grasp_masks = with_grasp_masks
        
        self._load_dicts()
        self._load_split()

    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        from OCID_VLG.OCID_sub_class_dict import cnames, subnames, instance_to_class
        cnames_inv = {int(v):k for k,v in cnames.items()}
        subnames_inv = {v:k for k,v in subnames.items()}
        self.class_names = cnames 
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = instance_to_class
        os.chdir(cwd)

    def _load_split(self):
        refer_data = json.load(open(os.path.join(self.refer_dir, self.split_map[self.split])))
        self.seq_paths, self.img_names, self.scene_ids = [], [], []
        self.bboxes, self.grasps = [], []
        self.sent_to_index, self.sent_indices = {}, []
        self.rgb_paths, self.depth_paths, self.mask_paths = [], [], []
        self.targets, self.sentences, self.semantics, self.objIDs = [], [], [], []
        n = 0
        for item in refer_data['data']:
            seq_path, im_name = item['image_filename'].split(',')
            self.seq_paths.append(seq_path)
            self.img_names.append(im_name)
            self.scene_ids.append(item['image_filename'])
            self.bboxes.append(item['box'])
            self.grasps.append(item['grasps'])
            self.objIDs.append(item['answer'])
            self.targets.append(item['target'])
            self.sentences.append(item['question'])
            self.semantics.append(item['program'])
            self.rgb_paths.append(os.path.join(seq_path, "rgb", im_name))
            self.depth_paths.append(os.path.join(seq_path, "depth", im_name))
            self.mask_paths.append(os.path.join(seq_path, "seg_mask_instances_combi", im_name))
            self.sent_indices.append(item['question_index'])
            self.sent_to_index[item['question_index']] = n
            n += 1

    def get_index_from_sent(self, sent_id):
        return self.sent_to_index[sent_id]

    def get_sent_from_index(self, n):
        return self.sent_indices[n]
    
    def _load_sent(self, sent_id):
        n = self.get_index_from_sent(sent_id)
        
        scene_id = self.scene_ids[n]
       
        img_path = os.path.join(self.root_dir, self.rgb_paths[n])
        img = self.get_image_from_path(img_path)
        
        x, y, w, h = self.bboxes[n]
        bbox = np.asarray([x, y, x+w, y+h])
        
        sent = self.sentences[n]
        
        target = self.targets[n]
        target_idx = self.class_instance_names[target]
        objID = self.objIDs[n]
        
        grasps = np.asarray(self.grasps[n])
        
        result = {'img': self.transform_img(img) if self.transform_img else img, 
                  'grasps':  grasps,
                  'grasp_rects': self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'sentence': sent,
                  'target': target,
                  'objID': objID,
                  'bbox': bbox,
                  'target_idx': target_idx,
                  'sent_id': sent_id,
                  'scene_id': scene_id,
                 }
        
        if self.with_depth:
            depth_path = os.path.join(self.root_dir, self.depth_paths[n])
            depth = self.get_depth_from_path(depth_path)
            result = {**result, 'depth': torch.from_numpy(depth) if self.transform_img else depth}

        if self.with_segm_mask:
            mask_path = os.path.join(self.root_dir, self.mask_paths[n])
            msk_full = self.get_mask_from_path(mask_path)
            msk = np.where(msk_full == objID, True, False)
            result = {**result, 'mask': torch.from_numpy(msk) if self.transform_img else msk}

        if self.with_grasp_masks:
            grasp_masks = self.transform_grasp.generate_masks(result['grasp_rects'])
            result = {**result, 'grasp_masks': grasp_masks}
        
        return result

    def __len__(self):
        return len(self.sent_indices)
    
    def __getitem__(self, n):
        sent_id = self.get_sent_from_index(n)
        return self._load_sent(sent_id)
    
    @staticmethod
    def transform_grasp_inv(grasp_pt):
        pass
    
    @functools.lru_cache(maxsize=None)
    def get_image_from_path(self, path):
        img_bgr = cv2.imread(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    @functools.lru_cache(maxsize=None)
    def get_mask_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    @functools.lru_cache(maxsize=None)
    def get_depth_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000. # mm -> m
    
    def get_image(self, n):
        img_path = os.path.join(self.root_dir, self.imgs[n])
        return self.get_image_from_path(img_path)
    
    def get_annotated_image(self, n, text=True):
        sample = self.__getitem__(n)
        
        img, sent, grasps, bbox = sample['img'], sample['sentence'], sample['grasps'], sample['bbox']
        if self.transform_img:
            img = np.asarray(tfn.to_pil_image(img))

        tmp = img.copy()
        for entry in sample['grasps']:
                ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
                tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
                tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
        
        tmp = cv2.rectangle(tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
        if text:
            tmp = cv2.putText(tmp, sent, (0,10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
        return tmp

    def visualization(self, n, save_path):
        s = self.__getitem__(n)

        rgb = s['img']
        depth = (0xff * s['depth'] / 3).astype(np.uint8)
        ii = self.get_annotated_image(n, text=False)
        sentence = s['sentence']
        msk = s['mask']
        msk_img = (rgb * 0.3).astype(np.uint8).copy()
        msk_img[msk, 0] = 255

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(msk_img)
        ax.set_title('Segm Mask')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(ii)
        ax.set_title('Box & Grasp')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(s['grasp_masks']['pos'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Position')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(s['grasp_masks']['qua'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(s['grasp_masks']['ang'], cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title('Angle')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(s['grasp_masks']['wid'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"sample_{n}.png"))
