import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy
import yolo.config as cfg

class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        
		self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
		
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
		
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
		
		# True of False
        self.flipped = cfg.FLIPPED
        
		# 'train'
		self.phase = phase
        
		# True of False
		self.rebuild = rebuild
		
        self.cursor = 0 # data set의 어는 부분인지 체크
        self.epoch = 1	# data set 한번 끝까지 완성
        self.gt_labels = None
        
		self.prepare()

	# batch로 묶는 부분 인듯?
    def get(self):
		# batch_size x image_size x image_size x 3, 
		# e.g. 10x448x448x3		
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
		
		# batch_size x cell_size x cell_size x 25(20+5), 
		# e.g. 10x448x448x3	
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
		
        count = 0
		
		# batch size 만큼 반복 (count를 증가시키며)
		
		# gt_labels = 
        #    gt_labels.append({'imname': imname, 'label': label, #    'flipped': False})
		
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
			
			# image_Read -> batch로 묶기
            images[count, :, :, :] = self.image_read(imname, flipped)
            
			
			labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
			
            count += 1
            self.cursor += 1
			
			# label 끝까지 돌면, 새로 랜덤하게 섞고, cursor 0으로 옮기고 다시 시작
			# epoch는 1 증가
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
      
		return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		# ???
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image
	
	# flip 만들고, 램덤으로 섞기 
    def prepare(self):
        gt_labels = self.load_labels()
        
		if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        
		np.random.shuffle(gt_labels)
        
		self.gt_labels = gt_labels
        
		return gt_labels

    def load_labels(self):
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

		# cache file이 존재하고, rebuild 옵션이 아닐때,		
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = cPickle.load(f)
            return gt_labels
		# 끝
		
        print('Processing gt_labels from: ' + self.data_path)
		
		# cache 경로 없으면 경로 생성
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
		
		# train 모드와 test 모드 구분하여 txt파일 name 경로 찾기
        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main','trainval.txt')
        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main','test.txt')
		
		# txt 읽기에서 한줄한줄 읽으며, 모든 imgae index 모두 읽기
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        # gt_labels list 생성
		gt_labels = []
		
		# index에 맞게 label과 num 생성
		# image index들에서 index 하나나 읽으며, 반복
		# num은 한 이미지 xml의 object 갯수
        for index in self.image_index:
            
			#label[y_ind, x_ind, 0] = 1, 
            #label[y_ind, x_ind, 1:5] = boxes
            #label[y_ind, x_ind, 5 + cls_ind] = 1
			
			label, num = self.load_pascal_annotation(index)
			
            if num == 0:
                continue
            
			imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
			
			# gt_labels 리스트에 추가
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
			
        print('Saving gt_labels to: ' + cache_file)
		
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_labels, f)
        
		return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
		# JPEGImages 폴더에서 jpg 이미지들 불러오기
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
		
		# 설정된 image_size에 대해서 읽은 이미지 크기(h,w) 몇 배인가?
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

		#(7x7 * (20+2*5)) -> groun truth이기에 (7x7 * (20+1*5))
        label = np.zeros((self.cell_size, self.cell_size, 25))
		
		# Annotations 폴더에서 xml 파일 읽기
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
			# max(~,0) -> 음수 제거
			# min(xmin-1,image-1) -> image 전체 사이즈 넘지않게 
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
			cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
			
			# x,y,w,h가 boxes에 저장
			boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
			
			# x,y가 몇 번 grid에 있는지
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            
			# label 0번째에 입력됬으면, skip
			if label[y_ind, x_ind, 0] == 1:
                continue
				
			#
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
