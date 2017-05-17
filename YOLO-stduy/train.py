import tensorflow as tf
import datetime
import os
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
		
		# 있으면, 파일 경로 
		# 없으면, None
        self.weights_file = cfg.WEIGHTS_FILE
		# [] 괄호 왜 필요?
        self.max_iter = cfg.MAX_ITER[]
		# LEARNING_RATE = 0.0001
        self.initial_learning_rate = cfg.LEARNING_RAT'E
		# DECAY_STEPS = 30000
        self.decay_steps = cfg.DECAY_STEPS
		# DECAY_RATE = 0.1
        self.decay_rate = cfg.DECAY_RATE
		# STAIRCASE = True	# ?? 체크 필요
        self.staircase = cfg.STAIRCASE
		# SUMMARY_ITER = 10	#	??
        self.summary_iter = cfg.SUMMARY_ITER
		# SAVE_ITER = 1000	#	??
        
		
		#  output 경로, datetime 확인 필요
		self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
			
		# 위에 경로 없으면, 경로 생성	
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
		
		# 위의 경로에 'config.txt' 생성
        self.save_cfg()

		# 변수 생성
        self.variable_to_restore = tf.global_variables()
		
		'''
		 Save a Checkpoint
		In order to emit a checkpoint file that may be used to later restore a model for further training or evaluation, we instantiate a "tf.train.Saver".
			saver = tf.train.Saver()
		'''
		'''
		In the training loop, the "tf.train.Saver.save" method will periodically be called to write a checkpoint file to the training directory with the current values of all the trainable variables.
			saver.save(sess, FLAGS.train_dir, global_step=step)
		'''
		'''
		At some later point in the future, training might be resumed by using the "tf.train.Saver.restore" method to reload the model parameters.
			saver.restore(sess, FLAGS.train_dir)
		'''
		
		# 트레이닝된 변수들과 함께 .ckpt로 저장
		
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
		
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
		
		'''
		Visualize the Status

		In order to emit the events files used by TensorBoard, all of the summaries (in this case, only one) are collected into a single Tensor during the graph building phase.
			summary = tf.summary.merge_all()
		'''
		'''
		And then after the session is created, a "tf.summary.FileWriter" may be instantiated to write the events files, which contain both the "graph" itself and the "values" of the summaries.
			summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
		'''
		'''
		Lastly, the events file will be updated with new summary values every time the summary is evaluated and the output passed to the writer's "add_summary()" function.
			summary_str = sess.run(summary, feed_dict=feed_dict)
			summary_writer.add_summary(summary_str, step)
		'''
		'''
		When the events files are written, TensorBoard may be run against the training folder to display the values from the summaries.
		'''
		
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)
		
		# global step을 변수로 생각??? []는??? 초기화상태에서 트레이닝없이 유지???
        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
		###########
		self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        
		# net.total_loss에 대해서 GD를 통해 global step에 lr만큼 weight등 업데이트
		self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)
        
		
		###########
		self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        
		self.averages_op = self.ema.apply(tf.trainable_variables())
        
		with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)
		
		###########		
		
		# gpu 관련 config 인듯??
        gpu_options = tf.GPUOptions()
		config = tf.ConfigProto(gpu_options=gpu_options)
		
		# config 따라 session 생성 및 변수 초기화
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

		# .ckpt 읽기
        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)
		
		# 그래프 그림 그리기? 
		# self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)	
        self.writer.add_graph(self.sess.graph)

		
		
    def train(self):
		
		# timer class를 사용하기 위해 선언
        train_timer = Timer()
        load_timer = Timer()
		
		# 최대 iteration+1 까지 반복
        for step in xrange(1, self.max_iter + 1):
			
			# date를 로딩하는데 걸리는 시간 측정
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            
			# training input 으로...
			feed_dict = {self.net.images: images, self.net.labels: labels}
			
			# 10, 20, 30, 40, 50 ...
            if step % self.summary_iter == 0:
				# 100,200,300,400 ... 에만 print
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
                        ' Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

				# self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)
				
                self.writer.add_summary(summary_str, step)

			# 대부분 이렇게 training 근데 10번째에 summary 추가
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=self.global_step)

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__ # ??
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

	
	
	
	
    yolo = YOLONet()      			# Network building
	
    pascal = pascal_voc('train')	# Data setting

    solver = Solver(yolo, pascal)	# Solver setting

	
	
	
	
	
    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
