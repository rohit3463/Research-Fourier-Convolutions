import sklearn
import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
from Train_module import trainModule
from Test_module import testModule
import logger as Logger

class Logging:
	def __init__(self, config):
		self.trainlogdir = config["train_logdir"]
		self.testlogdir = config["test_logdir"]
		self.train_logger = Logger(self.trainlogdir)
		self.test_logger = Logger(self.testlogdir)

	def write_train_tensorboard(self, results_dict, step):
		for tag, value in results_dict.items():
			self.train_logger.scalar_summary(tag, value, step+1)

	def write_test_tensorboard(self, results_dict):
		for tag, value in results_dict.items():
			self.test_logger.scalar_summary(tag, value, step+1)

	def write_images(self, images, step):
		images_dict = {}
		for i,image in enumerate(images):
			images_dict["image/"+str(i)] = image
		for tag, images in images_dict.items():
			logger.image_summary(tag, images, step+1)

class Train:
	def __init__(self, config):
		self.config = config
		self.test_interval = self.config["test_interval"]
		self.save_interval = self.config["save_interval"]
		self.testing = self.config["to_test"]
		self.current_epoch = self.get_latest_epoch()
		self.model = trainModule(self.config)
		self.total_epochs = self.config["total_epochs"]
		self.logger = Logging(self.config)

	def run(self):
		start = time.time()
		index = 0
		for i in range(self.current_epoch, self.total_epochs):
			prev_time = time.time()
			results = self.model.run(self.logger)
			index += 1
			time_taken = float(time.time() - prev_time)
			if self.testing:
				if i % self.test_interval == 0:
					self.model.test(self.logger)
			self.save_model(i)
		total_time = time.time() - start
		print("Total time taken : {}, for : {}, time per iteration : {}".format(total_time, index, total_time/index))

	def save_model(self, epochs = 1):
		if epochs % self.save_interval == 0:
			self.model.save_model()
			with open('temp.log','a') as f:
				f.write('\n')
				f.write("Epochs done : [{}]".format(epochs))

	def get_latest_epoch(self):
		if os.path.exists('temp.log'):
			self.config["pretrained"] = True
			with open('temp.log', 'r') as f:
				lines = f.read().splitlines()
				last_line = lines[-1]
				return last_line.split('[')[1].split(']')[0]
		else:
			return 0

