## imports
from generator import generatorModule

class trainModule:
	def __init__(self, config):
	   '''
	   ## Defines all the attributes to be accessed from a diferent class such as :-
	   1.define or load model.
	   2.load_weights if required.
	   3.define metrics.
	   4.define cost function
	   5.Set a learning rate and optimizer.
	   6.define save model and epochs after which to save model
	   Below is a sample code for initialization
	   '''
	    self.config = config
	    self.batch_size = self.config["batch_size"]
	    self.save_dir = self.config["save_dir"]
	    if self.config["define_model"]:
	    	model = self.load_model()
	    else:
	    	model = self.define_model()
	    if self.config["pretrained"]:
	    	self.load_weights()
	    self.cost = self.cost_fn()
	    self.optimizer = self.optimizer_def()

	def load_model(self):
		'''
		code to import a pre-saved model architecture
		>>> return a model architecture
		'''
		pass
	def define_model(self):
		'''
		if building model it must be defined here.
		>>> return a model architecture.
		'''
		pass
	def load_weights(Self):
		'''
		code to load weights into model architecture is required.
		>>> returns a loaded model.
		'''
		pass
	def run(self, logger):
		''' 
		This function runs a single input from generator into the model
		to write into tensorboard = {name:value}
		>>> returns [(actual,predicted), to write into tensorboard]
		'''
		self.train_gen = generatorModule(config["train_file"], config["batch_size"])
		for batch in self.train_gen():
			logger.write_train_tensorboard(self.convert_results_dict(results))
		pass
	def cost_fn(self):
		'''
		This defines the cost function to be called from run function.
		>>> returns cost based on prediction from model.
		'''
		pass
	def optimizer_def(self):
		'''
		This defines a learning rate and optimizer.
		>>> returns optimizer variable.
		'''
		pass

	def convert_results_dict(self, results, type='train'):
		if type == 'train':
			pass
		elif type == 'test':
			pass

	def test(self, logger):
		self.test_gen = generatorModule(config["test_file"], config["batch_size"])
		for batch in self.test_gen():
			
		logger.write_images(images)
		logger.write_test_tensorboard(self.convert_results_dict(results, type='test'))

	def save_model(self):
		'''
		logic to save model on each call to this function.
		returns boolean Saved or not
		'''
		pass






