import torch
import torchvision

class generatorModule:
	def __init__(self, file_name, batch_size, threads = 1, type='train'):
		'''
		Define files to read in order for generator to function
		makes numpy arrays from file inputs
		support for files such as :-
		1. csv_file reader.
		2. tfrecord file reader.
		'''
		self.file = file_name
		self.batch_size = batch_size
        self.download = True
        if type == 'train':
            self.train = True
        else:
            self.train = False
        self.gen = self.gen_fn()

	def gen_fn(self):
		'''
		actual generator initialization.
		>>> returns generator function variable. 
		'''
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST('Data/', train=self.train, download=self.download,
                                           transform=torchvision.transforms.Compose(torchvision.transforms.ToTensor(),
                                                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.batch_size, shuffle=True)

		
	def __call__(self):
		'''
		running loop to be called for each input
		must yield input each time called.
		'''
        for batch_idx, example in enumerate(self.gen):
            yield batch_idx, example