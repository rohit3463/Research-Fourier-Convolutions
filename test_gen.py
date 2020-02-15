from datagen_mnist import generatorModule

genObj = generatorModule(file_name=None,batch_size=16, type='test')

for batch_idx, (example_data, example_targets) in genObj():
    print(example_data.shape)
    print(example_targets.shape)