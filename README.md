# my_mnist
A simple hand made CNN for MNIST classification using numpy skimage and script

程序仍是demo状态。
请将mnist数据集源文件放在./mnist文件夹中，并在terminal运行 ./mnist_transfer.sh转换为npy格式，数据文件将被保存在当前目录：
test_image.npy
test_label.npy
train_image.npy
train_label.npy


然后将在terminal运行./train.sh执行训练，训练好的权重文件保存为npz格式（这里已经有一个训练好的权重my_mnist_weight.npz，在测试集上的准确率为91.6%）

运行./infer.sh执行预测操作，预测结果将保存在当前目录下./result.txt，分别为test_image的每张图片的预测结果。

运行./test_accuracy.sh执行当前目录下权重文件在测试集的准确率。

