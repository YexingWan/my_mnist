import numpy as np
import scipy.signal as signal
from skimage.measure import block_reduce

class My_mnist():

    def __init__(self,X,Y,batch_size=256,lr=0.01,C = 0.01,momumtum = 0.9):
        # X.shape: (num,channel,h,w)



        idx = np.random.permutation(X.shape[0])
        self._X = X[idx]
        self._Y = Y[idx]


        self._C = C
        self._momumtum = momumtum
        self._lr = lr
        self._batch_size = batch_size

        self.__gradient_list = []

        # weight_shape:(output_channel,input_channel,h,w)
        self.__w1 = np.random.normal(0,1,(16,1,3,3))
        self.__feature_conv1_nopool = None
        self.__maxpooling1_mask = None
        self.__feature_conv1 = None

        self.__w2 = np.random.normal(0,1,(16,16,3,3))
        self.__feature_conv2_nopool = None
        self.__maxpooling2_mask = None
        self.__feature_conv2 = None

        self.__w3 = np.random.normal(0,1,(8,16,1,1))
        self.__feature_conv3 = None

        self.__dense_w = np.random.normal(0,1,(8*7*7,10))
        self.loss = []


    def __conv(self,input_m,weight):
        assert(input_m.shape[1]==weight.shape[1])
        b_s, c, h, w = input_m.shape
        o_c, _, k_s, _ = weight.shape
        output_f = np.empty((b_s,o_c,h,w))
        for img_idx in range(b_s):
            img = input_m[img_idx]
            for k_idx in range(o_c):
                kernel = weight[k_idx]
                output_f[img_idx,k_idx,:,:] = np.array([signal.convolve2d(img[channel],kernel[channel],mode='same') for channel in range(c)]).sum(axis=0)
        return output_f


    def __fc(self,input_m,w):
        if len(input_m.shape) != 2:
            input_m = np.reshape(input_m,(input_m.shape[0],-1))
        r = np.dot(input_m,w)
        return r


    def __softmax(self,input_m):
        #input.shape: (batch_size, class_num)
        max = np.max(input_m,axis=1).reshape(-1,1)
        exp = np.exp(input_m - max)
        s = np.sum(exp,axis=1).reshape((-1,1))
        return exp / s

    # cross_entropy + self.C * L2
    def __loss(self,batch_pred,batch_Y):
        assert(batch_pred.shape[0]==batch_Y.shape[0])
        m = batch_pred.shape[0]
        log_p = np.log2(np.clip(batch_pred, 1e-12, 1))
        # loss = (-1/m)*(np.sum(log_p[range(len(batch_Y)), batch_Y]))+self._C * 2 * np.sqrt(np.sum(np.square(self.__w1))+np.sum(np.square(self.__w2))+np.sum(np.square(self.__w3))+np.sum(np.square(self.__dense_w)))
        loss = (-1 / m) * np.sum(log_p[range(len(batch_Y)), batch_Y])

        return loss


    def __ReLU(self,input_m):
        return np.maximum(input_m,0)


    def __maxpooling(self,input_maxtrix):
        result = block_reduce(input_maxtrix, (1, 1, 2, 2), np.max)
        test = result.repeat(2, axis=2).repeat(2, axis=3)
        mask = np.equal(input_maxtrix, test)
        return result, mask



    def __gradient_logit(self,target,predict):
        m = target.shape[0]
        predict[range(m), target] -= 1
        return  predict/m


    def __gradient_fc_w(self,next_grad):
          p = self.__feature_conv3.shape[1]*self.__feature_conv3.shape[2]*self.__feature_conv3.shape[3]
          features = self.__feature_conv3.reshape(-1, p)
          return features.T.dot(next_grad)


    def __gradient_fc_x(self, next_gard):
        return next_gard.dot(self.__dense_w.T)


    # 返回输入ReLU前的参数对loss的梯度
    def __gradient_relu(self,o,next_grad):
        return np.array(o > 0,dtype=int) * next_grad

    def __gradient_maxpooling(self,mask,next_grad):
        g  = next_grad.repeat(2,2).repeat(2,3)
        return g*mask



    '''
    计算卷积kernel对loss的梯度
    输入参数:
    inputs: 卷积层的输入features (number,channel,h,w)
    next_gradient: 之后要乘起来的梯度
    weight_shape: (output_channel,input_channel,h,w)
    '''
    def __gradient_conv_weight(self,inputs,weight_shape,next_grad):

        grad = np.empty(weight_shape)
        if weight_shape[2] != 1:
            inputs = np.pad(inputs, ((0, 0),(0, 0), (1,1), (1, 1)), 'constant')
        # assert(mask.shape == outputs.shape)

        # """
        # 每个kernel扫过的位置是一样的，上不同位置（不同h,w）的weight所"扫过"的范围不一样，不是全图
        # 范围可以用offset获得
        # """
        # for h in range(weight_shape[2]):
        #     for w in range(weight_shape[3]):
        #
        #         #对于某一个 h和w，算出offset，获得对应原图的位置
        #         offset = (weight_shape[2]-h-1,weight_shape[3]-w-1)
        #         related = inputs[:, : , h:(inputs.shape[2]-offset[0]), w:(inputs.shape[3] - offset[1])] # 256*16*7*7
        #
        #         #每个kernel对应output的不同channel，而output的不同channel有自己的mask（ReLU+Maxpooling）
        #         for k in range(weight_shape[0]):
        #             # 对于某一个kernel k，对应每个output的channel k
        #             # 为每个output的channel k， 构建对应的input上的mask（因为每个channel k上的mask不同）:
        #             final_mask = (self.__gradient_relu(outputs)[:,k] * mask[:,k]).reshape((self._batch_size,1,outputs.shape[2],outputs.shape[3])).repeat(inputs.shape[1],axis = 1)
        #             grad[k, :, h, w] = np.sum(np.sum(related * final_mask,axis=0)*(last_grad[k]),axis=(1,2))

        for k in range(weight_shape[0]):
            #for c in range(weight_shape[1]):
            for h in range(weight_shape[2]):
                for w in range(weight_shape[3]):
                    offset = (weight_shape[2] - h - 1, weight_shape[3] - w - 1)
                    related = inputs[:, :, h:(inputs.shape[2] - offset[0]),w:(inputs.shape[3] - offset[1])] #(batch_size,1,f_size,f_size)
                    #第K个kernel上的每个weight对应output上第K个channel的所有点都有独立的梯度
                    grad[k, :, h, w] = (related * next_grad[:,k].reshape(self._batch_size,1,next_grad.shape[2],next_grad.shape[3]) ).sum(axis = (0,2,3))

        return grad



    def __gradient_conv_x_as_sample(self,weight,sample_shape,next_grad):
        # print(sample_shape)
        # print(weight.shape)
        # print(next_grad.shape)
        grad = np.zeros(shape=sample_shape)
        #对于每个kernel
        for k in range(weight.shape[0]):
            grad += weight[k].sum(axis=(1,2))\
                .reshape(1,-1,1,1)\
                .repeat(grad.shape[3],3)\
                .repeat(grad.shape[2],2)\
                .repeat(grad.shape[0],0) * next_grad[:,k].reshape(-1,1,grad.shape[2],grad.shape[3])
        return grad

    def __forward_pass(self, batch):

        # 第一层卷积
        self.__feature_conv1_nopool = self.__ReLU(self.__conv(batch, self.__w1))
        self.__feature_conv1, self.__maxpooling1_mask = self.__maxpooling(self.__feature_conv1_nopool)

        # 第二层卷积
        self.__feature_conv2_nopool = self.__ReLU(self.__conv(self.__feature_conv1, self.__w2))
        self.__feature_conv2, self.__maxpooling2_mask = self.__maxpooling(self.__feature_conv2_nopool)

        # 第三层卷积
        self.__feature_conv3 = self.__ReLU(self.__conv(self.__feature_conv2, self.__w3))

        # 全连接
        predict = self.__softmax(self.__fc(self.__feature_conv3, self.__dense_w))

        return predict

    def __back_propagation(lr = 0.001, momuntun = 0.9):

        pass


    def train(self,epoch = 10):
        batch_num = self._X.shape[0] // self._batch_size
        for e in range(epoch):
            print(f'running epoch {e}...')
            for n in range(batch_num):
                start = n * self._batch_size
                end = (n + 1) * self._batch_size
                batch_X = self._X[start:end]
                batch_Y = self._Y[start:end]
                predict_Y = self.__forward_pass(batch_X)
                loss = self.__loss(predict_Y,batch_Y)
                print(f"loss:{loss}")


                grad_output_fc_as = self.__gradient_logit(batch_Y,predict_Y) #(256,10)

                grad_weight_fc = self.__gradient_fc_w(next_grad = grad_output_fc_as) #(392*10)


                grad_output_conv3_relu_as = self.__gradient_fc_x(next_gard=grad_output_fc_as) # shape: (256*392)



                #(256*392)
                grad_output_conv3_relu_as = grad_output_conv3_relu_as.reshape((self._batch_size,
                                                                               self.__feature_conv3.shape[1],
                                                                               self.__feature_conv3.shape[2],
                                                                               self.__feature_conv3.shape[3]))

                grad_output_conv3_as =  self.__gradient_relu(o = self.__feature_conv3,
                                                          next_grad=grad_output_conv3_relu_as)


                grad_weight_conv3 = self.__gradient_conv_weight(inputs=self.__feature_conv2,
                                                                weight_shape=self.__w3.shape,
                                                                next_grad = grad_output_conv3_as)





                grad_output_conv2_maxpooling_as = self.__gradient_conv_x_as_sample(weight=self.__w3,
                                                                                   sample_shape=self.__feature_conv2.shape,
                                                                                   next_grad=grad_output_conv3_as)
                # gradient矩阵变大/对到每个sample
                grad_output_conv2_relu_as = self.__gradient_maxpooling(mask=self.__maxpooling2_mask,next_grad=grad_output_conv2_maxpooling_as)



                grad_output_conv2_as= self.__gradient_relu(o = self.__feature_conv2_nopool,next_grad=grad_output_conv2_relu_as)


                grad_weight_conv2 = self.__gradient_conv_weight(inputs=self.__feature_conv1,
                                                                weight_shape=self.__w2.shape,
                                                                next_grad=grad_output_conv2_as)





                grad_output_conv1_maxpooling_as = self.__gradient_conv_x_as_sample(weight=self.__w2,
                                                                                   sample_shape=self.__feature_conv1.shape,
                                                                                   next_grad=grad_output_conv2_as)
                # gradient矩阵变大/对到每个sample
                grad_output_conv1_relu_as = self.__gradient_maxpooling(mask = self.__maxpooling1_mask,
                                                                       next_grad=grad_output_conv1_maxpooling_as)

                grad_output_conv1_as = self.__gradient_relu(o=self.__feature_conv1_nopool,
                                                            next_grad=grad_output_conv1_relu_as)


                grad_weight_conv1 = self.__gradient_conv_weight(inputs=batch_X,
                                                                weight_shape=self.__w1.shape,
                                                                next_grad=grad_output_conv1_as)




                self.__dense_w -= 0.001 * grad_weight_fc
                self.__w3 -= 0.001 * grad_weight_conv3
                self.__w2 -= 0.001 * grad_weight_conv2
                self.__w1 -= 0.001 * grad_weight_conv1


                #self.__back_propagation()
                # print(f"=============batch {n}===============")
                # print("gradient of fc_weight")
                # print(grad_weight_fc)
                # print()
                # print("gradient of output_conv3")
                # print(reshape_grad_output_conv3)
                # print()
                # print("gradient of weight_conv3")
                # print(grad_weight_conv3)
                # print()
                # print("gradient of output_conv2")
                # print(grad_output_conv2)
                # print()
                # print('weight of conv2')
                # print(self.__w2)
                # print()
                # print('feature of conv1 output')
                # print(self.__feature_conv1)
                # print()
                # print("gradient of weight_conv2")
                # print(grad_weight_conv2)
                # print()
                # print("gradient of weight_conv1")
                # print(grad_weight_conv1)
                # print()
                # print("gradient of weight_conv1")
                # print(grad_output_conv1)

    def infer(self):
        batch_num = self._X // self._batch_size
        for n in range(batch_num):
            start = n * self._batch_size
            end = (n + 1) * self._batch_size
            batch_X = self._X[start:end]
            pre = self.__forward_pass(batch_X)


    def test(self):
        pass

    def load_weight(self):
        pass


test_img = './test_image.npy'
test_label = './test_label.npy'
train_img = './train_image.npy'
train_label = './train_label.npy'

X = np.load(train_img)
Y = np.load(train_label)
model = My_mnist(X=X,Y=Y)
model.train()
# model.infer()

