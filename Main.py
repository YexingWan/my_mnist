import numpy as np
import scipy.signal as signal
from skimage.measure import block_reduce
from statistics import stdev


class My_mnist():

    def __init__(self,X,Y,test_X,test_Y,batch_size=512,lr=0.0001,C = 0.0005,momentum = 0.9):
        # X.shape: (num,channel,h,w)
        idx = np.random.permutation(X.shape[0])
        self._X = X[idx]
        #Y is label from 0 to number of classes
        self._Y = Y[idx]

        self._test_X = test_X
        self._test_Y = test_Y

        self._C = C
        self._momentum = momentum
        self._lr = lr
        self._batch_size = batch_size




        self._loss = []

        # weight_shape:(output_channel,input_channel,h,w)
        self.__w1 = np.random.normal(0.01,np.sqrt(2. / 784),(32,1,3,3))
        self.__feature_conv1_nopool = None
        self.__maxpooling1_mask = None
        self.__feature_conv1 = None

        # self.__w2 = np.random.normal(0.001,np.sqrt(2/1152),(16,16,3,3))
        # self.__feature_conv2_nopool = None
        # self.__maxpooling2_mask = None
        # self.__feature_conv2 = None

        # self.__w3 = np.random.normal(0.001,np.sqrt(2/392),(8,16,3,3))
        # self.__feature_conv3 = None

        #self.__dense_w = np.random.normal(0.001,np.sqrt(2/10),(16*14*14,10))

        #self.__dense_w1 = np.random.normal(0.1, 1, (16*14*14,256))
        #self.__dense_w2 = np.random.normal(0.1, 1, (256, 10))
        self.__dense_w1 =np.random.normal(0.01,np.sqrt(2. / 6272),(32*14*14,256))
        self.__dense_w2 =np.random.normal(0.01,np.sqrt(2. / 256),(256, 10))


        self.__weight_list = []
        self.__weight_list.append(self.__dense_w2)
        self.__weight_list.append(self.__dense_w1)
        self.__weight_list.append(self.__w1)


        self.__gradient_list = [0] * len(self.__weight_list)
        self.__accumulated_momemtum = [0] * len(self.__weight_list)


        # self.__accumulated_momemtum = [0]*len(self.__weight_list)
        #
        # self.__feature1 = None
        # self.__feature2 = None




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
        # predict_Y is one-hot encoded, origin batch Y is label from 0 to predict_Y.shape[1] (number of classes)
        assert(batch_pred.shape[0]==batch_Y.shape[0])
        m = batch_pred.shape[0]
        log_p = np.log2(np.clip(batch_pred, 1e-14, 1))
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
        p = predict.copy()
        p[range(m), target] -= 1
        return  p/m


    def __gradient_fc_w(self,next_grad,feature):
          #p = self.__feature_conv3.shape[1]*self.__feature_conv3.shape[2]*self.__feature_conv3.shape[3]
          #features = self.__feature_conv3.reshape(-1, p)
          return feature.T.dot(next_grad)


    def __gradient_fc_x(self, weight,next_gard):
        return next_gard.dot(weight.T)


    # 返回输入ReLU前的参数对loss的梯度
    def __gradient_relu(self,o,next_grad):
        grad = np.array(o > 0, dtype=int) * next_grad
        return grad

    def __gradient_maxpooling(self,mask,next_grad):
        g  = next_grad.repeat(2,2).repeat(2,3)
        return g*mask



    '''
    计算卷积kernel对loss的梯度
    输入参数:
    inputs: 卷积层的输入features (number,channel,h,w)
    next_gradient: 之后要乘起来的梯度,要sample-wise（batch）
    weight_shape: (output_channel,input_channel,h,w)
    '''
    def __gradient_conv_weight(self,inputs,weight_shape,next_grad):

        grad = np.empty(weight_shape)
        if weight_shape[2] != 1:
            inputs = np.pad(inputs, ((0, 0),(0, 0), (1,1), (1, 1)), 'constant')
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

        grad = np.zeros(shape=sample_shape)
        for c in range(sample_shape[1]):
            roted_kernel = np.rot90(weight[:,c],2,axes=(1,2))[np.newaxis,:,:,:]
            grad[:, c] = self.__conv(next_grad, roted_kernel).squeeze(axis=1)

        return grad

    def __forward_pass(self, batch):

        # 第一层卷积
        self.__feature_conv1_nopool = self.__ReLU(self.__conv(batch, self.__w1))
        self.__feature_conv1, self.__maxpooling1_mask = self.__maxpooling(self.__feature_conv1_nopool)

        # 第二层卷积
        # self.__feature_conv2_nopool = self.__ReLU(self.__conv(self.__feature_conv1, self.__w2))
        # self.__feature_conv2, self.__maxpooling2_mask = self.__maxpooling(self.__feature_conv2_nopool)

        # 第三层卷积
        #self.__feature_conv3 = self.__ReLU(self.__conv(self.__feature_conv2, self.__w3))
        #self.__feature_conv3 = self.__ReLU(self.__conv(self.__feature_conv1, self.__w3))

        reshaped_feature_conv_1 = np.reshape(self.__feature_conv1,(self._batch_size,-1))
        # 全连接

        # self.feature_dense = self.__fc(reshaped_feature_conv3, self.__dense_w)
        # predict = self.__softmax(self.feature_dense)


        self.__feature_dense_1 = self.__ReLU(self.__fc(reshaped_feature_conv_1, self.__dense_w1))
        self.__feature_dense_2 = self.__ReLU(self.__fc(self.__feature_dense_1, self.__dense_w2))
        predict = self.__softmax(self.__feature_dense_2)


        return predict



    def __test(self, test_X, test_Y):
        batch_num = len(test_Y) // self._batch_size
        loss = 0
        succ = 0
        for n in range(batch_num):
            start = n * self._batch_size
            end = (n + 1) * self._batch_size
            # batch_X = np.reshape(self._X[start:end], (-1,self._X[start:end].shape[2] * self._X[start:end].shape[3]))
            batch_X = test_X[start:end]
            batch_Y = test_Y[start:end]
            pre = self.__forward_pass(batch_X)
            loss += self.__loss(pre, batch_Y)
            succ+=np.sum(np.equal(np.argmax(pre, axis=1), batch_Y))
        loss = loss/batch_num
        self._loss.append(loss)
        print(f"accuracy:{succ/len(test_Y)}; test loss:{loss}")



    def train(self,epoch = 10):
        batch_num = self._X.shape[0] // self._batch_size
        for e in range(epoch):
            print(f'running epoch {e}...')
            for n in range(batch_num):
                print(f'batch{n}')
                start = n * self._batch_size
                end = (n + 1) * self._batch_size
                #batch_X = np.reshape(self._X[start:end], (-1,self._X[start:end].shape[2] * self._X[start:end].shape[3]))
                batch_X = self._X[start:end]
                batch_Y = self._Y[start:end]
                predict_Y = self.__forward_pass(batch_X)

                # if n%5==0:
                #     self.__test(self._test_X,self._test_Y)

                gradient_output_dense_2_relu_as = self.__gradient_logit(batch_Y,predict_Y) #(256,10)

                gradient_output_dense_2_as = self.__gradient_relu(o=self.__feature_dense_2,next_grad=gradient_output_dense_2_relu_as)

                gradient_dense_w2 = self.__gradient_fc_w(next_grad=gradient_output_dense_2_as,feature=self.__feature_dense_1)

                self.__gradient_list[0] = (gradient_dense_w2)

                gradient_output_dense_1_relu_as = self.__gradient_fc_x(next_gard=gradient_output_dense_2_as,weight=self.__dense_w2)

                gradient_output_dense_1_as = self.__gradient_relu(o=self.__feature_dense_1, next_grad=gradient_output_dense_1_relu_as)

                gradient_dense_w1 = self.__gradient_fc_w(next_grad=gradient_output_dense_1_as,feature = self.__feature_conv1.reshape(self._batch_size,-1))

                self.__gradient_list[1] = (gradient_dense_w1)


                grad_output_conv1_maxpooling_as = self.__gradient_fc_x(next_gard=gradient_output_dense_1_as,weight=self.__dense_w1)

                grad_output_conv1_maxpooling_as = grad_output_conv1_maxpooling_as.reshape((self._batch_size,
                                                                               self.__feature_conv1.shape[1],
                                                                               self.__feature_conv1.shape[2],
                                                                               self.__feature_conv1.shape[3]))

                grad_output_conv1_relu_as = self.__gradient_maxpooling(mask=self.__maxpooling1_mask,next_grad=grad_output_conv1_maxpooling_as)


                grad_output_conv1_as =  self.__gradient_relu(o = self.__feature_conv1_nopool,
                                                          next_grad=grad_output_conv1_relu_as)

                gradient_w1 = self.__gradient_conv_weight(inputs=batch_X,
                                                                weight_shape=self.__w1.shape,
                                                                next_grad = grad_output_conv1_as)

                self.__gradient_list[2] = gradient_w1

                self.__back_prop()
            self.__test(self._test_X, self._test_Y)

                # grad_output_conv2_maxpooling_as = self.__gradient_conv_x_as_sample(weight=self.__w3,
                #                                                                    sample_shape=self.__feature_conv2.shape,
                #                                                                    next_grad=grad_output_conv3_as)
                # # gradient矩阵变大/对到每个sample
                # grad_output_conv2_relu_as = self.__gradient_maxpooling(mask=self.__maxpooling2_mask,next_grad=grad_output_conv2_maxpooling_as)
                #
                #
                #
                # grad_output_conv2_as= self.__gradient_relu(o = self.__feature_conv2_nopool,next_grad=grad_output_conv2_relu_as)
                #
                # gradient_w2 = self.__gradient_conv_weight(inputs=self.__feature_conv1,
                #                                                 weight_shape=self.__w2.shape,
                #                                                 next_grad=grad_output_conv2_as)


                # grad_output_conv1_maxpooling_as = self.__gradient_conv_x_as_sample(weight=self.__w3,
                #                                                                    sample_shape=self.__feature_conv1.shape,
                #                                                                    next_grad=grad_output_conv3_as)
                # # gradient矩阵变大/对到每个sample
                # grad_output_conv1_relu_as = self.__gradient_maxpooling(mask = self.__maxpooling1_mask,
                #                                                        next_grad=grad_output_conv1_maxpooling_as)
                #
                # grad_output_conv1_as = self.__gradient_relu(o=self.__feature_conv1_nopool,
                #                                             next_grad=grad_output_conv1_relu_as)
                #
                # gradient_w1 = self.__gradient_conv_weight(inputs=batch_X,
                #                                                 weight_shape=self.__w1.shape,
                #                                                 next_grad=grad_output_conv1_as)
                #




                # print('gradient_dense')
                # print(gradient_dense)
                # print()
                # print('gradient_w3')
                # print(gradient_w3)
                # print()
                # print('gradient_w2')
                # print(gradient_w3)
                # print()
                # print('gradient_w1')
                # print(gradient_w3)
                # print()
                # print('learning rate')
                # print(self._lr)

                # self.__dense_w -= 0.0001 * self._lr * gradient_w_dense
                # self.__w3 -= 0.0001 * gradient_w3
                # self.__w2 -= 0.00001 * gradient_w2

                # self.__w1 -= self._lr * gradient_w1
                # self.__dense_w1 -= self._lr * gradient_dense_w1
                # self.__dense_w2 -= self._lr * gradient_dense_w2




    def infer(self):
        batch_num = self._X // self._batch_size
        for n in range(batch_num):
            start = n * self._batch_size
            end = (n + 1) * self._batch_size
            batch_X = self._X[start:end]
            pre = self.__forward_pass(batch_X)



    def __back_prop(self):
        for idx in range(len(self.__weight_list)):
            self.__accumulated_momemtum[idx] = (self._momentum * self.__accumulated_momemtum[idx]) - (self._C * self._lr * self.__weight_list[idx]) - (self._lr * self.__gradient_list[idx])
            self.__weight_list[idx] += self.__accumulated_momemtum[idx]
            #self.__weight_list[idx] -= (self._lr * self.__weight_list[idx]+ self._C * self._lr * self.__weight_list[idx])



    def load_weight(self):
        pass


    def save_weight(self):
        pass


test_img = './test_image.npy'
test_label = './test_label.npy'
train_img = './train_image.npy'
train_label = './train_label.npy'

X = np.load(train_img)
Y = np.load(train_label)
test_X = np.load(test_img)
test_Y = np.load(test_label)
model = My_mnist(X=X,Y=Y,test_X = test_X, test_Y = test_Y)
model.train()
# model.infer()

