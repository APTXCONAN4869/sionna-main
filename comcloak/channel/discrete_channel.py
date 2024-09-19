import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from comcloak.channel.utils import expand_to_rank
import numpy as np

# import tensorflow as tf


from comcloak.constants import GLOBAL_SEED_NUMBER 

class CustomOperations:
    
    class CustomXOR(Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            if a.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                z = (a + b) % 2
            else:
                z = torch.abs(a - b)
            return z

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output
    # STEBinarizer（Straight-Through Estimator Binarizer）是一种在神经网络中用于处理二值化操作的技术。
    # STE代表Straight-Through Estimator，它是一种用于在反向传播中处理不可微操作的技术。
    class STEBinarizer(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            z = torch.where(x < 0.5, torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device))
            return z

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    
    class SampleErrors(torch.nn.Module):
        def __init__(self, eps=1e-10, temperature=1.0):
            super().__init__()
            self._eps = eps
            self._temperature = temperature

        def forward(self, pb, shape):
            torch.manual_seed(GLOBAL_SEED_NUMBER)
            # torch.cuda.manual_seed(GLOBAL_SEED_NUMBER)
            u1 = torch.rand(shape, dtype=torch.float32)
            u2 = torch.rand(shape, dtype=torch.float32)

            # tf.random.set_seed(GLOBAL_SEED_NUMBER)
            # u1_tf=tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
            # u2_tf=tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
            # u1_np=u1_tf.numpy()
            # u2_np=u2_tf.numpy()
            # u1=torch.from_numpy(u1_np)
            # u2=torch.from_numpy(u2_np)

            u = torch.stack((u1, u2), dim=-1)

            # 采样Gumbel分布
            q = -torch.log(-torch.log(u + self._eps) + self._eps)
            p = torch.stack((pb, 1 - pb), dim=-1).unsqueeze(1).expand(shape[0], shape[1], 2)
            a = (torch.log(p + self._eps) + q) / self._temperature

            # 应用softmax
            e_cat = F.softmax(a, dim=-1)

            # 通过直通估计器对最终值进行二值化
            return CustomOperations.STEBinarizer.apply(e_cat[..., 0])

class BinaryMemorylessChannel(nn.Module):
    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100., dtype=torch.float32, **kwargs):
        super(BinaryMemorylessChannel,self).__init__(**kwargs)

        assert isinstance(return_llrs, bool), "return_llrs must be bool."
        self._return_llrs = return_llrs

        assert isinstance(bipolar_input, bool), "bipolar_input must be bool."
        self._bipolar_input = bipolar_input

        assert llr_max >= 0., "llr_max must be a positive scalar value."
        self.llr_max = llr_max
        self.dtype = dtype

        if self._return_llrs:
            assert dtype in (torch.float16, torch.float32, torch.float64), \
                "LLR outputs require non-integer dtypes."
        else:
            if self._bipolar_input:
                assert dtype in (torch.float16, torch.float32, torch.float64,
                                 torch.int8, torch.int16, torch.int32, torch.int64), \
                    "Only signed dtypes are supported for bipolar inputs."
            else:
                assert dtype in (torch.float16, torch.float32, torch.float64,
                                 torch.uint8, torch.int16, torch.int32, torch.int64,
                                 torch.int8, torch.int16, torch.int32, torch.int64), \
                    "Only real-valued dtypes are supported."

        self.check_input = True  # check input for consistency (i.e., binary)

        self._eps = 1e-9  # small additional term for numerical stability
        self.temperature = torch.tensor(0.1, dtype=torch.float32)  # for Gumble-softmax

    @property
    def llr_max(self):
        """Maximum value used for LLR calculations."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value):
        """Maximum value used for LLR calculations."""
        assert value >= 0, 'llr_max cannot be negative.'
        self._llr_max = value

    @property
    def temperature(self):
        """Temperature for Gumble-softmax trick."""
        return self._temperature.item()

    @temperature.setter
    def temperature(self, value):
        """Temperature for Gumble-softmax trick."""
        assert value >= 0, 'temperature cannot be negative.'
        self._temperature = torch.tensor(value, dtype=torch.float32)
    #########################
    # Utility methods
    #########################

    def _check_inputs(self, x):
        """Check input x for consistency, i.e., verify
        that all values are binary of bipolar values."""
        x = x.float()
        if self.check_input:
            if self._bipolar_input:
                assert torch.all(torch.logical_or(x == -1, x == 1)), "Input must be bipolar {-1, 1}."
            else:
                assert torch.all(torch.logical_or(x == 0, x == 1)), "Input must be binary {0, 1}."
            # input datatype consistency should be only evaluated once
            self.check_input = False

    # 使用方法
    @staticmethod
    def custom_xor(a, b):
        return CustomOperations.CustomXOR.apply(a, b)       

    @staticmethod
    def _ste_binarizer(x):
        """Straight through binarizer to quantize bits to int values."""
        return CustomOperations.STEBinarizer.apply(x)

    def _sample_errors(self, pb, shape):
        """Samples binary error vector with given error probability e.
        This function is based on the Gumble-softmax "trick" to keep the
        sampling differentiable."""
        torch.manual_seed(2023)
        # torch.cuda.manual_seed(2023)
        u1 = torch.rand(shape)
        u2 = torch.rand(shape)
        
        # tf.random.set_seed(GLOBAL_SEED_NUMBER)
        # u1_tf=tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        # u2_tf=tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        # u1_np=u1_tf.numpy()
        # u2_np=u2_tf.numpy()
        # u1=torch.from_numpy(u1_np)
        # u2=torch.from_numpy(u2_np)

        u = torch.stack((u1, u2), dim=-1)

        # sample Gumble distribution
        q = -torch.log(-torch.log(u + self._eps) + self._eps)
        p = torch.stack((pb, 1 - pb), dim=-1)
        p = p.unsqueeze(0).expand(q.shape)
        a = (torch.log(p + self._eps) + q) / self.temperature

        # apply softmax
        e_cat = F.softmax(a, dim=-1)

        # binarize final values via straight-through estimator
        return self._ste_binarizer(e_cat[..., 0])  # only take the first class
    
    #########################
    # Keras layer functions
    #########################

    # 这段代码定义了一个 build 方法了，用于验证输入的形状是否正确
    # 它主要检查第二个输入（错误概率 pb）的形状，确保其最后一维的长度为 2

    def build(self, input_shapes):
        """Verify correct input shapes"""

        pb_shapes = input_shapes[1]
        # allow tuple of scalars as alternative input
        if isinstance(pb_shapes, (tuple, list)):
            if not len(pb_shapes) == 2:
                raise ValueError("Last dim of pb must be of length 2.")
        else:
            if len(pb_shapes) > 0:
                if not pb_shapes[-1] == 2:
                    raise ValueError("Last dim of pb must be of length 2.")
            else:
                raise ValueError("Last dim of pb must be of length 2.")
            
    def forward(self, inputs):
        """Apply discrete binary memoryless channel to inputs."""

        x, pb = inputs

        # allow pb to be a tuple of two scalars
        if isinstance(pb, (tuple, list)):
            pb0 = pb[0]
            pb1 = pb[1]
        else:
            pb0 = pb[...,0]
            pb1 = pb[...,1]
        
        # 假设pb0和pb1是PyTorch张量
        pb0 = pb0.float()  # 确保pb0是浮点数
        pb1 = pb1.float()  # 确保pb1是浮点数
        pb0 = torch.clamp(pb0, 0., 1.)  # 将pb0的值限制在0和1之间
        pb1 = torch.clamp(pb1, 0., 1.)  # 将pb1的值限制在0和1之间

        # check x for consistency (binary, bipolar)
        self._check_inputs(x)

        e0 = self._sample_errors(pb0,x.shape)
        e1 = self._sample_errors(pb1, x.shape)

        if self._bipolar_input:
            neutral_element = torch.tensor(-1, dtype=x.dtype)
        else:
            neutral_element = torch.tensor(0, dtype=x.dtype)    

        # mask e0 and e1 with input such that e0 only applies where x==0    
        e = torch.where(x == neutral_element, e0, e1)
        e = e.to(dtype=x.dtype)

        if self._bipolar_input:
            # flip signs for bipolar case
            y = x * (-2*e + 1)
        else:
            # XOR for binary case
            y = self.custom_xor(x, e)

        # if LLRs should be returned
        if self._return_llrs:
            if not self._bipolar_input:
                y = 2 * y - 1  # transform to bipolar
            # Remark: Sionna uses the logit definition log[p(x=1)/p(x=0)]
            # 计算LLRs的组成部分
            y0 = -(torch.log(pb1 + self._eps) - torch.log(1 - pb0 - self._eps))
            y1 = (torch.log(1 - pb1 - self._eps) - torch.log(pb0 + self._eps))

            # multiply by y to keep gradient
            # 使用torch.where实现条件选择
            y = torch.where(y == 1, y1, y0).to(dtype=y.dtype) * y

            # and clip output llrs
            # 将LLR的值限制在范围内
            y = torch.clamp(y, min=-self._llr_max, max=self._llr_max)        

        return y

class BinarySymmetricChannel(BinaryMemorylessChannel):
    def __init__(self,return_llrs=False,bipolar_input=False,llr_max=100.,dtype=torch.float32,**kwargs):
        #继承父类的__init__()
        super().__init__(return_llrs=return_llrs,
                         bipolar_input=bipolar_input,
                         llr_max=llr_max,
                         dtype=dtype,
                         **kwargs)
    #########################
    # Keras layer functions
    #########################

    def build(self,input_shapes):
        """"Verify correct input shapes"""
        pass # nothing to verify here
    def forward(self,inputs):
        """Apply discrete binary symmetric channel, i.e., randomly flip
        bits with probability pb."""
        """"应用离散二进制对称信道,即以pb概率随机翻转位"""

        x,pb = inputs

        # the BSC is implemented by calling the DMC with symmetric pb
        # BSC（二元对称信道）是通过使用对称的pb（比特翻转概率）调用DMC（离散记忆少信道）来实现的
        # 这里的“对称的pb”意味着信道的翻转概率p对于输入0和1是相同的，即信道以相同的概率将输入0翻转为1，或将输入1翻转为0。

        """"在二元对称信道(BSC)中,通常有两种状态:输入0被保持为0,或被翻转为1;输入1被保持为1,或被翻转为0。
        如果翻转概率p对于两种输入都是相同的,那么信道就是对称的。

        在实现上,可以构建一个离散记忆少信道(DMC),并设置其状态转移概率矩阵为对称的,以此来模拟BSC的行为。

        在数学上,如果用p表示翻转概率,那么BSC的状态转移概率可以表示为:
        从状态0(输入0)翻转到状态1的概率是p   从状态1(输入1)翻转到状态0的概率也是p  保持原始状态的概率是1-p"""
        pb = pb.to(x.dtype)
        pb = torch.stack((pb,pb), dim=-1)
        y = super().forward((x,pb))

        return y

class BinaryZChannel(BinaryMemorylessChannel):
    def __init__(self, return_llrs=False, bipolar_input=False,llr_max=100.,dtype=torch.float32, **kwargs):

        super().__init__(return_llrs=return_llrs,
                         bipolar_input=bipolar_input,
                         llr_max=llr_max,
                         dtype=dtype,
                         **kwargs)
    #########################
    # Keras layer functions
    #########################
    def build(self, input_shapes):
        """Verify correct input shapes"""
        pass # nothing to verify here

    def forward(self, inputs):
        """Apply discrete binary symmetric channel, i.e., randomly flip
        bits with probability pb."""

        x, pb = inputs

        # the Z is implemented by calling the DMC with p(1|0)=0
        pb = pb.to(x.dtype)
        pb = torch.stack((torch.zeros_like(pb), pb), dim=-1)
        y = super().forward((x, pb))

        return y        
  

class BinaryErasureChannel(BinaryMemorylessChannel):
    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100, dtype=torch.float32):
        super().__init__()
        self.return_llrs = return_llrs
        self.bipolar_input = bipolar_input
        self.llr_max = llr_max
        self.dtype = dtype

        assert dtype in (torch.float16, torch.float32, torch.float64,
                         torch.int8, torch.int16, torch.int32, torch.int64), \
               "Unsigned integers are currently not supported."
    #########################
    # Keras layer functions
    #########################

    def forward(self, inputs):

        x,pb = inputs


        # Example validation of input x
        if not self.bipolar_input:
            assert torch.all((x == 0) | (x == 1)), "Input x must be binary (0 or 1)."
        else:
            assert torch.all((x == -1) | (x == 1)), "Input x must be bipolar (-1 or 1)."

        # Example validation of pb
        # clip for numerical stability
        pb = pb.float().clamp(0., 1.)

        # sample erasure pattern
        e = self._sample_errors(pb, x.size())

        # if LLRs should be returned
        # remark: the Sionna logit definition is llr = log[p(x=1)/p(x=0)]
        if self.return_llrs:
            if not self.bipolar_input:
                x = 2 * x - 1
            x = x.to(torch.float32) * self.llr_max  # calculate llrs

            # erase positions by setting llrs to 0
            y = torch.where(e == 1, torch.tensor(0, dtype=torch.float32), x)
        else:
            if self.bipolar_input:
                erased_element = torch.tensor(0, dtype=x.dtype) 
            else:
                erased_element=torch.tensor(-1, dtype=x.dtype)
            y = torch.where(e == 0, x, erased_element)

        return y


