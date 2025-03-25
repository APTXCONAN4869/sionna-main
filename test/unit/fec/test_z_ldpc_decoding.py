#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
print("Current directory:", os.getcwd())
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")
import unittest
import numpy as np
import torch
import copy
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 1  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

import scipy as sp

from comcloak.fec.ldpc.decoding import LDPCBPDecoder, LDPC5GDecoder
from comcloak.fec.ldpc.encoding import LDPC5GEncoder
from comcloak.supplement import RaggedTensor
from comcloak.fec.utils import GaussianPriorSource, load_parity_check_examples
from comcloak.utils import BinarySource

class TestBPDecoding(unittest.TestCase):
    "Testcases for LDPCBPDecoder class."

    # def test_dtypes(self):
    #     """Test against correct dtypes:
    #     - input parameters (must be int etc.)
    #     - parity-check matrix is only allowed to contain binary values
    #     """

    #     # Raise error if PCM contains other elements than 0,1
    #     pcm = np.random.uniform(0, 2, [100, 150]).astype(int)
    #     pcm[10, 20] = 2
    #     with self.assertRaises(AssertionError):
    #         dec = LDPCBPDecoder(pcm)

    #     # raise error if llrs are not tf.float32
    #     batch_size = 100
    #     n = 64
    #     k = 32
    #     pcm = np.random.uniform(0, 2, [n-k, n]).astype(int)
    #     dec = LDPCBPDecoder(pcm)

    #     llr = torch.randint(0, 100, size=(batch_size, n)).type(torch.int32)
    #     with self.assertRaises(AssertionError):
    #         dec(llr)

    #     # raise error if input shape does not match PCM dim
    #     batch_size = 100
    #     n = 64
    #     k = 32
    #     pcm = np.random.uniform(0, 2, [n-k, n]).astype(int)
    #     dec = LDPCBPDecoder(pcm)

    #     llr = torch.rand(size=(batch_size, (n+1))).type(torch.float32)
    #     with self.assertRaises(AssertionError):
    #         dec(llr)

    # def test_CN(self):
    #     """Test that CN function works correctly (i.e., extrinsic and sign preserving). Must be done for all node types.

    #     Test CN-degree 2 as well for all types. Must be a forwarding node
    #     """
    #     Ntrials = 1  # nb trials
    #     k = 12
    #     n = 24
    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc)

    #     # test cn_update_tanh
    #     for _ in range(Ntrials):
    #         # msg = np.random.normal(size=[10]) #generate random inputs
    #         msg = [-0.84183812, - 1.01420765, 1.94299806, 1.6476993, 0.30150094, 1.32848432,
    #                             - 1.00009697, 1.56274849, - 1.81601112, - 1.54949935]
    #         x = RaggedTensor.from_row_splits(
    #                                 values=torch.tensor(msg, dtype=torch.float32),
    #                                 row_splits=torch.tensor([0, len(msg)]))
            
            
    #         y1 = dec._cn_update_tanh(x)
    #         y2 = dec._cn_update_phi(x)
    #         y3 = dec._cn_update_minsum(x.expand_dims(1))
    #         y3.flat_values = y3.flat_values.squeeze()

    #         # both CN functions should yield same results (minsum does NOT!)
    #         self.assertTrue(np.allclose(y1.flat_values.numpy(), y2.flat_values.numpy(), atol=1e-4))

    #         # check that sign is correct (treat 0 as positive)
    #         s = 2*(np.array(msg) >= 0).astype(int) - 1
    #         s = s*np.prod(s)
    #         y1_s = 2*(y1.flat_values.numpy() >= 0).astype(int) - 1
    #         y2_s = 2*(y2.flat_values.numpy() >= 0).astype(int) - 1
    #         y3_s = 2*(y3.flat_values.numpy() >= 0).astype(int) - 1

    #         # ignore cases where all CN messages are small; otherwise the sign
    #         # becomes random
    #         if np.sum(np.abs(y1.flat_values.numpy())) > 1e-3:
    #             self.assertTrue(np.allclose(s, y1_s)), "sign tanh"
    #             self.assertTrue(np.allclose(s, y2_s)), "sign phi"
    #             self.assertTrue(np.allclose(s, y3_s)), "sign minsum"


    #         # test that exact zero input leads to exact zero output
    #         msg[-1] = 0.
    #         x = RaggedTensor.from_row_splits(values=torch.tensor(msg, dtype=torch.float32),
    #                                         row_splits=torch.tensor([0, len(msg)]))
    #         y1 = dec._cn_update_tanh(x).flat_values.numpy()
    #         y2 = dec._cn_update_phi(x).flat_values.numpy()

    #         # minsum needs batch dim
    #         y3 = dec._cn_update_minsum(x.expand_dims(1)).flat_values.squeeze().numpy()
    #         # y3 = torch.squeeze(y3, dim=2).flat_values.numpy()
    #         # the tanh-implementation is numerically not exact for exact 0
    #         # inputs
    #         self.assertTrue(np.array_equal(y1[:-1], np.zeros_like(y1[:-1])))
    #         self.assertTrue(np.array_equal(y2[:-1], np.zeros_like(y2[:-1])))
    #         self.assertTrue(np.array_equal(y3[:-1], np.zeros_like(y3[:-1])))

    # def test_int_state(self):
    #     """Test internal state functionality of decoder.
    #     This implies that Nx1 iterations yield exact same result as N
    #     iterations."""
    #     batch_size = 1
    #     Niter = 5
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     dec = LDPCBPDecoder(pcm, num_iter=Niter)
    #     dec2 = LDPCBPDecoder(pcm, num_iter=1, stateful=True)

    #     # llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
    #     np.random.seed(1)
    #     llr = np.random.normal(size=[batch_size, n], loc=4.2, scale=1)
    #     llr = torch.tensor(llr)
    #     res1 = dec(llr)

    #     res2, msg_vn = dec2([llr, None]) # iter 0 to init msg_vn

    #     for i in range(Niter-1):  # remaining iterations
    #         res2, _ = dec2([llr, msg_vn])
    #     # results must be the same, otherwise the internal state is not
    #     # correctly recovered
    #     self.assertTrue(np.allclose(res1, res2))

    # def test_phi(self):
    #     """Test that phi is self-inverse."""
    #     x = np.arange(0.01, 16.6, 0.01)
    #     y = LDPCBPDecoder._phi(None, x)
    #     z = LDPCBPDecoder._phi(None, y)
    #     self.assertTrue(np.allclose(x, z))

    # def test_VN(self):
    #     """Test that VN function works correctly (i.e., extrinsic).
    #     """
    #     Ntrials = 1000  # nb trials
    #     k = 12
    #     n = 24
    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc)

    #     # test vn updates
    #     for _ in range(Ntrials):
    #         msg = np.random.normal(size=[10]) #generate random inputs
    #         msg_ch = np.random.normal(size=[1]) #generate random inputs

    #         x = RaggedTensor.from_row_splits(
    #                                     values=torch.tensor(msg, dtype=torch.float32),
    #                                     row_splits=torch.tensor([0, len(msg)]))

    #         y = dec._vn_update(x, msg_ch).flat_values.numpy()

    #         y_ref = np.sum(msg) - msg + msg_ch
    #         self.assertTrue(np.allclose(y_ref, y, atol=1e-5))

    # def test_batch(self):
    #     """Test that batch of codewords yields the same results for each batch
    #     sample."""

    #     batch_size = 100
    #     Niter = 10
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     dec = LDPCBPDecoder(pcm)
    #     # llr = tf.random.normal([1, n], mean=4.2, stddev=1)
    #     llr = torch.normal(mean=4.2, std=1, size=(1, n))
    #     llr = torch.tile(llr, [batch_size, 1])
    #     x = dec(llr).numpy()
    #     for i in range(batch_size):
    #         # if decoder runs on GPU, the reduce_prod/reduce_sum in the GPU
    #         # yields slightly different result (probably due to scheduling).
    #         # This leads to slightly different results within one batch
    #         # which is further amplified with more iterations.
    #         self.assertTrue(np.allclose(x[0, :], x[i, :], atol=1e-4))

    # def test_gradient(self):
    #     """Test that gradient is accessible and not None."""

    #     batch_size = 100
    #     pcm, k, n, _ = load_parity_check_examples(2) 

    #     # Check that trainable parameter works as expected
    #     dec = LDPCBPDecoder(pcm, trainable=True)  # Trainable decoder
    #     self.assertGreater(len(list(dec.parameters())), 0, "Trainable variables should exist")

    #     dec = LDPCBPDecoder(pcm, trainable=False)  # Non-trainable decoder
    #     self.assertEqual(len(list(dec.parameters())), 0, "Trainable variables should not exist")

    #     cns = ['boxplus', 'boxplus-phi', 'minsum']
    #     trainable = [True, False]

    #     for cn in cns:
    #         for t in trainable:
    #             dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=False)

    #             # Randomly generate LLR input
    #             llr = torch.normal(mean=4.2, std=1.0, size=(batch_size, n))
    #             llr = llr.requires_grad_()

    #             # Forward pass
    #             x = dec(llr)
    #             grads = torch.autograd.grad(outputs=x.sum().requires_grad_(), inputs=dec.parameters(), create_graph=True, allow_unused=True)
    #             # Check that gradients exist
    #             self.assertIsNotNone(grads, "Gradients should not be None")
    #             # Compute gradients with respect to trainable parameters
    #             if t:  # Trainable case
                    
    #                 self.assertGreater(len(grads), 0, "No gradients found for trainable variables")

    #                 # Check each gradient is not None
    #                 for g in grads:
    #                     self.assertIsNotNone(g, "Gradient is None for a parameter")
    #             else:  # Non-trainable case
    #                 # Check that no gradients exist
    #                 self.assertTrue(all(g is None for g in grads), "Gradients should not exist")

    def test_gradient(self):
        """Test that gradient is accessible and not None."""

        # 设置测试参数
        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2) 

        # 检查可训练参数是否正常工作
        dec = LDPCBPDecoder(pcm, trainable=True)  # 可训练解码器
        self.assertGreater(len(list(dec.parameters())), 0, "可训练变量应该存在")

        dec = LDPCBPDecoder(pcm, trainable=False)  # 非可训练解码器
        self.assertEqual(len(list(dec.parameters())), 0, "可训练变量不应该存在")

        # 定义要测试的校验节点类型和是否可训练
        cns = ['boxplus', 'boxplus-phi', 'minsum']
        trainable = [True, False]

        # 遍历所有可能的校验节点类型和可训练状态
        for cn in cns:
            for t in trainable:
                # 创建LDPCBPDecoder对象
                dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=False)

                # 生成随机的LLR
                llr = torch.normal(mean=4.2, std=1.0, size=(batch_size, n))

                # 启用梯度跟踪
                llr = llr.requires_grad_()

                # 前向传播
                x = dec(llr)

                # 计算可训练参数的梯度
                if t:  # 可训练情况
                    grads = torch.autograd.grad(outputs=x.sum(), inputs=dec.parameters(), create_graph=True, allow_unused=True)
                    
                    # 检查梯度是否存在
                    self.assertIsNotNone(grads, "梯度不应该为None")
                    self.assertGreater(len(grads), 0, "没有找到可训练变量的梯度")

                    # 检查每个梯度是否为None
                    print(param.requires_grad for param in dec.parameters())
                    grads = [g for g in grads if g is not None]
                    for g in grads:
                        self.assertIsNotNone(g, "参数的梯度为None")
                else:  # 非可训练情况
                    grads = torch.autograd.grad(outputs=x.sum(), inputs=dec.parameters(), create_graph=True, allow_unused=True)
                    # 检查没有梯度存在
                    self.assertTrue(all(g is None for g in grads), "应该不存在梯度")


    # def test_all_erasure(self):
    #     """Test that all-erasure (llr=0) cw yields constant all-zero output."""

    #     batch_size = 100
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     cns = ['boxplus', 'boxplus-phi', 'minsum']
    #     trainable = [True, False]
    #     for cn in cns:
    #         for t in trainable:
    #             dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn)
    #             llr = torch.zeros([batch_size, n])
    #             x = dec(llr)
    #             self.assertTrue(np.array_equal(x.numpy(), llr.numpy()))

    # def test_hard_out(self):
    #     """Test hard-out flag yields hard-decided output."""

    #     batch_size = 100
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     cns = ['boxplus', 'boxplus-phi','minsum']
    #     trainable = [True, False]
    #     for cn in cns:
    #         for t in trainable:
    #             dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=True)

    #             # test that all zero CW yields hard-decided all-zero cw
    #             llr = -10.*torch.ones([batch_size, n])  # all-zero input
    #             x = dec(llr).numpy()
    #             self.assertTrue(np.array_equal(x, np.zeros_like(x)))

    #             # test that for arbitrary input only 0,1 values are returned
    #             # llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
    #             llr = torch.normal(mean=4.2, std=1, size=(batch_size, n))
    #             x = dec(llr).numpy()
    #             #x contains only {0,1}
    #             self.assertTrue(np.array_equal(x, x.astype(bool)))

    # def test_output_dim(self):
    #     """Test that output dim is n."""
    #     batch_size = 100
    #     Niter = 10
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     dec = LDPCBPDecoder(pcm)
    #     # llr = tf.random.normal([batch_size, n], mean=1., stddev=1)
    #     llr = torch.normal(mean=1, std=1, size=(batch_size, n))
    #     dec = LDPCBPDecoder(pcm, track_exit=False)
    #     x = dec(llr)
    #     self.assertTrue(np.shape(x)[1] == n)

    # def test_multi_dim(self):
    #     """Test that 2+D Tensors are correctly handled."""

    #     pcm, k, n, _ = load_parity_check_examples(2)
    #     dec = LDPCBPDecoder(pcm)
    #     shapes = [[10, 2, 3, n], [1, 4, n], [10, 2, 3, 3, n]]

    #     for s in shapes:
    #         # llr = tf.random.normal(s, mean=0, stddev=1)
    #         llr = torch.normal(mean=0, std=1, size=s)
    #         llr_ref = torch.reshape(llr, [-1, n])

    #         c = dec(llr)
    #         c_ref = dec(llr_ref)
    #         s[-1] = n
    #         c_ref = torch.reshape(c_ref, s)
    #         self.assertTrue(np.allclose(c.numpy(), c_ref.numpy(), atol=0.001))

    #     # and verify that wrong last dimension raises an error
    #     with self.assertRaises(AssertionError):
    #         s = [10, 2, n-1]
    #         # llr = tf.random.normal(s, mean=0, stddev=1)
    #         llr = torch.normal(mean=0, std=1, size=s)
    #         c = dec(llr)

    # def test_all_zero(self):
    #     """Test all-zero cw without noise yields all-zero info bits."""
    #     batch_size = 100
    #     pcm, k, n, _ = load_parity_check_examples(2)

    #     cns = ['boxplus', 'boxplus-phi','minsum']
    #     trainable = [True, False]
    #     for cn in cns:
    #         for t in trainable:
    #             dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=True)
    #             # init with all-zero and large LLRs/logits (=high SNR)
    #             llr = -10. * torch.ones([batch_size, n])
    #             x = np.zeros_like(llr)
    #             x_hat = dec(llr)
    #             self.assertTrue(np.array_equal(x, x_hat.numpy()))

    # def test_dtype2(self):
    #     """Test that output dtype can be flexible"""
    #     batch_size = 100
    #     pcm, k, n, _ = load_parity_check_examples(2)
    #     dec_32 = LDPCBPDecoder(pcm, output_dtype=torch.float32)
    #     dec_64 = LDPCBPDecoder(pcm, output_dtype=torch.float64)
    #     # llr_32 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
    #     #                             tf.cast(n, dtype=tf.int32)],
    #     #                             dtype=tf.float32)
    #     llr_32 = torch.tensor(np.random.uniform(size=[batch_size,n]),dtype=torch.float32)
    #     # llr_64 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
    #     #                             tf.cast(n, dtype=tf.int32)],
    #     #                             dtype=tf.float64)
    #     llr_64 = torch.tensor(np.random.uniform(size=[batch_size,n]),dtype=torch.float64)

    #     # output for both inputs is tf.float32
    #     u_32 = dec_32(llr_32)
    #     u_64 = dec_32(llr_64)
    #     self.assertTrue(u_32.dtype is torch.float32)
    #     self.assertTrue(u_64.dtype is torch.float32)

    #     # output for both inputs is tf.float64
    #     u_32 = dec_64(llr_32)
    #     u_64 = dec_64(llr_64)
    #     self.assertTrue(u_32.dtype is torch.float64)
    #     self.assertTrue(u_64.dtype is torch.float64)

    # def test_sparse(self):
    #     """Test that parity-check matrix can be also scipy.sparse mat."""
    #     batch_size = 10
    #     Niter = 10
    #     pcm, k, n, _ = load_parity_check_examples(3)
    #     source = GaussianPriorSource()

    #     # generate sparse parity-check matrices
    #     pcm_csc = sp.sparse.csc_matrix(pcm)
    #     pcm_csr = sp.sparse.csr_matrix(pcm)

    #     # instantiate decoders with different pcm datatypes
    #     dec = LDPCBPDecoder(pcm, num_iter=Niter)
    #     dec_csc = LDPCBPDecoder(pcm_csc, num_iter=Niter)
    #     dec_csr = LDPCBPDecoder(pcm_csr, num_iter=Niter)

    #     llr = source([[batch_size, n], 0.9])

    #     # and decode the same llrs with each decoder
    #     res = dec(llr)
    #     res_csc = dec_csc(llr)
    #     res_csr = dec_csr(llr)

    #     # results must be the same
    #     self.assertTrue(np.allclose(res, res_csc))
    #     self.assertTrue(np.allclose(res, res_csr))

    # def test_llrmax(self):
    #     """Test that llr_max can be set."""
    #     pcm, _, n, _ = load_parity_check_examples(0)
    #     # no iteration: decoder returns clipped llrs
    #     dec = LDPCBPDecoder(pcm, num_iter=0, hard_out=False)

    #     # test default value
    #     llr_max_def = dec.llr_max.numpy() # get default value
    #     x = torch.ones((1, n))*100
    #     y = dec(x).numpy()  # run 0 iterations
    #     np.max(y) == llr_max_def

    #     # set new llr_max
    #     llr_maxs = [17., 45.3, 78]
    #     for l in llr_maxs:
    #         dec.llr_max = l
    #         y = dec(x).numpy()  # run 0 iterations
    #         print(np.abs(np.max(y)-l) < 1e-6)

    # def test_cn_minsum(self):
    #     """Test min_sim implementation of CN update.
    #     Test that double min is correctly processed, zeros are detected and
    #     that signs are also correctly handled."""

    #     # init dummy decoder
    #     pcm, _, _, _ = load_parity_check_examples(0)
    #     dec = LDPCBPDecoder(pcm, cn_type="minsum")

    #     # test messages for CN function
    #     m_in = RaggedTensor.from_row_splits(values=torch.tensor([1, -2, 3, -1, 2, .3, 2, 0, 1, -1, 2.3],dtype=torch.float32), 
    #                                         row_splits=torch.tensor([0, 4, 7, 11]))
    #     # m_in = torch.tensor([[1, -2, 3, -1], [2, .3, 2], [0, 1, -1, 2.3]],
    #     #                     torch.float32)

    #     # apply decoder
    #     m_out = dec._cn_update_minsum(m_in.expand_dims(dim=-1)).to_list()
    #     m_out = np.array([t.numpy() for t in m_out], dtype=object)
    #     # reference decoder
    #     m_in = np.array([t.numpy() for t in m_in.to_list()], dtype=object)
    #     m_ref = copy.deepcopy(m_in)
    #     for i, m in enumerate(m_in):
    #         for j in range(len(m)):
    #             x = np.abs(np.copy(m))
    #             x[j] = 1000  # large value
    #             s = np.sign(np.copy(m))
    #             s[j] = 1
    #             s = np.prod(s)
    #             m_ref[i][j] = np.min(x) * s

    #     # and compare both results
    #     for i, (a, b) in enumerate(zip(m_ref, m_out)):
    #         self.assertTrue(np.allclose(a, b[:, 0]))


# class TestBPDecoding5G(unittest.TestCase):
#     """Checks LDPC5GDecoding layer.
#     Remark: As this layer inherits from BPDecoding many cases are covered by
#     previous tests."""

    # def test_encoding(self):
    #     """Test that encoded info bits can be reconstructed after decoding
    #     (assuming no/little noise)."""

    #     batch_size = 100

    #     # k, n
    #     params = [[64, 128], [64, 180], [167, 201], [439, 800], [948, 1024],
    #               [3893, 7940], [6530, 10023], [8448, 23000], [955, 1024],
    #               [1900, 2000]]

    #     # generate random bits
    #     for ret_info in [True, False]:
    #         src = BinarySource()
    #         for p in params:
    #             k = p[0]
    #             n = p[1]
    #             enc = LDPC5GEncoder(k, n)
    #             dec = LDPC5GDecoder(enc, hard_out=True, return_infobits=ret_info)
    #             b = src([batch_size, k])
    #             c = enc(b)
    #             x = 2*c -1  # BPSK (neg. sign due to  sionna llr definition)
    #             llr = 5 * x  # scale as we have no noise -> larger LLRs
    #             b_hat = dec(llr)
    #             if ret_info:
    #                 self.assertTrue(np.array_equal(b.numpy(), b_hat.numpy()))
    #             else:
    #                 self.assertTrue(np.array_equal(c.numpy(), b_hat.numpy()))

    # def test_dimensions(self):
    #     """Test for dimension mismatched between input_shape and k, n."""

    #     batch_size = 100
    #     n = 128
    #     k = 64
    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc)

    #     llr = torch.rand([torch.tensor(batch_size, dtype=torch.int32), 
    #                      torch.tensor(n + 1, dtype=torch.int32)]).type(torch.float32)

    #     with self.assertRaises(AssertionError):
    #         dec(llr)

    #     # varying batch-sizes should be supported
    #     llr = torch.rand([torch.tensor(batch_size + 1, dtype=torch.int32), 
    #                      torch.tensor(n, dtype=torch.int32)]).type(torch.float32)
    #     dec(llr)

    # def test_multi_dim(self):
    #     """Test that 2+D Tensors are correctly handled."""

    #     k = 100
    #     n = 200
    #     shapes =[[10, 20, 30, n], [1, 40, n], [10, 2, 3, 4, 3, n]]
    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc, num_iter=10)
    #     source = GaussianPriorSource()

    #     for s in shapes:
    #         llr = source([s, 1])
    #         llr_ref = torch.reshape(llr, [-1, n])

    #         c = dec(llr)
    #         c_ref = dec(llr_ref)
    #         s[-1] = k
    #         c_ref = torch.reshape(c_ref, s)
    #         self.assertTrue(np.allclose(c.numpy(), c_ref.numpy(), atol=0.01))

    #     # and verify that wrong last dimension raises an error
    #     with self.assertRaises(BaseException):
    #         s = [10, 2, k-1]
    #         llr = torch.normal(mean=0, std=1, size=s)
    #         c = dec(llr)

    # def test_gradient(self):
    #     """Test that gradient is accessible and not None."""

    #     batch_size = 100
    #     n = 128
    #     k = 64
    #     enc = LDPC5GEncoder(k, n)

    #     cns = ['boxplus', 'boxplus-phi', 'minsum']
    #     trainable = [True, False]
    #     for cn in cns:
    #         for t in trainable:
    #             dec = LDPC5GDecoder(enc,
    #                                 trainable=t,
    #                                 cn_type=cn,
    #                                 hard_out=False)
    #             llr = torch.normal(mean=4.2, std=1, size=(batch_size, n))

    #             # with tf.GradientTape() as tape:
    #             x = dec(llr)
    #             grads = torch.autograd.grad(x, dec.trainable_variables)

    #             # check that gradients exist
    #             self.assertIsNotNone(grads)

    #             # check that gradients are provided
    #             if t:  # if trainable we should get gradients
    #                 self.assertTrue(len(grads) > 0), "no gradient found"

    #                 # and check that array is not None
    #                 for g in grads:
    #                     self.assertTrue(not g is None), "grad is None"
    #             else:
    #                 self.assertTrue(len(grads) == 0), \
    #                                         "gradient should not exist"

    # def test_dtype(self):
    #     """Test that output dtype can be flexible."""
    #     batch_size = 100
    #     n = 128
    #     k = 64
    #     enc = LDPC5GEncoder(k, n)
    #     dec_32 = LDPC5GDecoder(enc, output_dtype=torch.float32)
    #     dec_64 = LDPC5GDecoder(enc, output_dtype=torch.float64)
    #     # llr_32 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
    #     #                          tf.cast(n, dtype=tf.int32)],
    #     #                          dtype=tf.float32)
    #     llr_32 = torch.rand(size=(torch.tensor(batch_size, dtype=torch.int32), 
    #                                 torch.tensor(n, dtype=torch.int32))).type(torch.float32)
    #     # llr_64 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
    #     #                          tf.cast(n, dtype=tf.int32)],
    #     #                          dtype=tf.float64)
    #     llr_64 = torch.rand(size=(torch.tensor(batch_size, dtype=torch.int32), 
    #                                 torch.tensor(n, dtype=torch.int32))).type(torch.float64)

    #     # output for both inputs is tf.float32
    #     u_32 = dec_32(llr_32)
    #     u_64 = dec_32(llr_64)
    #     self.assertTrue(u_32.dtype is torch.float32)
    #     self.assertTrue(u_64.dtype is torch.float32)

    #     # output for both inputs is tf.float64
    #     u_32 = dec_64(llr_32)
    #     u_64 = dec_64(llr_64)
    #     self.assertTrue(u_32.dtype is torch.float64)
    #     self.assertTrue(u_64.dtype is torch.float64)

    # def test_full_cw_ratematching(self):
    #     """Test that if return_infobit==False, the full codeword is returned.

    #     We test this for zero iterations, to see if all internal reshapes are correctly recovered before returning the estimate.
    #     """
    #     batch_size = 100
    #     params =[[64,128], [64, 180], [167, 201], [439, 800], [948, 1024],
    #              [3893, 7940], [6530, 10023], [8448, 23000]]

    #     for p in params:
    #         k = p[0]
    #         n = p[1]
    #         enc = LDPC5GEncoder(k, n)
    #         dec = LDPC5GDecoder(enc,
    #                             hard_out=False,
    #                             return_infobits=False,
    #                             num_iter=0)
    #         llr = torch.normal(size=(batch_size, n), mean=4.2, std=1)
    #         # check if return after 0 iterations equals input
    #         c_hat = dec(llr)
    #         self.assertTrue(np.array_equal(c_hat.numpy(), llr.numpy()))

    # def test_dtype_flexible(self):
    #     """Test that output_dtype can be flexible and
    #     only floats are supported."""
    #     batch_size = 100
    #     k = 100
    #     n = 200
    #     source = GaussianPriorSource()

    #     enc = LDPC5GEncoder(k, n)

    #     dtypes_supported = (torch.float16, torch.float32, torch.float64)

    #     for dt_in in dtypes_supported:
    #         for dt_out in dtypes_supported:
    #             llr = source([[batch_size, n], 0.5])
    #             llr = llr.type(dt_in)

    #             dec = LDPC5GDecoder(enc, output_dtype=dt_out)

    #             x = dec(llr)

    #             self.assertTrue(x.dtype == dt_out)

    #     # test that complex inputs raise error
    #     llr = source([[batch_size, n], 0.5])
    #     llr_c = torch.complex(llr, torch.zeros_like(llr))
    #     dec = LDPC5GDecoder(enc, output_dtype=torch.float32)

    #     with self.assertRaises(AssertionError):
    #         dec(llr_c)

    # def test_pruning(self):
    #     """Test degree-1 VN pruning"""

    #     batch_size = 100
    #     ks = [100, 400, 800, 2000, 4000, 8000]
    #     rs = [0.34, 0.5, 0.75, 0.9]
    #     source = GaussianPriorSource()

    #     for k in ks:
    #         for r in rs:

    #             n = int(k/r)

    #             enc = LDPC5GEncoder(k, n)
    #             dec = LDPC5GDecoder(enc,
    #                                 prune_pcm=True,
    #                                 hard_out=False,
    #                                 num_iter=10)

    #             dec_ref = LDPC5GDecoder(enc,
    #                                     prune_pcm=False,
    #                                     hard_out=False,
    #                                     num_iter=10)

    #             llr = source([[batch_size, n], 0.5])
    #             x = dec(llr)
    #             x_ref = dec_ref(llr)

    #             # allow small difference as iterative error can accumulate after
    #             # multiple iterations
    #             diff = torch.mean(torch.math.abs(x-x_ref)).numpy()
    #             self.assertTrue(diff < 5e-2)

    # def test_pruning(self):
    #     """Test output interleaver."""

    #     bs = 10
    #     source = BinarySource()

    #     # k, n, m
    #     params = [[12, 20, 1], [200, 250, 2], [345, 544, 4], [231, 808, 8]]

    #     for (k, n, m) in params:
    #         enc_ref = LDPC5GEncoder(k, n)  # no mapper
    #         enc = LDPC5GEncoder(k, n, m)
    #         dec_ref = LDPC5GDecoder(enc_ref, cn_type="minsum")
    #         dec = LDPC5GDecoder(enc, cn_type="minsum")
    #         dec_cw = LDPC5GDecoder(enc, cn_type="minsum", return_infobits=False)

    #         u = source([bs, k])
    #         c = enc(u)
    #         c_ref = enc_ref(u)
    #         # emulate tx (no noise/scaling due to minsum required)
    #         y = 2*c-1
    #         y_ref = 2*c_ref-1

    #         u_hat = dec(y)
    #         c_hat = dec_cw(y)
    #         u_hat_ref = dec_ref(y_ref)

    #         self.assertTrue(np.array_equal(u_hat.numpy(),
    #                                        u_hat_ref.numpy()))

    #         # also verify that codeword is correctly returned
    #         self.assertTrue(np.array_equal(c_hat.numpy(),
    #                                        c.numpy()))

    #         # and verify that c and c_ref are different for m>1
    #         if m > 1:
    #             self.assertFalse(np.array_equal(c.numpy(),
    #                                             c_ref.numpy()))

    # def test_int_state(self):
    #     """Test internal state functionality of decoder.
    #     This implies that Nx1 iterations yields exact same result as N
    #     iterations."""
    #     batch_size = 1
    #     Niter = 5
    #     k = 100
    #     n = 200

    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc, num_iter=Niter)
    #     dec2 = LDPC5GDecoder(enc, num_iter=1, stateful=True)

    #     llr = torch.normal(size=(batch_size, n), mean=4.2, std=1)

    #     res1 = dec(llr)

    #     res2, msg_vn = dec2([llr, None])  # iter 0 to init msg_vn

    #     for i in range(Niter-1):  # remaining iterations
    #         res2, msg_vn = dec2([llr, msg_vn])

    #     # results must be the same, otherwise the internal state is not
    #     # correctly recovered
    #     self.assertTrue(np.allclose(res1,res2))

    # def test_llrmax(self):
    #     """Test that llr_max can be set."""
    #     k = 12
    #     n = 20
    #     enc = LDPC5GEncoder(k, n)
    #     dec = LDPC5GDecoder(enc, hard_out=False, num_iter=0)

    #     # test default value
    #     llr_max_def = dec.llr_max.numpy() # get default value
    #     x = torch.ones((1, n))*100
    #     y = dec(x).numpy()  # run 0 iterations
    #     np.max(y) == llr_max_def

    #     # set new llr_max
    #     llr_maxs = [17., 45.3, 78]
    #     for l in llr_maxs:
    #         dec.llr_max = l
    #         y = dec(x).numpy()  # run 0 iterations
    #         print(np.abs(np.max(y)-l) < 1e-6)

if __name__ == '__main__':
    unittest.main()