
import torch
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")
# import pytest
import unittest
import numpy as np
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')
import comcloak
from comcloak.mimo import MaximumLikelihoodDetector
from comcloak.mimo import MaximumLikelihoodDetectorWithPrior
from comcloak.mapping import Constellation
from scipy.special import logsumexp
from scipy.stats import unitary_group

class TestSymbolMaximumLikelihoodDetector(unittest.TestCase):

    def test_vecs(self):
        """
        Test the list of all possible vectors of symbols build by the baseclass
        at init
        """
        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            num_points = 2 ** num_bits_per_symbol
            L = np.zeros([num_points ** num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points ** (num_streams - k - 1)
                tile_const = num_points ** k
                for j in range(tile_const):
                    for i, p in enumerate(C.points):
                        min_index = j * num_points * tile_point + (i * tile_point)
                        max_index = j * num_points * tile_point + ((i + 1) * tile_point)
                        L[min_index:max_index, k] = p
            return L

        for num_bits_per_symbol in (2, 4):
            for num_streams in (1, 2, 3, 4):
                ref_vecs = build_vecs(num_bits_per_symbol, num_streams)
                ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                test_vecs = ml._vecs
                max_dist = np.abs(test_vecs - ref_vecs)
                self.assertTrue(np.allclose(max_dist, 0.0, atol=1e-5))

    def test_output_dimensions(self):
        for num_bits_per_symbol in (2, 4):
            num_points = 2 ** num_bits_per_symbol
            for num_streams in (1, 2, 3, 4):
                for num_rx_ant in (4, 16, 32):
                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                    batch_size = 8
                    dim1 = 3
                    dim2 = 5
                    y = torch.complex( torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant]),dtype=torch.float32),
                                       torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant]),dtype=torch.float32))
                    h = torch.complex( torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant, num_streams]),dtype=torch.float32),
                                       torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant, num_streams]),dtype=torch.float32))

                    s = torch.eye(num_rx_ant, dtype=torch.complex64)
                    logits = ml((y, h, s))
                    self.assertEqual(logits.shape, torch.Size([batch_size, dim1, dim2, num_streams, num_points]))

                    s = torch.eye(num_rx_ant, dtype=torch.complex64).expand(batch_size, dim1, dim2, -1, -1)
                    logits = ml((y, h, s))
                    self.assertEqual(logits.shape, torch.Size([batch_size, dim1, dim2, num_streams, num_points]))

    def test_logits_calc_eager(self):
        "Test exponents calculation"
        np.random.seed(42)

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2 ** num_bits_per_symbol
            L = np.zeros([num_points ** num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points ** (num_streams - k - 1)
                tile_const = num_points ** k
                for j in range(tile_const):
                    for i, p in enumerate(points):
                        min_index = j * num_points * tile_point + (i * tile_point)
                        max_index = j * num_points * tile_point + ((i + 1) * tile_point)
                        L[min_index:max_index, k] = p

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:, j], p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, c

        batch_size = 16
        for num_bits_per_symbol in (2, 4):
            for num_streams in (1, 2, 3, 4):
                for num_rx_ant in (2, 16, 32):
                    ref_vecs, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    y = np.random.normal(size=[batch_size, num_rx_ant]) + 1j * np.random.normal(size=[batch_size, num_rx_ant])
                    h = np.random.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j * np.random.normal(size=[batch_size, num_rx_ant, num_streams])
                    e = np.random.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0) * np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1, -2))
                    ref_exp = ref_exp.real
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    ref_app = logsumexp(ref_exp, axis=-3)
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_app(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits

                    test_logits = call_sys_app(torch.tensor(y, dtype=torch.complex64),
                                               torch.tensor(h, dtype=torch.complex64),
                                               torch.tensor(s, dtype=torch.complex64)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_maxlog(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits

                    test_logits = call_sys_maxlog(torch.tensor(y, dtype=torch.complex64),
                                                  torch.tensor(h, dtype=torch.complex64),
                                                  torch.tensor(s, dtype=torch.complex64))
                    max_values, _ = test_logits
                    test_logits = max_values.numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))


class TestMaximumLikelihoodDetectorWithPrior(unittest.TestCase):

    def test_vecs_ind(self):
        """
        Test the list of all possible vectors of symbol indices built by the
        base class at init
        """
        def build_vecs_ind(num_bits_per_symbol, num_streams):
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i in range(num_points):
                        min_index = j*num_points*tile_point + (i*tile_point)
                        max_index = j*num_points*tile_point + ((i+1)*tile_point)
                        L[min_index:max_index, k] = [k, i]
            return L

        for num_bits_per_symbol in (2, 4):
            for num_streams in (1, 2, 3, 4):
                ref_vecs = build_vecs_ind(num_bits_per_symbol, num_streams)
                ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                test_vecs = ml._vecs_ind.numpy()  # 转为 numpy 数组
                max_dist = np.abs(test_vecs - ref_vecs)
                self.assertTrue(np.allclose(max_dist, 0.0, atol=1e-5))

    def test_output_dimensions(self):
        for num_bits_per_symbol in (2, 4):
            num_points = 2**num_bits_per_symbol
            for num_streams in (1, 2, 3, 4):
                for num_rx_ant in (4, 16, 32):
                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                    batch_size = 8
                    dim1 = 3
                    dim2 = 5
                    y = torch.complex(torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant]),dtype=torch.float32),
                                      torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant]),dtype=torch.float32))
                    h = torch.complex(torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant, num_streams]),dtype=torch.float32),
                                      torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_rx_ant, num_streams]),dtype=torch.float32))
                    prior = torch.tensor(np.random.normal(size = [batch_size, dim1, dim2, num_streams, num_points]),dtype=torch.float32)

                    s = torch.eye(num_rx_ant, dtype=torch.complex64)
                    inputs = y, h, prior, s
                    logits = ml(inputs)
                    self.assertEqual(logits.shape, (batch_size, dim1, dim2, num_streams, num_points))

                    s = torch.eye(num_rx_ant, dtype=torch.complex64).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    s = s.expand(batch_size, dim1, dim2, num_rx_ant, num_rx_ant) 
                    inputs = y, h, prior, s
                    logits = ml(inputs)
                    self.assertEqual(logits.shape, (batch_size, dim1, dim2, num_streams, num_points))

    def test_logits_calc_eager(self):
        "Test exponents calculation"
        torch.manual_seed(42)
        np.random.seed(42)

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            L_ind = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i, p in enumerate(points):
                        min_index = j*num_points*tile_point + (i*tile_point)
                        max_index = j*num_points*tile_point + ((i+1)*tile_point)
                        L[min_index:max_index, k] = p
                        L_ind[min_index:max_index, k] = [k, i]

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:, j], p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, L_ind, c

        batch_size = 16
        for num_bits_per_symbol in (2, 4):
            for num_streams in (1, 2, 3, 4):
                for num_rx_ant in (2, 16, 32):
                    num_points = 2**num_bits_per_symbol
                    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    num_vecs = ref_vecs.shape[0]

                    y = np.random.normal(size=[batch_size, num_rx_ant]) + 1j*np.random.normal(size=[batch_size, num_rx_ant])
                    h = np.random.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*np.random.normal(size=[batch_size, num_rx_ant, num_streams])
                    prior = np.random.normal(size=[batch_size, num_streams, num_points])
                    e = np.random.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0) * np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))

                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1, -2))
                    ref_exp = ref_exp.real

                    prior_ = []
                    for i in range(batch_size):
                        prior_.append([])
                        for j in range(num_vecs):
                            prior_[-1].append([])
                            for k in range(num_streams):
                                prior_[-1][-1].append(prior[i, ref_vecs_ind[j, k][0], ref_vecs_ind[j, k][1]])
                    prior_ = np.array(prior_)
                    prior_ = np.sum(prior_, axis=-1)
                    ref_exp = ref_exp + prior_
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)

                    ref_app = logsumexp(ref_exp, axis=-3)
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_app(y, h, prior, s):
                        inputs = y, h, prior, s
                        test_logits = ml(inputs)
                        return test_logits

                    test_logits = call_sys_app(torch.tensor(y, dtype=torch.cfloat),
                                                torch.tensor(h, dtype=torch.cfloat),
                                                torch.tensor(prior, dtype=torch.float32),
                                                torch.tensor(s, dtype=torch.cfloat)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_maxlog(y, h, prior, s):
                        inputs = y, h, prior, s
                        test_logits = ml(inputs)
                        return test_logits

                    test_logits = call_sys_maxlog(torch.tensor(y, dtype=torch.cfloat),
                                                  torch.tensor(h, dtype=torch.cfloat),
                                                  torch.tensor(prior, dtype=torch.float32),
                                                  torch.tensor(s, dtype=torch.cfloat))
                    max_values, _ = test_logits
                    test_logits = max_values.numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))



if __name__ == '__main__':
    unittest.main()
