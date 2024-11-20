import os
print("Current directory:", os.getcwd())
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")
# import pytest
import unittest
import numpy as np
import torch
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')
from comcloak.fec.scrambling import Descrambler, Scrambler, TB5GScrambler
from comcloak.utils import BinarySource
from comcloak.nr import generate_prng_seq

class TestScrambler(unittest.TestCase):

    def test_sequence_dimension(self):
        """Test against correct dimensions of the sequence"""
        seq_lengths = [1, 100, 256, 1000, int(1e4)]
        batch_sizes = [1, 100, 256, 1000, int(1e4)]

        # keep_State=True
        for seq_length in seq_lengths:
            # init new scrambler for new sequence size;
            # only different batch_sizes are allowed in this mode
            s = Scrambler(binary=False)
            for batch_size in batch_sizes:
                llr = torch.rand(batch_size, seq_length)
                # build scrambler
                x = s(llr).detach().numpy()
                self.assertTrue(np.array_equal(np.array(x.shape), [batch_size, seq_length]))
        
        # keep_state=False
        s = Scrambler(binary=False, keep_state=False)
        for seq_length in seq_lengths:
            for batch_size in batch_sizes:
                llr = torch.rand(batch_size, seq_length)
                # build scrambler
                x = s(llr).detach().numpy()
                self.assertTrue(np.array_equal(np.array(x.shape), [batch_size, seq_length]))

    def test_sequence_offset(self):
        """Test that scrambling sequence has no offset, i.e., equal likely 0s
        and 1s"""
        seq_length = int(1e4)
        batch_size = int(1e2)
        for seed in (None, 1337, 1234, 1003):
            for keep_state in (False, True):
                s = Scrambler(seed=seed, keep_state=keep_state, binary=True)
                llr = torch.rand(batch_size, seq_length)
                s(llr)
                # generate a random sequence
                x = s(torch.zeros_like(llr))
                self.assertAlmostEqual(np.mean(x.detach().numpy()), 0.5, places=2)

    def test_sequence_batch(self):
        """Test that scrambling sequence is random per batch sample iff
        keep_batch_dims=True."""

        seq_length = int(1e6)
        batch_size = int(1e1)
        llr = torch.rand(batch_size, seq_length)
        
        for keep_state in (False, True):
            s = Scrambler(keep_batch_constant=False, keep_state=keep_state, binary=True)
            # generate a random sequence
            x = s(torch.zeros_like(llr))
            for i in range(batch_size - 1):
                for j in range(i + 1, batch_size):
                    # each batch sample must be different
                    self.assertAlmostEqual(np.mean(np.abs(x[i].detach().numpy() - x[j].detach().numpy())), 0.5, places=2)
        # test that the pattern is the same of option keep_batch_constant==True
        for keep_state in (False, True):
            s = Scrambler(keep_batch_constant=True, keep_state=keep_state, binary=True)
            # generate a random sequence
            x = s(torch.zeros_like(llr))
            for i in range(batch_size - 1):
                for j in range(i + 1, batch_size):
                    self.assertTrue(np.sum(np.abs(x[i].detach().numpy() - x[j].detach().numpy())) == 0)

    def test_sequence_realization(self):
        """Test that scrambling sequences are random for each new realization.
        """

        seq_length = int(1e5)
        batch_size = int(1e2)
        s = Scrambler(keep_state=False, binary=True)
        llr = torch.rand(batch_size, seq_length)
        # generate a random sequence
        x1 = s(torch.zeros_like(llr))
        x2 = s(torch.zeros_like(llr))
        self.assertAlmostEqual(np.mean(np.abs(x1.detach().numpy() - x2.detach().numpy())), 0.5, places=3)

    def test_inverse(self):
        """Test that scrambling can be inverted/removed.
        2x scrambling must result in the original sequence (for binary and
         LLRs).
        """
        seq_length = int(1e5)
        batch_size = int(1e2)
        
        # Binary scrambling
        b = (torch.rand(batch_size, seq_length) > 0.5).float()
        for keep_batch in (False, True):
            s = Scrambler(binary=True, keep_batch_constant=keep_batch, keep_state=True)
            x = s(b)
            x = s(x)
            self.assertTrue(np.array_equal(x.detach().numpy(), b.detach().numpy()))
            
            # Non-binary (LLR) scrambling
            # check soft-value scrambling (flip sign)
            s = Scrambler(binary=False, keep_batch_constant=keep_batch, keep_state=True)
            llr = torch.rand(batch_size, seq_length)
            x = s(llr)
            x = s(x)
            self.assertTrue(np.array_equal(x.detach().numpy(), llr.detach().numpy()))

    def test_llr(self):
        """Test that scrambling works for soft-values (sign flip)."""
        s = Scrambler(binary=False, seed=12345)
        b = torch.ones(100, 200)
        x = s(b)
        s2 = Scrambler(binary=True, seed=12345)
        res = -2. * s2(torch.zeros_like(x)) + 1
        self.assertTrue(np.array_equal(x.detach().numpy(), res.detach().numpy()))

    def test_keep_state(self):
        """Test that keep_state works as expected.
        Iff keep_state==True, the scrambled sequences must be constant."""
        seq_length = int(1e5)
        batch_size = int(1e2)
        llr = torch.randint(-100, 100, (batch_size, seq_length)).float()
        
        s = Scrambler(binary=True, keep_state=True)
        res1 = s(torch.zeros_like(llr))
        res2 = s(torch.zeros_like(llr))
        self.assertTrue(np.array_equal(res1.detach().numpy(), res2.detach().numpy()))
        
        s = Scrambler(binary=True, keep_state=False)
        _ = s(llr)
        res1 = s(torch.zeros_like(llr))
        _ = s(llr)
        res2 = s(torch.zeros_like(llr))
        self.assertFalse(np.array_equal(res1.detach().numpy(), res2.detach().numpy()))

    def test_seed(self):
        """Test that seed generates reproducible results."""
        seq_length = int(1e5)
        batch_size = int(1e2)
        b = torch.zeros([batch_size, seq_length])

        s1 = Scrambler(seed=1337, binary=True, keep_state=False)
        res_s1_1 = s1(b)
        res_s1_2 = s1(b)
        # new realization per call
        self.assertFalse(np.array_equal(res_s1_1.numpy(), res_s1_2.numpy()))

        # if keep_state=True, the same seed should lead to the same sequence
        s2 = Scrambler(seed=1337, binary=True, keep_state=True)
        res_s2_1 = s2(b)
        s3 = Scrambler(seed=1337)
        res_s3_1 = s3(b)
        # same seed lead to same sequence
        self.assertTrue(np.array_equal(res_s2_1.numpy(), res_s3_1.numpy()))

         # but with random seed it gives a new sequence for each init
        s4 = Scrambler(seed=None, binary=True, keep_state=True)
        res_s4_1 = s2(b)
        s5 = Scrambler(seed=None)
        res_s5_1 = s5(b)
        # same seed lead to same sequence
        self.assertFalse(np.array_equal(res_s4_1.numpy(), res_s5_1.numpy()))

        # for keep_State=False, even the same seed leads to new results
        s6 = Scrambler(seed=1337, binary=True, keep_state=False)
        res_s6_1 = s6(b)
        # different seed generates new sequence
        self.assertFalse(np.array_equal(res_s6_1.numpy(), res_s2_1.numpy()))

        # init with same seed as previous random seed
        s7 = Scrambler(seed=None, binary=True, keep_state=True)
        res_s7_1 = s7(b)
        s8 = Scrambler(seed=s7.seed, binary=True, keep_state=True)
        res_s8_1 = s8(b)
        # same seed lead to same sequence
        self.assertTrue(np.array_equal(res_s7_1.numpy(), res_s8_1.numpy()))

        # test that seed can be also provided to call
        seed = 987654
        s9 = Scrambler(seed=45234, keep_state=False)
        s10 = Scrambler(seed=76543, keep_state=True)
        x1 = s9([b, seed]).numpy()
        x2 = s9([b, seed+1]).numpy()
        x3 = s9([b, seed]).numpy()
        x4 = s10([b, seed]).numpy()
        self.assertFalse(np.array_equal(x1, x2)) # different seed
        self.assertTrue(np.array_equal(x1, x3)) # same seed
        self.assertTrue(np.array_equal(x1, x4)) # same seed (keep_state=f)

        # test that random seed allows inverse
        x5 = s9([b, seed])
        x6 = s9([b, seed]).numpy()
        # same seed
        self.assertTrue(np.array_equal(x5, x6)) # identity
        # different seed
        x7 = s9([b, seed+1])
        self.assertFalse(np.array_equal(x5, x7)) # identity
        # same seed again
        x8 = s9([b, seed+1])
        self.assertTrue(np.array_equal(x7, x8)) # identity

    def test_dtype(self):
        """Test that variable dtypes are supported."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        dt_supported = [torch.float16, torch.float32, torch.float64]
        for dt in dt_supported:
            for dt_in in dt_supported:
                for dt_out in dt_supported:
                    b = torch.zeros([batch_size, seq_length], dtype=dt_in)
                    s1 = Scrambler(dtype=dt)
                    s2 = Descrambler(s1, dtype=dt_out)
                    x = s1(b)
                    y = s2(x)
                    assert (x.dtype==dt)
                    assert (y.dtype==dt_out)

    def test_descrambler(self):
        """"Test that descrambler works as expected."""
        seq_length = int(1e2)
        batch_size = int(1e1)

        b = torch.zeros([batch_size, seq_length])
        s1 = Scrambler()
        s2 = Descrambler(s1)
        x = s1(b)
        y = s2(x)
        assert (np.array_equal(b.numpy(), y.numpy()))

        x = s1([b, 1234])
        y = s2(x)
        assert (not np.array_equal(b.numpy(), y.numpy()))

        # check if seed is correctly retrieved from scrambler
        s3 = Scrambler(seed=12345)
        s4 = Descrambler(s3)
        x = s3(b)
        y = s4(x)
        assert (np.array_equal(b.numpy(), y.numpy()))

    def test_descrambler_nonbin(self):
        """"Test that descrambler works with non-binary."""
        seq_length = int(1e2)
        batch_size = int(1e1)

        b = torch.zeros([batch_size, seq_length])

        # scrambler binary, but descrambler non-binary
        scrambler = Scrambler(seed=1235456, binary=True)
        descrambler = Descrambler(scrambler, binary=False)
        # with explicit seed
        s = 8764
        y = scrambler([b, s])
        z = descrambler([2*y-1, s]) # bspk
        z = 1 + z # remove bpsk
        assert (np.array_equal(b.numpy(), z.numpy()))
        #without explicit seed
        y = scrambler(b)
        z = descrambler(2*y-1) # bspk
        z = 1 + z # remove bpsk
        assert (np.array_equal(b.numpy(), z.numpy()))

        # scrambler non-binary, but descrambler
        scrambler = Scrambler(seed=1235456, binary=False)
        descrambler = Descrambler(scrambler, binary=True)
        s = 546342
        y = scrambler([2*b-1, s]) # bspk
        y = 0.5*(1 + y) # remove bpsk
        z = descrambler([y, s])
        assert (np.array_equal(b.numpy(), z.numpy()))
        #without explicit seed
        y = scrambler(2*b-1) # bspk
        y = 0.5*(1 + y) # remove bpsk
        z = descrambler(y)
        y = 1 + y # remove bpsk
        assert (np.array_equal(b.numpy(), z.numpy()))

    def test_scrambler_binary(self):
        """Test that binary flag can be used as input"""
        seq_length = int(1e2)
        batch_size = int(1e1)
        
        b = torch.ones(batch_size, seq_length)

        # scrambler binary, but descrambler non-binary
        
        scrambler = Scrambler(seed=1245, binary=True)
        
        s = 1234
        x1 = scrambler(b) # binary scrambling
        x2 = scrambler([b, s]) # binary scrambling different seed
        x3 = scrambler([b, s, True]) # binary scrambling different seed
        x4 = scrambler([b, s, False]) # non-binary scrambling different seed

        assert (not np.array_equal(x1.numpy(), x2.numpy())) # different seed
        assert (np.array_equal(x2.numpy(), x3.numpy())) # same seed
        # same but "bpsk modulated"
        assert (not np.array_equal(x1.numpy(), 0.5*(1+x4.numpy())))

    def test_explicit_sequence(self):
        """Test that explicit scrambling sequence can be provided."""

        bs = 10
        seq_length = 123

        # with/ without implicit broadcasting
        shapes = [[bs, seq_length], [seq_length]]
        for s in shapes:
            seq = np.ones(s)

            x = np.zeros([bs, seq_length])
            scrambler1 = Scrambler(seed=1245, sequence=seq, binary=True)
            y1 = scrambler1(x)

            # for all-zero input, output sequence equals scrambling sequence
            if len(s)==1:
                y = y1.numpy()[0,:] # if implicit broadcasting is tested
            else:
                y = y1
            self.assertTrue(np.array_equal(seq, y))

            # check that seed has no influence
            scrambler2 = Scrambler(seed=1323, sequence=seq, binary=True)
            y2 = scrambler2(x)
            self.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))

        # test descrambler with new random sequence
        seq = generate_prng_seq(seq_length, 42)

        for b in [True, False]:
            scrambler = Scrambler(sequence=seq, binary=b)
            descrambler = Descrambler(scrambler, binary=b)
            x = np.ones([bs, seq_length])
            y = scrambler(x)
            y2 = scrambler([x, 1337]) # explicit seed should not have any impact
            z = descrambler(y)

            self.assertFalse(np.array_equal(x, y.numpy()))
            self.assertTrue(np.array_equal(x, z.numpy()))
            self.assertTrue(np.array_equal(y.numpy(), y2.numpy()))

class TestTB5GScrambler(unittest.TestCase):

    def test_sequence_dimension(self):
        """Test against correct dimensions of the sequence"""
        seq_lengths = [1, 100, 256, 1000, int(1e4)]
        batch_sizes = [1, 100, 256, 1000, int(1e4)]

        s = TB5GScrambler()
        for seq_length in seq_lengths:
            for batch_size in batch_sizes:
                llr = torch.rand((batch_size, seq_length))
                x = s(llr).detach().numpy()
                self.assertTrue(np.array_equal(x.shape, [batch_size, seq_length]))

    def test_sequence_batch(self):
        """Test that scrambling sequence the same for all batch samples."""
        seq_length = int(1e3)
        batch_size = int(100)

        s = TB5GScrambler(binary=True)

        x = s(torch.zeros((batch_size, seq_length)))
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                self.assertTrue(torch.sum(torch.abs(x[i, :] - x[j, :])) == 0)

    def test_sequence_realization(self):
        """Test that scrambling sequences are random for different init values."""
        seq_length = int(1e2)
        batch_size = int(10)
        n_rnti_ref = 1337
        n_id_ref = 42
        s = TB5GScrambler(n_rnti_ref, n_id_ref, binary=True)
        x_ref = s(torch.zeros((batch_size, seq_length)))

        for _ in range(100): # randomly init new scramblers
            n_rnti=np.random.randint(0, 2**16-1)
            n_id=np.random.randint(0, 2**10-1)
            if n_rnti==n_rnti_ref and n_id==n_id_ref:
                continue # skip evaluation if ref init parameters are selected
            s = TB5GScrambler(n_rnti, n_id, binary=True)
            # generate a random sequence
            x = s(torch.zeros((batch_size, seq_length)))
            # and the sequence must be different
            self.assertFalse(np.array_equal(x_ref.detach().numpy(), x.detach().numpy()))

    def test_inverse(self):
        """Test that scrambling can be inverted/removed.
           2x scrambling must result in the original sequence (for binary and LLRs).
        """
        seq_length = int(1e3)
        batch_size = int(1e2)

        #check binary scrambling
        b = torch.randint(0, 2, (batch_size, seq_length)).float()
        s = TB5GScrambler(binary=True)
        b = torch.tensor(torch.greater(torch.tensor(0.5), b), dtype=torch.float32)
        x = s(b)
        x = s(x)
        self.assertTrue(np.array_equal(x.numpy(), b.numpy()))

        #check soft-value scrambling (flip sign)
        s = TB5GScrambler(binary=False)
        llr = torch.rand((batch_size, seq_length))
        x = s(llr)
        x = s(x)
        self.assertTrue(np.array_equal(x.numpy(), llr.numpy()))

    def test_dtype(self):
        """Test that variable dtypes are supported."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        dt_supported = [torch.float16, torch.float32, torch.float64]
        for dt in dt_supported:
            for dt_in in dt_supported:
                for dt_out in dt_supported:
                    b = torch.zeros([batch_size, seq_length], dtype=dt_in)
                    s1 = TB5GScrambler(dtype=dt)
                    s2 = Descrambler(s1, dtype=dt_out)
                    x = s1(b)
                    y = s2(x)
                    self.assertEqual(x.dtype, dt)
                    self.assertEqual(y.dtype, dt_out)

    def test_descrambler(self):
        """Test that descrambler works as expected."""
        seq_length = int(1e2)
        batch_size = int(1e1)

        b = torch.zeros([batch_size, seq_length])
        s1 = TB5GScrambler()
        s2 = Descrambler(s1)
        x = s1(b)
        y = s2(x)
        assert (np.array_equal(b.numpy(), y.numpy()))

    def test_descrambler_nonbin(self):
        """"Test that descrambler works with non-binary."""
        seq_length = int(1e2)
        batch_size = int(1e1)

        b = torch.zeros([batch_size, seq_length])

        # scrambler binary, but descrambler non-binary
        scrambler = Scrambler(binary=True)
        descrambler = Descrambler(scrambler, binary=False)

        y = scrambler(b)
        z = descrambler(2*y-1) # bspk
        z = 1 + z # remove bpsk
        assert (np.array_equal(b.numpy(), z.numpy()))

        # scrambler non-binary, but descrambler
        scrambler = Scrambler(binary=False)
        descrambler = Descrambler(scrambler, binary=True)
        y = scrambler(2*b-1) # bspk
        y = 0.5*(1 + y) # remove bpsk
        z = descrambler(y)
        y = 1 + y # remove bpsk
        assert (np.array_equal(b.numpy(), z.numpy()))

    def test_scrambler_binary(self):
        """Test that binary flag can be used as input."""
        seq_length = int(1e2)
        batch_size = int(1e1)

        b = torch.ones([batch_size, seq_length])

        # scrambler binary, but descrambler non-binary

        scrambler = TB5GScrambler(binary=True)

        x1 = scrambler(b) # binary scrambling
        x2 = scrambler([b]) # binary scrambling
        x3 = scrambler([b, True]) # binary scrambling
        x4 = scrambler([b, False]) # non-binary scrambling

        assert (np.array_equal(x1.numpy(), x2.numpy()))
        assert (np.array_equal(x2.numpy(), x3.numpy()))
        assert (np.array_equal(x1.numpy(), 0.5*(1+x4.numpy())))

    def test_5gnr_reference(self):
        """Test against 5G NR reference."""
        bs = 2
        l = 100

        # check valid inputs
        n_rs = [0, 10, 65535]
        n_ids = [0, 10, 1023]
        s_old = None
        for n_r in n_rs:
            for n_id  in n_ids:
                s = TB5GScrambler(n_id=n_id, n_rnti=n_r)(torch.zeros((bs, l)))
                # verify that new sequence is unique
                if s_old is not None:
                    self.assertFalse(np.array_equal(s, s_old))
                s_old = s

        # test against invalid inputs
        n_rs = [-1, 1.2, 65536] # invalid
        n_ids = [0, 10, 1023] # valid
        for n_r in n_rs:
            for n_id in n_ids:
                with self.assertRaises(AssertionError):
                    s = TB5GScrambler(n_id=n_id, n_rnti=n_r)(torch.zeros((bs, l)))

        n_rs = [0, 10, 65535] # valid
        n_ids = [-1, 1.2, 1024] # invalid
        for n_r in n_rs:
            for n_id  in n_ids:
                with self.assertRaises(AssertionError):
                    s = TB5GScrambler(n_id=n_id, n_rnti=n_r)(torch.zeros((bs, l)))

        # test against reference example
        n_rnti = 20001
        n_id = 41
        l = 100
        s_ref = np.array([0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
                          1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
                          1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.,
                          0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
                          0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
                          0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
                          1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
                          1., 1., 1., 1., 1., 1., 1., 0., 0.])
        s = TB5GScrambler(n_id=n_id, n_rnti=n_rnti)(torch.zeros((1, l)))
        s = torch.squeeze(s, axis=0) # remove batch-dim
        self.assertTrue(np.array_equal(s, s_ref))

        # and test against wrong parameters
        s = TB5GScrambler(n_id=n_id, n_rnti=n_rnti+1)(torch.zeros((1, l)))
        s = torch.squeeze(s, axis=0) # remove batch-dim
        self.assertFalse(np.array_equal(s, s_ref))
        s = TB5GScrambler(n_id=n_id+1, n_rnti=n_rnti)(torch.zeros((1, l)))
        s = torch.squeeze(s, axis=0) # remove batch-dim
        self.assertFalse(np.array_equal(s, s_ref))

        # test that PUSCH and PDSCH are the same for single cw mode
        s_ref = TB5GScrambler(n_id=n_id,
                         n_rnti=n_rnti,
                         channel_type="PUSCH")(torch.zeros((1, l)))

        # cw_idx has no impact in uplink
        s = TB5GScrambler(n_id=n_id,
                        n_rnti=n_rnti,
                        codeword_index=1,
                        channel_type="PUSCH")(torch.zeros((1, l)))
        self.assertTrue(np.array_equal(s_ref, s))

        # downlink equals uplink for cw_idx=0
        s = TB5GScrambler(n_id=n_id,
                        n_rnti=n_rnti,
                        codeword_index=0,
                        channel_type="PDSCH")(torch.zeros((1, l)))
        self.assertTrue(np.array_equal(s_ref, s))

        # downlink is different uplink for cw_idx=1
        s = TB5GScrambler(n_id=n_id,
                        n_rnti=n_rnti,
                        codeword_index=1,
                        channel_type="PDSCH")(torch.zeros((1, l)))
        self.assertFalse(np.array_equal(s_ref, s))

    def test_multi_user(self):
        """Test multi-stream functionality.
        If n_rnti and n_id are provided as list of ints, the axis=-2 dimension
        is interpreted as independent stream."""

        seq_length = int(1e3)
        batch_size = 13

        n_rntis = [1, 38282, 1337, 36443]
        n_ids = [123, 42, 232, 134]

        u = torch.zeros([batch_size, 2, len(n_rntis), seq_length])
        s_ref = np.zeros([batch_size, 2, len(n_rntis), seq_length])

        # generate batch of multiple streams individually
        for idx,(n_rnti, n_id) in enumerate(zip(n_rntis, n_ids)):
            s_ref[..., idx,:] = TB5GScrambler(n_id=n_id,
                              n_rnti=n_rnti)(u[...,0,:]).numpy()

        # run scrambler one-shot with list of n_rnti/n_id
        scrambler = TB5GScrambler(n_id=n_ids,
                                  n_rnti=n_rntis)
        s = scrambler(u).numpy()

        # scrambling sequences should be equivalent
        self.assertTrue(np.array_equal(s, s_ref))

        # also test descrambler
        u_hat = Descrambler(scrambler)(s).numpy()

        self.assertTrue(np.array_equal(u_hat, np.zeros_like(u_hat)))

if __name__ == '__main__':
    unittest.main()