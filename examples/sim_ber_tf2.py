import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.experimental.numpy import log10 as _log10
from tensorflow.experimental.numpy import log2 as _log2
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Current directory:", os.getcwd())
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("./")
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    # os.system("pip install sionna")
    import sionna as sn
from sionna.utils.metrics import count_errors, count_block_errors
from sionna.mapping import Mapper, Constellation
import time
from sionna import signal
import matplotlib.pyplot as plt


# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# For the implementation of the Keras models
from tensorflow.keras import Model
def hard_decisions(llr):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex tf.DType
        Tensor of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    zero = tf.constant(0, dtype=llr.dtype)

    return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)

# def sim_ber(mc_fun,
#             ebno_dbs,
#             batch_size,
#             max_mc_iter,
#             soft_estimates=False,
#             num_target_bit_errors=None,
#             num_target_block_errors=None,
#             distribute=None,
#             forward_keyboard_interrupt=True,
#             verbose=True,
#             dtype=tf.complex64):
#     # pylint: disable=line-too-long
#     # utility function to print progress

#     # distributed execution should not be done in Eager mode
#     # XLA mode seems to have difficulties with TF2.13
#     @tf.function(jit_compile=False)
#     def _run_distributed(strategy, mc_fun, batch_size, ebno_db):
#         # use tf.distribute to execute on parallel devices (=replicas)
#         outputs_rep = strategy.run(mc_fun,
#                                    args=(batch_size, ebno_db))
#         # copy replicas back to single device
#         b = strategy.gather(outputs_rep[0], axis=0)
#         b_hat = strategy.gather(outputs_rep[1], axis=0)
#         return b, b_hat


#     # support multi-device simulations by using the tf.distribute package
#     if distribute is None: # disabled per default
#         run_multigpu = False
#     # use strategy if explicitly provided
#     elif isinstance(distribute, tf.distribute.Strategy):
#         run_multigpu = True
#         strategy = distribute # distribute is already a tf.distribute.strategy
#     else:
#         run_multigpu = True
#         # use all available gpus
#         if distribute=="all":
#             gpus = tf.config.list_logical_devices('GPU')
#         # mask active GPUs if indices are provided
#         elif isinstance(distribute, (tuple, list)):
#             gpus_avail = tf.config.list_logical_devices('GPU')
#             gpus = [gpus_avail[i] for i in distribute if i < len(gpus_avail)]
#         else:
#             raise ValueError("Unknown value for distribute.")

#         # deactivate logging of tf.device placement
#         if verbose:
#             print("Setting tf.debugging.set_log_device_placement to False.")
#         tf.debugging.set_log_device_placement(False)
#         # we reduce to the first device by default
#         strategy = tf.distribute.MirroredStrategy(gpus,
#                             cross_device_ops=tf.distribute.ReductionToOneDevice(
#                                                 reduce_to_device=gpus[0].name))

#     # reduce max_mc_iter if multi_gpu simulations are activated
#     if run_multigpu:
#         num_replicas = strategy.num_replicas_in_sync
#         max_mc_iter = int(np.ceil(max_mc_iter/num_replicas))
#         print(f"Distributing simulation across {num_replicas} devices.")
#         print(f"Reducing max_mc_iter to {max_mc_iter}")

#     ebno_dbs = tf.cast(ebno_dbs, dtype.real_dtype)
#     batch_size = tf.cast(batch_size, tf.int32)
#     num_points = tf.shape(ebno_dbs)[0]
#     bit_errors = tf.Variable(   tf.zeros([num_points], dtype=tf.int64),
#                                 dtype=tf.int64)
#     block_errors = tf.Variable( tf.zeros([num_points], dtype=tf.int64),
#                                 dtype=tf.int64)
#     nb_bits = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
#                             dtype=tf.int64)
#     nb_blocks = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
#                             dtype=tf.int64)

#     # track status of simulation (early termination etc.)
#     status = np.zeros(num_points)

#     # measure runtime per SNR point
#     runtime = np.zeros(num_points)

#     # ensure num_target_errors is a tensor
#     if num_target_bit_errors is not None:
#         num_target_bit_errors = tf.cast(num_target_bit_errors, tf.int64)
#     if num_target_block_errors is not None:
#         num_target_block_errors = tf.cast(num_target_block_errors, tf.int64)

#     try:
#         # simulate until a target number of errors is reached
#         for i in tf.range(max_mc_iter):
#             # runtime[i] = time.perf_counter() # save start time
#             iter_count = -1 # for print in verbose mode
            
#             for ii in tf.range(num_points):

#                 iter_count += 1

#                 if run_multigpu: # distributed execution
#                     b, b_hat = _run_distributed(strategy,
#                                                 mc_fun,
#                                                 batch_size,
#                                                 ebno_dbs[ii])
#                 else:
#                     outputs = mc_fun(batch_size=batch_size, ebno_db=ebno_dbs[ii])
#                     # assume first and second return value is b and b_hat
#                     # other returns are ignored
#                     b = outputs[0]
#                     b_hat = outputs[1]

#                 if soft_estimates:
#                     b_hat = hard_decisions(b_hat)

#                 # count errors
#                 bit_e = count_errors(b, b_hat)
#                 block_e = count_block_errors(b, b_hat)

#                 # count total number of bits
#                 bit_n = tf.size(b)
#                 block_n = tf.size(b[...,-1])

#                 # update variables
#                 bit_errors = tf.tensor_scatter_nd_add(  bit_errors, [[ii]],
#                                                     tf.cast([bit_e], tf.int64))
#                 block_errors = tf.tensor_scatter_nd_add(  block_errors, [[ii]],
#                                                 tf.cast([block_e], tf.int64))
#                 nb_bits = tf.tensor_scatter_nd_add( nb_bits, [[ii]],
#                                                     tf.cast([bit_n], tf.int64))
#                 nb_blocks = tf.tensor_scatter_nd_add( nb_blocks, [[ii]],
#                                                 tf.cast([block_n], tf.int64))
            
#             # if (i+1) % 10 == 0:
#             #     # calculate BER / BLER
#             #     ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
#             #     bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)
#             #     # replace nans (from early stop)
#             #     ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
#             #     bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)
#             #     # broadcast snr if ber is list
#             #     plot_ber(ber)

#     # Stop if KeyboardInterrupt is detected and set remaining SNR points to -1
#     except KeyboardInterrupt as e:

#         # Raise Interrupt again to stop outer loops
#         if forward_keyboard_interrupt:
#             raise e
#         print("\nSimulation stopped by the user " \
#               f"@ EbNo = {ebno_dbs[i].numpy()} dB.")
#         # overwrite remaining BER / BLER positions with -1
#         for idx in range(i+1, num_points):
#             bit_errors = tf.tensor_scatter_nd_update( bit_errors, [[idx]],
#                                                     tf.cast([-1], tf.int64))
#             block_errors = tf.tensor_scatter_nd_update( block_errors, [[idx]],
#                                                     tf.cast([-1], tf.int64))
#             nb_bits = tf.tensor_scatter_nd_update( nb_bits, [[idx]],
#                                                     tf.cast([1], tf.int64))
#             nb_blocks = tf.tensor_scatter_nd_update( nb_blocks, [[idx]],
#                                                     tf.cast([1], tf.int64))
#     ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
#     bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)
#     # replace nans (from early stop)
#     ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
#     bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)
#     # broadcast snr if ber is list
#     plot_ber(ber, ebno_dbs)
#     return ber, bler
# def sim_ber(mc_fun,
#             ebno_dbs,
#             batch_size,
#             max_mc_iter,
#             soft_estimates=False,
#             num_target_bit_errors=None,
#             num_target_block_errors=None,
#             target_ber=None,
#             target_bler=None,
#             early_stop=True,
#             graph_mode=None,
#             distribute=None,
#             verbose=True,
#             forward_keyboard_interrupt=True,
#             callback=None,
#             dtype=tf.complex64):
#     # pylint: disable=line-too-long
#     """Simulates until target number of errors is reached and returns BER/BLER.

#     The simulation continues with the next SNR point if either
#     ``num_target_bit_errors`` bit errors or ``num_target_block_errors`` block
#     errors is achieved. Further, it continues with the next SNR point after
#     ``max_mc_iter`` batches of size ``batch_size`` have been simulated.
#     Early stopping allows to stop the simulation after the first error-free SNR
#     point or after reaching a certain ``target_ber`` or ``target_bler``.

#     Input
#     -----
#     mc_fun: callable
#         Callable that yields the transmitted bits `b` and the
#         receiver's estimate `b_hat` for a given ``batch_size`` and
#         ``ebno_db``. If ``soft_estimates`` is True, `b_hat` is interpreted as
#         logit.

#     ebno_dbs: tf.float32
#         A tensor containing SNR points to be evaluated.

#     batch_size: tf.int32
#         Batch-size for evaluation.

#     max_mc_iter: tf.int32
#         Maximum number of Monte-Carlo iterations per SNR point.

#     soft_estimates: bool
#         A boolean, defaults to `False`. If `True`, `b_hat`
#         is interpreted as logit and an additional hard-decision is applied
#         internally.

#     num_target_bit_errors: tf.int32
#         Defaults to `None`. Target number of bit errors per SNR point until
#         the simulation continues to next SNR point.

#     num_target_block_errors: tf.int32
#         Defaults to `None`. Target number of block errors per SNR point
#         until the simulation continues

#     target_ber: tf.float32
#         Defaults to `None`. The simulation stops after the first SNR point
#         which achieves a lower bit error rate as specified by ``target_ber``.
#         This requires ``early_stop`` to be `True`.

#     target_bler: tf.float32
#         Defaults to `None`. The simulation stops after the first SNR point
#         which achieves a lower block error rate as specified by ``target_bler``.
#         This requires ``early_stop`` to be `True`.

#     early_stop: bool
#         A boolean defaults to `True`. If `True`, the simulation stops after the
#         first error-free SNR point (i.e., no error occurred after
#         ``max_mc_iter`` Monte-Carlo iterations).

#     graph_mode: One of ["graph", "xla"], str
#         A string describing the execution mode of ``mc_fun``.
#         Defaults to `None`. In this case, ``mc_fun`` is executed as is.

#     distribute: `None` (default) | "all" | list of indices | `tf.distribute.strategy`
#         Distributes simulation on multiple parallel devices. If `None`,
#         multi-device simulations are deactivated. If "all", the workload will
#         be automatically distributed across all available GPUs via the
#         `tf.distribute.MirroredStrategy`.
#         If an explicit list of indices is provided, only the GPUs with the given
#         indices will be used. Alternatively, a custom `tf.distribute.strategy`
#         can be provided. Note that the same `batch_size` will be
#         used for all GPUs in parallel, but the number of Monte-Carlo iterations
#         ``max_mc_iter`` will be scaled by the number of devices such that the
#         same number of total samples is simulated. However, all stopping
#         conditions are still in-place which can cause slight differences in the
#         total number of simulated samples.

#     verbose: bool
#         A boolean defaults to `True`. If `True`, the current progress will be
#         printed.

#     forward_keyboard_interrupt: bool
#         A boolean defaults to `True`. If `False`, KeyboardInterrupts will be
#         catched internally and not forwarded (e.g., will not stop outer loops).
#         If `False`, the simulation ends and returns the intermediate simulation
#         results.

#     callback: `None` (default) | callable
#         If specified, ``callback`` will be called after each Monte-Carlo step.
#         Can be used for logging or advanced early stopping. Input signature of
#         ``callback`` must match `callback(mc_iter, snr_idx, ebno_dbs,
#         bit_errors, block_errors, nb_bits, nb_blocks)` where ``mc_iter``
#         denotes the number of processed batches for the current SNR point,
#         ``snr_idx`` is the index of the current SNR point, ``ebno_dbs`` is the
#         vector of all SNR points to be evaluated, ``bit_errors`` the vector of
#         number of bit errors for each SNR point, ``block_errors`` the vector of
#         number of block errors, ``nb_bits`` the vector of number of simulated
#         bits, ``nb_blocks`` the vector of number of simulated blocks,
#         respectively. If ``callable`` returns `sim_ber.CALLBACK_NEXT_SNR`, early
#         stopping is detected and the simulation will continue with the
#         next SNR point. If ``callable`` returns
#         `sim_ber.CALLBACK_STOP`, the simulation is stopped
#         immediately. For `sim_ber.CALLBACK_CONTINUE` continues with
#         the simulation.

#     dtype: tf.complex64
#         Datatype of the callable ``mc_fun`` to be used as input/output.

#     Output
#     ------
#     (ber, bler) :
#         Tuple:

#     ber: tf.float32
#         The bit-error rate.

#     bler: tf.float32
#         The block-error rate.

#     Raises
#     ------
#     AssertionError
#         If ``soft_estimates`` is not bool.

#     AssertionError
#         If ``dtype`` is not `tf.complex`.

#     Note
#     ----
#     This function is implemented based on tensors to allow
#     full compatibility with tf.function(). However, to run simulations
#     in graph mode, the provided ``mc_fun`` must use the `@tf.function()`
#     decorator.

#     """

#     # utility function to print progress
#     def _print_progress(is_final, rt, idx_snr, idx_it, header_text=None):
#         """Print summary of current simulation progress.

#         Input
#         -----
#         is_final: bool
#             A boolean. If True, the progress is printed into a new line.
#         rt: float
#             The runtime of the current SNR point in seconds.
#         idx_snr: int
#             Index of current SNR point.
#         idx_it: int
#             Current iteration index.
#         header_text: list of str
#             Elements will be printed instead of current progress, iff not None.
#             Can be used to generate table header.
#         """
#         # set carriage return if not final step
#         if is_final:
#             end_str = "\n"
#         else:
#             end_str = "\r"

#         # prepare to print table header
#         if header_text is not None:
#             row_text = header_text
#             end_str = "\n"
#         else:
#             # calculate intermediate ber / bler
#             ber_np = (tf.cast(bit_errors[idx_snr], tf.float64)
#                         / tf.cast(nb_bits[idx_snr], tf.float64)).numpy()
#             ber_np = np.nan_to_num(ber_np) # avoid nan for first point
#             bler_np = (tf.cast(block_errors[idx_snr], tf.float64)
#                         / tf.cast(nb_blocks[idx_snr], tf.float64)).numpy()
#             bler_np = np.nan_to_num(bler_np) # avoid nan for first point

#             # load statuslevel
#             # print current iter if simulation is still running
#             if status[idx_snr]==0:
#                 status_txt = f"iter: {idx_it:.0f}/{max_mc_iter:.0f}"
#             else:
#                 status_txt = status_levels[int(status[idx_snr])]

#             # generate list with all elements to be printed
#             row_text = [str(np.round(ebno_dbs[idx_snr].numpy(), 3)),
#                         f"{ber_np:.4e}",
#                         f"{bler_np:.4e}",
#                         np.round(bit_errors[idx_snr].numpy(), 0),
#                         np.round(nb_bits[idx_snr].numpy(), 0),
#                         np.round(block_errors[idx_snr].numpy(), 0),
#                         np.round(nb_blocks[idx_snr].numpy(), 0),
#                         np.round(rt, 1),
#                         status_txt]

#         # pylint: disable=line-too-long, consider-using-f-string
#         print("{: >9} |{: >11} |{: >11} |{: >12} |{: >12} |{: >13} |{: >12} |{: >12} |{: >10}".format(*row_text), end=end_str)

#     # distributed execution should not be done in Eager mode
#     # XLA mode seems to have difficulties with TF2.13
#     @tf.function(jit_compile=False)
#     def _run_distributed(strategy, mc_fun, batch_size, ebno_db):
#         # use tf.distribute to execute on parallel devices (=replicas)
#         outputs_rep = strategy.run(mc_fun,
#                                    args=(batch_size, ebno_db))
#         # copy replicas back to single device
#         b = strategy.gather(outputs_rep[0], axis=0)
#         b_hat = strategy.gather(outputs_rep[1], axis=0)
#         return b, b_hat

#      # init table headers
#     header_text = ["EbNo [dB]", "BER", "BLER", "bit errors",
#                    "num bits", "block errors", "num blocks",
#                    "runtime [s]", "status"]

#     # replace status by text
#     status_levels = ["not simulated", # status=0
#             "reached max iter       ", # status=1; spacing for impr. layout
#             "no errors - early stop", # status=2
#             "reached target bit errors", # status=3
#             "reached target block errors", # status=4
#             "reached target BER - early stop", # status=5
#             "reached target BLER - early stop", # status=6
#             "callback triggered stopping"] # status=7


#     # check inputs for consistency
#     assert isinstance(early_stop, bool), "early_stop must be bool."
#     assert isinstance(soft_estimates, bool), "soft_estimates must be bool."
#     assert dtype.is_complex, "dtype must be a complex type."
#     assert isinstance(verbose, bool), "verbose must be bool."

#     # target_ber / target_bler only works if early stop is activated
#     if target_ber is not None:
#         if not early_stop:
#             print("Warning: early stop is deactivated. Thus, target_ber " \
#                   "is ignored.")
#     else:
#         target_ber = -1. # deactivate early stopping condition
#     if target_bler is not None:
#         if not early_stop:
#             print("Warning: early stop is deactivated. Thus, target_bler " \
#                   "is ignored.")
#     else:
#         target_bler = -1. # deactivate early stopping condition

#     if graph_mode is None:
#         graph_mode="default" # applies default graph mode
#     assert isinstance(graph_mode, str), "graph_mode must be str."

#     if graph_mode=="default":
#         pass # nothing to do
#     elif graph_mode=="graph":
#         # avoid retracing -> check if mc_fun is already a function
#         if not isinstance(mc_fun, tf.types.experimental.GenericFunction):
#             mc_fun = tf.function(mc_fun,
#                                  jit_compile=False,
#                                  experimental_follow_type_hints=True)
#     elif graph_mode=="xla":
#         # avoid retracing -> check if mc_fun is already a function
#         if not isinstance(mc_fun, tf.types.experimental.GenericFunction) or \
#            not mc_fun.function_spec.jit_compile:
#             mc_fun = tf.function(mc_fun,
#                                  jit_compile=True,
#                                  experimental_follow_type_hints=True)
#     else:
#         raise TypeError("Unknown graph_mode selected.")

#     # support multi-device simulations by using the tf.distribute package
#     if distribute is None: # disabled per default
#         run_multigpu = False
#     # use strategy if explicitly provided
#     elif isinstance(distribute, tf.distribute.Strategy):
#         run_multigpu = True
#         strategy = distribute # distribute is already a tf.distribute.strategy
#     else:
#         run_multigpu = True
#         # use all available gpus
#         if distribute=="all":
#             gpus = tf.config.list_logical_devices('GPU')
#         # mask active GPUs if indices are provided
#         elif isinstance(distribute, (tuple, list)):
#             gpus_avail = tf.config.list_logical_devices('GPU')
#             gpus = [gpus_avail[i] for i in distribute if i < len(gpus_avail)]
#         else:
#             raise ValueError("Unknown value for distribute.")

#         # deactivate logging of tf.device placement
#         if verbose:
#             print("Setting tf.debugging.set_log_device_placement to False.")
#         tf.debugging.set_log_device_placement(False)
#         # we reduce to the first device by default
#         strategy = tf.distribute.MirroredStrategy(gpus,
#                             cross_device_ops=tf.distribute.ReductionToOneDevice(
#                                                 reduce_to_device=gpus[0].name))

#     # reduce max_mc_iter if multi_gpu simulations are activated
#     if run_multigpu:
#         num_replicas = strategy.num_replicas_in_sync
#         max_mc_iter = int(np.ceil(max_mc_iter/num_replicas))
#         print(f"Distributing simulation across {num_replicas} devices.")
#         print(f"Reducing max_mc_iter to {max_mc_iter}")

#     ebno_dbs = tf.cast(ebno_dbs, dtype.real_dtype)
#     batch_size = tf.cast(batch_size, tf.int32)
#     num_points = tf.shape(ebno_dbs)[0]
#     bit_errors = tf.Variable(   tf.zeros([num_points], dtype=tf.int64),
#                                 dtype=tf.int64)
#     block_errors = tf.Variable( tf.zeros([num_points], dtype=tf.int64),
#                                 dtype=tf.int64)
#     nb_bits = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
#                             dtype=tf.int64)
#     nb_blocks = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
#                             dtype=tf.int64)

#     # track status of simulation (early termination etc.)
#     status = np.zeros(num_points)

#     # measure runtime per SNR point
#     runtime = np.zeros(num_points)

#     # ensure num_target_errors is a tensor
#     if num_target_bit_errors is not None:
#         num_target_bit_errors = tf.cast(num_target_bit_errors, tf.int64)
#     if num_target_block_errors is not None:
#         num_target_block_errors = tf.cast(num_target_block_errors, tf.int64)

#     try:
#         # simulate until a target number of errors is reached
#         for i in tf.range(num_points):
#             runtime[i] = time.perf_counter() # save start time
#             iter_count = -1 # for print in verbose mode
#             for ii in tf.range(max_mc_iter):

#                 iter_count += 1

#                 if run_multigpu: # distributed execution
#                     b, b_hat = _run_distributed(strategy,
#                                                 mc_fun,
#                                                 batch_size,
#                                                 ebno_dbs[i])
#                 else:
#                     outputs = mc_fun(batch_size=batch_size, ebno_db=ebno_dbs[i])
#                     # assume first and second return value is b and b_hat
#                     # other returns are ignored
#                     b = outputs[0]
#                     b_hat = outputs[1]

#                 if soft_estimates:
#                     b_hat = hard_decisions(b_hat)

#                 # count errors
#                 bit_e = count_errors(b, b_hat)
#                 block_e = count_block_errors(b, b_hat)

#                 # count total number of bits
#                 bit_n = tf.size(b)
#                 block_n = tf.size(b[...,-1])

#                 # update variables
#                 bit_errors = tf.tensor_scatter_nd_add(  bit_errors, [[i]],
#                                                     tf.cast([bit_e], tf.int64))
#                 block_errors = tf.tensor_scatter_nd_add(  block_errors, [[i]],
#                                                 tf.cast([block_e], tf.int64))
#                 nb_bits = tf.tensor_scatter_nd_add( nb_bits, [[i]],
#                                                     tf.cast([bit_n], tf.int64))
#                 nb_blocks = tf.tensor_scatter_nd_add( nb_blocks, [[i]],
#                                                 tf.cast([block_n], tf.int64))

#                 cb_state = sim_ber.CALLBACK_CONTINUE
#                 if callback is not None:
#                     cb_state = callback (ii, i, ebno_dbs, bit_errors,
#                                        block_errors, nb_bits,
#                                        nb_blocks)
#                     if cb_state in (sim_ber.CALLBACK_STOP,
#                                     sim_ber.CALLBACK_NEXT_SNR):
#                         # stop runtime timer
#                         runtime[i] = time.perf_counter() - runtime[i]
#                         status[i] = 7 # change internal status for summary
#                         break # stop for this SNR point have been simulated

#                 # print progress summary
#                 if verbose:
#                     # print summary header during first iteration
#                     if i==0 and iter_count==0:
#                         _print_progress(is_final=True,
#                                         rt=0,
#                                         idx_snr=0,
#                                         idx_it=0,
#                                         header_text=header_text)
#                         # print seperator after headline
#                         print('-' * 135)

#                     # evaluate current runtime
#                     rt = time.perf_counter() - runtime[i]
#                     # print current progress
#                     _print_progress(is_final=False, idx_snr=i, idx_it=ii, rt=rt)

#                 # bit-error based stopping cond.
#                 if num_target_bit_errors is not None:
#                     if tf.greater_equal(bit_errors[i], num_target_bit_errors):
#                         status[i] = 3 # change internal status for summary
#                         # stop runtime timer
#                         runtime[i] = time.perf_counter() - runtime[i]
#                         break # enough errors for SNR point have been simulated

#                 # block-error based stopping cond.
#                 if num_target_block_errors is not None:
#                     if tf.greater_equal(block_errors[i],
#                                         num_target_block_errors):
#                         # stop runtime timer
#                         runtime[i] = time.perf_counter() - runtime[i]
#                         status[i] = 4 # change internal status for summary
#                         break # enough errors for SNR point have been simulated

#                 # max iter have been reached -> continue with next SNR point
#                 if iter_count==max_mc_iter-1: # all iterations are done
#                     # stop runtime timer
#                     runtime[i] = time.perf_counter() - runtime[i]
#                     status[i] = 1 # change internal status for summary

#             # print results again AFTER last iteration / early stop (new status)
#             if verbose:
#                 _print_progress(is_final=True,
#                                 idx_snr=i,
#                                 idx_it=iter_count,
#                                 rt=runtime[i])

#             # early stop if no error occurred or target_ber/target_bler reached
#             if early_stop: # only if early stop is active
#                 if block_errors[i]==0:
#                     status[i] = 2 # change internal status for summary
#                     if verbose:
#                         print("\nSimulation stopped as no error occurred " \
#                               f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
#                     break

#                 # check for target_ber / target_bler
#                 ber_true =  bit_errors[i] / nb_bits[i]
#                 bler_true = block_errors[i] / nb_blocks[i]
#                 if ber_true <target_ber:
#                     status[i] = 5 # change internal status for summary
#                     if verbose:
#                         print("\nSimulation stopped as target BER is reached" \
#                               f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
#                     break
#                 if bler_true <target_bler:
#                     status[i] = 6 # change internal status for summary
#                     if verbose:
#                         print("\nSimulation stopped as target BLER is " \
#                               f"reached @ EbNo = {ebno_dbs[i].numpy():.1f} " \
#                               "dB.\n")
#                     break

#             # allow callback to end the entire simulation
#             if cb_state is sim_ber.CALLBACK_STOP:
#                 # stop runtime timer
#                 status[i] = 7 # change internal status for summary
#                 if verbose:
#                     print("\nSimulation stopped by callback function " \
#                           f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
#                 break

#     # Stop if KeyboardInterrupt is detected and set remaining SNR points to -1
#     except KeyboardInterrupt as e:

#         # Raise Interrupt again to stop outer loops
#         if forward_keyboard_interrupt:
#             raise e

#         print("\nSimulation stopped by the user " \
#               f"@ EbNo = {ebno_dbs[i].numpy()} dB.")
#         # overwrite remaining BER / BLER positions with -1
#         for idx in range(i+1, num_points):
#             bit_errors = tf.tensor_scatter_nd_update( bit_errors, [[idx]],
#                                                     tf.cast([-1], tf.int64))
#             block_errors = tf.tensor_scatter_nd_update( block_errors, [[idx]],
#                                                     tf.cast([-1], tf.int64))
#             nb_bits = tf.tensor_scatter_nd_update( nb_bits, [[idx]],
#                                                     tf.cast([1], tf.int64))
#             nb_blocks = tf.tensor_scatter_nd_update( nb_blocks, [[idx]],
#                                                     tf.cast([1], tf.int64))

#     # calculate BER / BLER
#     ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
#     bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)

#     # replace nans (from early stop)
#     ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
#     bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)

#     return ber, bler
import tensorflow as tf
import numpy as np
def sim_ber(mc_fun,
            ebno_dbs,
            batch_size,
            max_mc_iter,
            soft_estimates=False,
            num_target_bit_errors=None,
            num_target_block_errors=None,
            target_ber=None,
            target_bler=None,
            early_stop=True,
            distribute=None,
            forward_keyboard_interrupt=True,
            verbose=True,
            dtype=tf.complex64):


    # distributed execution should not be done in Eager mode
    # XLA mode seems to have difficulties with TF2.13
    @tf.function(jit_compile=False)
    def _run_distributed(strategy, mc_fun, batch_size, ebno_db):
        # use tf.distribute to execute on parallel devices (=replicas)
        outputs_rep = strategy.run(mc_fun,
                                   args=(batch_size, ebno_db))
        # copy replicas back to single device
        b = strategy.gather(outputs_rep[0], axis=0)
        b_hat = strategy.gather(outputs_rep[1], axis=0)
        return b, b_hat
    # target_ber / target_bler only works if early stop is activated
    if target_ber is not None:
        if not early_stop:
            print("Warning: early stop is deactivated. Thus, target_ber " \
                  "is ignored.")
    else:
        target_ber = -1. # deactivate early stopping condition
    if target_bler is not None:
        if not early_stop:
            print("Warning: early stop is deactivated. Thus, target_bler " \
                  "is ignored.")
    else:
        target_bler = -1. # deactivate early stopping condition

    # support multi-device simulations by using the tf.distribute package
    if distribute is None: # disabled per default
        run_multigpu = False
    # use strategy if explicitly provided
    elif isinstance(distribute, tf.distribute.Strategy):
        run_multigpu = True
        strategy = distribute # distribute is already a tf.distribute.strategy
    else:
        run_multigpu = True
        # use all available gpus
        if distribute=="all":
            gpus = tf.config.list_logical_devices('GPU')
        # mask active GPUs if indices are provided
        elif isinstance(distribute, (tuple, list)):
            gpus_avail = tf.config.list_logical_devices('GPU')
            gpus = [gpus_avail[i] for i in distribute if i < len(gpus_avail)]
        else:
            raise ValueError("Unknown value for distribute.")

        # deactivate logging of tf.device placement
        if verbose:
            print("Setting tf.debugging.set_log_device_placement to False.")
        tf.debugging.set_log_device_placement(False)
        # we reduce to the first device by default
        strategy = tf.distribute.MirroredStrategy(gpus,
                            cross_device_ops=tf.distribute.ReductionToOneDevice(
                                                reduce_to_device=gpus[0].name))

    # reduce max_mc_iter if multi_gpu simulations are activated
    if run_multigpu:
        num_replicas = strategy.num_replicas_in_sync
        max_mc_iter = int(np.ceil(max_mc_iter/num_replicas))
        print(f"Distributing simulation across {num_replicas} devices.")
        print(f"Reducing max_mc_iter to {max_mc_iter}")

    ebno_dbs = tf.cast(ebno_dbs, dtype.real_dtype)
    batch_size = tf.cast(batch_size, tf.int32)
    num_points = tf.shape(ebno_dbs)[0]
    bit_errors = tf.Variable(   tf.zeros([num_points], dtype=tf.int64),
                                dtype=tf.int64)
    block_errors = tf.Variable( tf.zeros([num_points], dtype=tf.int64),
                                dtype=tf.int64)
    nb_bits = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)
    nb_blocks = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)

    # track status of simulation (early termination etc.)
    status = np.zeros(num_points)

    # measure runtime per SNR point
    runtime = np.zeros(num_points)

    # ensure num_target_errors is a tensor
    if num_target_bit_errors is not None:
        num_target_bit_errors = tf.cast(num_target_bit_errors, tf.int64)
    if num_target_block_errors is not None:
        num_target_block_errors = tf.cast(num_target_block_errors, tf.int64)

    try:
        # simulate until a target number of errors is reached
        for ii in tf.range(max_mc_iter):
            for i in tf.range(num_points):
                runtime[i] = time.perf_counter() # save start time
                iter_count = -1 # for print in verbose mode
                iter_count += 1

                if run_multigpu: # distributed execution
                    b, b_hat = _run_distributed(strategy,
                                                mc_fun,
                                                batch_size,
                                                ebno_dbs[i])
                else:
                    outputs = mc_fun(batch_size=batch_size, ebno_db=ebno_dbs[i])
                    # assume first and second return value is b and b_hat
                    # other returns are ignored
                    b = outputs[0]
                    b_hat = outputs[1]

                if soft_estimates:
                    b_hat = hard_decisions(b_hat)

                # count errors
                bit_e = count_errors(b, b_hat)
                block_e = count_block_errors(b, b_hat)

                # count total number of bits
                bit_n = tf.size(b)
                block_n = tf.size(b[...,-1])

                # update variables
                bit_errors = tf.tensor_scatter_nd_add(  bit_errors, [[i]],
                                                    tf.cast([bit_e], tf.int64))
                block_errors = tf.tensor_scatter_nd_add(  block_errors, [[i]],
                                                tf.cast([block_e], tf.int64))
                nb_bits = tf.tensor_scatter_nd_add( nb_bits, [[i]],
                                                    tf.cast([bit_n], tf.int64))
                nb_blocks = tf.tensor_scatter_nd_add( nb_blocks, [[i]],
                                                tf.cast([block_n], tf.int64))
            
            # block-error based stopping cond.
                if num_target_block_errors is not None:
                    if tf.reduce_all(tf.greater_equal(block_errors, num_target_block_errors)):
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        status[i] = 4 # change internal status for summary
                        break # enough errors for SNR point have been simulated

                # max iter have been reached -> continue with next SNR point
                if iter_count==max_mc_iter-1: # all iterations are done
                    # stop runtime timer
                    runtime[i] = time.perf_counter() - runtime[i]
                    status[i] = 1 # change internal status for summary
            if (ii+1) % 10 == 0:
                # calculate BER / BLER
                ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
                bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)
                # replace nans (from early stop)
                ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
                bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)
                # broadcast snr if ber is list
                plot_ber(ber, ebno_dbs)
        

    # Stop if KeyboardInterrupt is detected and set remaining SNR points to -1
    except KeyboardInterrupt as e:

        # Raise Interrupt again to stop outer loops
        if forward_keyboard_interrupt:
            raise e
        print("\nSimulation stopped by the user " \
              f"@ EbNo = {ebno_dbs[i].numpy()} dB.")
        # overwrite remaining BER / BLER positions with -1
        for idx in range(i+1, num_points):
            bit_errors = tf.tensor_scatter_nd_update( bit_errors, [[idx]],
                                                    tf.cast([-1], tf.int64))
            block_errors = tf.tensor_scatter_nd_update( block_errors, [[idx]],
                                                    tf.cast([-1], tf.int64))
            nb_bits = tf.tensor_scatter_nd_update( nb_bits, [[idx]],
                                                    tf.cast([1], tf.int64))
            nb_blocks = tf.tensor_scatter_nd_update( nb_blocks, [[idx]],
                                                    tf.cast([1], tf.int64))
    ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
    bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)
    # replace nans (from early stop)
    ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
    bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)

    return ber, bler

sim_ber.CALLBACK_CONTINUE = None
sim_ber.CALLBACK_STOP = 2
sim_ber.CALLBACK_NEXT_SNR = 1

def plot_ber(ber, snr_db):
    """Plot error-rates.

    Input
    -----
    snr_db: ndarray
        Array of floats defining the simulated SNR points.
        Can be also a list of multiple arrays.

    ber: ndarray
        Array of floats defining the BER/BLER per SNR point.
        Can be also a list of multiple arrays.

    legend: str
        Defaults to "". Defining the legend entries. Can be
        either a string or a list of strings.

    ylabel: str
        Defaults to "BER". Defining the y-label.

    title: str
        Defaults to "Bit Error Rate". Defining the title of the figure.

    ebno: bool
        Defaults to True. If True, the x-label is set to
        "EbNo [dB]" instead of "EsNo [dB]".

    is_bler: bool
        Defaults to False. If True, the corresponding curve is dashed.

    xlim: tuple of floats
        Defaults to None. A tuple of two floats defining x-axis limits.

    ylim: tuple of floats
        Defaults to None. A tuple of two floats defining y-axis limits.

    save_fig: bool
        Defaults to False. If True, the figure is saved as `.png`.

    path: str
        Defaults to "". Defining the path to save the figure
        (iff ``save_fig`` is True).

    Output
    ------
        (fig, ax) :
            Tuple:

        fig : matplotlib.figure.Figure
            A matplotlib figure handle.

        ax : matplotlib.axes.Axes
            A matplotlib axes object.
    """
    
    
    if isinstance(ber, list):
        if not isinstance(snr_db, list):
            snr_db = [snr_db]*len(ber)
        # tile snr_db if not list, but ber is list
    line.set_data(snr_db, ber)
    plt.draw()
    plt.pause(1)
    # return fig, ax


NUM_BITS_PER_SYMBOL = 2 # QPSK
class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A keras model of an uncoded transmission over the AWGN channel.

        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.

        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the Keras model initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()

    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)

EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel



is_bler = False
ebno=True
ylabel = "BER"
legend="Uncoded"
fig, ax = plt.subplots(figsize=(16,10))
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_title("AWGN", fontsize=25)
if ebno:
    ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
else:
    ax.set_xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
ax.set_ylabel(ylabel, fontsize=25)
ax.legend(legend, fontsize=20)
# return figure handle
if is_bler:
    line_style = "--"
else:
    line_style = ""
line, = ax.semilogy([], [], line_style, linewidth=2)
ax.grid(which="both")
ber, bler = sim_ber(mc_fun=model_uncoded_awgn,
            ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
            batch_size=BATCH_SIZE,
            max_mc_iter=100,
            soft_estimates=True,
            num_target_bit_errors=None,
            num_target_block_errors=100,
            distribute=None,
            forward_keyboard_interrupt=True,
            verbose=True,
            dtype=tf.complex64)