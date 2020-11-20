import os
import time
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import queue
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)


import kaldi_io
from tf_block import batch_norm_wrapper, prelu
from ze_utils import set_cuda_visible_devices

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
VAR2STD_EPSILON = 0.00001


# noinspection PyAttributeOutsideInit
class Model(object):

    def __init__(self):
        self.graph = None

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

                    # Apply dropout
                    if i != len(kernel_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):

                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim
                    if i != len(embedding_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")

    @staticmethod
    def save_model(sess, output_dir, logger):
        if logger is not None:
            logger.info("Start saving graph ...")
        saver = tf.train.Saver()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = saver.save(sess, os.path.join(output_dir, 'model'))
        with open(os.path.join(output_dir, 'done'), 'wt') as fid:
            fid.write('done')
        if logger is not None:
            logger.info("Graph saved in path: %s" % save_path)

    def load_model(self, sess, input_dir, logger):
        if logger is not None:
            logger.info("Start loading graph ...")
        #saver = tf.train.import_meta_graph(os.path.join(input_dir, 'model.meta'))
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(input_dir, 'model.meta'))
        saver.restore(sess, os.path.join(input_dir, 'model'))
        self.graph = sess.graph
        self.input_x = self.graph.get_tensor_by_name("input_x:0")
        self.input_y = self.graph.get_tensor_by_name("input_y:0")
        self.num_classes = self.input_y.shape[1]
        self.learning_rate = self.graph.get_tensor_by_name("learning_rate:0")
        self.dropout_keep_prob = self.graph.get_tensor_by_name("dropout_keep_prob:0")
        self.phase = self.graph.get_tensor_by_name("phase:0")
        self.loss = self.graph.get_tensor_by_name("loss:0")
        self.optimizer = self.graph.get_operation_by_name("optimizer")
        self.accuracy = self.graph.get_tensor_by_name("accuracy/accuracy:0")
        self.embedding = [None] * 2  # TODO make this more general
        self.embedding[0] = self.graph.get_tensor_by_name("embed_layer-0/scores:0")
        self.embedding[1] = self.graph.get_tensor_by_name("embed_layer-1/scores:0")
        if logger is not None:
            logger.info("Graph restored from path: %s" % input_dir)

    def create_one_hot_output_matrix(self, labels):
        minibatch_size = len(labels)
        one_hot_matrix = np.zeros((minibatch_size, self.num_classes), dtype=np.int32)
        for i, lab in enumerate(labels):
            one_hot_matrix[i, lab] = 1
        return one_hot_matrix

    def print_models_params(self, input_dir, logger=None):
        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            self.load_model(sess, input_dir, logger)
            print('\n\nThe components are:\n')
            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print(v.name)
            print('\n')

    def get_models_weights(self, input_dir, logger=None):
        import h5py
        h5file = os.path.join(input_dir, 'model.h5')
        if os.path.exists(h5file):
            name2weights = {}

            def add2weights(name, mat):
                if not isinstance(mat, h5py.Group):
                    # print('%s  shape: %s' % (name, str(mat.shape)))
                    name2weights[name] = mat.value

            with h5py.File(h5file, 'r') as hf:
                hf.visititems(add2weights)
            return name2weights

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            self.load_model(sess, input_dir, logger)
            name2weights = {}
            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                name2weights[v.name] = sess.run(v)
                print('%s  shape: %s' % (v.name, str(name2weights[v.name].shape)))
            for i in range(5):
                for scope_name in ("frame_level_info_layer-%s" % i, "embed_layer-%s" % i):
                    for var_name in ("mean", "variance"):
                        name = '%s/%s:0' % (scope_name, var_name)
                        try:
                            name2weights[name] = sess.run(self.graph.get_tensor_by_name(name))
                            print('%s  shape: %s' % (name, str(name2weights[name].shape)))
                        except:
                            pass
            with h5py.File(h5file, 'w') as hf:
                for name, mat in name2weights.iteritems():
                    hf.create_dataset(name, data=mat.astype(np.float32))
            return name2weights

    def train_one_iteration(self, data_loader, args, logger):
        learning_rate = args.learning_rate
        print_interval = args.print_interval
        dropout_proportion = args.dropout_proportion
        input_dir = args.input_dir

        output_dir = args.output_dir
        random_seed = 4 * args.random_seed + args.random_seed % 3

        set_cuda_visible_devices(use_gpu=True, logger=logger)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
            if random_seed != 0:
                tf.set_random_seed(random_seed)
            self.load_model(sess, input_dir, logger)

            # Shuffle the data in each epoch
            minibatch_count = data_loader.count
            start_minibatch = 1
            dropout_keep_prob = 1 - dropout_proportion
            total_segments, minibatch_segments = 0, 0
            total_loss, minibatch_loss = 0, 0
            total_objective, minibatch_objective = 0, 0
            total_accuracy, minibatch_accuracy = 0, 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            total_disk_waiting = 0.0
            start_time = time.time()
            for minibatch_idx in range(minibatch_count):
                try:
                    disk_waiting = time.time()
                    batch_data, labels = data_loader.pop()
                    total_disk_waiting += time.time() - disk_waiting
                except queue.Empty:
                    logger.warning('Timeout reach when reading the minibatch index %d' % minibatch_idx)
                    continue
                if batch_data is None:
                    logger.warning('batch_data is None for the minibatch index %d' % minibatch_idx)
                    continue
                batch_labels = self.create_one_hot_output_matrix(labels)
                minibatch_segments += batch_data.shape[0]
                total_segments += batch_data.shape[0]
                total_segments_len += batch_data.shape[1]
                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels,
                             self.dropout_keep_prob: dropout_keep_prob, self.learning_rate: learning_rate,
                             self.phase: True}

                gpu_waiting = time.time()
                _, loss, accuracy = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                total_gpu_waiting += time.time() - gpu_waiting

                objective = -loss
                total_loss += loss
                minibatch_loss += loss
                total_objective += objective
                minibatch_objective += objective
                total_accuracy += accuracy
                minibatch_accuracy += accuracy
                end_minibatch = minibatch_idx + 1
                if end_minibatch % print_interval == 0:
                    cnt = end_minibatch - start_minibatch + 1
                    logger.info("Average training loss for minibatches %d-%d is %.4f over %d segments. Also, the "
                                "average training accuracy for these minibatches is %.4f and the average "
                                "objective function for these minibatches is %.4f. Average DISK waiting: %.1f "
                                "secs and average GPU waiting: %.1f secs for each minibatch." %
                                (start_minibatch, end_minibatch, minibatch_loss / cnt,
                                 minibatch_segments, minibatch_accuracy / cnt, minibatch_objective / cnt,
                                 total_disk_waiting / cnt, total_gpu_waiting / cnt))
                    start_minibatch = end_minibatch + 1
                    minibatch_segments = 0
                    minibatch_loss = 0
                    minibatch_accuracy = 0
                    minibatch_objective = 0
                    total_gpu_waiting = 0.0
                    total_disk_waiting = 0.0

            logger.info("Processed %d segments of average size %d into %d minibatches. Avg minibatch size was %d." %
                        (total_segments, total_segments_len / minibatch_count, minibatch_count,
                         total_segments / minibatch_count))

            logger.info("Overall average training loss is %.4f over %d segments. Also, the overall "
                        "average training accuracy is %.4f." % (total_loss / minibatch_count,
                                                                total_segments, total_accuracy / minibatch_count))

            logger.info("Overall average objective function is %.4f over %d segments." %
                        (total_objective / minibatch_count, total_segments))

            Model.save_model(sess, output_dir, logger)

            logger.info("Elapsed time for processing whole training minibatches is %.2f minutes." %
                        ((time.time() - start_time) / 60.0))

    def eval(self, data_loader, input_dir, use_gpu, logger):

        set_cuda_visible_devices(use_gpu=use_gpu, logger=logger)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            self.load_model(sess, input_dir, logger)

            # Shuffle the data in each epoch
            minibatch_count = data_loader.count
            total_segments = 0
            total_loss = 0
            total_accuracy = 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            total_disk_waiting = 0.0
            start_time = time.time()
            for minibatch_idx in range(minibatch_count):
                try:
                    disk_waiting = time.time()
                    batch_data, labels = data_loader.pop()
                    total_disk_waiting += time.time() - disk_waiting
                except queue.Empty:
                    logger.warning('Timeout reach when reading minibatch index %d' % minibatch_idx)
                    continue
                if batch_data is None:
                    logger.warning('batch_data is None for minibatch index %d' % minibatch_idx)
                    continue
                batch_labels = self.create_one_hot_output_matrix(labels)
                total_segments += batch_data.shape[0]
                total_segments_len += batch_data.shape[1]
                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 1.0,
                             self.phase: False}

                gpu_waiting = time.time()
                loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                total_gpu_waiting += time.time() - gpu_waiting
                total_loss += loss
                total_accuracy += accuracy

            logger.info("Processed %d segments of average size %d into %d minibatches. Avg minibatch size was %d." %
                        (total_segments, total_segments_len / minibatch_count, minibatch_count,
                         total_segments / minibatch_count))

            logger.info("Overall average loss is %.4f over %d segments. Also, the overall "
                        "average accuracy is %.4f." % (total_loss / minibatch_count, total_segments,
                                                       total_accuracy / minibatch_count))

            logger.info("Elapsed time for processing whole training minibatches is %.2f minutes." %
                        ((time.time() - start_time) / 60.0))

    def make_embedding(self, input_stream, output_stream, model_dir, min_chunk_size, chunk_size, use_gpu, logger):
        start_time = time.time()
        set_cuda_visible_devices(use_gpu=use_gpu, logger=logger)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        if not use_gpu:
            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2

        with tf.Session(config=config) as sess:
            self.load_model(sess, model_dir, logger)

            total_segments = 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            num_fail = 0
            num_success = 0
            for key, mat in kaldi_io.read_mat_ark(input_stream):
                logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
                total_segments += 1

                num_rows = mat.shape[0]
                if num_rows == 0:
                    logger.warning("Zero-length utterance: '%s'" % key)
                    num_fail += 1
                    continue

                if num_rows < min_chunk_size:
                    logger.warning("Minimum chunk size of %d is greater than the number of rows in utterance: %s" %
                                   (min_chunk_size, key))
                    num_fail += 1
                    continue
                this_chunk_size = chunk_size
                if num_rows < chunk_size:
                    logger.info("Chunk size of %d is greater than the number of rows in utterance: %s, "
                                "using chunk size of %d" % (chunk_size, key, num_rows))
                    this_chunk_size = num_rows
                elif chunk_size == -1:
                    this_chunk_size = num_rows

                num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))
                # logger.info("num_chunks: %d" % num_chunks)
                xvector_avg = 0
                tot_weight = 0.0

                for chunk_idx in range(num_chunks):
                    # If we're nearing the end of the input, we may need to shift the
                    # offset back so that we can get this_chunk_size frames of input to
                    # the nnet.
                    offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
                    if offset < min_chunk_size:
                        continue
                    # logger.info("offset: %d" % offset)
                    sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
                    data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
                    total_segments_len += sub_mat.shape[0]
                    feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
                    gpu_waiting = time.time()
                    xvector = sess.run(self.embedding[1], feed_dict=feed_dict)
                    xvector = xvector[0]
                    # logger.info("xvector: %s" % str(xvector.shape))
                    total_gpu_waiting += time.time() - gpu_waiting
                    tot_weight += offset
                    xvector_avg += offset * xvector

                xvector_avg /= tot_weight
                kaldi_io.write_vec_flt(output_stream, xvector_avg, key=key)
                num_success += 1

            logger.info("Processed %d features of average size %d frames. Done %d and failed %d" %
                        (total_segments, total_segments_len / total_segments, num_success, num_fail))

            logger.info("Total time for neural network computations is %.2f minutes." %
                        (total_gpu_waiting / 60.0))

            logger.info("Elapsed time for extracting whole embeddings is %.2f minutes." %
                        ((time.time() - start_time) / 60.0))


    def make_embedding_frame_num(self, input_stream, output_stream, model_dir, min_chunk_size, chunk_size, use_gpu, logger):
        frame_num = 300
        offset = 100
        start_time = time.time()
        set_cuda_visible_devices(use_gpu=use_gpu, logger=logger)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        if not use_gpu:
            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2

        with tf.Session(config=config) as sess:
            self.load_model(sess, model_dir, logger)

            total_segments = 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            num_fail = 0
            num_success = 0
            for key, mat in kaldi_io.read_mat_ark(input_stream):
                logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
                total_segments += 1

                num_rows = mat.shape[0]
                if num_rows == 0:
                    logger.warning("Zero-length utterance: '%s'" % key)
                    num_fail += 1
                    continue

                this_frame_num = frame_num
                if num_rows < frame_num:
                    logger.info("Frame number of %d is greater than the number of rows in utterance: %s, "
                                "using frame number of %d" % (frame_num, key, num_rows))
                    this_frame_num = num_rows           
                
                
                sub_mat = mat[offset: offset+this_frame_num, :]
                data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
                total_segments_len += sub_mat.shape[0]
                feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
                gpu_waiting = time.time()
                xvector = sess.run(self.embedding[1], feed_dict=feed_dict)
                xvector = xvector[0]
                total_gpu_waiting += time.time() - gpu_waiting

                kaldi_io.write_vec_flt(output_stream, xvector, key=key)
                num_success += 1

            logger.info("Processed %d features of average size %d frames. Done %d and failed %d" %
                        (total_segments, total_segments_len / total_segments, num_success, num_fail))

            logger.info("Total time for neural network computations is %.2f minutes." %
                        (total_gpu_waiting / 60.0))

            logger.info("Elapsed time for extracting whole embeddings is %.2f minutes." %
                        ((time.time() - start_time) / 60.0))





# noinspection PyAttributeOutsideInit
class ModelWithoutDropout(Model):

    def __init__(self):
        super(ModelWithoutDropout, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [128, 128, 128, 128, 3 * 128]
        kernel_sizes = [3, 3, 5, 1, 1]
        embedding_sizes = [128, 128]

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelWithoutDropoutTdnn(Model):

    def __init__(self):
        super(ModelWithoutDropoutTdnn, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 3, 3, 1, 1]
        embedding_sizes = [512, 512]
        dilation_rates = [1, 2, 3, 1, 1]

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.convolution(h, w, dilation_rate=[dilation_rates[i]], padding="SAME",
                                             name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelWithoutDropoutPRelu(Model):

    def __init__(self):
        super(ModelWithoutDropoutPRelu, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = prelu(h, shared=False)
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = prelu(h, shared=False)
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelL2LossWithoutDropoutPRelu(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutPRelu, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            l2_loss = tf.constant(0.0)

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = prelu(h, shared=False)
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = prelu(h, shared=False)
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelL2LossWithoutDropoutLRelu(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutLRelu, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            l2_loss = tf.constant(0.0)

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelL2LossWithoutDropoutLReluAttention(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttention, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            l2_loss = tf.constant(0.0)

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            
            Model.save_model(sess, output_dir, logger)
            sess.graph.finalize()
        if logger is not None:
            logger.info("Building finished.")


# noinspection PyAttributeOutsideInit
class ModelL2LossWithoutDropoutReluHeInit(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutReluHeInit, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            l2_loss = tf.constant(0.0)

            h = self.input_x

            # Frame level information Layer
            prev_dim = input_feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    # he_normal init
                    fan_in = kernel_size * prev_dim
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=tf.sqrt(2.0 / fan_in)), name="w")
                    # he_uniform init
                    limit = tf.sqrt(6.0 / fan_in)
                    b = tf.Variable(tf.random_uniform([layer_size], minval=-limit, maxval=limit), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    # he_normal init
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=tf.sqrt(2.0 / prev_dim)), name="w")
                    # he_uniform init
                    limit = tf.sqrt(6.0 / prev_dim)
                    b = tf.Variable(tf.random_uniform([out_dim], minval=-limit, maxval=limit), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                # glorot_normal: https://keras.io/initializers/
                stddev = tf.sqrt(2.0 / (prev_dim + num_classes))
                w = tf.Variable(tf.truncated_normal([prev_dim, num_classes], stddev=stddev), name="w")
                # glorot_uniform
                limit = tf.sqrt(6.0 / (prev_dim + num_classes))
                b = tf.Variable(tf.random_uniform([num_classes], minval=-limit, maxval=limit), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")

  
# noinspection PyAttributeOutsideInit
class ModelL2LossWithoutDropoutLReluAttention_63dim(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttention_63dim, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            l2_loss = tf.constant(0.0)

            h = self.input_x[:, :, 0:mfcc_dim]
            
            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)
            sess.graph.finalize()

        if logger is not None:
            logger.info("Building finished.")
            
  

# noinspection PyAttributeOutsideInit           
class ModelL2LossWithoutDropoutLReluAttentionPhoneme(Model):

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhoneme, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [96, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]
            posterior=self.input_x[:, :, mfcc_dim:]
            num_phoneme=posterior.shape[2].value


            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    if i == 0:
                        kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]


                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')

                        """
                        h_splits = tf.split(h, num_or_size_splits=layer_size, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=3)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=3)
                            h_reshape=tf.concat([h_reshape, h_q], axis=3)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape)  #[None, None, layer_size]
                        """
                        
                        h_splits = tf.split(h, num_or_size_splits=num_phoneme, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=2)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=2)
                            h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape, name='layeroutput')  #[None, None, layer_size]
                        
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

                    else: 
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size


            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")

# noinspection PyAttributeOutsideInit  
class ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [96, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]
            poster=self.input_x[:, :, mfcc_dim:]
            
            
            poster_temp=tf.gather(poster, indices=[0,33,34], axis=2)
            posterior=tf.concat((tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2), tf.expand_dims(poster[:,:,1],axis=2)),axis=2)

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,2],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[3,4,5,12,25], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[6,8,14], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[7,18], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[13,15,32,29], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            poster_temp=tf.gather(poster, indices=[9,35,38,39], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)

            poster_temp=tf.gather(poster, indices=[10,11], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,16],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,17],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[19,26], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[20,37], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[21,23], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,22],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,24],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[27,36], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,28],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,30],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,31],axis=2)),axis=2)
            

            
            num_phoneme=posterior.shape[2].value
            print(num_phoneme)


            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    if i == 0:
                        kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]


                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')


                        """
                        h_splits = tf.split(h, num_or_size_splits=layer_size, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=3)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=3)
                            h_reshape=tf.concat([h_reshape, h_q], axis=3)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape)  #[None, None, layer_size]
                        """
                        
                        
                        h_splits = tf.split(h, num_or_size_splits=num_phoneme, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=2)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=2)
                            h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape, name='layeroutput')  #[None, None, layer_size]
                        
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

                    else: 
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size




            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")

# noinspection PyAttributeOutsideInit  
class ModelL2LossWithoutDropoutLReluAttentionPhonemeClusterPequal(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeClusterPequal, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [80, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]

            
            num_phoneme=19

            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    if i == 0:
                        kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]


                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')


                        
                        h_splits = tf.split(h, num_or_size_splits=num_phoneme, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=2)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=2)
                            h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]

                        h = tf.reduce_mean( h_reshape, axis=2, name='layeroutput')  #[None, None, layer_size]
                        
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

                    else: 
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size




            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")




class ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster2layer(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster2layer, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [96, 168, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]
            poster=self.input_x[:, :, mfcc_dim:]
                
            
            poster_temp=tf.gather(poster, indices=[0,33,34], axis=2)
            posterior=tf.concat((tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2), tf.expand_dims(poster[:,:,1],axis=2)),axis=2)

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,2],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[3,4,5,12,25], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[6,8,14], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[7,18], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[13,15,32,29], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            poster_temp=tf.gather(poster, indices=[9,35,38,39], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)

            poster_temp=tf.gather(poster, indices=[10,11], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,16],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,17],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[19,26], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[20,37], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[21,23], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,22],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,24],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[27,36], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,28],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,30],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,31],axis=2)),axis=2)
            
            
            num_phoneme=posterior.shape[2].value
            print(num_phoneme)


            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    if i == 0 or i==1:
                        kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]


                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')


                        """
                        h_splits = tf.split(h, num_or_size_splits=layer_size, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=3)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=3)
                            h_reshape=tf.concat([h_reshape, h_q], axis=3)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape)  #[None, None, layer_size]
                        """
                        
                        
                        h_splits = tf.split(h, num_or_size_splits=num_phoneme, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=2)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=2)
                            h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape, name='layeroutput')  #[None, None, layer_size]
                        
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

                    else: 
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size




            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                #init = tf.initializers.GlorotUniform()
                #w = tf.Variable(init(shape=[prev_dim, num_classes]))
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")

            


# noinspection PyAttributeOutsideInit  
class ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitask(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitask, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            l2_loss_phoneme = tf.constant(0.0)
            h_share = self.input_x[:, :, 0:mfcc_dim]
            poster=self.input_x[:, :, mfcc_dim:]
            
            
            poster_temp=tf.gather(poster, indices=[0,33,34], axis=2)
            posterior=tf.concat((tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2), tf.expand_dims(poster[:,:,1],axis=2)),axis=2)

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,2],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[3,4,5,12,25], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[6,8,14], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[7,18], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[13,15,32,29], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            poster_temp=tf.gather(poster, indices=[9,35,38,39], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)

            poster_temp=tf.gather(poster, indices=[10,11], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,16],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,17],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[19,26], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[20,37], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[21,23], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,22],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,24],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[27,36], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,28],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,30],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,31],axis=2)),axis=2)
           
            
            num_phoneme=posterior.shape[2].value
            print(num_phoneme)
            
            
            # Frame level TDNN Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                
                #if i == 0 or i==1 or i==2 or i==3:
                if i == 0 or i== 1:
                
                    with tf.variable_scope("frame_level_share-%s" % i):
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                        conv = tf.nn.conv1d(h_share, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h_share = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h_share = tf.nn.leaky_relu(h_share, alpha=0.2, name='lrelu')
                        h_share = batch_norm_wrapper(h_share, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size
                    
                else:
                    h=h_share
                    h_phoneme = h_share
                    prev_dim_phoneme = prev_dim
                    with tf.variable_scope("frame_level_phoneme-%s" % i):
                        kernel_shape = [kernel_size, prev_dim_phoneme, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim_phoneme = layer_size
                
                    with tf.variable_scope("frame_level_speaker-%s" % i):
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                        conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h_phoneme = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h_phoneme = tf.nn.leaky_relu(h_phoneme, alpha=0.2, name='lrelu')
                        h_phoneme = batch_norm_wrapper(h_phoneme, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

            #### phoneme recognition network
            # frame level DNN layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("phoneme_embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([1, prev_dim_phoneme, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")
                    
                    conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h_phoneme = tf.nn.bias_add(conv, b)
 

                    # Apply L2 loss
                    if i == 0:
                        l2_loss_phoneme += 0.1 * tf.nn.l2_loss(w)
                        l2_loss_phoneme += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss_phoneme += tf.nn.l2_loss(w)
                        l2_loss_phoneme += tf.nn.l2_loss(b)

                    h_phoneme = tf.nn.leaky_relu(h_phoneme, alpha=0.2, name='lrelu')
                    h_phoneme = batch_norm_wrapper(h_phoneme, decay=0.95, is_training=self.phase)

                    prev_dim_phoneme = out_dim

            # Softmax
            with tf.variable_scope("phoneme_output"):
                w = tf.get_variable("w", shape=[1, prev_dim_phoneme, num_phoneme],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([1, prev_dim_phoneme,num_phoneme], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_phoneme]), name="b")

                # Apply L2 loss
                l2_loss_phoneme += tf.nn.l2_loss(w)
                l2_loss_phoneme += tf.nn.l2_loss(b)
                
                
                conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                scores_phoneme = tf.nn.bias_add(conv, b, name="scores")

                #scores_phoneme = tf.nn.xw_plus_b(h_phoneme, w, b, name="scores")

                predictions_phoneme = tf.argmax(scores_phoneme, 1, name="predictions")

            losses_phoneme = tf.nn.softmax_cross_entropy_with_logits(logits=scores_phoneme, labels=posterior)
            
            
            # Normal loss function
            loss_phoneme = tf.reduce_mean(losses_phoneme, name="phoneme_loss")
            loss_phoneme = tf.reduce_mean(loss_phoneme, name="phoneme_loss")


            #### speaker recognition network
            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            

            self.loss = tf.reduce_mean(loss + beta * l2_loss + (loss_phoneme + beta * l2_loss_phoneme ), name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")
            
            
            
            

class ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitaskPequal(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitaskPequal, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [96, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            l2_loss_phoneme = tf.constant(0.0)
            h_share = self.input_x[:, :, 0:mfcc_dim]
            poster=self.input_x[:, :, mfcc_dim:]
            
            
            poster_temp=tf.gather(poster, indices=[0,33,34], axis=2)
            posterior=tf.concat((tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2), tf.expand_dims(poster[:,:,1],axis=2)),axis=2)

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,2],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[3,4,5,12,25], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[6,8,14], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[7,18], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[13,15,32,29], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            poster_temp=tf.gather(poster, indices=[9,35,38,39], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)

            poster_temp=tf.gather(poster, indices=[10,11], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,16],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,17],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[19,26], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[20,37], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[21,23], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,22],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,24],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[27,36], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,28],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,30],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,31],axis=2)),axis=2)
           
            
            num_phoneme=posterior.shape[2].value
            
            
            
            # Frame level TDNN Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                
                if i == 0 or i==1 or i==2 or i==3:
                #if i == 0 or i==1 or i==2:
                #if i == 0 or i== 1:
                    if i==0:
                        with tf.variable_scope("frame_level_share-%s" % i):
                            kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                            w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                            b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")
    
                            conv = tf.nn.conv1d(h_share, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                            h_share = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]
    
    
                            # Apply nonlinearity and BN
                            h_share = tf.nn.leaky_relu(h_share, alpha=0.2, name='lrelu')
    
    
                            
                            h_splits = tf.split(h_share, num_or_size_splits=num_phoneme, axis=2)
                            h_reshape=tf.expand_dims(h_splits[0], axis=2)
                            for q in range(1,len(h_splits)):
                                h_q=tf.expand_dims(h_splits[q], axis=2)
                                h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]
    
                            h_share = tf.reduce_mean( h_reshape, axis=2, name='layeroutput')  #[None, None, layer_size]
                            
                            h_share = batch_norm_wrapper(h_share, decay=0.95, is_training=self.phase)
    
                            prev_dim = layer_size
                        
                    else:
                        with tf.variable_scope("frame_level_share-%s" % i):
                            kernel_shape = [kernel_size, prev_dim, layer_size]
                            w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                            b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                            conv = tf.nn.conv1d(h_share, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                            h_share = tf.nn.bias_add(conv, b)

                            # Apply nonlinearity and BN
                            h_share = tf.nn.leaky_relu(h_share, alpha=0.2, name='lrelu')
                            h_share = batch_norm_wrapper(h_share, decay=0.95, is_training=self.phase)

                            prev_dim = layer_size
                    
                else:
                    h=h_share
                    h_phoneme = h_share
                    prev_dim_phoneme = prev_dim
                    with tf.variable_scope("frame_level_speaker-%s" % i):
                        kernel_shape = [kernel_size, prev_dim_phoneme, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim_phoneme = layer_size
                
                    with tf.variable_scope("frame_level_phoneme-%s" % i):
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
    
                        conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h_phoneme = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h_phoneme = tf.nn.leaky_relu(h_phoneme, alpha=0.2, name='lrelu')
                        h_phoneme = batch_norm_wrapper(h_phoneme, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

            #### phoneme recognition network
            # frame level DNN layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("phoneme_embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([1, prev_dim_phoneme, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")
                    
                    conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h_phoneme = tf.nn.bias_add(conv, b)
 

                    # Apply L2 loss
                    if i == 0:
                        l2_loss_phoneme += 0.1 * tf.nn.l2_loss(w)
                        l2_loss_phoneme += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss_phoneme += tf.nn.l2_loss(w)
                        l2_loss_phoneme += tf.nn.l2_loss(b)

                    h_phoneme = tf.nn.leaky_relu(h_phoneme, alpha=0.2, name='lrelu')
                    h_phoneme = batch_norm_wrapper(h_phoneme, decay=0.95, is_training=self.phase)

                    prev_dim_phoneme = out_dim

            # Softmax
            with tf.variable_scope("phoneme_output"):
                w = tf.get_variable("w", shape=[1, prev_dim_phoneme, num_phoneme],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([1, prev_dim_phoneme,num_phoneme], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_phoneme]), name="b")

                # Apply L2 loss
                l2_loss_phoneme += tf.nn.l2_loss(w)
                l2_loss_phoneme += tf.nn.l2_loss(b)
                
                
                conv = tf.nn.conv1d(h_phoneme, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                scores_phoneme = tf.nn.bias_add(conv, b, name="scores")

                #scores_phoneme = tf.nn.xw_plus_b(h_phoneme, w, b, name="scores")

                predictions_phoneme = tf.argmax(scores_phoneme, 1, name="predictions")

            losses_phoneme = tf.nn.softmax_cross_entropy_with_logits(logits=scores_phoneme, labels=posterior)
            
            
            # Normal loss function
            loss_phoneme = tf.reduce_mean(losses_phoneme, name="phoneme_loss")
            loss_phoneme = tf.reduce_mean(loss_phoneme, name="phoneme_loss")


            #### speaker recognition network
            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            

            self.loss = tf.reduce_mean(loss + beta * l2_loss + (loss_phoneme + beta * l2_loss_phoneme ), name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")
            
     

class ModelL2LossWithoutDropoutLReluAttentionPhonemeVector(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeVector, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [512, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]
            poster=self.input_x[:, :, mfcc_dim:]
            
            
            poster_temp=tf.gather(poster, indices=[0,33,34], axis=2)
            posterior=tf.concat((tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2), tf.expand_dims(poster[:,:,1],axis=2)),axis=2)

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,2],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[3,4,5,12,25], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[6,8,14], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[7,18], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[13,15,32,29], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            poster_temp=tf.gather(poster, indices=[9,35,38,39], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)

            poster_temp=tf.gather(poster, indices=[10,11], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,16],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,17],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[19,26], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[20,37], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[21,23], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,22],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,24],axis=2)),axis=2)
            
            poster_temp=tf.gather(poster, indices=[27,36], axis=2)
            posterior=tf.concat((posterior,tf.expand_dims(tf.reduce_sum(poster_temp,axis=2),axis=2)),axis=2)
            

            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,28],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,30],axis=2)),axis=2)
            posterior=tf.concat((posterior, tf.expand_dims(poster[:,:,31],axis=2)),axis=2)
           

            num_phoneme=posterior.shape[2].value

            h=tf.concat((h, posterior),axis=2)
            feature_dim=h.shape[2].value
            
            
            # Frame level information Layer
            prev_dim = feature_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                #w = tf.Variable(tf.truncated_normal([prev_dim,num_classes], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")




class ModelL2LossWithoutDropoutLReluAttentionPatten(Model):  

    def __init__(self):
        super(ModelL2LossWithoutDropoutLReluAttentionPhonemeClusterPatten, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        layer_sizes = [96, 512, 512, 512, 6 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        beta = 0.0002
        mfcc_dim=23

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            
            l2_loss = tf.constant(0.0)
            h = self.input_x[:, :, 0:mfcc_dim]
            
            num_phoneme=19
            print(num_phoneme)


            # Frame level information Layer
            prev_dim = mfcc_dim
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    if i == 0:
                        
                        # apply attention, calculate posterior
                        
                        w_att = tf.get_variable("w_att", shape=[1, prev_dim, num_phoneme],
                                    initializer=tf.contrib.layers.xavier_initializer())
                        #w_att = tf.Variable(tf.truncated_normal([1, prev_dim,num_phoneme], stddev=0.1), name="w_att")
                        b_att = tf.Variable(tf.constant(0.1, shape=[num_phoneme]), name="b_att")

                               
                        conv_att = tf.nn.conv1d(h, w_att, stride=1, padding="SAME", name="conv_att-layer-%s" % i)
                        scores_att = tf.nn.bias_add(conv_att, b_att, name="scores_att")


                        posterior = tf.nn.softmax(scores_att, axis=2, name="posterior")
                        
                        
                        # TDNN layer
                        
                        kernel_shape = [kernel_size, prev_dim, num_phoneme*layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_phoneme*layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)     # shape [None, None, num_phoneme*layer_size]


                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')



                        """
                        h_splits = tf.split(h, num_or_size_splits=layer_size, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=3)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=3)
                            h_reshape=tf.concat([h_reshape, h_q], axis=3)    #[None, None, num_phoneme, layer_size]

                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape)  #[None, None, layer_size]
                        """
                        
                        
                        h_splits = tf.split(h, num_or_size_splits=num_phoneme, axis=2)
                        h_reshape=tf.expand_dims(h_splits[0], axis=2)
                        for q in range(1,len(h_splits)):
                            h_q=tf.expand_dims(h_splits[q], axis=2)
                            h_reshape=tf.concat([h_reshape, h_q], axis=2)    #[None, None, num_phoneme, layer_size]

                        
                        
                        h = tf.einsum('ijk,ijkl->ijl', posterior, h_reshape, name='layeroutput')  #[None, None, layer_size]
                        
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size

                    else: 
                        kernel_shape = [kernel_size, prev_dim, layer_size]
                        w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                        b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                        conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                        h = tf.nn.bias_add(conv, b)

                        # Apply nonlinearity and BN
                        h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                        h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                        prev_dim = layer_size




            prev_dim /= 2
            prev_dim = int(prev_dim)

            # apply self attention
            with tf.variable_scope("attention"):
                b = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="b")
                v = tf.Variable(tf.constant(0.1, shape=[prev_dim]), name="v")
                # Note: the dimension of the w needs more experiments, here we simply use a square matrix
                w = tf.Variable(tf.truncated_normal([prev_dim, prev_dim], stddev=0.1), name="w")
                h1, h2 = tf.split(h, 2, axis=2)
                non_linearity = tf.nn.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl', h1, w), b), name="non_linearity")
                attention = tf.nn.softmax(tf.einsum('ijk,k->ij', non_linearity, v), name="attention")

            h_m = tf.einsum('ijk,ij->ik', h2, attention)
            h_s = tf.subtract(tf.einsum('ijk,ij->ik', tf.square(h2), attention), tf.square(h_m), name="stats_var")
            h = tf.concat([h_m, tf.sqrt(h_s + VAR2STD_EPSILON)], 1)

            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    # Apply L2 loss
                    if i == 0:
                        l2_loss += 0.1 * tf.nn.l2_loss(w)
                        l2_loss += 0.1 * tf.nn.l2_loss(b)
                    else:
                        l2_loss += tf.nn.l2_loss(w)
                        l2_loss += tf.nn.l2_loss(b)

                    h = tf.nn.leaky_relu(h, alpha=0.2, name='lrelu')
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                # Apply L2 loss
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)

            # Normal loss function
            loss = tf.reduce_mean(losses, name="orig_loss")
            self.loss = tf.reduce_mean(loss + beta * l2_loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)

        if logger is not None:
            logger.info("Building finished.")
