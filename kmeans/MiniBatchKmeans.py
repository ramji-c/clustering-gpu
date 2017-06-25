# coding=utf-8
""" MiniBatch K-means implementation using tensorflow
    __author__: Ramji C
    Date: 21-May-2017
    Sequential algorithm:
    Given: k, mini-batch size b, iterations t, data set X
    Initialize each c ∈ C with an x picked randomly from X
     v ← 0
     for i = 1 to t do
           M ← b examples picked randomly from X
           for x ∈ M do
                 d[x] ← f (C, x)           // Cache the center nearest to x
           end for
           for x ∈ M do
                 c ← d[x]                    // Get cached center for this x
                 v[c] ← v[c] + 1         // Update per-center counts
                 η ← 1 / v[c]              // Get per-center learning rate
                 c ← (1 − η)c + ηx      // Take gradient step
           end for
     end for"""
import tensorflow as tf
import random
import numpy as np
import argparse
import os
import pickle


class MiniBatchKMeans:
    """MiniBatch K-means algorithm - implemented using tensorflow (GPU)
        requires: cudaToolkitv8.0, cudaDNNv5.1, tensorflow-gpu and CUDA enabled GPU"""

    def __init__(self, input_file, model_dir, k, n_iter, batch_size):
        """Initialization function
        :param input_file file containing data to be clustered
        :param model_dir directory where model file should be stored
        :param k number of clusters
        :param n_iter number of iterations
        :param batch_size size of MiniBatch"""

        # initialize tensorflow graph
        self.graph = tf.get_default_graph()

        # initialize input params
        self.input_file = input_file
        self.model_dir = model_dir
        self.k = k
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_samples, self.n_features = self.load_data(self.input_file)

        # initialize tensorflow params
        tf.logging.set_verbosity(tf.logging.INFO)
        self.DTYPE = tf.int32
        self.centroids = tf.Variable(initial_value=self.init_centroids(),
                                     dtype=tf.float64,
                                     name="centroids")
        self.data_queue_1 = tf.FIFOQueue(self.batch_size, tf.float64, name="dataQueue1")
        self.data_queue_2 = tf.FIFOQueue(self.batch_size, tf.float64, name="dataQueue2")
        self.inertia = tf.Variable(initial_value=0,
                                   dtype=tf.float64,
                                   name="inertia")
        self.centroid_cache = tf.Variable(initial_value=np.zeros([self.batch_size]),
                                          dtype=self.DTYPE,
                                          expected_shape=[self.batch_size],
                                          validate_shape=True,
                                          name="centroidCache")
        self.samples_per_centroid = tf.Variable(initial_value=np.zeros([self.k]),
                                                dtype=self.DTYPE,
                                                expected_shape=[self.k],
                                                validate_shape=True,
                                                name="perCentroidCount")
        self.learning_rate = tf.Variable(initial_value=np.zeros([self.k]),
                                         dtype=tf.float64,
                                         expected_shape=[self.k],
                                         validate_shape=True,
                                         name="learningRate")

    def validate_input(self):
        """validate input file"""
        # TODO: input file validation
        pass

    def load_data(self, input_file):
        """load input data from file and return sample count. Input file should be a numpy ndarray
            :return number of samples loaded"""
        with open(input_file, 'rb') as filehandle:
            self.data = pickle.load(filehandle)
        return self.data.shape

    def _loop_cond(self, itr):
        """callable loop condition checker for tf.while_loop
            :param itr loop control variable
            :param centroids current centroids
            :return 'True' if loop control variable is under limit, 'False' otherwise"""
        return tf.Print(tf.less(itr, self.n_iter), [itr], message="iteration: ")

    def _loop_body(self, itr):
        """body of tf.while_loop. contains the core logic of MiniBatch K-means
            :param itr loop control variable
            :returns incremented loop variable"""
        # reset inertia
        reset_inertia = tf.assign(ref=self.inertia, value=0, name="resetInertia")
        # sample mini-batch for this iteration and add to data queues
        mini_batch = self.sample_minibatch()
        with tf.control_dependencies([mini_batch, reset_inertia]):
            enqueue_op1 = self.data_queue_1.enqueue_many(mini_batch, name="putToQueue1")
            # enqueue_op2 = self.data_queue_2.enqueue_many(mini_batch, name="putToQueue2")
        # cache nearest centroids
        with tf.control_dependencies([enqueue_op1]):
            updated_centroid_cache = self.cache_centroids()
            # updated_centroid_cache = tf.Print(updated_centroid_cache, [updated_centroid_cache],
            #                                   summarize=self.batch_size, message="centroid cache: ")
        # compute new centroids
        with tf.control_dependencies([updated_centroid_cache]):
            updated_centroids = tf.map_fn(fn=self.update_centroids,
                                          elems=updated_centroid_cache,
                                          dtype=self.DTYPE,
                                          parallel_iterations=1,
                                          back_prop=False,
                                          swap_memory=True,
                                          name="updateCentroids")
            # updated_centroids = tf.Print(updated_centroids, [updated_centroids], summarize=self.k * self.n_features)
            # print current inertia
            print_inertia = tf.Print(self.inertia, [self.inertia], message="inertia: ")
        # TODO break loop if inertia is unchanged for 3 or more iterations
        # return incremented loop counter
        with tf.control_dependencies([updated_centroids, print_inertia]):
            return tf.add(itr, 1)

    def sample_minibatch(self):
        """sample mini-batch from samples
            :returns Tensor containing the current mini-batch"""
        sample_indices = [random.randrange(self.n_samples) for i in range(self.batch_size)]
        return tf.concat([self.data.getrow(i).toarray() for i in sample_indices], axis=0, name="miniBatch")

    def init_centroids(self):
        """Initialize centroids by selecting 'k' data points from input
            :returns Tensor containing initial centroids"""
        centroid_indices = [random.randrange(self.n_samples) for i in range(self.k)]
        return tf.concat([self.data.getrow(i).toarray() for i in centroid_indices], axis=0, name="centroids")

    def cache_centroids(self):
        """find the closest centroid for each sample in mini-batch
            :returns updated centroid cache"""
        centroid_indices = []
        # find nearest centroid for each sample
        for i in range(self.batch_size):
            sample = self.data_queue_1.dequeue(name="getNextSample")
            # add sample to data queue 2
            enqueue_op = self.data_queue_2.enqueue(sample, name="addToQueue2")
            with tf.control_dependencies([enqueue_op]):
                centroid_indices.append(tf.cast(self.find_nearest_centroid(sample),
                                                dtype=tf.int32))
        # update centroid cache
        return tf.scatter_update(ref=self.centroid_cache,
                                 indices=[i for i in range(0, self.batch_size)],
                                 updates=centroid_indices,
                                 name="updateCentroidCache")

    def find_nearest_centroid(self, sample):
        """find the nearest centroid for the given sample.
            uses an inner function to calculate pairwise distances - squared euclidean distance
            :param sample data point whose nearest centroid is to be found
            :returns index of the nearest centroid"""

        def calc_dist_to_centroid(centroid):
            """calculate squared euclidean distance between input centroid and current data point
            :returns pairwise distance between centroid and data point"""
            return tf.reduce_sum(tf.squared_difference(sample, centroid), name="sumOfSquares")

        distances_ = tf.map_fn(calc_dist_to_centroid, self.centroids, parallel_iterations=self.k, swap_memory=True)
        # update inertia
        self.inertia = tf.assign_add(ref=self.inertia, value=tf.reduce_min(distances_), name="updateInertia")
        # print_inertia = tf.Print(self.inertia, [tf.reduce_min(distances_), self.inertia])
        with tf.control_dependencies([self.inertia]):
            return tf.argmin(input=distances_, axis=0, name="nearestCentroid")

    def update_centroids(self, centroid):
        """compute updated values for centroids as mean of assigned samples
            :param centroid - centroid to be updated
            :returns updated centroids Tensor"""
        sample = self.data_queue_2.dequeue()
        # update per centroid count
        per_centroid_count = tf.scatter_nd_add(self.samples_per_centroid,
                                               indices=[[centroid]],
                                               updates=[1],
                                               name="incrementPerCenterCount")
        # update per center learning rate
        with tf.control_dependencies([per_centroid_count]):
            learning_rate = tf.squeeze(tf.cast(1 / tf.slice(per_centroid_count, [centroid], [1]), tf.float64))
            # learning_rate = tf.Print(learning_rate, [learning_rate], message="learning rate: ")
        tf.scatter_nd_update(self.learning_rate, [[centroid]], [learning_rate], name="updateLearningRate")
        # compute new centroids
        updated_centroids = tf.scatter_nd_update(self.centroids,
                                                 indices=[centroid],
                                                 updates=tf.add(tf.scalar_mul(scalar=(1 - learning_rate),
                                                                              x=tf.slice(input_=self.centroids,
                                                                                         begin=[centroid, 0],
                                                                                         size=[1, self.n_features])),
                                                                tf.scalar_mul(scalar=learning_rate, x=sample)))
        with tf.control_dependencies([updated_centroids]):
            return tf.Print(centroid, [centroid], message="updating centroid: ")

    def clear_(self):
        """clears loop variant data - reset Tensors"""
        # tf.scatter_update(ref=self.centroid_cache,
        #                   indices=[i for i in range(self.batch_size)],
        #                   updates=np.zeros([self.batch_size]))
        tf.assign(self.inertia, 0, name="resetInertia")

    def train(self):
        """cluster samples and build a model of centroids
            :returns Tensor of final centroids"""
        itr = 0
        with self.graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                saver = tf.train.Saver({"centroids": self.centroids})
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                cluster_op = tf.while_loop(cond=self._loop_cond, body=self._loop_body, loop_vars=[itr],
                                           parallel_iterations=1, swap_memory=True)
                sess.run(cluster_op)
                saver.save(sess, self.model_dir + 'centroids.mdl')
                sess.close()

    def get_model(self):
        """load trained tensorflow graph and retrieve centroids
            :returns centroids as ndarray"""
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver({"centroids": self.centroids})
            saver.restore(sess, self.model_dir + 'centroids.mdl')
            save_op = tf.Print(self.centroids, [self.centroids])
            cluster_centers_ = sess.run(save_op)
            np.save("cluster_centers.npy", cluster_centers_)
            sess.close()
        return cluster_centers_


# code for command line testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniBatch K-means implementation",
                                     usage="MiniBatchKmeans input_file model_dir k n_iter batch_size")
    parser.add_argument("input_file", help="file containing numpy ndarray")
    parser.add_argument("--model-dir", default=os.path.dirname(__file__), help="directory where model should be saved")
    parser.add_argument("-k", help="# of clusters")
    parser.add_argument("--n-iter", help="# of iterations")
    parser.add_argument("--batch-size", help="size of mini-batch")
    args = parser.parse_args()
    mbkmeans_obj = MiniBatchKMeans(input_file=args.input_file,
                                   model_dir=args.model_dir,
                                   k=int(args.k),
                                   n_iter=int(args.n_iter),
                                   batch_size=int(args.batch_size))
    mbkmeans_obj.train()
    # mbkmeans_obj.get_model()