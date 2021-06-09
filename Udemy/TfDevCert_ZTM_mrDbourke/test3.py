#%%
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import rank
import tensorflow_datasets
#import tensorflow_datasets as tfds|
tf.__version__
# %%
scaler = tf.constant(7)
scaler
# %%
scaler.ndim
vector = tf.constant([10,10])
vector
vector.ndim
matrix = tf.constant([[10,7],[7,10]])
matrix.ndim
matr2 = tf.constant([[10.,7],[7,10],[8.,9.]], dtype= tf.float16)
matr2.ndim
matr2

tensor = tf.constant([[[1,2,3,], [4,5,6,]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])

changetensor = tf.Variable([10,7])
unchangeten= tf.constant([10,7])
changetensor, unchangeten

changetensor[0]
changetensor[0].assign(7)
unchangeten[0].assign(11)
#%%  Creating random tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
random_1

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape =(3,2))
random_1, random_2, random_2 == random_1

# %% Shuffling the order of tensors...
not_shuffeled = tf.constant([[10,7],[3,4],[2,5]])
not_shuffeled
not_shuffeled.ndim

# %% shuffle
tf.random.set_seed(1)
tf.random.shuffle(not_shuffeled, 1, 'bulloxed')

# %%homne work:
random_2.normal(shape =(4,4))
#%% other ways to create tensors
tf.ones([10,7])
tf.zeros([7,7])
#%%
numpy_A = np.arange(1,25, dtype=np.int32)
numpy_A

A = tf.constant(numpy_A, shape=(2,3,4))
B = tf.constant(numpy_A)
A,B

# %%getting info from tensor:
A.shape
rank_4_tensor = tf.zeros(shape=[2,3,4,5])
rank_4_tensor
print('then, the first element')
rank_4_tensor[0]
#%%
rank_4_tensor.ndim, rank_4_tensor.shape, tf.size(rank_4_tensor)


# %% Get various attributes
rank_4_tensor.dtype



# %% tensors can be indexed
rank_4_tensor[:2, :2, :2,:2]
rank_4_tensor[0]
#%%
rk2 = tf.ones(shape=[2,4])
rk2
rk2.ndim

# %%
rk2[:,-1]
#%%
rk3 = rk2[..., tf.newaxis]
rk3
# %%
np.min(rk3)
tf.reduce_mean(rk3)
# %%
E = tf.constant(np.random.randint(0,100, size = 50))
E

# %%
import tensorflow_probability as tfp
tfp.stats.variance(E)
# %%
tf.math.reduce_std(tf.cast(E, dtype=tf.float32))
# %%
tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
# %% find the highest value in the tensor
tf.argmax(F)
F[tf.argmax(F)]

# %%
F[tf.argmin(F)]