# %% md   代码环境tf.1.14.0

'''
如何用显卡跑代码:
首先卸了tensorflow
安装tensorflow-gpu==对应版本号
然后运行代码就行了,他会自动有gpu了

'''


import warnings
warnings.filterwarnings("ignore") #忽略警告
# 基于自注意力机制的翻译系统

# %% md

## 1. 数据处理
# - 读取数据
# - 分别保存为inputs，outputs

# %%
#cmn 就是中英文翻译的数据集了.
with open('./cmn.txt', 'r', encoding='utf8') as f:
    data = f.readlines()

# %%

from tqdm import tqdm

inputs = []
outputs = []
for line in tqdm(data[:10000]):
    [en, ch] = line.strip('\n').split('\t')
    inputs.append(en.replace(',', ' ,')[:-1].lower())
    outputs.append(ch[:-1])

# %%

print(inputs[:10])

# %%

print(outputs[:10])

# %% md

### 1.1 英文分词
# 我们将英文用空格隔开即可，但是需要稍微修改一下，将大写字母全部用小写字母代替。在上文中使用
# `.lower
# `进行了替代。
#
# ```py
for line in tqdm(data):
    [en, ch] = line.strip('\n').split('\t')
    inputs.append(en[:-1].lower())
    outputs.append(ch[:-1])

# ```
# 此处我们只需要将英文用空格分开即可。

# %%

inputs = [en.split(' ') for en in inputs]

# %%

print(inputs[:10])

# %% md

### 1.2 中文分词
# - 中文分词选择结巴分词工具。
# ```py
import jieba

outputs = [[char for char in jieba.cut(''.join(line)) if char != ' '] for line in outputs]
# ```
# # - 也可以用hanlp。
# # ```py
# from pyhanlp import *
#
# outputs = [[term.word for term in HanLP.segment(line) if term.word != ' '] for line in outputs]
# # ```
# - 或者按字分词？

# %%
#
# import jieba
#
# jieba_outputs = [[char for char in jieba.cut(line) if char != ' '] for line in outputs[-10:]]
# print(jieba_outputs)

# %%

# outputs = [[char for char in jieba.cut(line) if char != ' '] for line in tqdm(outputs)]


# %% md

### 1.3 生成字典


# %%

def get_vocab(data, init=['<PAD>']):
    vocab = init
    for line in tqdm(data):
        for word in line:
            if word not in vocab:
                vocab.append(word)
    return vocab


SOURCE_CODES = ['<PAD>']
TARGET_CODES = ['<PAD>', '<GO>', '<EOS>'] #go是句子开始标志,eos是句子结束标志
encoder_vocab = get_vocab(inputs, init=SOURCE_CODES)
decoder_vocab = get_vocab(outputs, init=TARGET_CODES)

# %%

print(encoder_vocab[:10])
print(decoder_vocab[:10])

# %% md

### 1.4 数据生成器

# 翻译系统解码器端需要输入和输出，因此解码器需要拼凑出输入序列和输出序列。

# %%



#应该decoder_inputs,decoder_targets都加上go和eos
encoder_inputs = [[encoder_vocab.index(word) for word in line] for line in inputs]
decoder_inputs = [[decoder_vocab.index('<GO>')] + [decoder_vocab.index(word) for word in line] for line in outputs]
decoder_targets = [[decoder_vocab.index(word) for word in line] + [decoder_vocab.index('<EOS>')] for line in outputs]

# %%

print(decoder_inputs[:4])
print(decoder_targets[:4])

# %%

import numpy as np


def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4):
    batch_num = len(encoder_inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        en_input_batch = encoder_inputs[begin:end]
        de_input_batch = decoder_inputs[begin:end]
        de_target_batch = decoder_targets[begin:end]
        max_en_len = max([len(line) for line in en_input_batch])
        max_de_len = max([len(line) for line in de_input_batch])
        en_input_batch = np.array([line + [0] * (max_en_len - len(line)) for line in en_input_batch])
        de_input_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_input_batch])
        de_target_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_target_batch])
        yield en_input_batch, de_input_batch, de_target_batch


# %%

batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4)
next(batch)

# %% md

## 2. 构建模型

# %%

import tensorflow as tf

# %% md

# ### 2.1 构造建模组件
# 下面代码实现了图片结构中的各个功能组件。
# ![image.png](attachment: image.png)

#### layer norm层

# %%

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


# %% md

#### embedding层

# %%

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad: #xavier_initializer 一种初始化方法
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


# %% md

#### multihead层
# 该层实现了下面功能：
# ![image.png](attachment: image.png)

# %%

def multihead_attention(key_emb,
                        que_emb,
                        queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key_emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(que_emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


# %% md

#### feedforward

# 两层全连接，用卷积模拟加速运算，也可以使用dense层。

# %%

def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


# %% md

# #### label_smoothing.
# 对于训练有好处，将0变为接近零的小数，1
# 变为接近1的数，原文：
#
# During
# training, we
# employed
# label
# smoothing
# of
# value ls = 0.1[36].This
# hurts
# perplexity, as the
# model
# learns
# to
# be
# more
# unsure, but
# improves
# accuracy and BLEU
# score.
#
#
# # %%

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


# %% md

### 2.2 搭建模型
# 模型实现下图结构：
# ![image.png](attachment: image.png)

# %%

class Graph():
    def __init__(self, is_training=True):
        tf.reset_default_graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.input_vocab_size = arg.input_vocab_size
        self.label_vocab_size = arg.label_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.max_length = arg.max_length
        self.lr = arg.lr
        self.dropout_rate = arg.dropout_rate

        # input placeholder
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        self.de_inp = tf.placeholder(tf.int32, shape=(None, None))

        # Encoder
        with tf.variable_scope("encoder"):
            # embedding
            self.en_emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                    scope="enc_embed")
            self.enc = self.en_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")
            # Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            #Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.enc = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.en_emb,
                                                   queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False)

            # Feed Forward
            self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Decoder
        with tf.variable_scope("decoder"):
            # embedding
            self.de_emb = embedding(self.de_inp, vocab_size=self.label_vocab_size, num_units=self.hidden_units,
                                    scale=True, scope="dec_embed")
            self.dec = self.de_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.de_inp)[1]), 0), [tf.shape(self.de_inp)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="dec_pe")
            # Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            # Multihead Attention ( self-attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.dec = multihead_attention(key_emb=self.de_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope='self_attention')

            #Multihead Attention ( vanilla attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.dec = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope='vanilla_attention')

                    # Feed Forward
            self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units])

        # Final linear projection
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


# %% md

## 3. 训练模型


# %% md

### 3.1 参数设定

# %%

def create_hparams():
    params = tf.contrib.training.HParams(
        num_heads=8,
        num_blocks=6,
        # vocab
        input_vocab_size=50,
        label_vocab_size=50,
        # embedding size
        max_length=100,
        hidden_units=512,
        dropout_rate=0.2,
        lr=0.0003,
        is_training=True)
    return params


arg = create_hparams()
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)

# %% md

### 3.2 模型训练

# %%

import os

epochs = 0
batch_size = 64

g = Graph(arg)

saver = tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    if os.path.exists('./logs/model2.meta'):
        print('已经有模型了,所以下面直接进行finetune')
        saver.restore(sess, './logs/model2')
    writer = tf.summary.FileWriter('tensorboard/lm', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch_num = len(encoder_inputs) // batch_size
        batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size)
        for i in tqdm(range(batch_num)):
            encoder_input, decoder_input, decoder_target = next(batch)
            feed = {g.x: encoder_input, g.y: decoder_target, g.de_inp: decoder_input}
            cost, _ = sess.run([g.mean_loss, g.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        if (k + 1) % 5 == 0:
            print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
    saver.save(sess, './logs/model2')
    writer.close()

# %% md

### 3.3 模型推断

# %%
import tensorflow as tf
arg.is_training = False

g = Graph(arg)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './logs/model2')
    while True:
        line = input('输入测试英语句子: ')
        if line == 'exit': break
        line = line.lower().replace(',', ' ,').strip('\n').split(' ')
        tmp=[]
        for i in line:
            try:
                tmp.append(encoder_vocab.index(i))
            except:
                continue

        x=np.array(tmp)
        x = x.reshape(1, -1)
        de_inp = [[decoder_vocab.index('<GO>')]]
        while True:
            y = np.array(de_inp)
            preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
            if preds[0][-1] == decoder_vocab.index('<EOS>'):
                break
            de_inp[0].append(preds[0][-1])#输入每一次,加进去预测出来的一个字.
        got = ''.join(decoder_vocab[idx] for idx in de_inp[0][1:])
        print(got)

# %%


