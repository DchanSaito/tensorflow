# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 50000

# グラフ用
figy = np.zeros([21])
figdif = np.zeros([21])

# 変数などの設定
y = tf.Variable(tf.cast(0, tf.float32))
x = tf.placeholder(tf.float32)

# オペレーション
update_y = tf.assign(y, tf.add(y, tf.div(tf.cast(1, tf.float32), x)))

# log2との誤差
dif = tf.sub(tf.log(tf.cast(2, tf.float32)), y)

# sessionの初期化
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# 計算開始
i = 0
for _ in range(N):
    i += 1
    tmp = -i if i % 2 == 0 else i

    sess.run(update_y, feed_dict={x: tmp})
    if i <= 20:
        figy[i] = sess.run(y)
        figdif[i] = sess.run(dif)
    if i % 10 == 0:
        print('STEP: %d, RES: %f, DIF: %f' %(i, sess.run(y), sess.run(dif)))


plt.plot(figy, label = "result")
plt.plot(figdif, label = "dif")
plt.legend()
plt.ylim([-2,2])
plt.savefig("result.png")
