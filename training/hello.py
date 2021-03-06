# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def predict(x):
    result = 0.0
    for n in range(0, 5):
        result += w_val[n][0] * x**n
    return result

x = tf.placeholder(tf.float32, [None, 5])
# 変数の定義
w = tf.Variable(tf.zeros([5, 1]))
# y = xw 行列の積
y = tf.matmul(x, w)
# データ
t = tf.placeholder(tf.float32, [None, 1])

# 誤差
loss = tf.reduce_sum(tf.square(y - t))

# 勾配降下法によるパラメータ最適化
train_step = tf.train.AdamOptimizer().minimize(loss)

# 実行環境のセッション
sess = tf.Session()
# 変数の初期化
sess.run(tf.initialize_all_variables())

# トレーニングデータ
train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4, 25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
# 12行1列に変換
train_t = train_t.reshape([12, 1])

# データの正規化
train_x = np.zeros([12, 5])
for row, month in enumerate(range(1, 13)):
    for col, n in enumerate(range(0, 5)):
        train_x[row][col] = month ** n

i = 0
# トレーニング開始
for _ in range(200000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
        print ('Step: %d, Loss: %f' % (i, loss_val))

# 変数を入れることで値を取り出すことができる
w_val = sess.run(w)
print w_val

fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
subplot.set_xlim(1, 12)
subplot.scatter(range(1, 13), train_t)
linex = np.linspace(1, 12, 100)
liney = predict(linex)
subplot.plot(linex, liney)
plt.savefig("test.png")
