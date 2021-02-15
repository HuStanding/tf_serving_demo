## 一、TensorFlow Serving介绍

###  1.1 TensorFlow serving的what和why

**1）why**

当我们训练完一个tensorflow（模型后，需要把它做成一个服务，让使用者通过某种方式来调用你的模型，而不是直接运行你的代码（因为你的使用者不一定懂怎样安装），这个过程需要把模型部署到服务器上。常用的做法如使用flask、ornado等web框架创建一个服务器app，这个app在启动后就会一直挂在后台，然后等待用户使用客户端POST一个请求上来,接着调用你的模型，得到推理结果后以json的格式把结果返回给用户。

这个做法对于简单部署来说代码量不多，对于不熟悉web框架的朋友来说随便套用一个模板就能写出来，但是也会有一些明显的缺点：

- 需要在服务器上重新安装项目所需的所有依赖。
- 当接收到并发请求的时候，服务器可能要后台启动多个进程进行推理，造成资源紧缺。
- 不同的模型需要启动不同的服务

**2）what**

TensorFlow Serving是GOOGLE开源的一个服务系统，适用于部署机器学习模型，灵活、性能高、可用于生产环境。

TensorFlow Serving不需要其它环境依赖，导出的模型就能直接在TensorFlow Serving上使用，接收输入，返回输出，无需写任何部署代码。

它具有以下特性：

- 支持多种模型服务策略,比如用最新版本/所有版本/指定版本, 以及动态策略更新、模型的增删等
- 自动加载/卸载模型
- Batching
- 多种平台支持(非TF平台)
- 为gRPC expose port 8500，为REST API expose port 8501

### 1.2 tensorflow serving原理简介（how）

![image-20201021203254023](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjxuvhihipj316e0lsdkr.jpg)

TF Serving的工作流程主要分为以下几个步骤：

- Source会针对需要进行加载的模型创建一个Loader，Loader中会包含要加载模型的全部信息；
- Source通知Manager有新的模型需要进行加载；
- Manager通过版本管理策略（Version Policy）来确定哪些模型需要被下架，哪些模型需要被加载；
- Manger在确认需要加载的模型符合加载策略，便通知Loader来加载最新的模型；
- 客户端像服务端请求模型结果时，可以指定模型的版本，也可以使用最新模型的结果；

### 1.3 tf模型的保存、加载和转换

1）ckpt和pb的区别

ckpt：

首先这种模型文件是依赖 TensorFlow 的，只能在其框架下使用；

pb：

它具有语言独立性，可独立运行，封闭的序列化格式，任何语言都可以解析它，它允许其他语言和深度学习框架读取、继续训练和迁移 TensorFlow 的模型；

保存为 PB 文件时候，模型的变量都会变成固定的，导致模型的大小会大大减小

2）ckpt保存和加载

保存ckpt：

```python
import tensorflow as tf

x = tf.placeholder(tf.float32, [1, 2], name='input_x')
y = tf.placeholder(tf.float32, [1, 2], name='input_y')
z = tf.Variable([[1.0, 1.0]], name='var_z')
a = x + y
tf.identity(a, name="output_a")
b = x - y
tf.identity(b, name="output_b")

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
result_del = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_del)

tf.train.Saver().save(sess, './ckpt_model/model.ckpt')
'''
checkpoint 文本文件，记录了模型文件的路径信息列表
model.ckpt.data-00000-of-00001 网络参数值
model.ckpt.index 文件保存了当前参数名和索引
model.ckpt.meta 保存模型的网络结构
'''
```



加载ckpt：

```python
import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('./ckpt_model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)

    x = sess.graph.get_tensor_by_name("input_x:0")
    y = sess.graph.get_tensor_by_name("input_y:0")

    a = sess.graph.get_tensor_by_name("output_a:0")
    b = sess.graph.get_tensor_by_name("output_b:0")

    result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
    result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)
```



3）pb保存和加载

保存pb文件：

```python
'''
saved_model.pb 保存图形结构
variables 保存训练所习得的权重。
'''

import tensorflow as tf

x = tf.placeholder(tf.float32, [1, 2], name='input_x')
y = tf.placeholder(tf.float32, [1, 2], name='input_y')
z = tf.Variable([[1.0, 1.0]], name='var_z')
a = x + y
tf.identity(a, name="output_a")
b = x - y
tf.identity(b, name="output_b")

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)

tf.saved_model.simple_save(sess,
                           "./pb_model/",
                           inputs={
                               "input_x": x,
                               "input_y": y
                           },
                           outputs={
                               "result_add": a,
                               "result_sub": b
                           })
```

加载pb文件：

```python
import tensorflow as tf

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./pb_model")
    graph = tf.get_default_graph()

    x = sess.graph.get_tensor_by_name("input_x:0")
    y = sess.graph.get_tensor_by_name("input_y:0")

    a = sess.graph.get_tensor_by_name("output_a:0")
    b = sess.graph.get_tensor_by_name("output_b:0")

    result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
    result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)
```



4）ckpt转换成pb文件

```python
import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('./ckpt_model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)

    x = sess.graph.get_tensor_by_name("input_x:0")
    y = sess.graph.get_tensor_by_name("input_y:0")

    a = sess.graph.get_tensor_by_name("output_a:0")
    b = sess.graph.get_tensor_by_name("output_b:0")

    result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
    result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

    tf.saved_model.simple_save(sess,
                               "./ckpt2pb/",
                               inputs={
                                   "input_x": x,
                                   "input_y": y
                               },
                               outputs={
                                   "result_add": a,
                                   "result_sub": b
                               })
    print(result_add)
    print(result_sub)
```



### 1.4 自己动手部署服务

目前TF Serving有Docker、APT（二级制安装）和源码编译三种方式，但考虑实际的生产环境项目部署和简单性，推荐使用Docker方式。

（1）拉取TF Serving 镜像

docker pull tensorflow/serving

（2）查看镜像是否存在

docker images

![image-20201021203508538](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjxuvso23bj31b605ital.jpg)

（3）模型存放目录规则

models文件就可以作为我们存放模型的目录，对其位置无要求；

但是，models目录下，我们最好按照这个目录树的结构存放；

二级目录：add_sub 我们模型的功能名或者业务名（用来区分不同模型）

三级目录：20201016 只要是数字就行 （用来区分同一模型的不同版本）

四级目录：存放模型的pb文件（四五级目录存储模型就是这个样子）

五级目录：variables 存放模型的参数

![image-20201021203543927](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjxuvvy1baj30s60cudhn.jpg)

（4）一行命令启动接口服务

```
docker run \

-p 8502:8501 \

--mount type=bind,source=/home/liuziyu004/self_project/docker_make/models/add_sub,target=/models/add_sub \

-e MODEL_NAME=add_sub \

-t tensorflow/serving &


```

参数解释如下

```
-p 8502:8501

其中（1）-p tf serving 监听端口；（2）8502是访问接口时的端口号，可以修改，（3）是8501为 REST API 模式，不要修改

source=/home/liuziyu004/self_project/docker_make/models/add_sub

source参数为本地模型存放的地址，到二级目录即可；注意：这个地方一定要绝对目录地址

target=/models/add_sub

挂载在docker中的目录，访问接口时，构成接口链接的一部分 http://{ip}{port}/v1/{target}:predict

-e MODEL_NAME=add_sub

添加docker中的环境变量

-t tensorflow/serving

启动镜像tensorflow/serving
```

（5）接口效果验证

1）看docker中容器是否启动：

![image-20201021203632284](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjxuvzmwtbj31h606g40f.jpg)

2）接口链接构成方式：http://{ip地址}{port端口}/v1/{docker启动时target}:predict

3）访问接口postman效果：

![image-20201021203648186](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjxuw1886zj30pm0n4tba.jpg)


（6）停止服务

根据（5）的第一步，我们可以知道服务的容器id为582b4b234e5d，执行以下两条命令即可
docker stop 582b4b234e5d

docker rm 582b4b234e5d 
