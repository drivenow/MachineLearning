#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("mode","train","train|inference")

# Hyperparameters
logdir = ""
nb_epoch = 600
issync = FLAGS.issync
learning_rate = 0.001


def main(_):
    #regist ps,worker
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
  
  
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
          global_step = tf.Variable(0, name='global_step', trainable=False)
          #******************1.build_model here***********************
          
          
          
          
          #*******************2.model optimizer && train_op**************************
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          if issync == 1:
              #同步模式计算更新梯度
              grads_and_vars = optimizer.compute_gradients(loss_op)
              rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=len(worker_hosts),
                                                    replica_id=FLAGS.task_index,
                                                    total_num_replicas=len(worker_hosts),
                                                    use_locking=True)
              train_op = rep_op.apply_gradients(grads_and_vars,
                                           global_step=global_step)
              init_token_op = rep_op.get_init_tokens_op()
              chief_queue_runner = rep_op.get_chief_queue_runner()
          else:
              #异步模式计算更新梯度
              train_op = optimizer.minimize(loss_op,
                                           global_step=global_step)
    
        #******************3.model saver and summary**************************
        saver = tf.train.Saver()
        tf.summary.scalar('loss', loss_op)
        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        
        
        
              
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        if FLAGS.mode == "train":
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   logdir=logdir,
                                   init_op=init_op,
                                   summary_op=None,
                                   saver=saver,
                                   global_step=global_step,
                                   stop_grace_secs=300,
                                   save_model_secs=10)
        else:
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   logdir=logdir,
                                   summary_op=summary_op,
                                   saver=saver,
                                   global_step=global_step,
                                   stop_grace_secs=300,
                                   save_model_secs=0)

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            #*******************4.run op**********************
            step = 0
            while  step < nb_epoch:
                print("step: %d, weight: %f, biase: %f, loss: %f" %())
        #别忘了关闭监视器
        sv.stop()

if __name__ == "__main__":
  tf.app.run()

