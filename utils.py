
import numpy as np
import tensorflow as tf
import math
import cv2
import matplotlib.pyplot as plt
import time
import random


def relabeling(im):
    img=im[:,:,0]
    re_img=np.zeros((img.shape[0],img.shape[1],1), np.uint8)
    re_img[np.where(img==1)]=1
    re_img[np.where(img==2)]=2
    re_img[np.where(img==20)]=2
    re_img[np.where(img==3)]=3
    re_img[np.where(img==17)]=3
    re_img[np.where(img==10)]=3
    re_img[np.where(img==15)]=3
    re_img[np.where(img==4)]=4
    re_img[np.where(img==5)]=5
    re_img[np.where(img==6)]=5
    re_img[np.where(img==7)]=6
    re_img[np.where(img==14)]=6
    re_img[np.where(img==8)]=7
    re_img[np.where(img==9)]=8
    re_img[np.where(img==13)]=9
    re_img[np.where(img==16)]=9
    re_img[np.where(img==16)]=9
    re_img[np.where(img==18)]=10
    re_img[np.where(img==19)]=11
    re_img[np.where(img==22)]=12
    re_img[np.where(img==24)]=13
    re_img[np.where(img==25)]=14
    re_img[np.where(img==23)]=15
    re_img[np.where(img==26)]=15
    re_img[np.where(img==29)]=16
    re_img[np.where(img==32)]=17
    re_img[np.where(img==35)]=17
    re_img[np.where(img==34)]=18
    re_img[np.where(img==37)]=19
    
    return re_img


# In[3]:


def l_oneshot(label):
    arr=np.zeros((label[:,:,0].shape[0]*label[:,:,0].shape[1],20),np.uint8)
    flat_label=label[:,:,0].flatten()
    for x in range(len(flat_label)):
        arr[x,flat_label[x]]=1
    arr=np.reshape(arr,(label[:,:,0].shape[0],label[:,:,0].shape[1],20))
    return arr

def im_resize(img,s):
#     x=np.minimum(img.shape[0],img.shape[1])
#     re_img=img[0:x,0:x]
    im=cv2.resize(img, (s,s), interpolation=cv2.INTER_NEAREST)
    return im


# In[4]:


def depth_resize(img,s):
#     x=np.minimum(img.shape[0],img.shape[1])
#     re_img=img[0:x,0:x]
    im=np.reshape(cv2.resize(img, (s,s), interpolation=cv2.INTER_LINEAR),(s,s,1))
    return im


# In[5]:


def d_resize(img,s):
#     x=np.minimum(img.shape[0],img.shape[1])
#     re_img=img[0:x,0:x]
    im=cv2.resize(img, (s,s), interpolation=cv2.INTER_LINEAR)
    return im


# In[6]:


def l_oneshot_13(label):
    arr=np.zeros((label.shape[0]*label.shape[1],14),np.uint8)
    flat_label=label.flatten()
    for x in range(len(flat_label)):
        arr[x,flat_label[x]]=1
    arr=np.reshape(arr,(label.shape[0],label.shape[1],14))
    return arr


# In[7]:


def bn(layer, use_bn):
     return tf.nn.relu(layer)


# In[8]:


def bn_sig(layer, use_bn):
#     mean,var=tf.nn.moments(layer,[1,2])
#     exp_mean=tf.expand_dims(mean,1)
#     exp_var=tf.expand_dims(var,1)
#     exp_mean_1=tf.expand_dims(exp_mean,2)
#     exp_var_1=tf.expand_dims(exp_var,2)
#     layer_1=tf.nn.sigmoid((layer-exp_mean_1)/exp_var_1)
#     layer_1=tf.nn.sigmoid(layer-exp_mean_1)
    return tf.nn.sigmoid(layer)


# In[9]:


def bn_relu(layer, use_bn):
    mean,var=tf.nn.moments(layer,[1,2])
    exp_mean=tf.expand_dims(mean,1)
    exp_var=tf.expand_dims(var,1)
    exp_mean_1=tf.expand_dims(exp_mean,2)
    exp_var_1=tf.expand_dims(exp_var,2)
    # layer_1=tf.nn.relu((layer-exp_mean_1)/exp_var_1)
    layer_1=tf.nn.relu(layer-exp_mean_1)
    return layer_1


# In[10]:


def bn_elu(layer, use_bn):
    mean,var=tf.nn.moments(layer,[1,2])
    exp_mean=tf.expand_dims(mean,1)
    exp_var=tf.expand_dims(var,1)
    exp_mean_1=tf.expand_dims(exp_mean,2)
    exp_var_1=tf.expand_dims(exp_var,2)
    layer_1=tf.nn.elu(layer-exp_mean_1)
    return layer_1


# In[11]:


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)


# In[12]:


def unpool_layer(x, raveled_argmax, out_shape):
    batch = out_shape[0]
    height = out_shape[1]
    width = out_shape[2]
    channels = out_shape[3]
    one_like_mask = tf.ones_like(raveled_argmax)
    batch_range = tf.reshape(tf.to_int64(tf.range(batch)), shape=[batch, 1, 1, 1])
    b = one_like_mask * batch_range
    y = raveled_argmax // (width * channels)
    w = raveled_argmax % (width * channels) // channels
    feature_range = tf.to_int64(tf.range(channels))
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(x)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, w, f]), [4, updates_size]))
    values = tf.reshape(x, [updates_size])
    ret = tf.scatter_nd(indices, values, tf.to_int64(out_shape))
    return ret


# In[13]:


def act_func_c(layer,is_use):
    return tf.nn.relu(layer)


# In[14]:


def act_func_d(layer,is_use):
    if is_use == True:
        mean,var=tf.nn.moments(layer,[1,2])
        var_1=tf.reshape(mean,[-1,1,1,tf.to_int64(var.shape[1])])
        mean_1=tf.reshape(mean,[-1,1,1,tf.to_int64(mean.shape[1])])
        layer_1=layer-mean_1
        return tf.nn.relu(layer_1)
    else:
        return tf.nn.relu(layer)


# In[15]:


def make_wei(name,ker_s,in_l_num,out_l_num):
    w= tf.get_variable('w_'+name, shape=[ker_s,ker_s,in_l_num,out_l_num], initializer=tf.contrib.layers.xavier_initializer())
    b=tf.get_variable('b_'+name, shape=[out_l_num], initializer=tf.contrib.layers.xavier_initializer())
    return w,b


# In[16]:


def get_conv(in_layer,k_s,in_n,out_n,l_n,name,is_train):
    w_l2=0
    for i in range(l_n):
        if i==0:
            a=in_n
        else:
            a=out_n
        w,b=make_wei(name+'_'+str(i+1),k_s,a,out_n)
        w_l2=w_l2+tf.reduce_sum(tf.square(w))
        in_layer=act_func_d(tf.nn.conv2d(in_layer,w, strides=[1,1,1,1], padding= 'SAME')+b,is_train)
    return in_layer,w_l2


# In[17]:


def get_deconv(in_l,k_s,i_s,edge,edge_n,in_n,out_n,name,is_train):
    w_edge_1,b_edge_1=make_wei(name+'_1',k_s,in_n+edge_n,in_n)
    w_edge_2,b_edge_2=make_wei(name+'_2',k_s,in_n,out_n)
    w_edge_3,b_edge_3=make_wei(name+'_3',1,out_n,out_n)
    w_edge_4,b_edge_4=make_wei(name+'_4',1,out_n,out_n)
    

 
    edge_1=tf.image.resize_images(in_l,[math.ceil(i_s),math.ceil(i_s)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    edge_2=tf.concat([edge,edge_1],3)
    edge_3=act_func_d(tf.nn.conv2d(edge_2,w_edge_1, strides=[1,1,1,1], padding= 'SAME')+b_edge_1,is_train)
    edge_4=act_func_d(tf.nn.conv2d(edge_3,w_edge_2, strides=[1,1,1,1], padding= 'SAME')+b_edge_2,is_train)
    edge_5=act_func_d(tf.nn.conv2d(edge_4,w_edge_3, strides=[1,1,1,1], padding= 'SAME')+b_edge_3,is_train)


    return edge_5


# In[18]:


def deconv(d_in_l,s_in_l,k_s,i_s,edge,edge_n,in_n,out_n,name):
    w_dns_1,b_dns_1=make_wei('dns_'+name+'_1',k_s,out_n*2,out_n)
    w_dns_2,b_dns_2=make_wei('dns_'+name+'_2',k_s,out_n,out_n)

    
    w_dep_1,b_dep_1=make_wei('dep_'+name+'_1',k_s,in_n+edge_n,out_n)
    w_dep_2,b_dep_2=make_wei('dep_'+name+'_2',k_s,out_n,out_n)
    w_dep_3,b_dep_3=make_wei('dep_'+name+'_3',k_s,out_n*2,out_n)



    w_seg_1,b_seg_1=make_wei('seg_'+name+'_1',k_s,in_n+edge_n,out_n)
    w_seg_2,b_seg_2=make_wei('seg_'+name+'_2',k_s,out_n,out_n)
    w_seg_3,b_seg_3=make_wei('seg_'+name+'_3',k_s,out_n*2,out_n)

    

 
    dep_1=tf.image.resize_images(d_in_l,[math.ceil(i_s),math.ceil(i_s)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    seg_1=tf.image.resize_images(s_in_l,[math.ceil(i_s),math.ceil(i_s)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    dep_2=tf.concat([dep_1,edge],3)
    seg_2=tf.concat([seg_1,edge],3)
    
    dep_3=act_func_d(tf.nn.conv2d(dep_2,w_dep_1, strides=[1,1,1,1], padding= 'SAME')+b_dep_1)
    dep_4=act_func_d(tf.nn.conv2d(dep_3,w_dep_2, strides=[1,1,1,1], padding= 'SAME')+b_dep_2)
    
    
    seg_3=act_func_d(tf.nn.conv2d(seg_2,w_seg_1, strides=[1,1,1,1], padding= 'SAME')+b_seg_1)
    seg_4=act_func_d(tf.nn.conv2d(seg_3,w_seg_2, strides=[1,1,1,1], padding= 'SAME')+b_seg_2)

    
    dns_1=tf.concat([dep_4,seg_4],3)
    dns_2=act_func_d(tf.nn.conv2d(dns_1,w_dns_1, strides=[1,1,1,1], padding= 'SAME')+b_dns_1)
    dns_3=act_func_d(tf.nn.conv2d(dns_2,w_dns_2, strides=[1,1,1,1], padding= 'SAME')+b_dns_2)

    
    dep_5=tf.concat([dep_4,dns_3],3)
    seg_5=tf.concat([dep_4,dns_3],3)
    
    dep_6=act_func_d(tf.nn.conv2d(dep_5,w_dep_3, strides=[1,1,1,1], padding= 'SAME')+b_dep_3)
    seg_6=act_func_d(tf.nn.conv2d(seg_5,w_seg_3, strides=[1,1,1,1], padding= 'SAME')+b_seg_3)
    
    return dep_6,seg_6


# In[19]:


def acc(output,label):
    soft=tf.nn.softmax(output)
    max_output=tf.reduce_max(soft,1)
    max_output=tf.reshape(max_output,[-1,1])

    est=soft//max_output

    
    inter=est*label
    union_1=est+label
    union_2=tf.ones_like(union_1)
    union=tf.minimum(union_1,union_2)
    
    union_sum=tf.reduce_sum(union,0)
    label_sum=tf.reduce_sum(label,0)
    inter_sum=tf.reduce_sum(inter,0)

    return inter_sum, label_sum,union_sum


# In[20]:


def iou_mean(inter,union,label,ch):
    m_iou=0
    m_mean=0
    none_zero=np.ones_like(union)
    label=np.maximum(label,none_zero)
    union=np.maximum(union,none_zero)
    
    s_iou=inter/union
    s_mean=inter/label

    
    for m in range(ch):
        m_iou+=s_iou[m]
        m_mean+=s_mean[m]
    print("mean_iou_acc : ", '{:.9f}'.format(m_iou/ch) )       
    print("mean_mean_acc : ", '{:.9f}'.format(m_mean/ch))
    return s_iou,s_mean,m_iou/ch,m_mean/ch

