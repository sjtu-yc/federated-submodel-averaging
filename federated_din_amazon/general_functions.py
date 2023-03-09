import tensorflow as tf
import numpy as np
import DINMessage.Message as DINMessage
import DINMessage.Hyperparameters as DINHyperparameters
import DINMessage.Layer as DINLayer
import flatbuffers


def create_placeholders():
    """Creates the placeholders that we will use to inject the weights into the graph"""
    placeholders = []
    for var in tf.trainable_variables():
        placeholders.append(tf.placeholder_with_default(var, var.shape,
                                                        name="%s/%s" % ("FedAvg", var.op.name)))
    return placeholders


def assign_vars(local_vars, placeholders):
    """Utility to refresh local variables.

    Args:
        local_vars: List of local variables.

    Returns:
        refresh_ops: The ops to assign value of global vars to local vars.
    """
    reassign_ops = []
    for var, fvar in zip(local_vars, placeholders):
        reassign_ops.append(tf.assign(var, fvar))
    return tf.group(*(reassign_ops))


def prepare_data(input, target, max_len=None):
    # x: a list of sentences
    lengths_x = [len(s[2]) for s in input]
    seqs_mid = [inp[2] for inp in input]

    if max_len is not None:
        new_seqs_mid = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > max_len:
                new_seqs_mid.append(inp[2][l_x - max_len:])
                new_lengths_x.append(max_len)
            else:
                new_seqs_mid.append(inp[2])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    max_len_x = np.max(lengths_x)

    mid_his = np.zeros((n_samples, max_len_x)).astype('int64')
    mid_mask = np.zeros((n_samples, max_len_x)).astype('float32')
    for idx, s_x in enumerate(seqs_mid):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])

    return uids, mids, mid_his, mid_mask, np.array(target), np.array(lengths_x)


def din_attention(query, facts, mask, mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))   # [B, L]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])   # [B, L*2*H]
    queries = tf.reshape(queries, tf.shape(facts))      # [B, L, 2*H]
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)    # [B, L, 4*2*H]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')    # [B, L, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [B, 1, L]
    scores = d_layer_3_all  # [B, 1, L]

    key_masks = tf.expand_dims(mask, 1)     # [B, 1, L]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, L]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, L]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, L]*[B, L, 2*H] = [B, 1, 2*H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])   # [B, L]
        output = facts * tf.expand_dims(scores, -1)     # [B, L, 2*H]*[B, L, 1]
        output = tf.reshape(output, tf.shape(facts))    # [B, L, 2*H]
    return output

def create_DINMessage_from_dict(my_dict):
    """
    get a python dictionary as input
    return a binary message which can be used to create a flatbuffers by DINMessage.GetRootAsMonster(message, 0)
    """
    builder = flatbuffers.Builder(0)
    if 'hyperparameters' in my_dict:
        tmp_dict = my_dict['hyperparameters']
    
    if 'user_IDs' in my_dict:
        user_IDs = my_dict['user_IDs']
        user_IDs_length = len(user_IDs)
        DINMessage.MessageStartUserIDsVector(builder, user_IDs_length)
        i = user_IDs_length - 1
        while i >= 0:
            builder.PrependInt32(user_IDs[i])
            i -= 1
        user_IDs_vector = builder.EndVector(user_IDs_length)

    if 'item_IDs' in my_dict:
        item_IDs = my_dict['item_IDs']
        item_IDs_length = len(item_IDs)
        DINMessage.MessageStartItemIDsVector(builder, item_IDs_length)
        i = item_IDs_length - 1
        while i >= 0:
            builder.PrependInt32(item_IDs[i])
            i -= 1
        item_IDs_vector = builder.EndVector(item_IDs_length)

    if 'cate_IDs' in my_dict:
        cate_IDs = my_dict['cate_IDs']
        cate_IDs_length = len(cate_IDs)
        DINMessage.MessageStartCateIDsVector(builder, cate_IDs_length)
        i = cate_IDs_length - 1
        while i >= 0:
            builder.PrependInt32(cate_IDs[i])
            i -= 1
        cate_IDs_vector = builder.EndVector(cate_IDs_length)

    if 'user_IDs_appear_times' in my_dict:
        user_IDs_appear_times = my_dict['user_IDs_appear_times']
        user_IDs_appear_times_length = len(user_IDs_appear_times)
        DINMessage.MessageStartUserIDsAppearTimesVector(builder, user_IDs_appear_times_length)
        i = user_IDs_appear_times_length - 1
        while i >= 0:
            builder.PrependInt32(user_IDs_appear_times[i])
            i -= 1
        user_IDs_appear_times_vector = builder.EndVector(user_IDs_appear_times_length)

    if 'item_IDs_appear_times' in my_dict:
        item_IDs_appear_times = my_dict['item_IDs_appear_times']
        item_IDs_appear_times_length = len(item_IDs_appear_times)
        DINMessage.MessageStartItemIDsAppearTimesVector(builder, item_IDs_appear_times_length)
        i = item_IDs_appear_times_length - 1
        while i >= 0:
            builder.PrependInt32(item_IDs_appear_times[i])
            i -= 1
        item_IDs_appear_times_vector = builder.EndVector(item_IDs_appear_times_length)

    if 'cate_IDs_appear_times' in my_dict:
        cate_IDs_appear_times = my_dict['cate_IDs_appear_times']
        cate_IDs_appear_times_length = len(cate_IDs_appear_times)
        DINMessage.MessageStartCateIDsAppearTimesVector(builder, cate_IDs_appear_times_length)
        i = cate_IDs_appear_times_length - 1
        while i >= 0:
            builder.PrependInt32(cate_IDs_appear_times[i])
            i -= 1
        cate_IDs_appear_times_vector = builder.EndVector(cate_IDs_appear_times_length)
    
    if 'client_ID' in my_dict:
        None

    if 'local_loss' in my_dict:
        None

    if 'client_train_set_size' in my_dict:
        None

    if 'model_timestamp' in my_dict:
        model_timestamp = builder.CreateString(my_dict['model_timestamp'])

    if 'model_paras' in my_dict:
        assert 'layer_names' in my_dict
        array_list = my_dict['model_paras']  # a list of numpy arrays
        array_list_length = len(array_list)
        name_list = my_dict['layer_names']
        name_list_length = len(name_list)
        assert name_list_length == array_list_length

        name_string_list = []
        for i in range(0, name_list_length):
            name_string_list.append(builder.CreateString(name_list[i]))

        dimension_vector_list = []
        for i in range(0, array_list_length):
            dimension = array_list[i].shape
            dimension_length = len(dimension)
            DINLayer.LayerStartDimensionVector(builder, dimension_length)
            j = dimension_length - 1
            while j >= 0:
                builder.PrependInt32(dimension[j])
                j -= 1
            tmp_dimension = builder.EndVector(dimension_length)
            dimension_vector_list.append(tmp_dimension)

        data_vector_list = []
        for i in range(0,array_list_length):
            data_vector_list.append(builder.CreateByteVector(array_list[i].tobytes()))   # transform a numpy array into bytes and then into a Vector
            #data_vector_list.append(builder.CreateNumpyVector(array_list[i]))

        DINLayer_vector_list = []
        for i in range(0, array_list_length):
            DINLayer.LayerStart(builder)
            DINLayer.LayerAddName(builder, name_string_list[i])
            DINLayer.LayerAddDimension(builder, dimension_vector_list[i])
            DINLayer.LayerAddByteArray(builder, data_vector_list[i])
            DINLayer_vector_list.append(DINLayer.LayerEnd(builder))

        DINMessage.MessageStartModelParametersVector(builder, array_list_length)
        i = array_list_length - 1
        while i >= 0:
            builder.PrependUOffsetTRelative(DINLayer_vector_list[i])
            i -= 1
        tmp_vector = builder.EndVector(array_list_length)

    DINMessage.MessageStart(builder)
    if 'hyperparameters' in my_dict:
        DINMessage.MessageAddHyperparameters(builder, DINHyperparameters.CreateHyperparameters(builder,
                                                tmp_dict['communication_rounds'], tmp_dict['local_iter_num'],
                                                tmp_dict['train_batch_size'], tmp_dict['test_batch_size'],
                                                tmp_dict['predict_batch_size'], tmp_dict['predict_users_num'],
                                                tmp_dict['predict_ads_num'], tmp_dict['learning_rate'],
                                                tmp_dict['decay_rate'], tmp_dict['embedding_dim']))
    if 'user_IDs' in my_dict:
        DINMessage.MessageAddUserIDs(builder, user_IDs_vector)
    if 'item_IDs' in my_dict:
        DINMessage.MessageAddItemIDs(builder, item_IDs_vector)
    if 'cate_IDs' in my_dict:
        DINMessage.MessageAddCateIDs(builder, cate_IDs_vector)
    if 'user_IDs_appear_times' in my_dict:
        DINMessage.MessageAddUserIDsAppearTimes(builder, user_IDs_appear_times_vector)
    if 'item_IDs_appear_times' in my_dict:
        DINMessage.MessageAddItemIDsAppearTimes(builder, item_IDs_appear_times_vector)
    if 'cate_IDs_appear_times' in my_dict:
        DINMessage.MessageAddCateIDsAppearTimes(builder, cate_IDs_appear_times_vector)
    if 'client_ID' in my_dict:
        DINMessage.MessageAddClientID(builder, my_dict['client_ID'])
    if 'local_loss' in my_dict:
        DINMessage.MessageAddLocalLoss(builder, my_dict['local_loss'])
    if 'client_train_set_size' in my_dict:
        DINMessage.MessageAddClientTrainSetSize(builder, my_dict['client_train_set_size'])
    if 'model_timestamp' in my_dict:
        DINMessage.MessageAddModelTimestamp(builder, model_timestamp)
    if 'model_paras' in my_dict:
        DINMessage.MessageAddModelParameters(builder, tmp_vector)
    my_buf = DINMessage.MessageEnd(builder)

    message_head = builder.Finish(my_buf)   # returns builder.head()
    my_message = builder.Output()   # equal to builder.bytes[message_head:]

    return my_message

def restore_dict_from_DINMessage(my_message):
    """
    get a binary message as input
    return a python dictionary
    """
    my_dict = {}
    my_buf = DINMessage.Message.GetRootAsMessage(my_message, 0)

    tmp_buf = my_buf.Hyperparameters()
    if tmp_buf is not None:
        hyperparameters = {'communication_rounds': tmp_buf.CommunicationRounds(),
                           'local_iter_num': tmp_buf.LocalIterNum(),
                           'train_batch_size': tmp_buf.TrainBatchSize(),
                           'test_batch_size': tmp_buf.TestBatchSize(),
                           'predict_batch_size': tmp_buf.PredictBatchSize(),
                           'predict_users_num': tmp_buf.PredictUserNum(),
                           'predict_ads_num': tmp_buf.PredictAdsNum(),
                           'learning_rate': tmp_buf.LearningRate(),
                           'decay_rate': tmp_buf.DecayRate(),
                           'embedding_dim': tmp_buf.EmbeddingDim()}
        my_dict['hyperparameters'] = hyperparameters

    if my_buf.UserIDsLength() != 0:
        my_dict['user_IDs'] = my_buf.UserIDsAsNumpy().tolist()

    if my_buf.ItemIDsLength() != 0:
        my_dict['item_IDs'] = my_buf.ItemIDsAsNumpy().tolist()

    if my_buf.CateIDsLength() != 0:
        my_dict['cate_IDs'] = my_buf.CateIDsAsNumpy().tolist()

    if my_buf.UserIDsAppearTimesLength() != 0:
        my_dict['user_IDs_appear_times'] = my_buf.UserIDsAppearTimesAsNumpy().tolist()

    if my_buf.ItemIDsAppearTimesLength() != 0:
        my_dict['item_IDs_appear_times'] = my_buf.ItemIDsAppearTimesAsNumpy().tolist()

    if my_buf.CateIDsAppearTimesLength() != 0:
        my_dict['cate_IDs_appear_times'] = my_buf.CateIDsAppearTimesAsNumpy().tolist()

    my_dict['client_ID'] = my_buf.ClientID()

    my_dict['local_loss'] = my_buf.LocalLoss()

    my_dict['client_train_set_size'] = my_buf.ClientTrainSetSize()

    model_timestamp = my_buf.ModelTimestamp()
    if model_timestamp is not None:
        my_dict['model_timestamp'] = model_timestamp.decode()

    layer_names = []
    model_parameters_length = my_buf.ModelParametersLength()
    if model_parameters_length != 0:
        model_parameters = []
        for i in range(0, model_parameters_length):
            tmp_layer = my_buf.ModelParameters(i)
            layer_names.append(tmp_layer.Name())
            layer_dimension = []
            for j in range(0, tmp_layer.DimensionLength()):
                layer_dimension.append(tmp_layer.Dimension(j))
            model_parameter = np.frombuffer(tmp_layer.ByteArrayAsNumpy().tobytes(), dtype=np.float32)
            model_parameter.resize(tuple(layer_dimension))
            model_parameters.append(model_parameter)
        my_dict['layer_names'] = layer_names
        my_dict['model_paras'] = model_parameters

    return my_dict