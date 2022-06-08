import feature_columns as fc_lib
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from collections import defaultdict
from itertools import chain
import tensorflow as tf

def create_embedding_dict(feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns,is_varlen=0, to_list=False):
    group_embedding_dict = defaultdict(list)
    varlen_embedding_vec_dict={}
    eps = tf.constant(1e-8, tf.float32)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = sparse_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name]=sparse_embedding_dict[embedding_name](lookup_idx)
        if isinstance(fc, fc_lib.VarLenSparseFeat) and is_varlen==1:
            combiner=fc.combiner
            feature_length_name = fc.length_name
            maxlen=fc.maxlen
            seq_input = varlen_embedding_vec_dict[feature_name]
            uiseq_embed_list, user_behavior_length = seq_input,sparse_input_dict[feature_length_name]
            mask = tf.sequence_mask(user_behavior_length,maxlen, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))
            embedding_size = uiseq_embed_list.shape[-1]
            mask = tf.tile(mask, [1, 1, embedding_size])
            if combiner == "max":
                hist = uiseq_embed_list - (1 - mask) * 1e9
                vec = tf.reduce_max(hist, 1, keepdims=True)
                group_embedding_dict[fc.group_name].append(vec)
            else:
                hist = tf.reduce_sum(uiseq_embed_list * mask, 1, keepdims=False)
                if combiner == "mean":
                    hist = tf.compat.v1.div(hist, tf.cast(user_behavior_length, tf.float32) + eps)
                vec = tf.expand_dims(hist, axis=1)
                group_embedding_dict[fc.group_name].append(vec)
        else:
            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
        
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def get_dense_input(features, feature_columns):
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list