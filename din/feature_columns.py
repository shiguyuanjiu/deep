from collections import namedtuple,OrderedDict 
from tensorflow.python.keras.layers import Input

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim',  'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    def __new__(cls,name,vocabulary_size,embedding_dim,embeddings_initializer=None,embedding_name=None,group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_name is None:
            embedding_name=name

        return super(SparseFeat, cls).__new__(cls,name,vocabulary_size,embedding_dim,embeddings_initializer,embedding_name,group_name,trainable)


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim','maxlen','length_name','combiner', 
                             'embeddings_initializer','embedding_name','group_name', 'trainable'])):
    def __new__(cls,name,vocabulary_size,embedding_dim,maxlen,length_name,combiner='mean',embeddings_initializer=None,
                embedding_name=None,group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_name is None:
            embedding_name=name

        return super(VarLenSparseFeat, cls).__new__(cls,name,vocabulary_size,embedding_dim,maxlen,length_name,combiner,
                                                    embeddings_initializer,embedding_name,group_name, trainable)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
 
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()
    
def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name)
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name)
        else:
            raise TypeError("Invalid feature column type,got", type(fc))
    return input_features