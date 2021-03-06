#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import numpy as np
import os
import sys
import pandas as pd
import numpy as np


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, mask=None, training=True):
        attn_output = self.att(inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
              'embed_dim': embed_dim,
              'num_heads': num_heads,
              'ff_dim': ff_dim,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=seq_embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=pos_embed_dim, trainable=False,
                                        weights=[self.get_pos_matrix(maxlen, pos_embed_dim)])

    def call(self, x):
        seq, pssm = x
        maxlen = tf.shape(seq)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        seq = self.token_emb(seq)
        x = tf.concat([seq, pssm], -1)
        return x + positions
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions
    
    def get_pos_matrix(self, max_len, d_emb):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
            if pos != 0 else np.zeros(d_emb) 
                for pos in range(max_len)
                ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        return pos_enc
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': maxlen,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'pos_embed_dim': pos_embed_dim,
            'seq_embed_dim': seq_embed_dim,
        })
        return config



class cn(layers.Layer):
    def __init__(self, window_len, maxlen, vocab_size, embed_dim, num_heads, ff_dim, pos_embed_dim, seq_embed_dim):
        super(cn, self).__init__()
#         self.leaky_relu = layers.ELU()
        self.leaky_relu = layers.LeakyReLU()
#         self.initializer = tf.keras.initializers.HeUniform()
        self.embedding_layer_mut = TokenAndPositionEmbedding(window_len, vocab_size, 
                                                             embed_dim, pos_embed_dim, seq_embed_dim)
        self.embedding_layer_par = TokenAndPositionEmbedding(maxlen, vocab_size, 
                                                             embed_dim, pos_embed_dim, seq_embed_dim)
        self.trans_block_mut1 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.trans_block_par1 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.trans_block_mut2 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.trans_block_par2 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.trans_block_mut3 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.trans_block_par3 = TransformerBlock(embed_dim, num_heads, ff_dim)
#         self.trans_block_mut4 = TransformerBlock(embed_dim, num_heads, ff_dim)
#         self.trans_block_par4 = TransformerBlock(embed_dim, num_heads, ff_dim)
        
        self.sub_layer = layers.Subtract()
        self.mul_layer = layers.Multiply()
        self.concat_ = layers.Concatenate(axis=1)
        
#         self.permute1 = layers.Permute((2, 1))
        self.att_dense_mut = layers.Dense(window_len * 4, activation='softmax', kernel_initializer='he_uniform')
        self.att_dense_par = layers.Dense(maxlen, activation='softmax', kernel_initializer='he_uniform')
#         self.permute2 = layer.Permute((2, 1), name=name + '_att_vec')
#         self.att_out = layers.Multiply()
        
        self.mut_c1 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.mut_c2 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.mut_c3 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.mut_c4 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        
        self.par_c1 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.par_c2 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.par_c3 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        self.par_c4 = layers.Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')
        
        self.all_c1 = layers.Conv1D(1, 1, kernel_initializer='he_uniform')
        self.seq_c2 = layers.Conv1D(4, 1, kernel_initializer='he_uniform')
        
        self.dense1 = layers.Dense(64, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(4, activation='softmax', kernel_initializer='he_uniform')
        
    def call(self, inputs):
        mut0, mut1, par0, mut0_pssm, mut1_pssm, par0_pssm = inputs
        mut0_mask = create_padding_mask(mut0)
        mut1_mask = create_padding_mask(mut1)
        par0_mask = create_padding_mask(par0)
        
        mut0_pssm = tf.math.sigmoid(mut0_pssm)
        mut1_pssm = tf.math.sigmoid(mut1_pssm)
        par0_pssm = tf.math.sigmoid(par0_pssm)
        
        mut0 = self.embedding_layer_mut([mut0, mut0_pssm])
        mut1 = self.embedding_layer_mut([mut1, mut1_pssm])
        par0 = self.embedding_layer_par([par0, par0_pssm])
        
        mut0 = self.trans_block_mut1(mut0, mut0_mask)
        mut1 = self.trans_block_mut1(mut1, mut1_mask)
        par0 = self.trans_block_par1(par0, par0_mask)
        
        mut0 = self.trans_block_mut2(mut0, mut0_mask)
        mut1 = self.trans_block_mut2(mut1, mut1_mask)
        par0 = self.trans_block_par2(par0, par0_mask)
        
        mut0 = self.trans_block_mut3(mut0, mut0_mask)
        mut1 = self.trans_block_mut3(mut1, mut1_mask)
        par0 = self.trans_block_par3(par0, par0_mask)
        
#         mut0 = self.trans_block_mut4(mut0, mut0_mask)
#         mut1 = self.trans_block_mut4(mut1, mut1_mask)
#         par0 = self.trans_block_par4(par0, par0_mask)
        
        # ------par0 attention-------
        par0_att = layers.Permute((2, 1))(par0)
        par0_att = self.att_dense_par(par0_att)
        par0_att = layers.Permute((2, 1), name='par_att_vec')(par0_att)
        par0 = layers.Multiply()([par0, par0_att])
        
        # ------par0 res-block-------
        par0_shortcut = par0
        par0 = self.leaky_relu(self.par_c1(par0))
        par0 = layers.Add()([par0_shortcut, self.par_c2(par0)])
        par0 = self.leaky_relu(par0)
        par0 = layers.MaxPooling1D()(par0)
        
        par0_shortcut = par0
        par0 = self.leaky_relu(self.par_c3(par0))
        par0 = layers.Add()([par0_shortcut, self.par_c4(par0)])
        par0 = self.leaky_relu(par0)
        par0 = layers.MaxPooling1D()(par0)
        
#         par0 = layers.Flatten()(par0)

#         par0 = layers.Permute((2, 1))(par0)
        
        # -----mut0 mut1 substract and multiply---
        mut0_mut1_sub = layers.Subtract()([mut0, mut1])
        mut0_mut1_mul = layers.Multiply()([mut0, mut1])
        
        # -----mut0 mut1 attention------
        mut0_mut1 = layers.concatenate([mut0, mut1, mut0_mut1_sub, mut0_mut1_mul], axis=1)
        # here changed the mut0_mut1 attention !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mut0_mut1_att = layers.Permute((2, 1))(mut0_mut1)
        mut0_mut1_att = self.att_dense_mut(mut0_mut1_att)
        mut0_mut1_att = layers.Permute((2, 1), name='mut_att_vec')(mut0_mut1_att)
        mut0_mut1 = layers.Multiply()([mut0_mut1, mut0_mut1_att])
        
        # -----mut0 mut1 res-block------
        mut0_mut1_shortcut = mut0_mut1
        mut0_mut1 = self.leaky_relu(self.mut_c1(mut0_mut1))
        mut0_mut1 = layers.Add()([mut0_mut1_shortcut, self.mut_c2(mut0_mut1)])
        mut0_mut1 = self.leaky_relu(mut0_mut1)
        mut0_mut1 = layers.MaxPooling1D()(mut0_mut1)
        
#         mut0_mut1_shortcut = mut0_mut1
#         mut0_mut1 = self.leaky_relu(self.mut_c3(mut0_mut1))
#         mut0_mut1 = layers.Add()([mut0_mut1_shortcut, self.mut_c4(mut0_mut1)])
#         mut0_mut1 = self.leaky_relu(mut0_mut1)
#         mut0_mut1 = layers.MaxPooling1D()(mut0_mut1)

#         mut0_mut1 = layers.Permute((2, 1))(mut0_mut1)
    
#         mut0_mut1 = layers.Flatten()(mut0_mut1)
#         print(mut0_mut1.shape)
        
        mut_par = layers.Concatenate(axis=1)([mut0_mut1, par0])
##         mut_par = layers.GlobalAveragePooling1D()(mut_par)

#         mut_par = self.all_c1(mut_par)
#         mut_par = layers.Flatten()(mut_par)
        
#         mut_par = self.dense1(mut_par)
#         mut_par = layers.Dropout(0.1)(mut_par)
        
#         outputs = self.dense2(mut_par)
        mut_par = self.seq_c2(mut_par)
        mut_par = layers.GlobalAveragePooling1D()(mut_par)
        outputs = layers.Softmax()(mut_par)
        
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({  
            'maxlen': maxlen,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'window_len': window_len,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'pos_embed_dim': pos_embed_dim,
            'seq_embed_dim': seq_embed_dim,
        })
        return config



def categorical_focal_loss(alpha, gamma=2.):
    alpha = np.array(alpha, dtype=np.float32)
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
    
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    
        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed



window_len = 51
maxlen = 1024
vocab_size = 21
embed_dim = 64
num_heads = 4
ff_dim = 64
pos_embed_dim = 64
seq_embed_dim = 44

def build_model():
    input1 = layers.Input(shape=(window_len,))
    input2 = layers.Input(shape=(window_len,))
    input3 = layers.Input(shape=(maxlen,))
    input4 = layers.Input(shape=(window_len, 20))
    input5 = layers.Input(shape=(window_len, 20))
    input6 = layers.Input(shape=(maxlen, 20))
    
    common_nn = cn(window_len=window_len, maxlen=maxlen, vocab_size=vocab_size, 
                   embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, 
                   pos_embed_dim=pos_embed_dim, seq_embed_dim=seq_embed_dim)
    
#     output1, output2, output3 = common_nn([input1, input2, input3, input4, input5, input6])
    output = common_nn([input1, input2, input3, input4, input5, input6])
#     output2 = common_nn([input2, input1, input3])
#     output3 = common_nn([input1, input1, input3])
    
#     model = keras.Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])
#     model = keras.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=[output1, output2, output3])
    model = keras.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=output)
    return model

