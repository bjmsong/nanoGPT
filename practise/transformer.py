# 不完整：decoder部分没有实现

import numpy as np
import torch
from torch import nn, optim


def make_batch(sentences):
    """
    input_batch: (batch_size(1), src_len)
    output_batch: (batch_size(1), tgt_len)
    target_batch: (batch_size(1), tgt_len)
    """
    # 编码
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]

    # 转换成LongTensor
    return torch.LongTensor(input_batch), torch.LongTensor(
        output_batch), torch.LongTensor(target_batch)


def get_sinusoid_encoding_table(n_position, d_model):
    """
    返回位置编码的embedding矩阵
    """

    # 弧度，即位置编码的公共部分：pos/10000^(2i/d_model)
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    # (n_position, d_model)
    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # 偶数位置
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # 奇数位置
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q 和 seq_k 是q/k的序列，不一定一致，例如在cross attention中，q来自解码端，k来自编码端
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token，标记为1
    # unsqueeze(1): 在指定维度(这里是 1)处增加一个新的维度，扩展了张量的形状。(batch_size,len_k) -> (batch_size,1,len_k)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # expand: 将张量沿着某些维度进行扩展, expand方法并不会复制原始数据，而是使用广播的方式共享原始数据
    # (batch_size,1,len_k) -> (batch_size,len_q,len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # (batch_size, n_heads, seqLen, seqLen)
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0
        attention_score.masked_fill_(attn_mask, -1e9)
        # 在最后一个维度上进行softmax操作, (batch_size, n_heads, seqLen, seqLen)
        attn = nn.Softmax(dim=-1)(attention_score)
        # (batch_size, n_heads, seqLen, d_v)
        context = torch.matmul(attn, V)
        return context, attn

# 核心模块！
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 参数矩阵，用线性层来表示
        self.W_Q = nn.Linear(d_model, d_k*n_heads)
        self.W_K = nn.Linear(d_model, d_k*n_heads)
        self.W_V = nn.Linear(d_model, d_v*n_heads)
        self.linear = nn.Linear(d_v*n_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size()[0]
        """
        投影->分头->交换张量维度： (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        投影 self.W_Q(Q): (batch_size, seqLen, d_model) -> (batch_size, seqLen, d_k * n_heads)
        分头 self.W_Q(Q).view(batch_size, -1, n_heads, d_k): (batch_size,seqLen,d_k * n_heads)->(batch_size,seqLen,n_heads,d_k)
        交换张量维度 transpose(1,2): 维度1和2交换位置, (batch_size,seqLen,n_heads, d_k) -> (batch_size, n_heads, seqLen, d_k)
        """
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        w_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_K(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # (batch_size, seqLen, len_k) -> (batch_size, n_heads, seqLen, len_k)，就是把pad信息重复到了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = self.self_attention(q_s, w_s, v_s, attn_mask)
        # 多头注意力合并到一起，(batch_size, seqLen, n_heads*d_v)
        context = context.transpose(1, 2).continuous().view(batch_size, -1, n_heads*d_v)
        # (batch_size, seqLen, d_model)
        output = self.linear(context)
        return self.layer_norm(output) + residual, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # nn.Conv1d是用于一维输入的卷积层, 卷积核是一维。nn.Conv2d用于二维输入的卷积层，例如图像数据
        # 用kernel_size=1卷积层来代替全连接层：减少模型的参数数量
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        # (batch_size, seqLen, d_model) -> (batch_size, d_model, seqLen)
        output = nn.ReLu()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Args:
            enc_inputs: (batch_size,src_len,d_model)
            enc_self_attn_mask (_type_): _description_

        Returns:
            enc_outputs: (batch_size, seqLen, d_model)
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码: 以get_sinusoid_encoding_table()的结果作为embedding层
        # freeze=True：embedding层的权重被冻结，即在训练过程中不会更新
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)  # (batch_size, seqlen) -> (batch_size, seqlen, embedding_size)
        # 最后一位是填充的，用0表示位置
        enc_outputs = enc_outputs + self.pos_emb(torch.LongTensor([1, 2, 3, 4, 0]))
        # 标识句子中哪些位置是被填充(pad)的
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask) 
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 最后的输出层，输入维度是embedding size，输出维度是词表的大小
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outpus, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outpus)
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(
            -1,
            dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}  # 遍历index, key
    src_len = 5
    tgt_len = 5

    # 模型超参数
    d_model = 512  # Embedding Size
    d_ff = 512*4  # FeedForward dimension
    d_k = d_v = 64  # K(=Q), V向量的维度
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        # 梯度归零
        optimizer.zero_grad()
        # 因为只有一个样本，就不需要mini-batch遍历样本了
        # 前向传播
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs)
        """
        outputs: (tgt_len, tgt_vocab_len), 字典中每个字的概率，不需要归一化
        torch.tensor.contiguous(): 返回一个内存连续的张量, 内存连续的张量在计算时更加高效 
            torch.tensor.is_contiguous(): 判断是否内存连续
        torch.tensor.view(-1): 调整张量形状，将一个任意形状的张量变成一个一维张量 
        target_batch.contiguous().view(-1): (tgt_len)
        """
        # 计算loss
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:{}  cost={}'.format(epoch + 1, loss))
        # 反向传播
        loss.backward()
        # 更新模型权重
        optimizer.step()

    # test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    """
    predict.data.max(1, keepdim=True)[1]: (tgt_len, 1)
        torch.tensor.data: 返回一个不带梯度信息的张量
        torch.tensor.max(1, keepdims=True): 在第1个维度(即行)上找到最大值，并在结果张量中保留该维度
    """
    predict = predict.data.max(1, keepdim=True)[1]
    # predict.squeeze(): 从张量中移除维度大小为1的维度, (tgt_len, 1) -> (tgt_len)
    print(sentences[0], '->',
          [number_dict[n.item()] for n in predict.squeeze()])
