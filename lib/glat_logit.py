import torch.nn as nn
import torch.nn.functional as F
from lib.layers import GraphConvolution, EncoderLayer, GraphAttentionLayer, GraphAtt_Mutlihead_Basic, \
    PositionwiseFeedForward
import torch
# import Constants
import pdb
from torch.autograd import Variable

def get_non_pad_mask(seq, node_type):
    assert seq.dim() == 2
    # b, n
    mask = node_type != 2
    return Variable(mask.unsqueeze(-1).float())
    # return seq.ne(blank).unsqueeze(-1).type(torch.float)


def get_attn_key_pad_mask(seq_q, node_type):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.

    len_q = seq_q.size(1) # b, n
    padding_mask = node_type == 2
    # padding_mask = seq_k.eq(blank)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
                                                                    # b, n, n
    return Variable(padding_mask)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, non_pad_mask):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        x *= non_pad_mask

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)

        x *= non_pad_mask

        x = F.elu(x)


        return x


class GAT_Ensemble(nn.Module):
    def __init__(self, vocab_num, nfeat, nhid, noutput, dropout, alpha, nheads, GAT_num):
        """Dense version of GAT."""
        super(GAT_Ensemble, self).__init__()
        print("initialize GAT with module num = : ", GAT_num)

        self.GAT_num = GAT_num
        # self.GATs = nn.ModuleList()
        # print(type(vocab_num))
        # print(type(nfeat))
        self.embed = nn.Embedding(vocab_num, nfeat)

        # for num in range(self.GAT_num):
        #     model = GAT(nfeat, nhid, noutput, dropout, alpha, nheads)
        #     self.GATs.append(model)

        self.GATs = nn.ModuleList([
            GAT(nfeat, nhid, noutput, dropout, alpha, nheads)
            for _ in range(self.GAT_num)
        ])

    def forward(self, fea, adj, non_pad_mask):
        fea = fea.long()
        x = self.embed(fea)
        # x = x.squeeze(2)

        for GAT in self.GATs:
            x = GAT(x, adj, non_pad_mask)
        # for num in range(self.GAT_num):
        #     x = self.GATs[num](x, adj, non_pad_mask)
        return x


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):

        super().__init__()

        # n_position = len_max_seq + 1

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc = 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, slf_attn_mask, non_pad_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        # slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        # non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = src_seq
        # enc_output = torch.unsqueeze(enc_output, 0)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            # enc_output, enc_slf_attn = enc_layer(
            #     enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k,
                               d_v=d_v, dropout=dropout)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, slf_attn_mask, non_pad_mask):
        src_pos = src_seq
        enc_output, *_ = self.encoder(src_seq, src_pos, slf_attn_mask, non_pad_mask)

        return enc_output


# class Transformer_Ensemble(nn.Module):
#     def __init__(self, Trans_num, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64,
#                  dropout=0.1):
#         """Ensembled version of Trans."""
#         super(Transformer_Ensemble, self).__init__()
#         print("initialize Trans with module num = : ", Trans_num)
#
#         self.Trans_num = Trans_num
#         self.Trans = nn.ModuleList()
#
#         for num in range(self.Trans_num):
#             model = Transformer(d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers,
#                                 n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
#             self.Trans.append(model)
#
#     def forward(self, x, slf_attn_mask, non_pad_mask):
#         for num in range(self.Trans_num):
#             x = self.Trans[num](x, slf_attn_mask, non_pad_mask)
#         return x


class Connect_Cls(nn.Module):
    def __init__(self, in_features, mid_features, n_class, bias=True):
        super(Connect_Cls, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.n_class = n_class
        # self.balanced_ratio = 0.5
        self.FC = nn.Sequential(
            nn.Linear(2*self.in_features, self.mid_features),
            nn.BatchNorm1d(self.mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.mid_features, self.n_class)
        )

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, adj):
        B = input.size(0)
        N = input.size(1)
        D = input.size(2)

        conn_fea = torch.cat([input.repeat(1, 1, N).view(B, N * N, -1), input.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * D) # (B, N, N, 2D)
        conn_fea = conn_fea.view(B, -1, 2*D).view(-1, 2*D)
        conn_adj = adj.view(B, -1).view(-1)

        # pos_conn = torch.nonzero(conn_adj).squeeze(-1)
        # neg_conn = torch.nonzero(1-conn_adj).squeeze(-1)[:len(pos_conn)]
        #
        # pos_fea = conn_fea[pos_conn]
        # neg_fea = conn_fea[neg_conn]
        # total_fea = torch.cat([pos_fea, neg_fea], dim=0)

        # pdb.set_trace()

        x = self.FC(conn_fea)
        # x = self.FC(total_fea)

        # conn = torch.nonzero(adj)
        # pos_input = []
        # conn_num = len(conn)
        # if conn_num != 0:
        #     for i in range(conn_num):
        #         pos_input.append(torch.cat([input[conn[i][0]], input[conn[i][1]]], dim=-1).unsqueeze(0))
        #     pos_input = torch.cat(pos_input, dim=0)
        # else:
        #     pos_input = torch.zeros(size=(0,0), device='cuda')
        #
        # neg_input = []
        # disconn = torch.nonzero(1-adj)
        # if len(conn) <= len(disconn):
        #     disconn_num = len(conn) if len(conn) != 0 else 2
        # else:
        #     disconn_num = len(disconn)
        # for i in range(disconn_num):
        #     neg_input.append(torch.cat([input[disconn[i][0]], input[disconn[i][1]]], dim=-1).unsqueeze(0))
        # neg_input = torch.cat(neg_input, dim=0)
        #
        # total_input = torch.cat((pos_input, neg_input), dim=0)
        # x = self.FC1(total_input)
        # x = self.FC2(x)
        # return x, [len(pos_conn), len(neg_conn)]

        x = self.softmax(x)

        return x


class Pred_label(nn.Module):
    def __init__(self, model):
        super(Pred_label, self).__init__()
        # embed_shape_predicate = model.embed_predicate.weight.shape

        embed_shape_predicate_logit = model.embed_predicate_logit.weight.shape
        embed_shape_predicate = model.embed_predicate_logit.weight.shape

        embed_shape_entity = model.embed_predicate.weight.shape

        # self.FC = nn.Linear(embed_shape_predicate[1], embed_shape_predicate[1])
        # self.FC = nn.Linear(embed_shape_predicate_logit[0], embed_shape_predicate_logit[0])
        self.FC = nn.Linear(embed_shape_predicate[0], embed_shape_predicate[0])

        print("embed_shape_predicate: ", embed_shape_predicate)

        self.decoder_predicate = nn.Linear(embed_shape_predicate[1], embed_shape_predicate[0], bias=False)
        self.decoder_predicate.weight = model.embed_predicate.weight
        # self.decoder_predicate.weight = model.embed_predicate_logit.weight

        self.decoder_predicate_logit = nn.Linear(embed_shape_predicate_logit[1], embed_shape_predicate_logit[0], bias=False)
        # self.decoder_predicate_logit.weight = model.embed_predicate_logit.weight
        self.decoder_predicate_logit.weight.data = model.embed_predicate_logit.weight.data.t()

        self.decoder_entity = nn.Linear(embed_shape_entity[1], embed_shape_entity[0], bias=False)
        self.decoder_entity.weight = model.embed_entity.weight

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, h, node_type):
        h = self.FC(h)
        b_size, n_num, _ = h.size()

        # pdb.set_trace()

        predicate, predicate_order_list, entity, entity_order_list, blank, blank_order_list, _ = split(h, node_type)

        # pdb.set_trace()

        predicate_one_hogt = self.decoder_predicate(predicate)
        predicate_logit = self.decoder_predicate_logit(predicate)

        # predicate = self.decoder_predicate_logit(predicate)
        entity = self.decoder_entity(entity)
        if len(blank.size()) != 0:
            blank = self.decoder_entity(blank)
            blank_logits = self.softmax(blank)
            _, blank_labels = torch.max(blank_logits, dim=-1, keepdim=True)
        else:
            blank_labels = blank

        # lm_logits = combine(predicate, predicate_order_list, entity, entity_order_list, b_size, n_num)
        # pdb.set_trace()

        predicate_logits = predicate_logit
        entity_logits = entity

        predicate_conf = self.softmax(predicate_logits)
        entity_conf = self.softmax(entity_logits)

        # lm_logits = self.decoder(h)
        # predicate_logits = self.softmax(predicate)
        # entity_logits = self.softmax(entity)

        # pdb.set_trace()
        # if self.training:
        #     return predicate_logits, entity_logits
        # else:
        _, predicate_labels = torch.max(predicate_conf, dim=-1, keepdim=True)
        _, entity_labels = torch.max(entity_conf, dim=-1, keepdim=True)

        all_labels = combine(predicate_labels, predicate_order_list, entity_labels, entity_order_list, blank_labels, blank_order_list, b_size, n_num, predicate_labels)

        return predicate_conf, entity_conf, predicate_logits, entity_logits, all_labels


class GLAT_basic(nn.Module):
    def __init__(self, foc_type, att_type, d_model, nout,  n_head, d_k=64, d_v=64, dropout=0.1, d_inner=2048):

        super(GLAT_basic, self).__init__()

        self.slf_attn = GraphAtt_Mutlihead_Basic(
            n_head, d_model, d_k, d_v, foc_type, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, adj, non_pad_mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, adj, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        enc_output *= non_pad_mask

        return enc_output


class GLAT(nn.Module):
    def __init__(self, fea_dim, nhid_glat_g, nhid_glat_l, nout, dropout, nheads, type):
        super(GLAT, self).__init__()
        self.type = type
        if type == 0:
            self.GLAT_G = GLAT_basic("global", "do_product", fea_dim, nout, nheads, dropout=dropout)
        elif type == 1:
            self.GLAT_L = GLAT_basic("local", "do_product", fea_dim, nout, nheads, dropout=dropout)
        elif type == 2:
            self.GLAT_G = GLAT_basic("global", "do_product", fea_dim, nout, nheads, dropout=dropout)
            self.GLAT_L = GLAT_basic("local",  "do_product", fea_dim, nout, nheads, dropout=dropout)
            self.fc = nn.Linear(2 * int(nout), int(nout))
        else:
            raise("wrong model type")

        # self.GLAT_L = GAT(fea_dim, nhid, nout, dropout, alpha, nheads)
        # self.GLAT_G = EncoderLayer(d_model=nhid, d_inner=2048, n_head=nheads, d_k=64, d_v=64, dropout=dropout)


    def forward(self, x, adj, non_pad_mask=None, slf_attn_mask=None):
        if self.type == 0:
            x = self.GLAT_G(x, adj, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)  # output b, n, dim
        elif self.type == 1:
            x = self.GLAT_L(x, adj, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        elif self.type == 2:
            x_g = self.GLAT_G(x, adj, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask) # output b, n, dim
            x_l = self.GLAT_L(x, adj, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            x = torch.cat([x_g, x_l], dim=-1) # output b, n, 2*dim
            x = self.fc(x)
        return x


def split(fea, node_type, node_logit=[]):
    batch_size = fea.size(0)
    num_node = fea.size(1)
    fea_flatten = fea.view(batch_size*num_node, -1)
    dim = fea_flatten.size()[-1]

    if len(node_logit) != 0:
        node_logit_flatten = node_logit.view(batch_size*num_node, -1)
        dim_logit = node_logit_flatten.size()[-1]


    # torch.repeat(node_type, (-1, dim))
    node_type_flatten = node_type.view(-1, 1).expand(-1, dim)


    # order_list = torch.tensor(range(0, len(fea_flatten)))
    order_list = torch.Tensor(range(0, len(fea_flatten))).cuda()
    predicate_mask = node_type_flatten == 0
    entity_mask = node_type_flatten == 1
    blank_mask = node_type_flatten == 2


    predicate = fea_flatten[predicate_mask].view(-1, dim)
    if len(node_logit) != 0:
        # predicate_logit = node_logit[predicate_mask].view(-1, dim_logit)
        predicate_logit = node_logit.view(-1, 52)[predicate_mask.expand(-1, 52)].view(-1, 52)
    else:
        predicate_logit = []
    predicate_order_list = order_list[predicate_mask[:, 0]]

    entity = fea_flatten[entity_mask].view(-1, dim)
    entity_order_list = order_list[entity_mask[:, 0]]

    blank = fea_flatten[blank_mask].view(-1, dim)
    blank_order_list = order_list[blank_mask[:, 0]]

    if len(blank.size()) != 0:
        blank = blank.squeeze(-1)

    # pdb.set_trace()

    return predicate.squeeze(-1), predicate_order_list, entity.squeeze(-1), entity_order_list, blank, blank_order_list, predicate_logit


def combine(predicate, predicate_order_list, entity, entity_order_list, blank, blank_order_list, b_size, n_num, predicate_logit):

    # pdb.set_trace()
    if len(blank.size()) != 0:
        # fea_embed = torch.cat((entity, predicate, blank), 0)
        fea_embed = torch.cat((entity, predicate_logit, blank), 0)
        order_list = torch.cat((entity_order_list, predicate_order_list, blank_order_list), 0)
    else:
        # fea_embed = torch.cat((entity, predicate), 0)
        fea_embed = torch.cat((entity, predicate_logit), 0)
        order_list = torch.cat((entity_order_list, predicate_order_list), 0)

    # pdb.set_trace()
    new_fea = [f.unsqueeze(0) for _, f in sorted(zip(order_list, fea_embed))]

    # order_list = [o for o, _ in sorted(zip(order_list, fea_embed))]
    # pdb.set_trace()

    new_fea = torch.cat(new_fea, 0)
    new_fea = new_fea.view(b_size, n_num, new_fea.size(-1))

    return new_fea.squeeze(-1)


class GLAT_Seq(nn.Module):
    def __init__(self, vocab_num, fea_dim, nhid_glat_g, nhid_glat_l, nout, dropout, nheads, types):
        super(GLAT_Seq, self).__init__()
        self.num = len(types)
        # self.embed = nn.Embedding(vocab_num, fea_dim)
        self.embed_predicate = nn.Embedding(vocab_num[0], fea_dim)

        # self.embed_predicate_logit = nn.Linear(fea_dim, vocab_num[0])
        self.embed_predicate_logit = nn.Linear(vocab_num[0], fea_dim)
        # self.embed_predicate_logit = nn.Embedding(vocab_num[0], fea_dim)

        self.embed_entity = nn.Embedding(vocab_num[1], fea_dim)

        self.GLATs = nn.ModuleList()

        for i in range(self.num):
            model = GLAT(fea_dim, nhid_glat_g, nhid_glat_l, nout, dropout, nheads, int(types[i]))
            self.GLATs.append(model)



    def forward(self, fea, adj, node_type, node_logit, non_pad_mask=None, slf_attn_mask=None):
        fea = fea.long()
        # x = self.embed(fea)
        b_size, n_num = fea.size()

        # fea_flatten = fea.view(-1)
        # node_type_flatten = node_type.view(-1)
        #
        # order_list = torch.tensor(range(0, len(fea_flatten)))
        # predicate_mask = node_type_flatten == 0
        # entity_mask = node_type_flatten != 0
        # # pdb.set_trace()
        # predicate = fea_flatten[predicate_mask]
        # predicate_order_list = order_list[predicate_mask]
        #
        # entity = fea_flatten[entity_mask]
        # entity_order_list = order_list[entity_mask]

        predicate, predicate_order_list, entity, entity_order_list, blank, blank_order_list, predicate_logit = split(fea, node_type, node_logit)
        predicate = self.embed_predicate(predicate)
        entity = self.embed_entity(entity)
        predicate_logit = self.embed_predicate_logit(predicate_logit)

        if len(blank.size()) != 0:
            blank = self.embed_entity(blank)

        # fea_embed = torch.cat((entity, predicate), 0)
        # order_list = torch.cat((entity_order_list, predicate_order_list), 0)
        #
        # new_fea = [f.unsqueeze(0) for _, f in sorted(zip(order_list, fea_embed))]
        #
        # order_list = [o for o, _ in sorted(zip(order_list, fea_embed))]
        # # pdb.set_trace()
        #
        # new_fea = torch.cat(new_fea, 0)
        # new_fea = new_fea.view(b_size, n_num, new_fea.size(-1))

        new_fea = combine(predicate, predicate_order_list, entity, entity_order_list, blank, blank_order_list, b_size, n_num, predicate_logit)

        # pdb.set_trace()

        # new_fea = new_fea.view(b_size, n_num, -1)

        for num in range(self.num):
            x = self.GLATs[num](new_fea, adj, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        return x


class GLATNET(nn.Module):
    def __init__(self, vocab_num, feat_dim, nhid_glat_g, nhid_glat_l, nout, dropout, nheads, blank, types):
        """Dense version of GAT."""
        super(GLATNET, self).__init__()
        print("initialize GlATNET with types: ", types)

        self.GLAT_Seq = GLAT_Seq(vocab_num, feat_dim, nhid_glat_g, nhid_glat_l, nout, dropout, nheads, types)

        self.Pred_label = Pred_label(self.GLAT_Seq)
        self.Pred_connect = Connect_Cls(nhid_glat_g, int(nhid_glat_g/ 2), 3)

        self.blank = blank

    def forward(self, fea, adj, node_type, node_logit):
        slf_attn_mask = get_attn_key_pad_mask(seq_q=fea, node_type=node_type)
        non_pad_mask = get_non_pad_mask(seq=fea, node_type=node_type)

        x = self.GLAT_Seq(fea, adj, node_type, node_logit, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        # pdb.set_trace()

        pred_label = self.Pred_label(x, node_type)
        # pred_edge = self.Pred_connect(x, adj)

        # return pred_label, pred_edge
        return pred_label, None


class Baseline(nn.Module):
    def __init__(self, vocab_num, GAT_num, Trans_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads, blank, fc = False):
        """Dense version of GAT."""
        super(Baseline, self).__init__()
        print("initialize Encoder with GAT num ", GAT_num, " Bert num ", Trans_num)
        self.fc = fc
        if self.fc:
            self.embed = nn.Embedding(vocab_num, feat_dim)
            self.Pred_label = Pred_label(self)
            self.fcs = nn.ModuleList(
                nn.Linear(feat_dim, feat_dim/2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim/2, feat_dim/2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim/2, feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.GAT_Ensemble = GAT_Ensemble(vocab_num, feat_dim, nhid_gat, nhid_trans, dropout, alpha, nheads, GAT_num)
            self.Trans_Ensemble = Transformer(n_layers=Trans_num, d_word_vec=nhid_trans, d_model=nhid_trans, dropout=dropout)
            self.Pred_label = Pred_label(self.GAT_Ensemble)

        # self.Pred_label = Pred_label(self)

        print("nhid_trans: ", nhid_trans)
        print("int(nhid_trans/2): ", int(nhid_trans/2))

        self.Pred_connect = Connect_Cls(nhid_trans, int(nhid_trans/2), 3)

        self.blank = blank


    def forward(self, fea, adj):
        slf_attn_mask = get_attn_key_pad_mask(seq_k=fea, seq_q=fea, blank=self.blank)
        non_pad_mask = get_non_pad_mask(fea, blank=self.blank)

        # non_pad_mask = None
        # slf_attn_mask = None

        # x = self.embed(fea)
        if self.fc:
            x = self.fcs(fea)
        else:
            x = self.GAT_Ensemble(fea, adj, non_pad_mask)
            x = self.Trans_Ensemble(x, slf_attn_mask, non_pad_mask)

        pred_label = self.Pred_label(x)
        pred_edge = self.Pred_connect(x, adj)

        return pred_label, pred_edge
