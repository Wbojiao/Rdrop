import torch
from torch import nn
from torch.nn import functional as F
from modules import HanAttention
from transformers import RobertaConfig, RobertaModel
from transformers import BertConfig, BertModel
from transformers import AutoConfig, AutoModel
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class nomiLinear(nn.Module):

    def __init__(self, speechin, textin, out_features, device=None, dtype=None):
        super(nomiLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}


        self.speechin = speechin 
        self.textin = textin 

        self.in_features = speechin + textin
        self.out_features = out_features

        self.weight_1 = Parameter(torch.empty((self.in_features, self.in_features), **factory_kwargs))
        # self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.bias = None

        self.weight_2 = Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # init.uniform_(self.bias, -bound, bound)

    def _forward(self, input):
        output = F.linear(input, self.weight_1, self.bias)
        speech_op = F.linear(input[:, : self.speechin ], self.weight_1[:, : self.speechin ], self.bias)
        text_op = F.linear(input[:, self.speechin : ], self.weight_1[:, self.speechin : ], self.bias)
        return speech_op, text_op, output

    def forward(self, input):
        output = F.linear(input, self.weight_1, self.bias)
        speech_op = F.linear(input[:, : self.speechin ], self.weight_1[:, : self.speechin ], self.bias)
        text_op = F.linear(input[:, self.speechin : ], self.weight_1[:, self.speechin : ], self.bias)

        output = F.linear(output, self.weight_2, self.bias)
        speech_op = F.linear(speech_op, self.weight_2, self.bias)
        text_op = F.linear(text_op, self.weight_2, self.bias)        

        return speech_op, text_op, output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NIONM(nn.Module):

    def __init__(self, params):
        super(NIONM,self).__init__()

        self.max_sequence_length = params.max_sequence_length

        self.speech_embedding = nn.Sequential(nn.Linear(params.speech_w, params.embed_dim),
                                             nn.ReLU())

        self.speechs_batchnorm = nn.BatchNorm1d(params.speech_h)
        self.text_batchnorm = nn.BatchNorm1d(params.max_sequence_length)

        self.speech_multihead_attention = nn.MultiheadAttention(params.embed_dim,params.num_heads,dropout=params.dropout_rate)
        self.composition_speech_multihead_attention = nn.MultiheadAttention(params.hidden_size,params.num_heads,dropout=params.dropout_rate)
        self.composition_text_multihead_attention = nn.MultiheadAttention(params.hidden_size,params.num_heads,dropout=params.dropout_rate)

        self.layer_norm = nn.LayerNorm(params.embed_dim)
        self.dropout = nn.Dropout(params.dropout_rate)

        self.combined_linear = nn.Sequential(nn.Linear(4 * params.hidden_size,params.hidden_size*2),
                                             nn.ReLU(inplace=True),
                                             self.dropout,
                                             nn.Linear(2 * params.hidden_size,params.hidden_size),
                                             nn.ReLU(inplace=True),)

        self.attention_speech =  HanAttention( params.hidden_size)
        self.attention_text =  HanAttention( params.hidden_size)
      
        config = AutoConfig.from_pretrained(params.bert_model_path) 
        self.bert_model = AutoModel.from_pretrained(params.bert_model_path, config=config)
 

        #self.predictor = nomiLinear(params.hidden_size, params.hidden_size, params.n_classes)

        self.predictor = nn.Sequential(nn.Linear(params.hidden_size*2, params.hidden_size),
                                     nn.ReLU(inplace=True),
                                     self.dropout,
                                      nn.Linear(params.hidden_size, params.n_classes),
                                     )

        self.text_fc = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size),
                                     nn.ReLU(inplace=True),
                                     self.dropout,
                                      nn.Linear(params.hidden_size, params.n_classes),
                                     )

        self.speech_fc = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size),
                                       nn.ReLU(inplace=True),
                                       self.dropout,
                                       nn.Linear(params.hidden_size, params.n_classes),
                                )

    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)#将tensor维度互换

        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
        # BATCH,100,34  --- BATCH, 100, 768

        # text
        # BATCH,100,34  --- BATCH, 100, 768
        input_ids, attention_mask = sentences['input_ids'],sentences['attention_mask']
        outputs  = self.bert_model(input_ids)
        text_embedding = outputs.last_hidden_state
        text_enc = self.text_batchnorm(text_embedding)
        text_enc = self.dropout(text_enc)
        text_enc = text_embedding + text_enc
        # BATCH,100,34  --- BATCH, 100, 768

        # local inference layer
        # BATCH,100,768, BATCH,100,768
        speechs_combined , text_combined = speech_enc, text_enc

        # speech
  
        # BATCH,100,768
        speechs_combined = speechs_combined.permute(1,0,2)
        speechs_combined, _ = self.composition_speech_multihead_attention(speechs_combined,speechs_combined,speechs_combined)
        speechs_combined = speechs_combined.permute(1,0,2)
        speech_attention = self.attention_speech(speechs_combined)
        # BATCH,100,768


        # text
        # BATCH,100,768
        text_combined = text_combined.permute(1,0,2)
        text_combined, _ = self.composition_text_multihead_attention(text_combined,text_combined,text_combined)
        text_combined = text_combined.permute(1,0,2)
        text_attention = self.attention_text(text_combined)
        # BATCH,100,768

        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)

        # BATCH,768
        prob = self.predictor(cat_compose)
        speech_prob = self.speech_fc(speech_attention)
        text_prob = self.text_fc(text_attention)
        
        return speech_prob, text_prob, prob



