import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


class Blstm(nn.Module):

    def __init__(self, config):
        super(Blstm, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.blstm_layers = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.hidden_dropout_prob,
            bidirectional=True
        )

    def forward(
        self,
        input_ids,
        attention_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.word_embeddings(input_ids)
        embedding_output = self.dropout(embedding_output)
        seq_lengths = attention_mask.sum(dim=1)
        embedding_output = torch.nn.utils.rnn.pack_padded_sequence(embedding_output, seq_lengths.cpu(),
                                                                   batch_first=True, enforce_sorted=False)
        encoder_outputs = self.blstm_layers(embedding_output)
        sequence_output = encoder_outputs[0]
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(sequence_output, batch_first=True,
                                                              total_length=input_ids.shape[1])

        return sequence_output


class BlstmForTokenClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.encoder = Blstm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)  # *2 due to forward and backward LSTM

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None
    ):

        sequence_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
