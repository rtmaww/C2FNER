from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch



class MyBertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = nn.Sequential(nn.Linear(config.hidden_size, 256), nn.ReLU(), nn.Linear(256, config.num_labels))
        self.classifier = torch.nn.Parameter(torch.randn(config.hidden_size, config.num_labels), requires_grad=True)
        self.classifier.data.normal_(mean=0.0, std=config.initializer_range)


        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.hidden_states[-1] # -2, -3

        sequence_output = self.dropout(sequence_output)

        # logits = self.classifier(sequence_output)
        # logits_ = logits

        # # amsoftmax
        x = sequence_output.view(-1, sequence_output.size(-1))
        lb = labels.view(-1)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.classifier, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.classifier, w_norm)
        logits_ = torch.mm(x_norm, w_norm)
        logits = logits_.view(sequence_output.size(0), sequence_output.size(1), -1)

        lb_mask = lb < 0
        lb.masked_fill_(lb_mask, 0)
        delt_costh = torch.zeros_like(logits_).scatter_(1, lb.unsqueeze(1), -0.2)
        # delt_costh = -0.1 * ((labels.view(-1,1).repeat(1, self.num_labels) == torch.tensor([i  for i in range(self.num_labels)]).cuda() ) == 6 )
        # # costh_m =
        logits_ = 15 * (logits_ - delt_costh)
        # logits_ = 15* logits
        # logits = logits_.view(sequence_output.size(0), sequence_output.size(1), -1)




        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(weight=torch.tensor([1.0]*(self.num_labels-1)+[0.8]).cuda())
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits_.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





class MyBertForTokenClassification_clusterloss(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        cluster_center = torch.load("./cluster/models/coarseft_new/kmeans/bert-base-cased_layer-1_50eachclass_cluster_protos.pt")

        self.cluster_proto = nn.Parameter(cluster_center, requires_grad=True)
        self.use_cluster_loss = True
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 128)
        )

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cluster_labels=None,
        entity_masks=None,
        entity_cluster_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if cluster_labels is not None :


            embeds = outputs[0].unsqueeze(1).repeat(1,entity_masks.size(1),1,1) # (b,ne,l,h)
            entity_embeds = ( entity_masks.unsqueeze(-1) * embeds ).sum(-2) / (entity_masks.sum(-1, keepdim=True)+1e-10) # (b,ne,h)
            entity_cluster_labels = entity_cluster_labels.view(-1)
            valid_embeds = entity_embeds.view(-1, entity_embeds.size(-1))[entity_cluster_labels >= 0] # (n,h)
            valid_cluster_label = entity_cluster_labels[entity_cluster_labels >= 0]
            valid_cluster_proto = self.cluster_proto.data

            # valid_embeds = self.head(valid_embeds)
            # valid_cluster_proto = self.head(valid_cluster_proto)


            cluster_loss = self.cluster_margin_loss(valid_embeds, valid_cluster_proto, valid_cluster_label, margin=2., topk=200)




            if self.use_cluster_loss:
                loss += cluster_loss


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def cluster_margin_loss(self, valid_embeds, valid_cluster_proto, valid_cluster_label, margin=5., topk=10.):

        # valid_embed: (n, h)
        # valid_cluster_proto: (1000, h)
        # valid_cluster_label: (n, )

        valid_embeds = valid_embeds.unsqueeze(1).repeat(1, valid_cluster_proto.size(0), 1)  # (n, 1000, h)
        protos = valid_cluster_proto.unsqueeze(0).repeat(valid_embeds.size(0), 1, 1)  # (n, 1000, h)
        distance = torch.pow(valid_embeds - protos, 2).sum(-1).float().sqrt()  # (n, 1000)
        positive_dist = distance.gather(1, valid_cluster_label.view(-1, 1))  # (n, 1)

        ### margin loss
        # positive_mask = torch.eye(distance.size(-1))[valid_cluster_label.view(-1)].to(distance.device) # (n, 1000), only positive = 1

        ### class label for choosing negative
        # valid_cluster_class_label = valid_cluster_label.view(-1)/50
        class_cluster_num = valid_cluster_proto.size(0) // (self.num_labels-1)
        positive_mask = nn.functional.one_hot(valid_cluster_label.view(-1)//class_cluster_num, self.num_labels-1)  # (n,class_num)
        positive_mask = positive_mask.unsqueeze(-1).repeat(1,1,class_cluster_num).view(valid_embeds.size(0),-1) # (n,class_num*50)
        ## min as negative
        negative_dist = torch.min(
            distance.masked_fill(positive_mask.bool(), value=torch.tensor(torch.finfo(float).max)), dim=-1)[0]  # (n,)


        margin_loss = nn.MarginRankingLoss(margin=margin)
        target = torch.ones(negative_dist.size(0)).to(negative_dist.device)
        cluster_loss = margin_loss(negative_dist.view(-1), positive_dist.view(-1), target)


        return cluster_loss
