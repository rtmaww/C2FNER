from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLMPredictionHead
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch

from typing import Optional,Tuple
import random
from transformers.file_utils import ModelOutput
import os



class TokenClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    proto_logits: Optional[torch.FloatTensor] = None


class MyBertForTokenClassification_prototype(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        sample_path=config.sample_path
        print(sample_path)
        print("config.model_name",config.model_name,config.sample_id)
        if os.path.exists(f"{sample_path}/{config.sample_id}.{config.model_name}.new_cluster_means.pt"):
            self.base_cluster_means = nn.Parameter(torch.load(f"{sample_path}/{config.sample_id}.{config.model_name}.new_cluster_means.pt"),requires_grad=True)
            # self.base_cluster_covs = torch.load(f"{sample_path}/{config.sample_id}.{config.model_name}.new_cluster_covs.pt")
        else:
            print("load model cluster mean")
            print(f"models/{config.model_name}/cluster_means.pt")
            self.base_cluster_means = nn.Parameter(torch.load(f"models/{config.model_name}/cluster_means.pt"),requires_grad=True)

        self.cluster_fine_labels = torch.LongTensor(torch.load(f"{sample_path}/{config.sample_id}.{config.model_name}.rb_cluster_fine_labels"))
        print(self.base_cluster_means.size(0),self.cluster_fine_labels.size(0))
        assert self.base_cluster_means.size(0) == self.cluster_fine_labels.size(0)
        
        self.use_aug = config.use_aug
        if self.use_aug:
            self.sampled_embeds = torch.load(
                f"{sample_path}/{config.sample_id}.{config.model_name}.rb_sampled_embeds")
            self.sampled_labels = torch.load(
                f"{sample_path}/{config.sample_id}.{config.model_name}.rb_sampled_label_ids")
            # self.sampled_embeds.requires_grad = True
            self.sampled_prob_list = torch.ones(self.sampled_labels.size(0)) * 0.01

        self.label_idxs = torch.LongTensor([i for i in range(1, self.num_labels)])



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
        
        
        proto_logits, proto_loss = self.prototype_dist(sequence_output, labels)
        proto_logits = proto_logits.view(sequence_output.size(0), sequence_output.size(1), -1)
        
        if labels is not None and self.use_aug:
            
            # logits = self.classifier(sequence_output)

            sampled_mask = torch.bernoulli(self.sampled_prob_list).bool()
            sampled_embed = self.sampled_embeds[sampled_mask].float().to(labels.device)
            sampled_label = self.sampled_labels[sampled_mask].to(labels.device)

            sampled_logits, sampled_loss = self.prototype_dist(sampled_embed, sampled_label)

        ## proto loss
        proto_dist = torch.cdist(self.base_cluster_means, self.base_cluster_means) # (n_c, n_c)
        positive_mask = self.cluster_fine_labels.view(-1, 1) == self.cluster_fine_labels.view(1, -1).repeat(self.cluster_fine_labels.size(0), 1)  # (n_c, n_c)
        positive_dist = proto_dist.masked_select(positive_mask).view(-1)
        proto_dist_loss = positive_dist.mean()

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


                loss += proto_loss
                loss += proto_dist_loss

                
                if self.use_aug:
                    loss += sampled_loss
                
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            proto_logits=proto_logits
        )

    def prototype_dist(self, valid_embeds, labels=None):
        
        distance = torch.cdist(valid_embeds.view(-1, valid_embeds.size(-1)), self.base_cluster_means)  # (n,n_c)
        distance = distance.unsqueeze(1).repeat(1, self.num_labels - 1, 1)  # (n,c,n_c)

        self.label_idxs = self.label_idxs.to(distance.device)
        self.cluster_fine_labels = self.cluster_fine_labels.to(distance.device)
        label_idx = self.label_idxs.view(1, -1, 1).repeat(distance.size(0), 1, self.base_cluster_means.size(0))  # (n, c, n_c)
        label_mask = self.cluster_fine_labels.view(1, 1, -1).repeat(distance.size(0), self.num_labels - 1, 1) == label_idx  # (n, c, n_c)
        # label_mask = label_mask.to(distance.device)

        # distance_to_label = torch.min(distance.masked_fill(~label_mask, value=torch.tensor(torch.finfo(float).max)), dim=-1)[0] # (n,c)
        distance_to_label = (distance * label_mask).sum(-1) / label_mask.sum(-1)

        logits = -1 * distance_to_label  # (n,c)

        loss = None
        if labels is not None:
            labels = labels.view(-1)
            valid_logits = logits.view(-1, self.num_labels - 1)[labels > 0]
            valid_labels = labels[labels > 0] - 1
            loss_func = CrossEntropyLoss()
            loss = loss_func(valid_logits, valid_labels)


        return logits, loss

    def update_cluster_label(self):
        import collections
        k_num = 6
        k_threshold = 3
        
        proto_dist = torch.cdist(self.base_cluster_means, self.base_cluster_means)  # (n_c, n_c)
        topk_dist = torch.topk(proto_dist, dim=-1, k=k_num + 1, largest=False).indices

        for cluster_idx, label_id in enumerate(self.cluster_fine_labels):
            if label_id == -1:
                topk = topk_dist[cluster_idx]
                fine_class_num = collections.defaultdict(int)
                for fine_class_id in self.cluster_fine_labels[topk]:
                    if fine_class_id != -1:
                        fine_class_num[fine_class_id.item()] += 1
                max_class_id_num = sorted([(k, v) for k, v in fine_class_num.items()], key=lambda x: x[1], reverse=True)
                if not max_class_id_num:
                    continue
                max_class_id_num = max_class_id_num[0]
                # print(max_class_id_num[1],max_class_id_num[1])
                if max_class_id_num[1] >= k_threshold:
                    update = True
                    self.cluster_fine_labels[cluster_idx] = max_class_id_num[0]
                    print(f"Label {cluster_idx} to {max_class_id_num[0]}, valid_k={max_class_id_num[1]}/{k_num}")

        unlabeled_proto_num = (self.cluster_fine_labels == -1).sum()
        print(f"Unlabeled proto num: {unlabeled_proto_num}/{self.cluster_fine_labels.size(0)}")

