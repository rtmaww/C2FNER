from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
import json
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForSequenceClassification, \
    AutoConfig, default_data_collator, DataCollatorWithPadding, BertForTokenClassification



class DataCollatorForLMTokanClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        cluster_labels = [feature['cluster_labels'] for feature in features] if 'cluster_labels' in features[
            0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            # batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
            #                        ori_labels]
            batch['cluster_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                       cluster_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            # batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
            #                        ori_labels]
            batch["cluster_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                       cluster_labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


def torch_cov(input_vec: torch.tensor):
    x = input_vec - torch.mean(input_vec, axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0] - 1)
    return cov_matrix



def bulid_dataloader_token(task_name=None, train_file='', max_length=512):
    data_collator = DataCollatorForLMTokanClassification(
        tokenizer, pad_to_multiple_of=None
    )
    if task_name == None:
        data_files = {}
        data_files["train"] = train_file
        # data_files["validation"] = eval_file
        extension = train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset(task_name)

    # raw_datasets = load_dataset("glue", task_name)
    # sentence1_key, sentence2_key = task_to_keys[task_name]
    text_column_name = 'text'
    label_column_name = "label"
    padding = False
    max_length = 512

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        # ori_labels = []
        cluster_labels = []
        # gold_labels = examples[label_column_name]
        # if args.label_schema=='IO':
        #     gold_labels = [['I-{}'.format(l[2:]) if l !='O' else 'O' for l in label] for label in gold_labels]
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            cluster_label = examples["cluster_label"][i] if "cluster_label" in examples else [-1] * len(label)
            previous_word_idx = None
            label_ids = []
            cluster_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    cluster_ids.append(-1)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label2id[label[word_idx]])
                    cluster_ids.append(cluster_label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # label_ids.append(-1)
                    label_ids.append(label2id[label[word_idx]] if not label[word_idx].startswith("B-") else label2id[
                        "I-" + label[word_idx][2:]])
                    cluster_ids.append(cluster_label[word_idx])
                    # label_ids.append(label2id[label[word_idx]] )
                previous_word_idx = word_idx

            labels.append(label_ids)
            cluster_labels.append(cluster_ids)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["cluster_labels"] = cluster_labels

        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    # eval_dataset = processed_raw_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=batch_size
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    return train_dataloader



def retrieve_cluster_labels_robust(model, support_dataloader, base_cluster_means, base_cluster_covs, target_layer=-1, mapping=None):
    # total_entities = {}
    model.eval()
    pro_bar = tqdm(range(len(support_dataloader)))
    step = 0
    device = model.device


    all_entity_embeddings = []
    all_entity_labels = []
    entity_label_names = []
    for batch in support_dataloader:
        step += 1
        with torch.no_grad():
            batch = {key: value.to(device) for key, value in batch.items()}
            batch['output_hidden_states'] = True
            labels = batch.pop("labels")
            _ = batch.pop("cluster_labels")
            outputs = model(**batch)
            input_ids = batch['input_ids']
            # labels = batch["labels"]
            target_hidden_states = outputs.hidden_states[target_layer]

            # attention_mask = batch['attention_mask']
            valid_ids = labels != -100
            input_ids = input_ids[valid_ids]
            target_hidden_states = target_hidden_states[valid_ids]
            labels = labels[valid_ids]

            entity = ""
            entity_label = ""
            entity_embeddings = []
            last_label = None
            for token_id, label_id, token_hidden_states in zip(input_ids, labels, target_hidden_states):
                token_id = token_id.item()
                label_id = label_id.item()

                if id2label[label_id].startswith("B-"):
                    if entity:
                        all_entity_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                        all_entity_labels.append(IO_label2id["I-" + last_label[2:]])
                        entity_label_names.append("I-" + last_label[2:])
                        entity = ""
                        entity_embeddings = []
                    entity = tokenizer._convert_id_to_token(token_id)
                    entity_embeddings.append(token_hidden_states)

                elif id2label[label_id].startswith("I-"):
                    entity += "_" + tokenizer._convert_id_to_token(token_id)
                    entity_embeddings.append(token_hidden_states)
                else:
                    if entity:
                        all_entity_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                        all_entity_labels.append(IO_label2id["I-" + last_label[2:]])
                        entity_label_names.append("I-" + last_label[2:])
                        entity = ""
                        entity_embeddings = []
                last_label = id2label[label_id]
            if entity:
                all_entity_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                all_entity_labels.append(IO_label2id["I-" + last_label[2:]])
                entity_label_names.append("I-" + last_label[2:])
        pro_bar.update(1)

    all_entity_embeddings = torch.stack(all_entity_embeddings)  # (n, h)
    all_entity_labels = torch.LongTensor(all_entity_labels)  # (n,)

    c_cluster_num = base_cluster_means.size(0) // len(coarse_label2id)
    coarse_masks = torch.eye(base_cluster_means.size(0) // c_cluster_num).unsqueeze(-1).repeat(1, 1,c_cluster_num).view(base_cluster_means.size(0) // c_cluster_num, -1)  # (8, 400)
    coarse_masks = (1 - coarse_masks).bool().to(all_entity_embeddings.device)

    ####### start assigning cluster
    cluster_fine_labels = [-1] * base_cluster_means.size(0)
    cluster_nearest_dist = [-1] * base_cluster_means.size(0)
    cluster_fine_label_names = [-1] * base_cluster_means.size(0)

    ### first record nearest cluster statistics for each class
    class_entity_nearest_cluster = {}
    for support_idx in range(all_entity_labels.size(0)):
        support_embed = all_entity_embeddings[support_idx]  # (1,h)
        support_label = all_entity_labels[support_idx]
        support_label_name = entity_label_names[support_idx]
        if mapping is not None:
            label_mapped = mapping[support_label_name.split("-")[1]]
            if isinstance(label_mapped, list):
                coarse_id = [coarse_label2id[i] for i in label_mapped]
            else:
                coarse_id = coarse_label2id[label_mapped]
        else:
            coarse_id = coarse_label2id[support_label_name.split("-")[1]]
        dist = torch.pow(support_embed.view(1, -1).repeat(base_cluster_means.size(0), 1)
                         - base_cluster_means, 2).sum(-1).float().sqrt()  # (n_c,)

        # nearest_dist, nearest_index = torch.min(dist, dim=-1)
        if isinstance(coarse_id, list):
            mask = ~((~coarse_masks[coarse_id]).view(-1, coarse_masks.size(-1)).sum(0).bool())
        else:
            mask = coarse_masks[coarse_id]
        nearest_dist, nearest_index = torch.min(dist.masked_fill(mask, value=torch.tensor(torch.finfo(float).max)), dim=-1)
        if support_label.item() not in class_entity_nearest_cluster:
            class_entity_nearest_cluster[support_label.item()] = {"cluster_ids":[], "cluster_dists":[], "support_idxs":[]}
        class_entity_nearest_cluster[support_label.item()]["cluster_ids"].append(nearest_index.item())
        class_entity_nearest_cluster[support_label.item()]["cluster_dists"].append(nearest_dist.item())
        class_entity_nearest_cluster[support_label.item()]["support_idxs"].append(support_idx)

    for class_id, value in class_entity_nearest_cluster.items():
        cluster_ids = value["cluster_ids"]
        cluster_dists = value["cluster_dists"]
        for cluster_id in cluster_ids:
            if cluster_fine_labels[cluster_id] == -1 or cluster_fine_labels[cluster_id] == class_id:
                cluster_fine_labels[cluster_id] = class_id
            else: # conflict with other class
                conflict_class_id = cluster_fine_labels[cluster_id]
                current_dist = [dist for dist, cid in zip(cluster_dists, cluster_ids) if cid == cluster_id]
                conflict_dist = [dist for dist, cid in zip(class_entity_nearest_cluster[conflict_class_id]["cluster_dists"],
                                                                        class_entity_nearest_cluster[conflict_class_id]["cluster_ids"]) if cid == cluster_id]
                current_dist = sorted(current_dist)[0]
                conflict_dist = sorted(conflict_dist)[0]
                if current_dist < conflict_dist:
                        print(f"Replacing {cluster_id} from {IO_id2label[cluster_fine_labels[cluster_id]]} to {IO_id2label[class_id]} by distance.")
                        cluster_fine_labels[cluster_id] = class_id
                else:
                        continue

    ## check if all classes are assigned
    new_base_cluster_means = base_cluster_means
    new_base_cluster_covs = base_cluster_covs
    for class_id, value in class_entity_nearest_cluster.items():
        if class_id not in cluster_fine_labels:
            print(f"Poor unassigned class: {IO_id2label[class_id]}, particularly init a prototype for it.")
            new_prototype = [all_entity_embeddings[idx] for idx in value["support_idxs"]]
            new_prototype = torch.stack(new_prototype)
            new_mean = new_prototype.mean(0, keepdim=True)
            new_cov = torch_cov(new_prototype)
            new_base_cluster_means = torch.cat([new_base_cluster_means, new_mean], dim=0)
            new_base_cluster_covs = torch.cat([new_base_cluster_covs, new_cov.unsqueeze(0)], dim=0)

            cluster_fine_labels.append(class_id)

    if new_base_cluster_means.size(0) > base_cluster_means.size(0):
        torch.save(new_base_cluster_means, f"{support_path}/{support_name.split('.')[0]}.{save_model}.new_cluster_means.pt")
        ## too big, not save unless necessary
        # torch.save(new_base_cluster_covs, f"{support_path}/{support_name.split('.')[0]}.{save_model}.new_cluster_covs.pt")



    import os
    cluster_fine_label_names = [IO_id2label[label] if label != -1 else label for label in cluster_fine_labels]
    for i in range(len(coarse_label2id)):
        print(cluster_fine_label_names[i * c_cluster_num:(i + 1) * c_cluster_num])
    # print(cluster_fine_label_names)
    torch.save(cluster_fine_labels, f"{support_path}/{support_name.split('.')[0]}.{save_model}.rb_cluster_fine_labels")




def torch_cov(input_vec: torch.tensor):
    x = input_vec - torch.mean(input_vec, axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0] - 1)
    return cov_matrix


if __name__ == '__main__':
    save_model = "coarseft_cluster_2_1"
    source_model_name = 'models/coarseft_cluster_2_1'
    batch_size = 4
    device = 'cuda'

    ## conll
    # nerd_conll_mapping = {"person":"PER", "organization":"ORG", "other":"MISC", "building":"NOUN", "location":"LOC", "art":"MISC", "product": "MISC","event": "MISC"}
    # coarse_label2id = {'ORG': 0, 'MISC': 1, 'PER': 2, 'LOC': 3}

    ## onto
    # nerd_onto_mapping = {"person":"PERSON", "organization":"ORG", "other":["NORP","LANGUAGE", "LAW"], "building":"FAC", "location":["LOC","GPE"], "art":"WORK_OF_ART", "product":"PRODUCT","event":"EVENT"}
    # coarse_label2id = {"CARDINAL": 0, "PRODUCT": 1, "DATE":2, "LOC":3, "NORP":4, "GPE":5, "PERSON":6, "TIME":7, "ORG":8, "WORK_OF_ART":9, "QUANTITY":10, "PERCENT":11, "EVENT":12, "MONEY":13, "ORDINAL":14,"FAC":15, "LAW":16, "LANGUAGE":17}

    ## nerd
    # coarse_label2id = {"person":0, "organization":1, "other":2, "building":3, "location":4, "art":5, "product":6,"event":7}
    coarse_label2id = {"person":0, "organization":1, "other":2, "building":3, "location":4, "art":5, "product":6,"event":7}

    tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    config = AutoConfig.from_pretrained(source_model_name)
    model = BertForTokenClassification.from_pretrained(source_model_name,
                                                       from_tf=bool(".ckpt" in source_model_name),
                                                       config=config)

    total_tokens = {}
    model.to(device)

    support_path ="./dataset/fine/5shot/"
    label_list_path = f"./dataset/fine/labels.txt"

    with open(label_list_path, "r") as f:
        label_list = [l.strip() for l in f.readlines()]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    IO_label_list = [l.strip() for l in label_list if not l.strip().startswith("B-")]
    IO_label2id = {l: i for i, l in enumerate(IO_label_list)}
    IO_id2label = {v: k for k, v in IO_label2id.items()}
    for sample_id in [1,2,3]:
        support_name = f"{sample_id}.json"
        support_dataloader = bulid_dataloader_token(train_file=support_path + support_name)
        base_cluster_means = torch.load(f'{source_model_name}/cluster_means.pt')
        base_cluster_covs = torch.load(f'{source_model_name}/cluster_covs.pt')
        base_cluster_covs = base_cluster_covs.to(device)
        base_cluster_means = base_cluster_means.to(device)


        retrieve_cluster_labels_robust(model, support_dataloader, base_cluster_means, base_cluster_covs, target_layer=-1, mapping=None)
