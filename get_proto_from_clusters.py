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
        cluster_labels = [feature['cluster_labels'] for feature in features] if 'cluster_labels' in features[0].keys() else None
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



def bulid_dataloader_token(task_name=None, train_file='/root/honor_dataset/clear_train_1000.json', max_length=128):
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
            cluster_label = examples["cluster_label"][i]
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
                    label_ids.append(
                        label2id[label[word_idx]])
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
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    return train_dataloader



def get_protos_from_model(model, dataloader, target_layer=-1, cluster_num=1000):
    cluster_embeds = {i:[] for i in range(cluster_num)}
    model.eval()
    pro_bar = tqdm(range(len(dataloader)))
    step = 0
    device = model.device
    for batch in dataloader:
        step += 1
        with torch.no_grad():
            batch = {key: value.to(device) for key, value in batch.items()}
            batch['output_hidden_states'] = True
            labels = batch.pop("labels")
            cluster_labels = batch.pop("cluster_labels")
            outputs = model(**batch)
            input_ids = batch['input_ids']

            target_hidden_states = outputs.hidden_states[target_layer]

            valid_ids = labels != -100
            target_hidden_states = target_hidden_states[valid_ids]
            cluster_labels = cluster_labels[valid_ids]

            previous_cluster_id = -1
            token_embeds = []
            for  cluster_id, token_hidden_states in zip( cluster_labels, target_hidden_states):
                cluster_id = cluster_id.item()
                if cluster_id != previous_cluster_id:
                    if previous_cluster_id != -1:
                        cluster_embeds[previous_cluster_id].append(torch.stack(token_embeds).mean(dim=0))
                        token_embeds = []
                if cluster_id != -1:
                    token_embeds.append(token_hidden_states)
                previous_cluster_id = cluster_id
            # final
            if previous_cluster_id != -1:
                cluster_embeds[previous_cluster_id].append(torch.stack(token_embeds).mean(dim=0))

        pro_bar.update(1)

    cluster_protos = []
    for i in range(cluster_num):
        cluster_protos.append(torch.stack(cluster_embeds[i]).mean(dim=0))

    cluster_center = torch.stack(cluster_protos, dim=0)
    torch.save(cluster_center, f'{output_dir}/{save_name}_cluster_protos.pt')


if __name__ == '__main__':
    cluster_num = 400
    class_cluster_num = 50
    # task_name = 'sst2'
    cluster_method = 'kmeans'
    source_model_name = './bert-base-cased'
    cluster_model = "coarseft_new"
    batch_size = 128
    device = 'cuda'
    train_path = f"cluster/models/{cluster_model}/kmeans/layer-1_50eachclass_train.json"
    # eval_path = f"cluster/models/{cluster_model}/kmeans/layer-1_50eachclass_test.json"
    # label_list_path = "./dataset/few_NERD_transfer_data/new_CTF/coarse/labels.txt"
    label_list_path =  "./dataset/coarse/labels.txt"

    with open(label_list_path, "r") as f:
        label_list = [l.strip() for l in f.readlines()]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    target_layer = -1
    output_dir = f'cluster/models/{cluster_model}/{cluster_method}'
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_name = f'bert-base-cased_layer-1_{class_cluster_num}eachclass'
    # save_name = f'embedding_{cluster_num}'

    tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    config = AutoConfig.from_pretrained(source_model_name)
    model = BertForTokenClassification.from_pretrained(source_model_name,
                                                       from_tf=bool(".ckpt" in source_model_name),
                                                       config=config)

    train_dataloader = bulid_dataloader_token(train_file=train_path)

    total_tokens = {}
    model.to(device)

    get_protos_from_model(model, train_dataloader, cluster_num=cluster_num)

