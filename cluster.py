from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
import json
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForSequenceClassification, AutoConfig, default_data_collator, DataCollatorWithPadding, BertForTokenClassification
import collections

def bulid_dataloader_token(task_name=None, train_file='', max_length=128):
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=None
    )
    if task_name == None:
        data_files = {}
        data_files["train"] = train_file
        # data_files["validation"] = eval_file
        extension =  train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset(task_name)
    
    # raw_datasets = load_dataset("glue", task_name)
    # sentence1_key, sentence2_key = task_to_keys[task_name]
    text_column_name = 'text'
    label_column_name = "label"
    padding =   False
    # max_length = 512

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
        ori_labels = []
        # gold_labels = examples[label_column_name]
        # if args.label_schema=='IO':
        #     gold_labels = [['I-{}'.format(l[2:]) if l !='O' else 'O' for l in label] for label in gold_labels]
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # label_ids.append(-1)
                    label_ids.append(label2id[label[word_idx]] if not label[word_idx].startswith("B-") else label2id["I-"+label[word_idx][2:]])
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

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


def recluster_entity_by_labels(model, dataloader, target_layer=-1, class_cluster_num=100, cluster_method='kmeans'):

    total_entities = {}
    mention_embeddings = []
    mention_list = []
    mention_label_list = []
    entity2label = collections.defaultdict(list)
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
            outputs = model(**batch)
            input_ids = batch['input_ids']
            # labels = batch["labels"]
            target_hidden_states = outputs.hidden_states[target_layer]

            # attention_mask = batch['attention_mask']
            valid_ids = labels != -100
            input_ids = input_ids[valid_ids]
            target_hidden_states = target_hidden_states[valid_ids]
            labels = labels[valid_ids]

            entity = None
            entity_embeddings = []
            entity_label = None
            for idx, token_id, label_id, token_hidden_states in zip(range(input_ids.size(0)), input_ids, labels, target_hidden_states):
                token_id = token_id.item()
                label_id = label_id.item()

                if id2label[label_id].startswith("B-"):
                    if entity:
                        mention_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                        mention_list.append(entity)
                        mention_label_list.append(entity_label)
                        entity = None
                        entity_embeddings = []
                        entity_label = None

                    entity = tokenizer._convert_id_to_token(token_id)
                    entity_embeddings.append(token_hidden_states.to('cpu'))
                    entity_label = id2label[label_id][2:]

                elif id2label[label_id].startswith("I-"):
                    entity += "_" + tokenizer._convert_id_to_token(token_id)
                    entity_embeddings.append(token_hidden_states.to('cpu'))
                else:
                    if entity:
                        mention_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                        mention_list.append(entity)
                        mention_label_list.append(entity_label)
                        entity = None
                        entity_embeddings = []
                        entity_label = None
                last_label = id2label[label_id]

            if entity:
                mention_embeddings.append(torch.stack(entity_embeddings).mean(dim=0))
                mention_list.append(entity)
                mention_label_list.append(entity_label)
        pro_bar.update(1)


    class2mention_embed_list = collections.defaultdict(list)
    class_mention_id2origin_id = collections.defaultdict(dict)
    for mention_id in range(len(mention_list)):
        class_name = mention_label_list[mention_id]
        class2mention_embed_list[class_name].append(mention_embeddings[mention_id])
        class_mention_id = len(class2mention_embed_list[class_name]) - 1
        class_mention_id2origin_id[class_name][class_mention_id] = mention_id

    ### cluster for each class
    # class_cluster_num = 50
    print('################# cluster start #################')

    for class_num, (class_name, mention_embeddings) in enumerate(class2mention_embed_list.items()):
        print(class_name)

    cluster_centers = []
    mention_cluster_ids = [0] * len(mention_list)
    cluster2mentions = collections.defaultdict(list) # same to cluster2tokens
    for class_num, (class_name, mention_embeddings) in enumerate(class2mention_embed_list.items()):

        print(f"Clustering for class {class_num}, cluster_num={class_cluster_num}")
        print('class_name',class_name)
        token_embeddings = torch.stack(mention_embeddings)

        if cluster_method == 'kmeans':
            clusters = KMeans(n_clusters=class_cluster_num, random_state=0).fit(token_embeddings)
            token_embeddings_cluster_ids = clusters.predict(token_embeddings)
        else:
            clusters = AgglomerativeClustering(n_clusters=cluster_num).fit(token_embeddings)
            token_embeddings_cluster_ids = clusters.labels_
        print(torch.tensor(clusters.cluster_centers_).shape)
        cluster_centers.append(torch.tensor(clusters.cluster_centers_).to(model.device))

        # map to origin ids
        cluster_start_id = class_cluster_num * class_num
        for class_mention_id, class_cluster_id in enumerate(token_embeddings_cluster_ids):
            mention_cluster_ids[class_mention_id2origin_id[class_name][class_mention_id]] = cluster_start_id + class_cluster_id.item()
            cluster2mentions[cluster_start_id + class_cluster_id].append(mention_list[class_mention_id2origin_id[class_name][class_mention_id]])

    print('############ cluster end ############')

    generate_cluster_labels_by_order(eval_path, mention_cluster_ids)



def generate_cluster_labels_by_order(train_file, mention_cluster_ids):

    with open(train_file, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    new_lines = []
    mention_idx = 0
    for line in lines:
        if mention_idx >= len(mention_cluster_ids):
            print(mention_idx)
        words = line["text"]
        labels = line["label"]
        cluster_labels = []
        entity_cluster_id = -1
        for idx, label in enumerate(labels):
            if label.startswith("B-"):
                entity_cluster_id = mention_cluster_ids[mention_idx]
                mention_idx += 1
                cluster_labels.append(entity_cluster_id)
            elif label.startswith("I-"):
                cluster_labels.append(entity_cluster_id)
            else:
                cluster_labels.append(-1)

        new_lines.append({"text":words, "label":labels, "cluster_label": cluster_labels})

    assert mention_idx == len(mention_cluster_ids)


    with open(f'{output_dir}/{save_name}_train.json',"w") as f:
        for line in new_lines:
            f.write(json.dumps(line)+"\n")


if __name__ == '__main__':

    class_cluster_num = 50
    cluster_method = 'kmeans'
    model_name_or_path = 'models/coarseft_new'
    batch_size = 128
    max_length = 512
    device = 'cuda'
    train_path = "./dataset/coarse/train.json"
    eval_path = "./dataset/coarse/test.json"
    label_list_path = "./dataset/coarse/labels.txt"

    with open(label_list_path, "r") as f:
        label_list = [l.strip() for l in f.readlines() ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i:label for label, i in label2id.items()}
    
    target_layer=-1
    output_dir = f'cluster/{model_name_or_path}/{cluster_method}'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_name = f'layer{target_layer}_{class_cluster_num}eachclass'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = BertForTokenClassification.from_pretrained(model_name_or_path,
                                                               from_tf=bool(".ckpt" in model_name_or_path),
                                                            config=config)
    


    train_dataloader = bulid_dataloader_token(train_file=train_path, max_length=max_length)
    eval_dataloader = bulid_dataloader_token(train_file=eval_path, max_length=max_length)

    total_tokens = {}
    model.to(device)

    recluster_entity_by_labels(model, eval_dataloader, target_layer=target_layer, class_cluster_num=class_cluster_num, cluster_method=cluster_method)
