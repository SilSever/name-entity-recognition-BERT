from typing import List, Tuple, Dict

import numpy as np
import torch
import transformers as tr
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import utils
from config import Config


class NER:
    def __init__(
        self, feautres: List, labels: List, tag2idx: Dict, tag_values: List
    ) -> None:
        """
        NER init function
        :param feautres: features
        :param labels: labels
        :param tag2idx: Dict tag to id
        :param tag_values: label's dictionary
        """
        self.features = feautres
        self.labels = labels
        self.tag2idx = tag2idx
        self.tag_values = tag_values

        self.max_len = 75
        self.batch_size = 32
        self.epochs = 3
        self.max_grad_norm = 1.0
        self.finetuning = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tr.BertTokenizer.from_pretrained(
            Config.BERT_MODEL, do_lower_case=False
        )
        self.tokens, self.labels = self.tokens_and_labels()

        self.model = tr.BertForTokenClassification.from_pretrained(
            Config.BERT_MODEL,
            num_labels=len(self.tag2idx),
            output_attentions=False,
            output_hidden_states=False,
        ).to(self.device)

        self.train_dataloader, self.valid_dataloader = self.preprocessing()

        self.optimizer = self.set_optimizer()

        # self.train()

    def _compute_tokens_and_labels(self, sent: str, labs: List) -> Tuple[List, List]:
        """
        Supporting function to the main one
        :param sent: sentence to be tokenized
        :param labs: labels to be tokenized
        :return: tokens list and labels list
        """
        tk_sent = []
        tk_lab = []

        for word, label in zip(sent, labs):
            tk_words = self.tokenizer.tokenize(word)
            subwords = len(tk_words)

            tk_sent.extend(tk_words)
            tk_lab.extend([label] * subwords)

        return tk_sent, tk_lab

    def tokens_and_labels(self) -> Tuple[List, List]:
        """
        Compute the tokens and the labels to be trained
        :return: tokens list and labels list
        """
        tokenized_txts_labs = [
            self._compute_tokens_and_labels(sent, labs)
            for sent, labs in zip(self.features, self.labels)
        ]

        tokens = [token_label_pair[0] for token_label_pair in tokenized_txts_labs]
        labels = [token_label_pair[1] for token_label_pair in tokenized_txts_labs]

        return tokens, labels

    def preprocessing(self) -> Tuple[DataLoader, DataLoader]:
        """
        Preprocess the tokens in order to be batched
        :return: tr_batches, validation_batches
        """
        input_ids = pad_sequences(
            [self.tokenizer.convert_tokens_to_ids(token) for token in self.tokens],
            maxlen=self.max_len,
            dtype="long",
            value=0.0,
            truncating="post",
            padding="post",
        )

        tags = pad_sequences(
            [[self.tag2idx.get(lab) for lab in label] for label in self.labels],
            maxlen=self.max_len,
            value=self.tag2idx["PAD"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        attention_masks = [
            [float(i != 0.0) for i in input_id] for input_id in input_ids
        ]

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
            input_ids, tags, random_state=2018, test_size=0.1
        )
        tr_masks, val_masks, _, _ = train_test_split(
            attention_masks, input_ids, random_state=2018, test_size=0.1
        )

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)

        tr_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        tr_dataloader = DataLoader(
            tr_data, sampler=RandomSampler(tr_data), batch_size=self.batch_size
        )

        val_data = TensorDataset(val_inputs, val_masks, val_tags)
        val_dataloader = DataLoader(
            val_data, sampler=SequentialSampler(val_data), batch_size=self.batch_size
        )

        return tr_dataloader, val_dataloader

    def set_optimizer(self) -> tr.AdamW:
        """
        Set the parameters needed for the optimizer
        :return: the optimizer
        """
        if self.finetuning:
            params = list(self.model.named_parameters())
            optimizer_params = [
                {
                    "params": [
                        p
                        for n, p in params
                        if not any(nd in n for nd in ["bias", "gamma", "beta"])
                    ],
                    "weight_decay_rate": 0.01,
                },
                {
                    "params": [
                        p
                        for n, p in params
                        if any(nd in n for nd in ["bias", "gamma", "beta"])
                    ],
                    "weight_decay_rate": 0.0,
                },
            ]
        else:
            params = list(self.model.classifier.named_parameters())
            optimizer_params = [{"params": [p for n, p in params]}]

        optimizer = tr.AdamW(optimizer_params, lr=3e-5, eps=1e-8)
        return optimizer

    def train(self) -> tr.BertForTokenClassification:
        """
        Training function
        :return: None
        """
        total_steps = len(self.train_dataloader) * self.epochs

        scheduler = tr.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        loss_values, val_loss_values = [], []

        for _ in tqdm(range(self.epochs), desc="Epoch"):

            # Training
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                self.model.zero_grad()

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_grad_norm
                )

                self.optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            loss_values.append(avg_train_loss)

            # Validation
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            predictions, true_labels = [], []
            for batch in self.valid_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = self.model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                    )
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            eval_loss = eval_loss / len(self.valid_dataloader)
            val_loss_values.append(eval_loss)

            self._print_metrics(eval_loss, predictions, true_labels)

        utils.plot_losses(loss_values, val_loss_values)

        self.model.save_pretrained(Config.MODEL)
        return self.model

    def _print_metrics(
        self, eval_loss: float, predictions: List, true_labels: List
    ) -> None:
        """
        Print training metrics
        :param eval_loss: evaluations losses
        :param predictions: predictions
        :param true_labels: labels
        :return: None
        """
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [
            self.tag_values[p_i]
            for pred, lab in zip(predictions, true_labels)
            for p_i, l_i in zip(pred, lab)
            if self.tag_values[l_i] != "PAD"
        ]
        valid_tags = [
            self.tag_values[l_i]
            for lab in true_labels
            for l_i in lab
            if self.tag_values[l_i] != "PAD"
        ]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}\n".format(f1_score(pred_tags, valid_tags)))
