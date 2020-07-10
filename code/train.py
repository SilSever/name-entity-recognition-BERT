import utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers as tr
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class NER:
    def __init__(self, feautres, labels, tag2idx):
        self.features = feautres
        self.labels = labels
        self.tag2idx = tag2idx

        self.max_len = 75
        self.batch_size = 32
        self.epochs = 3
        self.max_grad_norm = 1.0

        self.tokenizer = tr.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.tokens, self.labels = self.tokens_and_labels()

        self.model = tr.BertForTokenClassification.from_pretrained(
                    "bert-base-cased",
                    num_labels=len(self.tag2idx),
                    output_attentions = False,
                    output_hidden_states = False
        )
        
        train_data, train_sampler, train_dataloader, valid_data, valid_sampler, valid_dataloader = self.preprocessing()

        self.train()

    def tokens_and_labels(self):

        def _compute_tokens_and_labels(sent, labs):
            tokenized_sentence = []
            labels = []

            for word, label in zip(sent, labs):

                # Tokenize the word and count # of subwords the word is broken into
                tokenized_word = self.tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)

                # Add the tokenized word to the final tokenized word list
                tokenized_sentence.extend(tokenized_word)

                # Add the same label to the new list of labels `n_subwords` times
                labels.extend([label] * n_subwords)

            return tokenized_sentence, labels

        
        tokenized_texts_and_labels = [
            _compute_tokens_and_labels(sent, labs)
            for sent, labs in zip(self.features, self.labels)
        ]


        tokens = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        return tokens, labels

    
    def preprocessing(self):
        self.input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in self.tokens],
                          maxlen=self.MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in self.labels],
                     maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

        self.attention_masks = [[float(i != 0.0) for i in ii] for ii in self.input_ids]

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                            random_state=2018, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                            random_state=2018, test_size=0.1)

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size)

        return train_data, train_sampler, train_dataloader, valid_data, valid_sampler, valid_dataloader

    def train(self):
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                    for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()