import torch
import transformers as tr
import numpy as np

from config import Config


class Predicter():
    def __init__(self, tag2idx, tag_values):
        self.test_sentence = """ 
        Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
        reporter for the network, about protests in Minnesota and elsewhere. 
        """
        self.tag2idx = tag2idx
        self.tag_values = tag_values

        self.model = tr.BertForTokenClassification.from_pretrained(
            Config.MODEL,
            num_labels=len(self.tag2idx),
            output_attentions=False,
            output_hidden_states=False
        )
        self.tokenizer = tr.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        tokenized_sentence = self.tokenizer.encode(self.test_sentence)

        self.predict(tokenized_sentence)

    def predict(self, tokenized_sentence):
        input_ids = torch.tensor([tokenized_sentence]).cuda()

        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        # join bpe split tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)

        for token, label in zip(new_tokens, new_labels):
            print("{}\t{}".format(label, token))
