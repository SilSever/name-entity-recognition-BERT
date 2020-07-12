from typing import List, Dict

import numpy as np
import torch
import transformers as tr

from config import Config


class Predicter:
    def __init__(self, test: List, labels: List, tag2idx: Dict, tag_values: List) -> None:
        """
        Predicter init function
        :param test: list of test sentences
        :param labels: list of test labels
        :param tag2idx: Dict tag to id
        :param tag_values: label's dictionary
        """

        self.test_sentence = """ 
        Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
        reporter for the network, about protests in Minnesota and elsewhere. 
        """

        self.test_sentences = test
        self.test_labels = labels
        self.tag2idx = tag2idx
        self.tag_values = tag_values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = tr.BertForTokenClassification.from_pretrained(
            Config.MODEL,
            num_labels=len(self.tag2idx),
            output_attentions=False,
            output_hidden_states=False,
        ).to(self.device)
        self.tokenizer = tr.BertTokenizer.from_pretrained(
            Config.BERT_MODEL, do_lower_case=False
        )

        self.predict()

    def predict(self) -> None:
        """
        Predict the NER classes
        :return: None
        """
        for sentence in self.test_sentences:
            tk_sentence = self.tokenizer.encode(sentence)
            self._predict(tk_sentence)

    def _predict(self, tk_sentence: List):
        """
        Supporting function to predict
        :param tk_sentence: tokenized sentences to predict
        :return: None
        """
        input_ids = torch.tensor([tk_sentence]).cuda()

        with torch.no_grad():
            output = self.model(input_ids)
        label_ind = np.argmax(output[0].to("cpu").numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])

        computed_tokens, computed_labels = [], []
        for token, label_id in zip(tokens, label_ind[0]):
            if token.startswith("##"):
                computed_tokens[-1] = computed_tokens[-1] + token[2:]
            else:
                computed_labels.append(self.tag_values[label_id])
                computed_tokens.append(token)

        self._write_predictions(computed_tokens, computed_labels, Config.PREDICTION)

    @staticmethod
    def _write_predictions(computed_tokens, computed_labels, out_path):
        with open(out_path, mode='w') as out_file:
            for token, label in zip(computed_tokens, computed_labels):
                out_file.write("{}\t{}".format(label, token))
            out_file.write("\n")
