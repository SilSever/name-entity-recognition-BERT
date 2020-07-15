from typing import List, Dict, Tuple

import numpy as np
import torch
import transformers as tr

from config import Config
from sklearn.metrics import f1_score


class Predicter:
    def __init__(self, test: List, labels: List, tag2idx: Dict, tag_values: List) -> None:
        """
        Predicter init function
        :param test: list of test sentences
        :param labels: list of test labels
        :param tag2idx: Dict tag to id
        :param tag_values: label's dictionary
        """
        self.test_sentences = test
        self.test_labels = labels
        self.tag2idx = tag2idx
        self.tag_values = tag_values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_sentence = """
        Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
        reporter for the network, about protests in Minnesota and elsewhere. 
        """

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
        with open(Config.PREDICTION, mode='w') as out_file:

            for tk_sentence, or_label in zip([self.test_sentence], [""]):
                tk_sentence = self.tokenizer.encode(tk_sentence)
                tokens, pr_labels, tmp = self._predict(tk_sentence)

                tmp_label = [self.tag2idx[lb] for lb in or_label]
                f1 = f1_score(tmp_label, tmp, average='micro')
                # print(pr_labels,"\n", tmp, "\n", or_label, "\n", tmp_label, "\n",  f1,"\n\n")

                for token, label in zip(tokens, pr_labels):
                    out_file.write("{}:{} ".format(token, label))
                out_file.write("\n")

    def _predict(self, tk_sentence: List) -> Tuple[List, List, List]:
        """
        Supporting function to predict
        :param tk_sentence: tokenized sentences to predict
        :return: predicted_tokens and predicted labels
        """
        input_ids = torch.tensor([tk_sentence]).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
        label_ind = np.argmax(output[0].to("cpu").numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])

        computed_tokens, computed_labels = [], []
        for token, label_id in zip(tokens[1:-1], label_ind[0][1:-1]):
            if token.startswith("##"):
                computed_tokens[-1] = computed_tokens[-1] + token[2:]
            else:
                computed_labels.append(self.tag_values[label_id])
                computed_tokens.append(token)

        return computed_tokens, computed_labels, label_ind[0][1:-1]

        # self._write_predictions(computed_tokens, computed_labels, Config.PREDICTION)
