import itertools

import spacy
from spacy import displacy
from spacy.tokens import Span, Doc

import random
import json
import re


import os,sys,io,re

from html.parser import HTMLParser

class TimexHTMLParser(HTMLParser):  # courtesy of @dzajic
    def __init__ (self):
        super().__init__()
        self.underlying = ""
        self.inTIMEX3 = False
        self.timex3 = []
        self.timex3_attr = []

    def handle_starttag(self, tag, attrs):
        if tag == "timex3":
            self.inTIMEX3 = True
            self.timex3_attr.append({k:v for (k,v) in attrs})

    def handle_endtag(self, tag):
        if tag == "timex3":
            self.inTIMEX3 = False

    def handle_data(self, data):
        if self.inTIMEX3:
            self.timex3.append((data, len(self.underlying), len(self.underlying + data) - 1))
        self.underlying += data

def normalize_timex(source, article_id):
    filename = f"/Users/jinzhao/schoolwork/lab-work/COVID19_DATA/heideltime_output/{source}.txt"
    with open(filename, 'r') as f:
        heideltime_parse = f.read()
    annotated = re.search("<TimeML>([\s\S]*)<\/TimeML>", heideltime_parse).group(1).strip('\n')
    annotated_articles = annotated.split('\n\n')

    timex_html_parser = TimexHTMLParser()
    timex_html_parser.feed(annotated_articles[article_id])
    normalized_dict = {}
    for i,(timex_text, start_char, end_char) in enumerate(timex_html_parser.timex3):
        # print(timex_html_parser.timex3_attr[i])
        # print(timex_text)
        # print(start_char)
        # print(end_char)
        # print("************")
        normalized_dict[timex_text] = timex_html_parser.timex3_attr[i]
    return normalized_dict

pronoun_set = {
  'I',
  'you',
  'my',
  'mine',
  'myself',
  'we',
  'us',
  'our',
  'ours',
  'ourselves',
  'you',
  'you',
  'your',
  'yours',
  'yourself',
  'you',
  'you',
  'your',
  'your',
  'yourselves',
  'he',
  'him',
  'his',
  'his',
  'himself',
  'she',
  'her',
  'her',
  'her',
  'herself',
  'it',
  'it',
  'its',
  'itself',
  'they',
  'them',
  'their',
  'theirs',
  'themself',
  'they',
  'them',
  'their',
  'theirs',
  'themselves'
}
def get_entity_options(tags):
    """ generating color options for visualizing the replaced pronouns and time stamps """

    def color_generator(number_of_colors):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        return color

    colors = {"ENT":"#E8DAEF"}

    color = color_generator(len(tags))
    for i in range(len(tags)):
        if tags[i].startswith("#"):
            colors[tags[i]] = '#ddd'
        else:
            colors[tags[i]] = color[i]

    options = {"ents": tags, "colors": colors}
    return options

def populate_doc(source:str, article_id:int):
    with open(f'/Users/jinzhao/schoolwork/lab-work/COVID19_DATA/e2e-coref_output/bert-base-cased_{source}_output.jsonl', 'r') as e2e_output_file:
        for i, line in enumerate(e2e_output_file):
            if i==article_id:
                jsonline = json.loads(line)
                break

    with open(f'/Users/jinzhao/schoolwork/lab-work/COVID19_DATA/temporal_model_output/bert-base-cased_{source}_temporal_auto_nodes.txt', 'r') as temporal_output_file:
        content = temporal_output_file.read()
        edge_lists = re.split("filename:[.\S\s]+?EDGE_LIST", content.strip())
        edge_lists = [x.strip() for x in edge_lists if x]

    edge_list = edge_lists[article_id]
    nlp = spacy.load("en_core_web_sm")
    sentences = jsonline["sentences"]

    words = [item for sublist in sentences for item in sublist]
    sent_starts = []
    for snt in sentences:
        sent_starts.extend([True] + [False] * (len(snt) - 1))
    spaces = [True] * len(words)
    assert len(sent_starts) == len(words)
    doc = Doc(nlp.vocab, words=words, spaces=spaces, sent_starts=sent_starts)
    edges = [line.split() for line in edge_list.strip().split('\n')]
    snts_len = [len(s) for s in sentences]
    acc_len = list(itertools.accumulate(snts_len))
    spans = []
    tags = set()

    normalized_timex_dict = normalize_timex(source, article_id)

    for edge in edges:
        if edge[0] in ("-7_-7_-7", "-1_-1_-1"):
            continue
        sentence_id_token, start_token, end_token = [int(e) for e in edge[0].split('_')]
        if sentence_id_token:
            start_offset = acc_len[sentence_id_token - 1] + start_token
            end_offset = acc_len[sentence_id_token - 1] + end_token
        else:
            start_offset = start_token
            end_offset = end_token

        token_type = edge[1]
        rel = edge[3]

        if token_type == 'Event':
            if edge[2] in ("-7_-7_-7", "-1_-1_-1"):
                time = " ".join(sentences[1][:2]) #DCT
            else:
                sentence_id_ref, start_ref, end_ref = [int(e) for e in edge[2].split('_')]
                tokens = sentences[sentence_id_ref][start_ref: end_ref + 1]
                time = " ".join(tokens)
                if time in normalized_timex_dict and normalized_timex_dict[time]['type'] == 'DATE':
                    time = normalized_timex_dict[time]['value']
            tags.add(f"#{rel} {time}")
            spans.append(Span(doc, start_offset, end_offset + 1, f"#{rel} {time}"))


    for cluster in jsonline["predicted_clusters"]:
        tag = " ".join(words[cluster[0][0]:cluster[0][1] + 1])
        for token_span in cluster: #decide on tag in the first loop
            w = " ".join(words[token_span[0]:token_span[1]+1]).lower()
            if any(ele.isupper() for ele in w) and w.lower() not in pronoun_set:
                tag = w
        for token_span in cluster:
            tags.add(tag)
            spans.append(Span(doc, token_span[0], token_span[1]+1, f"{tag}"))

    doc.set_ents(spacy.util.filter_spans(spans)) #filter here is used to resolve overlapping spans
    return doc, tags

def visualize(doc, tags):
    options = get_entity_options(list(tags))
    displacy.serve(doc, style="ent", options=options)

def output_modified_doc2txt(doc):
    output_string = ""
    for sentence in doc.sents:
        tokens = []
        for spacy_token in sentence:
            if spacy_token.ent_iob_ == 'B':
                if spacy_token.ent_type_.startswith('#'):
                    tokens.append(f"{spacy_token.text}({spacy_token.ent_type_})")
                else:
                    tokens.append(spacy_token.ent_type_)
            elif spacy_token.ent_iob_ == 'I':
                pass
            else:
                tokens.append(spacy_token.text)
        output_string += ' '.join(tokens) + '\n'
    return output_string

if __name__ == '__main__':
    doc, tags = populate_doc("business-standard", 2)
    output_modified_doc2txt(doc)
    visualize(doc, tags)
