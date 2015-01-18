#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from collections import defaultdict
import nltk


# Corpus consisting of list of real terms
def load_terms():
    """docstring for load_terms"""
    with open('input/small_terms.txt', 'r') as tf:
        terms_raw = tf.read().decode('utf-8')
    tagged_terms = terms_raw.split('\n')
    return tagged_terms


def get_pos_seq_freq(tagged_terms):
    """docstring for get_pos_seq_freq"""
    pos_seq_freq_dict = defaultdict(int)
    for tagged_term in tagged_terms:
        pos_seq = ' '.join(w.rsplit('/', 2)[1] for w in tagged_term.split())
        pos_seq_freq_dict[pos_seq] += 1
    return pos_seq_freq_dict


# Quizas seria buena idea sacar el 'del' del modelo.
def get_lemma_freq(tagged_sents):
    """docstring for get_lemma_freq"""
    lemma_freq_dict = defaultdict(int)
    for sent in tagged_sents:
        sent_div = sent.split()
        for word in sent_div:
            word_div = word.rsplit('/', 2)
            if word_div[0].isalnum():
                lemma_freq_dict[word_div[0].lower()] += 1
    return lemma_freq_dict


# Idem arriba. Sacar 'del' del modelo?
def get_affix_freq(unique_token_list, affix_type, affix_len):
    """docstring for get_affix_freq"""
    affix_freq_dict = defaultdict(int)

    for word in unique_token_list:
        if word.isalnum() and len(word) > affix_len:
            if affix_type == 'prefix':
                affix = word[:affix_len]
            elif affix_type == 'suffix':
                affix = word[-affix_len:]
            affix_freq_dict[affix] += 1
    return affix_freq_dict


def make_term_model(tagged_terms):
    """docstring for make_term_model"""
    terms = tagged_terms

    # Syntactical part
    pos_seq_freq = get_pos_seq_freq(terms)

    # Lexical part
    lemma_freq = get_lemma_freq(terms)

    # Morphological part
    lemma_f3 = get_affix_freq(lemma_freq.keys(), 'prefix', 3)
    lemma_f4 = get_affix_freq(lemma_freq.keys(), 'prefix', 4)
    lemma_f5 = get_affix_freq(lemma_freq.keys(), 'prefix', 5)
    lemma_l3 = get_affix_freq(lemma_freq.keys(), 'suffix', 3)
    lemma_l4 = get_affix_freq(lemma_freq.keys(), 'suffix', 4)
    lemma_l5 = get_affix_freq(lemma_freq.keys(), 'suffix', 5)

    return {'pos_freq': pos_seq_freq, 'lemma_freq': lemma_freq,
            'lemma_f3': lemma_f3, 'lemma_f4': lemma_f4, 'lemma_f5': lemma_f5,
            'lemma_l3': lemma_l3, 'lemma_l4': lemma_l4, 'lemma_l5': lemma_l5}


# General language corpus
def load_general():
    """docstring for load_general"""
    with open('input/big_reference.txt', 'r') as gcf:
        ref_raw = gcf.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in ref_raw.split('./Fp')]
    return tagged_sents


def make_general_model(tagged_general):
    # (POS tags not used, but expected as input)
    """docstring for make_general_model"""
    gen_corp = tagged_general

    # Doesn't use syntactical info

    # Lexical part
    lemma_freq = get_lemma_freq(gen_corp)

    # Morphological part
    lemma_f3 = get_affix_freq(lemma_freq.keys(), 'prefix', 3)
    lemma_f4 = get_affix_freq(lemma_freq.keys(), 'prefix', 4)
    lemma_f5 = get_affix_freq(lemma_freq.keys(), 'prefix', 5)
    lemma_l3 = get_affix_freq(lemma_freq.keys(), 'suffix', 3)
    lemma_l4 = get_affix_freq(lemma_freq.keys(), 'suffix', 4)
    lemma_l5 = get_affix_freq(lemma_freq.keys(), 'suffix', 5)

    return {'lemma_freq': lemma_freq,
            'lemma_f3': lemma_f3, 'lemma_f4': lemma_f4, 'lemma_f5': lemma_f5,
            'lemma_l3': lemma_l3, 'lemma_l4': lemma_l4, 'lemma_l5': lemma_l5}


# Corpus from which to extract candidates
def load_analysis():
    """docstring for load_analysis"""
    with open('input/small_domain.txt', 'r') as corpf:  # small corpus
        corp = corpf.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in corp.split('./Fp')]
    tagged_sents = [s.split() for s in tagged_sents]
    return tagged_sents


def chunk_sents(pos_sequence, tagged_sents):
    """docstring for chunk_sents"""
    grammar = r'TC: {%s}' % ''.join(['<%s>' % t for t in pos_sequence.split()])
    chunker = nltk.RegexpParser(grammar)
    chunks = []
    for sent in tagged_sents:
        lemtagdiv_sent = [tuple(w.rsplit('/', 2)) for w in sent]
        for chnk in chunker.parse(lemtagdiv_sent).subtrees():
            if str(chnk).startswith('(TC'):
                phrase = chnk.__unicode__()[4:-1]
                if '\n' in phrase:
                    phrase = ' '.join(phrase.split())
                chunks.append(phrase)
    return chunks


def calc_lexical_score(candidate, term_model, gen_model, s=0.001,
                       stoplist=['del']):
    """docstring for calc_lexical_score"""

    term_lemma_num = sum(term_model['lemma_freq'].values())
    gen_lemma_sum = sum(gen_model['lemma_freq'].values())

    lemma_score_list = []
    lemmas = [w.rsplit('/', 2)[0] for w in candidate.split()]
    for lem in lemmas:
        if lem in stoplist:  # TODO: if lem in stoplist; continue statement
            continue
        relative_lem_freq_in_terms = 0.0
        if lem in term_model['lemma_freq'].keys():
            relative_lem_freq_in_terms = \
                term_model['lemma_freq'][lem] / term_lemma_num

        relative_lem_freq_in_gen = 0.0
        if lem in gen_model['lemma_freq'].keys():
            relative_lem_freq_in_gen = \
                gen_model['lemma_freq'][lem] / gen_lemma_sum

        lemma_score = \
            relative_lem_freq_in_terms / (relative_lem_freq_in_gen + s)
        lemma_score_list.append(lemma_score)
    lemma_coef = sum(lemma_score_list) / len(lemma_score_list)

    lexical_coef = lemma_coef  # Falta la parte de palabras.

    return lexical_coef


def main():
    """docstring for main"""

    term_corp = load_terms()
    term_model = make_term_model(term_corp)

    gen_corp = load_general()
    gen_model = make_general_model(gen_corp)

    anal_corp = load_analysis()

    candidate_score = []

    pos_patterns = term_model['pos_freq'].keys()
    for pos_seq in pos_patterns:
        print pos_seq
        chunks = list(set(chunk_sents(pos_seq, anal_corp)))
        for candidate in chunks:
            lex_coef = calc_lexical_score(candidate, term_model, gen_model)
            # morph_coef = ...
            print candidate
            print lex_coef
            raw_input()

if __name__ == '__main__':
    main()
