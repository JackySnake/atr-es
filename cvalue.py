#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from collections import defaultdict
from math import log
import nltk
import binom


def load_tagged_sents():
    """docstring for load_tagged_senits"""
    with open('input/corpus_prueba.txt', 'r') as corpf:
        corp = corpf.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in corp.split('./Fp')]
    tagged_sents = [s.split() for s in tagged_sents]
    tagged_sents = [[tuple(w.split('/')) for w in s] for s in tagged_sents]
    return tagged_sents


def chunk_sents(tagged_sents, chunk_pattern, min_freq):
    """docstring for chunk_sents"""
    phrase_freq_dict = defaultdict(int)
    chunker = nltk.RegexpParser(chunk_pattern)
    for sent in tagged_sents:
        for chk in chunker.parse(sent).subtrees():
            if str(chk).startswith('(TC'):
                phrase = chk.__unicode__()[4:-1]
                phrase_freq_dict[phrase] += 1
    phrase_freq_dict = \
        dict([p for p in phrase_freq_dict.items() if p[1] >= min_freq])
    return phrase_freq_dict


def binom_ratio_filter(candidate_freq_dict, cutoff):
    """doctsring for binom_ratio_filter"""
    kept_phrases = []
    disc_phrases = []
    br_under_cutoff = binom.main(cutoff)
    for nom_phrase in candidate_freq_dict.keys():
        keep_phrase = True
        nom_phrase_div = nom_phrase.split()
        for lt_ref in nom_phrase_div:
            if lt_ref in br_under_cutoff.keys():
                keep_phrase = False
                break
        if keep_phrase is True:
            kept_phrases.append((nom_phrase, candidate_freq_dict[nom_phrase]),)
        else:
            disc_phrases.append((nom_phrase, candidate_freq_dict[nom_phrase]),)
    return dict(kept_phrases), dict(disc_phrases)


def remove_postags(tagged_phrase):
    clean_phrase = ' '.join([w.split('/')[0] for w in tagged_phrase.split()])
    return clean_phrase


# Remove POS tags from phrases (keys) and return dict.
def build_cval_input(phrase_freq_dict):
    """docstring for candidate_freq_dict"""
    cval_input_dict = {}
    for phrase in phrase_freq_dict.keys():
        #lema_phrase = ' '.join([w.split('/')[0] for w in phrase.split()])
        lema_phrase = remove_postags(phrase)
        cval_input_dict[lema_phrase] = phrase_freq_dict[phrase]
    return cval_input_dict


def remove_1word_candidates(phrase_freq_dict):
    """doctstring for candidate_freq_dict"""
    return dict(
        (k, v) for k, v in phrase_freq_dict.items() if len(k.split()) > 1)


def cvalue(freq_dict):
    term_cvalue = {}
    max_len_term = max(len(c.split()) for c in freq_dict.keys())
    for tc in [c for c in freq_dict.keys() if len(c.split()) == max_len_term]:
        cval = log(len(tc.split()), 2) * freq_dict[tc]
        term_cvalue[tc] = cval
    for term_len in reversed(range(1, max_len_term)):
        for tc in [c for c in freq_dict.keys() if len(c.split()) == term_len]:
            substring_of = [c for c in freq_dict.keys() if tc in c and tc != c]
            if len(substring_of) > 0:
                fa = freq_dict[tc]
                PTa = len(set(substring_of))
                fb = sum(freq_dict[c] for c in substring_of)
                for lc in substring_of:
                    for xc in [c for c in substring_of if lc != c]:
                        if lc in xc:
                            fb -= freq_dict[lc]
                #cval = log(len(tc.split()), 2) * (fa - fb) / PTa)
                cval = log(len(tc.split()), 2) * (fa - (1 / PTa * fb))  # same thing.
                term_cvalue[tc] = cval
            else:
                cval = log(len(tc.split()), 2) * freq_dict[tc]
                term_cvalue[tc] = cval
    return term_cvalue


def calc_precision(extracted_terms):
    """doctstring for calc_precision"""
    with open('input/términos en corpus.txt', 'r') as termf:
        terms_raw = termf.read().decode('utf-8')
    terms = [l.strip() for l in terms_raw.split('\n')]
    terms = [remove_postags(l).lower() for l in terms if len(l.split()) > 1]

    real_term_num = 0
    for exterm in extracted_terms:
        if exterm.lower() in terms:
            real_term_num += 1
    print (real_term_num / len(extracted_terms)) * 100


def main():
    """docstring for main"""
    sents = load_tagged_sents()
    pattern = r"""
        TC: {<NC>+<AQ>*(<PDEL><DA>?<NC>+<AQ>*)?}
        """
    phrase_freq = chunk_sents(sents, pattern, 2)

    #accepted_phrases = binom_ratio_filter(phrase_freq, 0.2)[0]
    #cval_input = build_cval_input(accepted_phrases)
    cval_input = build_cval_input(phrase_freq)

    cval_input = remove_1word_candidates(cval_input)

    cval_output = cvalue(cval_input)
    cval_output = sorted(
        cval_output.items(), key=lambda item: item[1], reverse=True)

    precision = calc_precision([t[0] for t in cval_output])

    #with open('br_cvals.txt', 'w') as f:
    #    f.write('\n'.join(
    #        str(round(c[1], 3))+'\t'+c[0].encode('utf-8') for c in cval_output))


if __name__ == '__main__':
    main()
