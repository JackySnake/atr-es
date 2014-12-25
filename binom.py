#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from collections import defaultdict
from scipy.stats import binom


def load_dom_ref():
    """docstring for load_dom_ref"""
    with open('input/corpus_prueba.txt', 'r') as domf:
        dom_raw = domf.read().decode('utf-8')
    dom_freq_dict = defaultdict(int)
    for word in dom_raw.split():
        if word.split('/')[1] in ('NC', 'AQ'):
            dom_freq_dict[word] += 1

    with open('input/referencia_ncyad.txt', 'r') as reff:
        ref_raw = reff.read().decode('utf-8')[:-1]  # Remove last \n
    ref_freq_dict = {}
    for word in ref_raw.split('\n'):
        word_div = word.split('/')
        if word_div[1] in ('NC', 'AQ'):
            ref_freq_dict[word_div[0]+'/'+word_div[1]] = int(word_div[2])

    return dom_freq_dict, ref_freq_dict


def binom_prob_dict(lematag_freq_dict):
    """docstring for binom_prob_dict"""
    lematag_binom_prob_dict = {}
    num_words = sum(lematag_freq_dict.values())
    for word in lematag_freq_dict.keys():
        word_freq = lematag_freq_dict[word]
        word_prob = word_freq / num_words
        word_binom_prob = round(binom.pmf(word_freq, num_words, word_prob), 5)
        lematag_binom_prob_dict[word] = word_binom_prob
    return lematag_binom_prob_dict


def sorted_freq_dict(freq_dict):
    """doctstring for sorted_freq_dict"""
    return sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)


def binom_ratio(dom_binom_dict, ref_binom_dict):
    """docstring for binom_ratio"""
    shared = []
    key_intersect = \
        set(dom_binom_dict.keys()).intersection(set(ref_binom_dict.keys()))
    for lempos in key_intersect:
        shared.append(
            (lempos, round(ref_binom_dict[lempos]/dom_binom_dict[lempos], 3)),)
    return dict(shared)


def main(cutoff):
    """doctstring main"""
    dom_freq, ref_freq = load_dom_ref()
    dom_binom = binom_prob_dict(dom_freq)
    ref_binom = binom_prob_dict(ref_freq)
    brat_dict = binom_ratio(dom_binom, ref_binom)
    brat_dict = dict((k, v) for k, v in brat_dict.items() if v < cutoff)
    return brat_dict


if __name__ == '__main__':
    pass
