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


# Remove POS tags from phrases (keys) and return dict.
def remove_pos_tags(phrase_freq_dict):
    """docstring for candidate_freq_dict"""
    cval_input_dict = {}
    for phrase in phrase_freq_dict.keys():
        lema_phrase = ' '.join([w.split('/')[0] for w in phrase.split()])
        cval_input_dict[lema_phrase] = phrase_freq_dict[phrase]
    return cval_input_dict


def remove_1word_candidates(phrase_freq_dict):
    """doctstring for candidate_freq_dict"""
    return dict(
        (k, v) for k, v in phrase_freq_dict.items() if len(k.split()) > 1)


def build_sorted_phrases(phrase_freq_dict):
    """docstring for build_sorted_phrases"""
    sorted_phrase_dict = defaultdict(list)
    for phrs in phrase_freq_dict.items():
        sorted_phrase_dict[len(phrs[0].split())].append(phrs)
    for num_words in sorted_phrase_dict.keys():
        sorted_phrase_dict[num_words] = sorted(sorted_phrase_dict[num_words],
                                               key=lambda item: item[1],
                                               reverse=True)
    return sorted_phrase_dict


# TODO: filter out candidates with c-value < threshold
def calc_cvalue(sorted_phrase_dict):
    """ See:
- Frantzi, Ananiadou, Mima (2000)- Automatic Recognition of Multi-Word Terms -
    the C-value-NC-value Method
- Barrón-Cedeño, Sierra, Drouin, Ananiadou (2009)- An Improved Term Recognition
    Method for Spanish"""
    cvalue_dict = {}
    triple_dict = {}  # 'candidate string': (f(b), t(b), c(b))
    max_num_words = max(sorted_phrase_dict.keys())

    # Longest candidates.
    for phrs_a, freq_a in sorted_phrase_dict[max_num_words]:
        cvalue_dict[phrs_a] = log(len(phrs_a.split()), 2) * freq_a
        for num_words in reversed(range(2, max_num_words)):
            for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                if phrs_b in phrs_a:
                    if phrs_b not in triple_dict.keys():  # create triple
                        triple_dict[phrs_b] = (freq_b, freq_a, 1)
                    else:                                 # update triple
                        fb, old_tb, old_cb = triple_dict[phrs_b]
                        triple_dict[phrs_b] = (fb, old_tb + freq_a, old_cb + 1)

    # Candidates with num. words < max num. words
    num_words_counter = max_num_words - 1
    while num_words_counter > 1:
        for phrs_a, freq_a in sorted_phrase_dict[num_words_counter]:
            if phrs_a not in triple_dict.keys():
                cvalue_dict[phrs_a] = log(len(phrs_a.split()), 2) * freq_a
            else:
                cvalue_dict[phrs_a] = log(len(phrs_a.split()), 2) * \
                    (freq_a - ((1/triple_dict[phrs_a][2])
                               * triple_dict[phrs_a][1]))
            for num_words in reversed(range(2, num_words_counter)):
                for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                    if phrs_b in phrs_a:
                        if phrs_b not in triple_dict.keys():  # create triple
                            triple_dict[phrs_b] = (freq_b, freq_a, 1)
                        else:                                 # update triple
                            fb, old_tb, old_cb = triple_dict[phrs_b]
# if/else below: If n(a) is the number of times a has appeared as nested, then
# t(b) will be increased by f(a) - n(a). Frantzi, et al (2000), end of p.5.
                            if phrs_a in triple_dict.keys():
                                triple_dict[phrs_b] = (
                                    fb,
                                    old_tb + freq_a - triple_dict[phrs_a][1],
                                    old_cb + 1)
                            else:
                                triple_dict[phrs_b] = (
                                    fb, old_tb + freq_a, old_cb + 1)
        num_words_counter -= 1

    return cvalue_dict


def main():
    """docstring for main"""

    # STEP 1.
    sents = load_tagged_sents()

    # STEP 2: Extract matching patterns above frequency threshold.
    pattern = r"""
        TC: {<NC>+<AQ>*(<PDEL><DA>?<NC>+<AQ>*)?}
        """
    phrase_freq = chunk_sents(sents, pattern, 2)

    # STEP 2: Remove chunks with words in stoplist (or binom ratio list).
    accepted_phrases = binom_ratio_filter(phrase_freq, 0.2)[0]

    # Remove 1 word chunks (necessary if pos pattern extracts 1 word chunks).
    accepted_phrases = remove_1word_candidates(accepted_phrases)

    # STEP 2: Remove POS tags from phrases.
    accepted_phrases = remove_pos_tags(accepted_phrases)

    # STEP 2: Order candidates first by number of words, then by frequency.
    sorted_phrases = build_sorted_phrases(accepted_phrases)

    # STEP 3: Calculate c-value
    cvalue_output = calc_cvalue(sorted_phrases)
    for x in sorted(cvalue_output.items(),
                    key=lambda item: item[1], reverse=True):
        print x[0], x[1]

    # TODO: test con ejemplo de juguete en Frantzi, et al (2000). Simplemente
    # por rigurosidad, ya que los resultados son iguales a los de la
    # implementación anterior, que pasaba el test.

    # TODO: implementar funciones de precisión (por trechos) y cobertura.

    # TODO: jugar con argumentos (patrón sintáctico, corte en lista de razones
    # de probabilidades binomiales, threshold de frequencia, threshold de
    # c-value, etc) y registrar valores de precisión y cobertura.


if __name__ == '__main__':
    main()
