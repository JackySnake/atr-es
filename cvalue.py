#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from collections import defaultdict
from math import log
import nltk
import binom


def load_tagged_sents():
    """docstring for load_tagged_sents"""
    #with open('input/small_domain.txt', 'r') as corpf:  # small corpus
    with open('input/big_domain.txt', 'r') as corpf:
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


def remove_str_postags(tagged_str):
    """docstring for remove_str_postags"""
    stripped_str = ' '.join([w.split('/')[0] for w in tagged_str.split()])
    return stripped_str


# Remove POS tags from phrases (keys) and return dict.
def remove_dict_postags(phrase_freq_dict):
    """docstring for remove_dict_postags"""
    new_dict = {}
    for phrase in phrase_freq_dict.keys():
        new_str = remove_str_postags(phrase)
        new_dict[new_str] = phrase_freq_dict[phrase]
    return new_dict


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


def calc_cvalue(sorted_phrase_dict, min_cvalue):
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
        cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
        if cvalue >= min_cvalue:
            cvalue_dict[phrs_a] = cvalue
            for num_words in reversed(range(1, max_num_words)):
                for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                    if phrs_b in phrs_a:
                        if phrs_b not in triple_dict.keys():  # create triple
                            triple_dict[phrs_b] = (freq_b, freq_a, 1)
                        else:                                 # update triple
                            fb, old_tb, old_cb = triple_dict[phrs_b]
                            triple_dict[phrs_b] = \
                                (fb, old_tb + freq_a, old_cb + 1)

    # Candidates with num. words < max num. words
    num_words_counter = max_num_words - 1
    while num_words_counter > 0:
        for phrs_a, freq_a in sorted_phrase_dict[num_words_counter]:
            if phrs_a not in triple_dict.keys():
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            else:
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * \
                    (freq_a - ((1/triple_dict[phrs_a][2])
                               * triple_dict[phrs_a][1]))
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            if cvalue >= min_cvalue:
                for num_words in reversed(range(1, num_words_counter)):
                    for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                        if phrs_b in phrs_a:
                            if phrs_b not in triple_dict.keys():  # make triple
                                triple_dict[phrs_b] = (freq_b, freq_a, 1)
                            else:                                 # updt triple
                                fb, old_tb, old_cb = triple_dict[phrs_b]
# if/else below: If n(a) is the number of times a has appeared as nested, then
# t(b) will be increased by f(a) - n(a). Frantzi, et al (2000), end of p.5.
                                if phrs_a in triple_dict.keys():
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a -
                                        triple_dict[phrs_a][1], old_cb + 1)
                                else:
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a, old_cb + 1)
        num_words_counter -= 1

    return cvalue_dict


def load_reference():
    """docstring for load_reference"""
    #with open('input/small_terms.txt', 'r') as rf:  # small corpus
    with open('input/big_terms.txt', 'r') as rf:
        ref_raw = rf.read().decode('utf-8')
    ref_list = ref_raw.split('\n')
    ref_list = [remove_str_postags(i.strip()) for i in ref_list]
    return ref_list


def generate_bins(item_list, num_bins):
    """docstring for generate_bins"""
    start_index = 0
    for bin_num in xrange(num_bins):
        end_index = start_index + len(item_list[bin_num::num_bins])
        yield item_list[start_index:end_index]
        start_index = end_index


def precision_recall_stats(reference_list, sorted_test_list, num_bins):
    """docstring for precision_recall_stats"""
    ref_set = set(reference_list)
    test_set = set(sorted_test_list)
    print 'Candidates:', len(test_set)
    print 'Real Terms:', len(test_set.intersection(ref_set))
    print '=' * 4

    for segment in generate_bins(sorted_test_list, num_bins):
        seg_pval = nltk.metrics.precision(ref_set, set(segment))
        print round(seg_pval, 2)
    print '=' * 4

    pval = nltk.metrics.precision(ref_set, test_set)
    rval = nltk.metrics.recall(ref_set, test_set)
    print round(pval, 2)
    print round(rval, 2)


def make_contextword_weight_dict(real_term_list, tagged_sents, valid_tags,
                                 context_size):
    """docstring for make_contextword_weight_dict"""
    # if context_size = 5, does that mean that it's 5 words to the left and 5
    # words to the right, or 2 words to left and 2 to the right?
    #context_size = int(context_size/2)
    context_word_dict = defaultdict(int)
    num_terms_seen = 0
    for term in real_term_list:
        for sent in tagged_sents:
            sent_str = ' '.join(w[0] for w in sent)
            if term in sent_str:
                term_split = term.split()
                for wt_idx in range(len(sent) - len(term_split)):
                    # wt_idx = wordtag_index.
                    word_size_window = [
                        w[0] for w in
                        sent[wt_idx:wt_idx+len(term_split)]]
                    if term_split == word_size_window:
                        left_context = sent[:wt_idx][-context_size:]
                        right_context = \
                            sent[wt_idx+len(term_split):][:context_size]
                        # TODO: cut at punctuation marks after which context
                        # words are not related to term. e.g:
                        # left_context = left_context.split(.,;:()etc)[-1]
                        # right_context = right_context.split(.,;:()etc)[0]
                        context = left_context + right_context
                        valid_words = [w[0] for w in context if
                                       w[1].lower() in valid_tags]
                        for word in valid_words:
                            context_word_dict[word] += 1
                        num_terms_seen += 1
                        break  #  1 term match per sentence
    context_word_dict = dict(  # Transform keys: freqs -> weights
        (k, v/num_terms_seen) for k, v in context_word_dict.items())
    return context_word_dict


def calc_ncvalue(cvalue_dict, tagged_sents, contextword_weight_dict,
                 valid_tags, context_size):
    """docstring for calc_ncvalue"""
    # if context_size = 5, does that mean that it's 5 words to the left and 5
    # words to the right, or 2 words to left and 2 to the right?
    #context_size = int(context_size/2)
    ncvalue_dict = {}
    for candidate in cvalue_dict.keys():
        ccw_freq_dict = defaultdict(int)  # ccw = candidate_context_words
        for sent in tagged_sents:
            sent_str = ' '.join(w[0] for w in sent)
            if candidate in sent_str:
                candidate_split = candidate.split()
                for wt_idx in range(len(sent) - len(candidate_split)):
                    word_size_window = [
                        w[0] for w in
                        sent[wt_idx:wt_idx+len(candidate_split)]]
                    if candidate_split == word_size_window:
                        left_context = sent[:wt_idx][-context_size:]
                        right_context = \
                            sent[wt_idx+len(candidate_split):][:context_size]
                        # TODO: see same bit in previous function.
                        context = left_context + right_context
                        valid_words = [w[0] for w in context if
                                       w[1].lower() in valid_tags]
                        for word in valid_words:
                            ccw_freq_dict[word] += 1
                        break  # 1 candidate match per sentence
        context_factors = []
        for word in ccw_freq_dict.keys():
            if word in contextword_weight_dict.keys():
                context_factors.append(
                    ccw_freq_dict[word] * contextword_weight_dict[word])
        ncvalue = (0.8 * cvalue_dict[candidate]) + (0.2 * sum(context_factors))
        ncvalue_dict[candidate] = ncvalue
    return ncvalue_dict


def run_experiment(phrase_pattern, min_freq, binom_cutoff,
                   min_cvalue, num_bins):
    """docstring for run_experiment"""
    # STEP 1.
    sents = load_tagged_sents()
    # STEP 2: Extract matching patterns above frequency threshold.
    phrase_freq = chunk_sents(sents, phrase_pattern, min_freq)
    # STEP 2: Remove chunks with words in stoplist (or binom ratio list).
    accepted_phrases = binom_ratio_filter(phrase_freq, binom_cutoff)[0]
    # STEP 2: Remove POS tags from phrases.
    accepted_phrases = remove_dict_postags(accepted_phrases)
    # STEP 2: Order candidates first by number of words, then by frequency.
    sorted_phrases = build_sorted_phrases(accepted_phrases)
    # STEP 3: Calculate c-value
    cvalue_output = calc_cvalue(sorted_phrases, min_cvalue)

    reference = load_reference()
    sorted_cval = sorted(
        cvalue_output.items(), key=lambda item: item[1], reverse=True)
    test = [k for k, v in sorted_cval]
    print phrase_pattern.strip()
    print min_freq, binom_cutoff, min_cvalue
    stats = precision_recall_stats(reference, test, num_bins)
    print

    cval_top_x_percent = 0.25
    cval_top = [cand_term for cand_term, cand_freq in
                sorted_cval[0:int(len(sorted_cval) * cval_top_x_percent)]]
    valid_postags = ['nc', 'aq', 'vm']
    context_word_weights = make_contextword_weight_dict(
        cval_top, sents, valid_postags, 5)
    ncvalue_output = calc_ncvalue(
        cvalue_output, sents, context_word_weights, valid_postags, 5)
    sorted_ncvalue = sorted(
        ncvalue_output.items(), key=lambda item: item[1], reverse=True)
    test = [k for k, v in sorted_ncvalue]
    stats = precision_recall_stats(reference, test, num_bins)
    print


def main():
    """docstring for main"""

    pattern = r"""
        TC: {<NC>+<AQ>*(<PDEL><DA>?<NC>+<AQ>*)?}
        """
    min_freq = 1
    binom_cutoff = 0.0
    min_cvalue = 5.0
    num_bins = 4

    run_experiment(pattern, min_freq, binom_cutoff,
                   min_cvalue, num_bins)

    # curiosamente, jugando con los valores de min_freq, binom_cutoff y
    # min_cvalue, frecuentemente se ve mejor precisión en los tramos primeros
    # y últimos del output de cvalue. ¿por qué? ¿cómo afectan las
    # preposiciones, y qué pasa si de entrada las elimino?

    # TODO: test con ejemplo de juguete en Frantzi, et al (2000). Simplemente
    # por rigurosidad, ya que los resultados son iguales a los de la
    # implementación anterior, que pasaba el test.

    # TODO: jugar con argumentos (patrón sintáctico, corte en lista de razones
    # de probabilidades binomiales, threshold de frequencia, threshold de
    # c-value, etc) y registrar valores de precisión y cobertura.

    # TODO: hacer función que haga la combinación de diferentes valores de
    # min_freq, binom_cutoff, min_cvalue

    # TODO: revisar patrones POS de términos validados y ver si se puede
    # jugar con el patrón de chunkeo.


if __name__ == '__main__':
    main()
