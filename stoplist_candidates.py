#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import defaultdict


def load_ref():
    with open('input/big_reference.txt', 'r') as reff:
        ref_raw = reff.read().decode('utf-8')
    ref_freq_dict = defaultdict(int)
    for word in ref_raw.split():
        if word.split('/')[1] in ('NC', 'AQ'):
            ref_freq_dict[word] += 1
    return ref_freq_dict


def sorted_freq_dict(freq_dict):
    """doctstring for sorted_freq_dict"""
    return sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)


if __name__ == '__main__':
    freq_dict = load_ref()
    sorted_fd = sorted_freq_dict(freq_dict)
    cutoff = int(0.1 * len(sorted_fd))
    top_ten_percent = sorted_fd[:cutoff]
    with open('stoplist_candidates.txt', 'a') as text_file:
        for i in top_ten_percent:
            text_file.write(str(i[1]) + '\t' + i[0].encode('utf-8') + '\n')
