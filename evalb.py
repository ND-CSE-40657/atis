#!/usr/bin/env python

import collections
import trees

def _brackets_helper(node, i, result):
    i0 = i
    if len(node.children) > 0:
        for child in node.children:
            i = _brackets_helper(child, i, result)
        j0 = i
        if len(node.children[0].children) > 0: # don't count preterminals
            result[node.label, i0, j0] += 1
    else:
        j0 = i0 + 1
    return j0

def brackets(t):
    result = collections.defaultdict(int)
    _brackets_helper(t.root, 0, result)
    return result

def score(ts1, ts2):
    c1 = c2 = m = 0
    if len(ts1) != len(ts2):
        raise ValueError("two lists of trees should have same length")
    for t1, t2 in zip(ts1, ts2):
        b1 = brackets(t1)
        b2 = brackets(t2)
        c1 += sum(b1.values())
        c2 += sum(b2.values())
        for b,c in b1.items():
            m += min(c, b2[b])
    return (m, c1, c2)

if __name__ == "__main__":
    import sys

    try:
        _, parsefilename, goldfilename = sys.argv
    except:
        sys.stderr.write("usage: evalb.py <parse-file> <gold-file>\n")
        sys.exit(1)

    parsetrees = []
    goldtrees = []
    for parseline, goldline in zip(open(parsefilename), open(goldfilename)):
        goldtree = trees.Tree.from_str(goldline)
        goldtrees.append(goldtree)
        if parseline.strip() == '':
            # Make a fake tree that at least gets TOP correct
            parsetree = trees.Tree(trees.Node('TOP', [trees.Node('UH', [trees.Node(w)]) for w in goldtree.leaves()]))
        else:
            parsetree = trees.Tree.from_str(parseline)
        parsetrees.append(parsetree)

    matchcount, parsecount, goldcount = score(parsetrees, goldtrees)

    print(f"{parsefilename}\t{parsecount} brackets")
    print(f"{goldfilename}\t{goldcount} brackets")
    print(f"matching\t{matchcount} brackets")
    print(f"precision\t{matchcount/parsecount}")
    print(f"recall\t{matchcount/goldcount}")
    print(f"F1\t{2*matchcount/(goldcount + parsecount)}")
