import sys

# If running on Kaggle, the following lines will download and preprocess the data files.
#!rm -rf hw3
#!git clone https://github.com/ND-CSE-40657/hw3 hw3
#!python hw3/preprocess.py < hw3/data/train.trees > hw3/data/train.trees.pre
#!python hw3/unknown.py < hw3/data/train.trees.pre > hw3/data/train.trees.pre.unk
#!python hw3/preprocess.py < hw3/data/dev.trees > hw3/data/dev.trees.pre
#sys.path.append('hw3')

import torch
import layers
import trees
import evalb
import collections
import random
import copy
import os

# Directories on Kaggle
#datadir = 'hw3/data'
#outdir = '/kaggle/working'

# Local directories
datadir = 'data'
outdir = '.'

trainfile = os.path.join(datadir, 'train.trees.pre.unk')
devfile = os.path.join(datadir, 'dev.trees.pre')
testfile = os.path.join(datadir, 'test.strings')
outfile = os.path.join(outdir, 'test.parsed')

class ParseFailure(Exception):
    pass

class FFN(torch.nn.Module):
    def __init__(self, idims, hdims, odims, residual=True):
        super().__init__()
        self.lin1 = layers.LinearLayer(idims, hdims)
        self.lin2 = layers.LinearLayer(hdims, odims)
        self.residual = residual
        
    def forward(self, inp):
        hid = torch.relu(self.lin1(inp))
        out = self.lin2(hid)
        if self.residual:
            return inp + out
        else:
            return out

class MHSelfAttentionLayer(torch.nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, nheads, dims):
        super().__init__()
        self.heads = torch.nn.ModuleList([layers.SelfAttentionLayer(dims) for h in range(nheads)])
        
    def forward(self, inp):
        return sum([h(inp) for h in self.heads]) / len(self.heads)
        
class Model(torch.nn.Module):
    """Neural parsing model."""
    def __init__(self, rules, vocab, dims):
        super().__init__()

        # CFG rules and mapping to numbers
        self.rules = rules
        self.rule_index = {r:i for (i,r) in enumerate(rules)}

        # Vocabulary (terminal alphabet) and mapping to numbers
        self.vocab = vocab
        self.vocab_index = {w:i for (i,w) in enumerate(self.vocab)}

        # Word embeddings
        self.wemb = layers.Embedding(len(self.vocab_index), dims)
        # Position embeddings
        self.pemb = layers.Embedding(100, dims)

        # Parameters for transformer encoder
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4*dims, dims),
            torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4*dims, dims),
            torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4*dims, dims),
            torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4*dims, dims),
            torch.nn.Dropout(0.1))
        
        # Parameters for output layer
        self.out = FFN(2*dims, dims, len(rules), residual=False)

    def encode(self, words):
        """Encode a string as a sequence of vectors.
        
        Argument:
        - words (list of n strs): input string
        
        Return:
        - tensor of size n,d: encoding of input string
        """

        # numberize words
        nums = torch.tensor([self.vocab_index[w] for w in words])
        
        # positions
        pos = torch.arange(len(nums))
        
        # embed
        emb = self.wemb(nums) + self.pemb(pos)

        # run layers
        return self.layers(emb)

    def score_rule(self, enc, lhs, rhs, i, j):
        """Compute the log-weight of a rule in context.

        Arguments:

        - enc (tensor of size n,d): encoding of input string, as
          returned by Model.encode()
        - lhs (str): rule left-hand side
        - rhs (tuple of str): rule right-hand side
        - i, j (int in [0,n)): start and end of substring spanned by lhs

        Return:
        - float: rule log-weight

        """
        
        if (lhs, rhs) not in self.rule_index: 
            raise ParseFailure("rule not in grammar")
        
        # Most papers take the difference of the first and last word,
        # but we're concatenating them.
        c = torch.cat([enc[i], enc[j-1]])
        
        # Two-layer feedforward network. There's no softmax at the end,
        # since the rule score is not a probability.
        o = self.out(c)
        r = self.rule_index[lhs, rhs]
        return o[r]

    def score_tree(self, tree):
        """Compute the log-weight of a tree.

        Argument:
        - tree (Tree): tree

        Return:
        - float: tree log-weight (sum of rule log-weights)
        """
        
        def visit(node, i):
            score = 0
            j = i
            if len(node.children) > 0:
                for child in node.children:
                    child_score, j = visit(child, j)
                    score += child_score
                rhs = tuple(child.label for child in node.children)
                score += self.score_rule(enc, node.label, rhs, i, j)
            else:
                j += 1
            return score, j
        enc = self.encode([leaf.label for leaf in tree.leaves()])
        score, _ = visit(tree.root, 0)
        return score

    def parse(self, words, mode='max'):
        """Parse a string. This function has two modes: In 'max' mode, it
        finds and returns the highest-weight tree. In 'sum' mode, it
        computes the log of the total weight of all trees.

        Arguments:
        - words (list of n strs): input string
        - mode (str): either 'max' or 'sum

        Return:
        - If mode == 'max': The highest-weight tree
        - If mode == 'sum': The log of the total weight of all trees

        Notes: 

        - The weight of a tree is the product of the weights of the
          rules in it.  
        - Since the weights can get big/small, we store them as log-weights.
        - Use torch.logaddexp() to add weights that are stored as log-weights.
        - Call Model.score_rule() to compute the log-weight of a rule.
        """
        
        raise ParseFailure()

traintrees = [trees.Tree.from_str(line) for line in open(trainfile)]
devtrees = [trees.Tree.from_str(line) for line in open(devfile)]

# Construct PCFG rules and vocab
rules = set()
vocab = set()
for t in traintrees:
    for node in t.bottomup():
        if len(node.children) > 0:
            rhs = tuple(child.label for child in node.children)
            rules.add((node.label, rhs))
        else:
            vocab.add(node.label)

# Change unknown words to <unk>
for t in devtrees:
    for n in t.leaves():
        if n.label not in vocab:
            n.label = "<unk>"
            
m = Model(rules, vocab, 256)
            
o = torch.optim.Adam(m.parameters(), lr=0.0003)

prev_dev_loss = best_dev_loss = None

for epoch in range(10):
    train_loss = 0
    random.shuffle(traintrees)
    m.train()
    for tree in traintrees:
        words = [node.label for node in tree.leaves()]
        try:
            tree_score = m.score_tree(tree)
            z = m.parse(words, mode='sum')
        except ParseFailure:
            continue
        loss = -(tree_score-z)
        o.zero_grad()
        loss.backward()
        o.step()
        train_loss += loss.item()

    dev_loss = 0
    dev_failed = 0
    m.eval()
    devparses = []
    for ti, tree in enumerate(devtrees):
        words = [node.label for node in tree.leaves()]
        try:
            tree_score = m.score_tree(tree)
            z = m.parse(words, mode='sum')
            loss = -(tree_score-z)
            dev_loss += loss.item()
        except ParseFailure:
            pass
            
        try:
            t = m.parse(words, mode='max')
            devparses.append(t)
            if ti == 1:
                print(t.pretty_print())
        except ParseFailure:
            # make a fall-back tree
            n = trees.Node('TOP', [trees.Node('UH', [trees.Node(w)]) for w in words])
            devparses.append(trees.Tree(n))
            dev_failed += 1
            
    dev_match, dev_goldcount, dev_parsecount = evalb.score(devtrees, devparses)
    # You can also print out the dev F1 score if you want; however, be aware that without running postprocess.py, the scores will be lower.
    dev_f1 = 2*dev_match/(dev_goldcount + dev_parsecount)
    print(f'train_loss={train_loss} dev_loss={dev_loss} dev_failed={dev_failed}', file=sys.stderr)

    if best_dev_loss is None or dev_loss < best_dev_loss:
        best_model = copy.deepcopy(m)
        print('saving new best model', file=sys.stderr)
        torch.save(m, os.path.join(outdir, f'model{epoch+1}.pt'))
        best_dev_loss = dev_loss
        
    if prev_dev_loss is not None and dev_loss > prev_dev_loss:
        print('halving learning rate', file=sys.stderr)
        o.param_groups[0]['lr'] /= 2
    prev_dev_loss = dev_loss
        
m = best_model

with open(outfile, 'w') as f:
    for line in open(testfile):
        words = line.split()
        words = [w if w in vocab else '<unk>' for w in words]
        try:
            t = m.parse(words, mode='max')
        except ParseFailure:
            # make a fake tree
            n = trees.Node('TOP', [trees.Node('UH', [trees.Node(w)]) for w in words])
            t = trees.Tree(n)
        print(t, file=f)
