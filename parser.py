import torch
import layers
import trees
import collections
import random
import copy

# If installed, this prints progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

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

        # Parameters for encoder
        self.emb = layers.Embedding(len(self.vocab_index), dims)
        self.rnn1 = layers.RNN(dims)

        # Parameters for rule model
        self.out1 = layers.LinearLayer(2*dims, dims)
        self.out2 = layers.LinearLayer(dims, len(rules))

    def encode(self, words):
        """Encode a string as a sequence of vectors.
        
        Argument:
        - words (list of n strs): input string
        
        Return:
        - tensor of size n,d: encoding of input string
        """

        # numberize words
        unk = self.vocab_index['<unk>']
        nums = torch.tensor([self.vocab_index.get(w, unk) for w in words])

        # look up word embeddings
        emb = self.emb(nums)

        # run RNN
        enc = self.rnn1.sequence(emb)
        
        return enc

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
        
        if (lhs, rhs) not in self.rule_index: return 0.
        
        # Most papers take the difference of the first and last word,
        # but we're concatenating them.
        c = torch.cat([enc[i], enc[j-1]])

        # Two-layer feedforward network. There's no softmax at the end,
        # since the rule score is not a probability.
        h = torch.tanh(self.out1(c))
        o = self.out2(h)
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

        raise NotImplementedError()

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--dev', dest='dev', type=str)
    parser.add_argument('--load', dest='load', type=str)
    parser.add_argument('--save', dest='save', type=str)
    parser.add_argument('infile', nargs='?', type=str, help='test data to parse')
    parser.add_argument('-o', '--outfile', type=str, help='write parses to file')
    args = parser.parse_args()
    
    if args.train:
        traintrees = [trees.Tree.from_str(line) for line in open(args.train)]

        # Construct PCFG rules
        rules = set()
        vocab = set()
        for t in traintrees:
            for node in t.bottomup():
                if len(node.children) > 0:
                    rhs = tuple(child.label for child in node.children)
                    rules.add((node.label, rhs))
                else:
                    vocab.add(node.label)

        m = Model(rules, vocab, 256)
    
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        devtrees = [trees.Tree.from_str(line) for line in open(args.dev)]
        
    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        o = torch.optim.Adam(m.parameters(), lr=0.0003)

        prev_dev_loss = best_dev_loss = None

        for epoch in range(10):
            train_loss = 0
            random.shuffle(traintrees)
            m.train()
            for tree in tqdm(traintrees):
                words = [node.label for node in tree.leaves()]
                tree_score = m.score_tree(tree)
                z = m.parse(words, mode='sum')
                loss = -(tree_score-z)
                o.zero_grad()
                loss.backward()
                o.step()
                train_loss += loss.item()

            dev_loss = 0
            dev_failed = 0
            m.eval()
            for ti, tree in enumerate(devtrees):
                words = [node.label for node in tree.leaves()]
                tree_score = m.score_tree(tree)
                z = m.parse(words, mode='sum')
                if z is None:
                    dev_failed += 1
                    continue
                if ti == 1:
                    t = m.parse(words, mode='max')
                    print(t.pretty_print())
                loss = -(tree_score-z)
                dev_loss += loss.item()
                
            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                best_dev_loss = dev_loss
                print('saving new best model', file=sys.stderr)
                if args.save:
                    torch.save(m, args.save)
            prev_dev_loss = dev_loss

            print(f'train_loss={train_loss} dev_loss={dev_loss} dev_failed={dev_failed}', file=sys.stderr)
            
        m = best_model

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for line in open(args.infile):
                words = line.split()
                t = m.parse(words, mode='max')
                if t is not None:
                    print(t, file=outfile)
                else:
                    print(file=outfile)
