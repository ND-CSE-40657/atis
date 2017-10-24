#!/usr/bin/env python

import sys, fileinput
import tree

for line in fileinput.input():
    try:
        t = tree.Tree.from_str(line)

        t.restore_unit()
        t.unbinarize()

        print(t)
    except:
        print("")
    
    
