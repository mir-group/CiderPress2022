import ase.io 
from io import StringIO
from collections import Counter
from ase.formula import Formula

structs = []

f = open('dat2_clean.txt', 'r')
txt = f.read()
f.close()

with open('g2_formulas.txt', 'r') as f:
    formulas = f.readlines()
    formulas = [Formula(form[:-1]).format('abc') for form in formulas]
print(formulas[0])
counter = Counter(formulas)
fobjs = [Formula(form) for form in formulas]

struct_strs = txt.split('\n\n')

matches = 0
numm = 0
good_matches = []
noformulas = []
for struct_str in struct_strs:
    name, struct_str = struct_str.split('\n', 1)
    name = name[:-4]
    if name.endswith('_s'):
        spin = 0
    elif name.endswith('_d'):
        spin = 1
    elif name.endswith('_t'):
        spin = 2
    elif name.endswith('_q'):
        spin = 3
    else:
        raise ValueError('Could not determine spin for %s' % name)
    structio = StringIO(struct_str)
    struct = ase.io.read(structio, format='xyz')
    #print(name, spin, struct)
    cformula = Formula(struct.get_chemical_formula()).format('abc')
    if Formula(cformula) in fobjs:
        numm += 1
    if cformula in counter:
        matches += counter[cformula]
        if counter[cformula] == 1:
            print('Found match {}'.format(cformula))
            good_matches.append((name, cformula))
        else:
            print('Found {} matches for {}'.format(counter[cformula], cformula))
    else:
        print('Formula not found {}'.format(cformula))
        noformulas.append(cformula)

print(matches)
print(len(good_matches))
print(len(noformulas))
print(numm)
