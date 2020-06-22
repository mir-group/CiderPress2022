import re

f = open('dat2.txt', 'r')
txt = f.read()
f.close()

final_match = 'Atom         x                y                z           '
#struct_files = re.split('={21}  [A-Za-z0-9]*?_.\.xyz  ={22}\n' + final_match + '\n', txt)[1:]
struct_files = re.split('={21}  [\-A-Za-z0-9]*?_.\.xyz  ={22}\n', txt)[1:]
struct_names = re.findall('[\-A-Za-z0-9]*?_.\.xyz', txt)
print(len(struct_names), len(struct_files))

clean_structs = []

full_str = ''

for i, struct in enumerate(struct_files):
    p = re.compile('\n \n(\n)*')
    struct = p.sub('', struct)
    size = len(struct.split('\n')) - 1
    struct = ('%d  %s  %s\n%d\n' % (i, struct_names[i], size)) + struct + '\n\n'
    full_str += struct

f = open('dat2_numbered.txt', 'w')
f.write(full_str)
f.close()
