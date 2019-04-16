"""
Defines two dictionaries for converting 
between text and integer sequences.
"""

char_map_str = """
' 0
<SPACE> 1
aa 2
ae 3
ah 4
ao 5
aw 6
ax 7
axh 8
axr 9
ay 10
bcl 11
ch 12
d 13
dcl 14
dh 15
dx 16
eh 17
el 18
em 19
en 20
eng 21
epi 22
er 23
ey 24
f 25
g 26
gcl 27
hh 28
hv 29
ih 30
ix 31
iy 32
jh 33
k 34
kcl 35
l 36
m 37
n 38
ng 39
nx 40
ow 41
oy 42
p 43
pau 44
pcl 45
q 46
r 47
s 48
sh 49
t 50
tcl 51
th 52
uh 53
uw 54
ux 55
v 56
w 57
y 58
z 59
zh 60
h 61
b 62
"""
# the "blank" character is mapped to 28

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch
index_map[2] = ' '