'''
vowpal wabbit
adam & julia

model verbal vomit:

#ruchu w rozwiązaniu (1- prawo, 2-lewo, 3-góra, 4-dół) | stan otoczenia (0-blank, 1-wall, 20 - furnace inactive,  21-furnace active, 30-table inactive, 31-table active)

przykład:

dla otoczenia
XXX
XW_
XFX

datalog:

2 | cellleft: 1 celldown: 20 cellup: 1 cellright: 0
lub jeśli kuchnia jest aktywna
4 | cellleft: 1 celldown: 21 cellup: 1 cellright: 0

można zwiększyć skanowany obszar o dodatkowy kwadrat, np:
XXXXX
XXXXX
XXW__
XXFXX
XXXXX

'''

# pvomit parser