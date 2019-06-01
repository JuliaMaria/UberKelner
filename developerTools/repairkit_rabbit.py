# it appears that rabbit can not operate on WSAD movement coordinates.
# Due to this technical inconvienience, the repair kit presence is required.
# The sole purpose of repair kit is to translate datamodel_rabbit.txt into datamodel_rabbit_repaired.txt

from os import path
import re

if __name__ == '__main__':
    print("Repair kit: datamodel translation executed...")

    # declare translation
    moves = {
        'W': '1',
        'S': '2',
        'A': '3',
        'D': '4'
    }

    # open files
    datamodel = open(path.join('..', 'data', 'datamodel_rabbit.txt'), 'r')
    datamodel_repaired = open(path.join('..', 'data', 'datamodel_rabbit_repaired.txt'), 'w')

    # parse all lines in datamodel and save repaired translations to datamodel_repaired
    for row in datamodel:

        features = re.sub(' +', ' ', row[1:])
        # translate movement
        datamodel_repaired.write(moves.get(row[0]) + features)

    # close files
    datamodel_repaired.close()
    datamodel.close()

    print("Repair kit: datamodel translation completed.")
