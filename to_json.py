import json
from functools import wraps
from glob import iglob
from pprint import pprint


def writer(f):
    def wrapper(*args, **kwargs):
        _j, name = f(*args)
        print(name)
        j = json.dumps(_j, indent=4, sort_keys=True,
                       ensure_ascii=True, *kwargs)
        if not name.endswith('.jsonj'):
            name += '.json'
        with open(name, 'w') as obj:
            print(obj)
            obj.write(j)
    return wrapper


def reader(f):
    def wrapper(*args):
        _j = f(*args)
        if not _j.endswith('.json'):
            _j += '.json'
        return json.load(open(_j))
    return wrapper


@reader
def get_json(_j): return _j


@writer
def set_json(_j, name): return _j, name


def explore():
    for outJ in iglob("HCI/*/*.json"):
        _j = get_json(outJ)

        yield _j


@writer
def dup_keys():
    """
    Input : json of all excels with dictionaries of their fields and corresponding units
    Output: json dictionary of each field more than one unit definition as key and unique unit definitions as value
    """
    altJ = []
    deleteme = set()
    for _j in explore():
        for k in _j:
            altJ = [deleteme.add((k, v['units']))
                    for k, v in _j[k]['data_dict'].items()]
    zz = sorted([[k, v] for k, v in (deleteme)], key=lambda x: x[0])
    zz = [x for x in zz if [z[0] for z in zz].count(x[0]) > 1]
    index = {z: [y[1] for y in list(filter(lambda x: x[0] == z, zz))if y[1] not in ['', None, 'NA', "Not available (NA)"]]
             for z in [z[0] for z in zz]}
    pop = ([k for k in index if len(index[k]) <= 1])
    pop.extend(['ind_id'])
    ''' list of things that are synonymous or that don't affect a unifom global mapping '''

    def kill(elim): return list(map(lambda k: index.pop(k), elim))
    okay = {'county_name': 'various synonyms',
            'geoname': 'various combinations, "county" exists in all of them. If they are blank digestion should skip(Confirm with Bryce). Could conform they have the same range', 'geotype': '== geoname', 'RSE': 'variations of "NA"', 'SE': 'variations of "NA"', 'county_fips': 'standardized dimensions', 'rse': 'one describes what and the other why'}
    notNeeded = {
        'region_name': 'we are not using region, do not need to encode'}
    kill(list(okay) + list(notNeeded))
    kill(pop)
    print(len(index))
    return index, 'col_map_conflicts'


@writer
def count_common_keys():

    altJ = []

    for _j in explore():
        for k in _j:
            [altJ.append(kv) for kv in _j[k]['data_dict']]

    common = {
        'all': {k: {'count': altJ.count(k)} for k in altJ}}
    # common['25-50%'] = {k: v for k,
    # v in common['all'].items() if .5 > (v['count'] / 41) > .25}
    common['>75%'] = {k: v for k,
                      v in common['all'].items() if (v['count'] / 41) > .75}
    common['>90%'] = {k: v for k, v in common['all'].items()
                      if (v['count'] / 41) >= .9}
    return common, 'common_keys'


@writer
def organize_common_keys():
    common = get_json("common_keys")
    universal = [x for x in common['>75%']]
    all_outputs = get_json("allOutputs")
    newDict = {}

    # print(list(nest_dictionaries
    # (all_outputs, 3)))
    for d in all_outputs:
        for k, v in d.items():
            # print(k)
            # newDict[k] = {sK}
            tempDict = {}
            for sK, sV in v.items():
                # print(sK, v[sK])
                # print(sV)
                if sK == 'data_dict':
                    tempDict['dimCommon'] = {x: sV[x]
                                             for x in v[sK] if x in universal}
                    tempDict['dimSpecific'] = {x: sV[x]
                                               for x in v[sK] if x not in universal}
                elif sK == 'indicator':
                    tempDict['indicator'] = v[sK]
            newDict[k] = tempDict
        # else:
        # else:
        # print(v[sK])
    return newDict, 'allOrganized'


def nest_dictionaries(d, n):

    n -= 1
    if n == 0:
        yield d
    if isinstance(d, list):
        for i in d:
            d = yield from nest_dictionaries(i, n)
    elif isinstance(d, dict):
        for k, v in d.items():
            d = yield from nest_dictionaries(v, n)


def find_nulls():
    altJ = []
    for _j in explore():
        for k in _j:
            for kk in _j[k]['data_dict']:
                data_field = _j[k]['data_dict'][kk]
                if data_field == None:
                    print(list(_j)[0], kk)


def combine_jsons():
    altJ = []
    for _j in explore():
        altJ.append(_j)
    set_json(altJ, 'allOutputs')


def pretty(json_mappable):
    print(json.dumps(json_mappable, sort_keys=True, indent=4, ensure_ascii=False))


def show_common():
    r = get_json('common_keys')
    r.pop('all')
    pretty(r)


if __name__ == '__main__':
    # set_json(count_common_keys(), 'common_keys', sort_keys=True)
    # show_common()
    # explore()
    # count_common_keys()
    # organize_common_keys()
    dup_keys()
#
