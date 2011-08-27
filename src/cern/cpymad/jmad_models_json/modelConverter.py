
import json

# some fixed conversions:
_VALUE_MAP = {'true' : True, 'false' : False}

def _convert_key(key):
    if key.startswith('@'):
        return key[1:]
    else:
        return key

def _try_float_convert(value):
    try:
        return float(value)
    except:
        return None

def _convert_value(value):
    if _VALUE_MAP.has_key(value):
        return _VALUE_MAP[value]
    
    newval = _try_float_convert(value)
    if not newval == None:
        return newval
    
    return value

def _convert_recursively(item):
    if  isinstance(item, list):
        newlist = []
        for nextItem in item:
            newlist.append(_convert_recursively(nextItem));
        return newlist
    elif isinstance(item, dict):
        newdict = {}    
        for key, value in item.items():
            newdict[_convert_key(key)] = _convert_recursively(value)
        return newdict
    else:
        return _convert_value(item)
        
        
def convert_dict(indict):
    '''
    converts a jmad-model definition to one which is more nicely readble from cpymad
    '''
    return _convert_recursively(indict)

def convert_file(infilename, outfilename):
    indict = json.loads(file(infilename, 'r').read())
    outdict = convert_dict(indict);
    file(outfilename, 'w').write(json.dumps(outdict))

if __name__ == "__main__":
    print "Converting lhc"
    convert_file('lhc-1.jmd.json', 'lhc-1-out.jmd.json')
    
